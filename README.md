# Curvemetrics

Welcome to Curvemetrics, a stablecoin and liquid staking derivative depeg detector for Curve's StableSwap pools. Curvemetrics consists of two primary components:

1. Metrics - Our metrics are designed to capture *leading* indicators of potential depegs.
2. Bayesian Online Changepoint Detection - Our *BOCD* models are trained on historical Curve data, and listen to changes in metrics data in *real-time* to detect potential changepoints (depegs).

We have exposed our metrics and corresponding raw data to an HTTP port for anyone to query (more details below). You may similarly listen to detected changepoints using our API, by following and turning notification on for our Twitter bot **link**, or signing up for **these** Telegram channels. The theoretical underpinning of Curvemetrics is available in our corresponding research paper **link**.

This codebase and research paper were developed by Xenophon Labs and sponsored by the Cuve Analytics team.

# Introduction

# Metrics

# Setup

To set up Curvemetrics on your own VM, follow the steps below.

## Requirements

First, activate the venv and download the requirements:

```
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Database Design

We use a PostgreSQL database, which we interact with using SQLAlchemy. Setting up PostgreSQL (https://www.digitalocean.com/community/tutorials/how-to-install-postgresql-on-ubuntu-20-04-quickstart):

```sudo apt-get install postgresql```
```sudo systemctl start postgresql.service```
```sudo -u postgres psql```

We use a DataWarehouse design with a few ``star`` tables and two ``dimension`` tables. Our star tables correspond to numeric data that we track ("facts"). We track raw data and metrics data as our facts. Our dimension tables store metadata on our "facts"; we have a `pools` dimension table and a `tokens` dimension table. The `pools` table tells us the name, symbol, address, etc., of each pool, as well as which tokens it holds. Similarly, our `tokens` table tells us metadata about each token. Our ``star`` tables reference our ``dimensions`` tables.

We have the following ``star`` tables, which track relevant numeric data:

- ``pool_data`` - This table (indexed by Messari's subgraph) gives us block-level data on the pool's reserves (i.e. how much of the pool is made up of DAI vs USDC). We use this data to construct entropy and inequality measurements on each pool.
- ``lp_events`` - This table (indexed by Convex-community's subgraph) gives us all the deposits and withdraws for any of the supported pools.
- ``swaps`` - This table (indexed by Convex-community's subgraph) gives us all the swaps for any of the supported pools.
- ``token_ohlcv`` - This table (pulled from CCXT exchagnes or Chainlink aggregators) gives us the price of each token over time. Data from CCXT is pulled on a minutely granularity, data from Chainlink depends on the update-frequency of the aggregator.
- ``snapshots`` - This table (indexed by Convex-community's volume subgraph) gives us the dailySnapshots on each Curve pool (particularly, we care about the virtual price).
- ``pool_metrics`` - This table serves all of the computed metrics for each Curve pool, described in the Metrics section.
- ``token_metrics`` - This table serves all of the computed metrics for each relevant token, described in the Metrics section.
- ``blockTimestamps`` - This table helps us by relating blocks to timestamps historically (so we don't blow up our RPC keys!).
- ``changepoints`` - This table holds the timestamps at which each model detected a changepoint for each pool.
- ``takers`` - This table holds information on all addresses that have submitted swaps to the supported pools on Curve, the amount they bought and sold, and their cumulative/mean 1d markouts. 

Each table is defined as a Python class (called an Entity), and created using the SQLAlchemy `declarative_base` class. See the Figma below for a better understanding of our database design.

<Figma!>

## Database and Table creation

First create the database with its corresponding user:

```CREATE USER <user> WITH PASSWORD '<pwd>';```
```ALTER USER <user> CREATEDB;```
```psql -U <user> -h localhost -d postgres```
```CREATE DATABASE <user>;```

<user> and <pwd> should be defined as 'PSQL_USER' and 'PSQL_PASSWORD' in your `.env`, since SQLAlchemy and Flask will need those to serve the data. Then run the below command from the root of the repository:

```
python3 -m curvemetrics.create_database
```

This calls the `Base.MetaData.create_all(.)` method from SQLAlchemy's `declarative_base` class, which loads all of the Entities into our db and creates/typesets the tables, as well as their indexes.

## Backfilling

Setting up the SQL database with backfilled raw data can take a while (on the order of a couple days). Backfilling involves three sequential steps:

1. Backfill the raw data.
2. Compute and backfill metrics.
3. Compute and backfill the `takers` table

Since metrics and the `takers` table require raw data to be filled before they can be computed, we must first backfill the raw data (this is the slow step). You can run the job to do so in the background with the following command:

```
nohup bash curvemetrics/scripts/backfill/batch_backfill_raw_data.sh <start> <end> > ./logs/raw.log 2>&1 &
```

Once this is completed (or has progressed enough for us to backfill some of the metrics), run:

```
nohup bash curvemetrics/scripts/backfill/batch_backfill_metrics.sh <start> <end> > ./logs/metrics.log 2>&1 &
```

And finally: 

```
nohup python3 -m curvemetrics.scripts.backfill.takers.py <start> <end> > ./logs/takers.log 2>&1 &
```

You can verify if any job failed for any date with:

```
find ./logs -type f -exec grep -L "Done :)" {} \; | sort
```

Which checks for our `Done :)` exit status. We split the backfilling jobs up into weekly/monthly jobs in the `.sh` scripts to prevent OOM errors. You can confirm that tables have been backfilled by using Postgres' interactive CLI:

```
psql -U <user> -h localhost
```

For example:

```
SELECT * FROM swaps LIMIT 10;
```

Or by looking at the example notebook in `./notebooks/analysis.ipynb`.

Warning: if you have the free Infura plan with 100,000 requests, you might hit your daily limit after backfilling for a few months. We use Infura to get Chainlink prices and query block timestamps (most block timestamps since 2022 are indexed in the `data/` directory as `.csv`s).

## Frontfilling

Frontfilling was set up to run using an hourly cron job. The job is set up using `crontab -e` to fill data every hour:

```
0 * * * * /usr/bin/bash /root/curve-lp-metrics/curvemetrics/scripts/frontfill/frontfill.sh
```

This job will perform the following operations in the following order:

1. Query the last hour of raw data and save it to our database.
2. Read the raw data and compute metrics, save them to our db.
3. Update the `takers` table and compute latest `sharkFlow`.

We have a 5 minute buffer we use to ensure that we are not losing any data (e.g. due to latency from theGraph).

## Resetting
If there was an issue with the database, you can drop all the tables and vacuum the database by running:

```
python3 -m curvemetrics.reset
```

# Flask App

Set up using nginx and supervisor in Ubuntu.

The Flask App uses SQLAlchemy to query our Postgres database.

Example query:

```
curl "http://172.104.8.91/pool_data?pool_id=0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7&start=1640995203&end=1641995203" > example.json
```

nginx setup
```
# curvemetrics config
server {
        listen 80;
        server_name 172.104.8.91;
        location / {
                include proxy_params;
                proxy_pass http://localhost:5000;
        }
}
```

supervisor setup
```
[program:curvemetrics]
command=/root/curve-lp-metrics/venv/bin/python3 -m curvemetrics.app.app
directory=/root/curve-lp-metrics
user=root
autostart=true
autorestart=true
redirect_stderr=true
```