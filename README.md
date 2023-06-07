# curve-lp-metrics
Regime shift detection metrics for Curve liquidity providers.

First, activate the venv and download the requirements:

```
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

# Dependencies
Don't need R

# Metrics

<Describe metrics>

# Data Gathering

## Database Design

Setting up PostgreSQL: https://www.digitalocean.com/community/tutorials how-to-install-postgresql-on-ubuntu-20-04-quickstart

```sudo apt-get install postgresql```
```sudo systemctl start postgresql.service```
```sudo -u postgres psql```
```CREATE USER <user> WITH PASSWORD '<pwd>';```
```ALTER USER <user> CREATEDB;```
```psql -U <user> -h localhost -d postgres```
```CREATE DATABASE <user>;```

<user> and <pwd> should be defined as 'PSQL_USER' and 'PSQL_PASSWORD' in your `.env`.

We use a DataWarehouse design with a few ``star`` tables and two ``dimension`` tables. Our star tables correspond to numeric data that we track ("facts"). We track raw data and metrics data. Our dimension tables store metadata on our "facts"; we have a `pools` dimension table and a `tokens` dimension table. The `pools` table tells us the name, symbol, address, etc., of each pool, as well as which tokens it holds. Similarly, our `tokens` table tells us metadata about each token.

We have the following ``star`` tables, which track relevant numeric data:

- ``pool_data`` - This table (indexed by Messari's subgraph) gives us block-level data on the pool's reserves (i.e. how much of the pool is made up of DAI vs USDC). We use this data to construct entropy and inequality measurements on each pool.
- ``lp_events`` - This table (indexed by Convex-community's subgraph) gives us all the deposits and withdraws for any of the supported pools.
- ``swaps`` - This table (indexed by Convex-community's subgraph) gives us all the swaps for any of the supported pools.
- ``token_ohlcv`` - This table (pulled from CCXT exchagnes or Chainlink aggregators) gives us the price of each token over time. Data from CCXT is pulled on a minutely granularity, data from Chainlink depends on the update-frequency of the aggregator.
- ``metrisc`` - This table serves all of the computed metrics described in the Metrics section, as well as rolling averages and rolling standard deviations for each metric (rolling window size of 1 hour, for other window sizes, rolling metrics must be computed on-the-fly).

## Creating the Raw database
Create a folder `database` with a single file `database.db`. This will be our SQLite database. Ensure that the supported pools in the `config.json` file are agreeable to you (add or remove pools you need or don't need).

Then run the below command:

```
python3 -m curvemetrics.create_database
```

This will create all of our SQL tables and populate our meta tables (i.e. the `tokens`, `pools`, and `pool_tokens` junction table). You can verify they exist and query their data by doing the following:

```
sqlite3 database/database.db
```

This will let you interact with the SQL database. View data by doing (e.g.):

```
SELECT * FROM tokens;
```

## Backfilling
Setting up the SQL database with backfilled raw data can take a while. You can run the job to do so in the background with the following command:

```
nohup bash curvemetrics/batch_backfill_raw_data.sh 2023-01-01 2023-05-01 > ./logs/backfill.log 2>&1 &
```

This backfills raw data (reserves, swaps, lp events) from January to May 2023, but you can specify whatever dates you want. This splits the work into daily jobs. Warning: if you have the free Infura plan with 100,000 requests, you'll probably hit your daily limit after backfilling roughly 11 months of data (5 tokens, 11 months)

You can verify if the job failed for any date with:

```
find ./logs -type f -exec grep -L "Done :)" {} \; | sort
```

Which checks for our `Done :)` exit status.

## Indexing
To ensure we can read timeseries data quickly for each token and pool, we create indexes on each pool. To create these indexes, run:

```
nohup python3 -m curvemetrics.create_indexes > ./logs/index.log 2>&1 &
```

## Resetting
If there was an issue with the database, you can drop all the tables and vacuum the database by running:

```
python3 -m curvemetrics.reset
```