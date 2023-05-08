# curve-lp-metrics
Regime shift detection metrics for Curve liquidity providers.

# Data Gathering

## Creating the Raw database
Create a folder `database` with a single file `rawdata.db`. This will be our SQLite database. Ensure that the supported pools in the `config.json` file are agreeable to you (add or remove pools you need or don't need).

Then run the below command:

```
python3 curvemetrics.create_raw_database
```

This will create all of our SQL tables and populate our meta tables (i.e. the `tokens`, `pools`, and `pool_tokens` junction table). You can verify they exist and query their data by doing the following:

```
sqlite3 database/rawdata.db
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

This backfills raw data (reserves, swaps, lp events) from January to May 2023, but you can specify whatever dates you want. This splits the work into daily jobs. You can verify if the job failed for any date with:

```
find ./logs -type f -exec grep -L "Done :)" {} \;
```

Which checks for our `Done :)` exit status.
