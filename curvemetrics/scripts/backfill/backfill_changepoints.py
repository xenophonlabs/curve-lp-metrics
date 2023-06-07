"""
Backfill changepoints for given time period into SQL database.
Assumes we have optimal hyperparameters for models in config.json.
"""
import argparse
import asyncio
import os
import json
import pandas as pd
import traceback

from datetime import datetime

from ...src.classes.metricsprocessor import MetricsProcessor
from ...src.classes.datahandler import DataHandler

STEP_SIZE = 10 # NOTE: increasing this risks losing txs, 10 is probably safe

def load_config():
    # Load the configuration
    with open(os.path.join(os.path.abspath('config.json')), "r") as config_file:
        config = json.load(config_file)
    return config

async def main(start: str, end: str):

    print(f"\n[{datetime.now()}] Processing metrics...\n")

    datahandler = DataHandler()
    token_metadata = datahandler.get_token_metadata()
    pool_metadata = datahandler.get_pool_metadata()

    start, end = datetime.fromisoformat(start), datetime.fromisoformat(end)
    start_ts, end_ts = datetime.timestamp(start), datetime.timestamp(end)

    metricsprocessor = MetricsProcessor(pool_metadata, token_metadata)

    config = load_config()

    print(f"\n[{datetime.now()}] Processing pool changepoints.\n")

    try:
        # Fetch and insert pool data
        # for pool in pool_metadata.keys():
        for pool in ["0xa1f8a6807c402e4a15ef4eba36528a3fed24e577", "0x971add32ea87f10bd192671630be3be8a11b8623"]:

            print(f"[{datetime.now()}] Processing pool {pool_metadata[pool]['name']}.")

            if pool_metadata[pool]['creationDate'] < start_ts:
                pool_start_ts = start_ts
                pool_end_ts = end_ts
            elif start_ts < pool_metadata[pool]['creationDate'] < start_ts:
                pool_start_ts = pool_metadata[pool]['creationDate']
                pool_end_ts = end_ts
            else:
                print(f"[{datetime.now()}] Pools {pool_metadata[pool]['name']} was created after the end date. Skipping...\n")
                continue
                
            # Edge cases

            if pool == "0xceaf7747579696a2f0bb206a14210e3c9e6fb269": # UST pool
                if 1653667380 < start_ts:
                    print(f"[{datetime.now()}] Pool {pool_metadata[pool]['name']}, UST no longer indexed after {1653667380}. Skipping...\n")
                    continue
                elif start_ts < 1653667380 < end_ts:
                    pool_start_ts = start_ts
                    pool_end_ts = 1653667380
            
            elif pool == "0xdcef968d416a41cdac0ed8702fac8128a64241a2": # FRAX
                if start_ts < 1656399149 < end_ts:
                    pool_start_ts = 1656399149
                    pool_end_ts = end_ts
                elif end_ts < 1656399149:
                    print(f"[{datetime.now()}] Pool {pool_metadata[pool]['name']}, no output token supply until {1653667380}. Skipping...\n")
                    continue
             
            elif pool == "0x0f9cb53ebe405d49a0bbdbd291a65ff571bc83e1": # USDN
                if 1675776755 < start_ts:
                    print(f"[{datetime.now()}] Pool {pool_metadata[pool]['name']}, USDN no longer indexed after {1675776755}. Skipping...\n")
                    continue
                elif start_ts < 1675776755 < end_ts:
                    pool_start_ts = start_ts
                    pool_end_ts = 1675776755

            print(f"[{datetime.now()}] Start time: {datetime.fromtimestamp(pool_start_ts)}")
            print(f"[{datetime.now()}] End time: {datetime.fromtimestamp(pool_end_ts)}")

            pool_data = datahandler.get_pool_data(pool, pool_start_ts, pool_end_ts)
            snapshots = datahandler.get_pool_snapshots(pool, pool_start_ts, pool_end_ts)

            ohlcvs = {}
            for token in pool_metadata[pool]['inputTokens']:
                ohlcv = datahandler.get_ohlcv_data(token, start=start_ts, end=pool_end_ts)
                ohlcvs[token] = ohlcv

            lp_share_price = metricsprocessor.lp_share_price(pool, pool_data, ohlcvs)
            datahandler.insert_pool_metrics(pd.DataFrame(lp_share_price), pool)

            cps = metricsprocessor.true_cps(lp_share_price, snapshots)
            datahandler.insert_changepoints(cps, pool, 'baseline', 'baseline')

            print(f"[{datetime.now()}] Finished processing pool {pool_metadata[pool]['name']}.\n")

        print(f"[{datetime.now()}] Done :)")
    
    except Exception as e:
        print(f"\n[{datetime.now()}] An error occurred during raw database backfilling:\n{traceback.print_exc()}\n")
    
    finally:
        datahandler.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backfill pool and token data in SQL table.')
    parser.add_argument('start', type=str, help='Start date in format YYYY-MM-DD HH:MM:SS')
    parser.add_argument('end', type=str, help='end date in format YYYY-MM-DD HH:MM:SS')
    args = parser.parse_args()
    asyncio.run(main(args.start, args.end))