"""
Backfill metrics for given time period into SQL database.
"""
import argparse
import asyncio
import traceback
import os
import json

from datetime import datetime

from ...src.classes.metricsprocessor import MetricsProcessor
from ...src.classes.datahandler import DataHandler

def load_config():
    # Load the configuration
    with open(os.path.join(os.path.abspath('config.json')), "r") as config_file:
        config = json.load(config_file)
    return config

async def main(start: str, end: str):

    print(f"\n[{datetime.now()}] Processing metrics...\n")

    config = load_config()

    datahandler = DataHandler()
    token_metadata = datahandler.get_token_metadata()
    pool_metadata = datahandler.get_pool_metadata()

    start, end = datetime.fromisoformat(start), datetime.fromisoformat(end)
    start_ts, end_ts = datetime.timestamp(start), datetime.timestamp(end)

    metricsprocessor = MetricsProcessor(pool_metadata, token_metadata)

    print(f"\n[{datetime.now()}] Processing pool metrics.\n")

    try:
        for pool in pool_metadata.keys():
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
            swaps_data = datahandler.get_swaps_data(pool, pool_start_ts, pool_end_ts)
            lp_data = datahandler.get_lp_data(pool, pool_start_ts, pool_end_ts)

            ohlcvs = {}
            tokens = set(swaps_data['tokenBought']).union(set(swaps_data['tokenSold'])).union(set(pool_metadata[pool]['inputTokens']))
            for token in tokens:
                ohlcv = datahandler.get_ohlcv_data(token, start=start_ts, end=pool_end_ts)
                ohlcvs[token] = ohlcv

            pool_metrics = metricsprocessor.process_metrics_for_pool(pool, pool_data, swaps_data, lp_data, ohlcvs)
            datahandler.insert_pool_metrics(pool_metrics, pool)

            print(f"[{datetime.now()}] Finished processing pool {pool_metadata[pool]['name']}.\n")

        for token in token_metadata.keys():

            token_start_ts, token_end_ts = start_ts, end_ts 

            print(f"[{datetime.now()}] Processing token {token_metadata[token]['symbol']}.")

            # TODO: refactor these if statements for all scripts

            if token_metadata[token]['symbol'] == "WETH":
                print(f"[{datetime.now()}] {token_metadata[token]['symbol']} assumed to be = ETH. Skipping...\n")
                continue

            elif token_metadata[token]['symbol'] in ["3Crv"]:
                print(f"[{datetime.now()}] We use virtual prices for {token_metadata[token]['symbol']}. Skipping...\n")
                continue
            
            elif token_metadata[token]['symbol'] == "USDN":
                if token_end_ts < 1635284389:
                    print(f"[{datetime.now()}] USDN only indexed from 1635284389 2021-10-26. Skipping...")
                    continue
                elif token_start_ts < 1635284389 < token_end_ts:
                    print(f"[{datetime.now()}] USDN only indexed from 1635284389 2021-10-26 21:39:49. Setting start ts to 1635284389.")
                
                if 1675776755 < start_ts:
                    print(f"[{datetime.now()}] USDN no longer indexed after {1675776755}. Skipping...\n")
                    continue
                elif start_ts < 1675776755 < end_ts:
                    print(f"[{datetime.now()}] USDN only indexed up to 1675776755 (2022-05-27 12:03:00 PM GMT-04:00 DST). Setting end time to {datetime.fromtimestamp(end_ts)}.")
                    token_end_ts = 1675776755

            elif token_metadata[token]['symbol'] == "UST":
                if 1653667380 < start_ts:
                    print(f"[{datetime.now()}] UST only indexed up to 1653667380 (2022-05-27 12:03:00 PM GMT-04:00 DST). Skipping...")
                    continue
                elif start_ts < 1653667380 < end_ts:
                    print(f"[{datetime.now()}] UST only indexed up to 1653667380 (2022-05-27 12:03:00 PM GMT-04:00 DST). Setting end time to {datetime.fromtimestamp(end_ts)}.")
                    token_end_ts = 1653667380
                
            elif token_metadata[token]['symbol'] == "cbETH":
                if token_start_ts < 1661444040 < token_end_ts:
                    print(f"[{datetime.now()}] cbETH only indexed from 1661444040 (2022 12:14:00 PM GMT-04:00 DST). Setting start ts to 1661444040.")
                    token_start_ts = 1661444040
                elif token_end_ts < 1661444040:
                    print(f"[{datetime.now()}] cbETH only indexed from 1661444040 (2022 12:14:00 PM GMT-04:00 DST). Skipping...")
                    continue
            
            elif token_metadata[token]['symbol'] in ["frxETH", "cvxCRV"]:
                # depend on the pool creation date.
                _, pool, _ = config['token_exchange_map'][datahandler.token_metadata[token]['symbol']]
                if pool_metadata[pool]['creationDate'] < start_ts:
                    token_start_ts = start_ts
                    token_end_ts = end_ts
                elif start_ts < pool_metadata[pool]['creationDate'] < start_ts:
                    token_start_ts = pool_metadata[pool]['creationDate']
                    token_end_ts = end_ts
                else:
                    print(f"[{datetime.now()}] Token {token_metadata[token]['symbol']} has no pricing data before {end_ts}. Skipping...\n")
                    continue

            print(f"[{datetime.now()}] Start time: {datetime.fromtimestamp(token_start_ts)}")
            print(f"[{datetime.now()}] End time: {datetime.fromtimestamp(token_end_ts)}")

            token_ohlcv = datahandler.get_ohlcv_data(token, token_start_ts, token_end_ts)
            token_metrics = metricsprocessor.process_metrics_for_token(token, token_ohlcv)
            datahandler.insert_token_metrics(token_metrics, token)

            print(f"[{datetime.now()}] Finished processing token {token_metadata[token]['symbol']}.\n")

        print(f"[{datetime.now()}] Done :)")
    
    except Exception as e:
        print(f"\n[{datetime.now()}] An error occurred during raw database backfilling: {traceback.print_exc()}\n")
    
    finally:
        datahandler.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backfill pool and token metrics in SQL tables.')
    parser.add_argument('start', type=str, help='Start date in format YYYY-MM-DD HH:MM:SS')
    parser.add_argument('end', type=str, help='end date in format YYYY-MM-DD HH:MM:SS')
    args = parser.parse_args()
    asyncio.run(main(args.start, args.end))