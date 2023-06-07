"""
Backfill raw data into SQL database.
"""
import argparse
import asyncio
import os
import json
import traceback

from datetime import datetime

from ...src.classes.datafetcher import DataFetcher
from ...src.classes.datahandler import DataHandler

STEP_SIZE = 10 # NOTE: increasing this risks losing txs, 10 is probably safe

def load_config():
    # Load the configuration
    with open(os.path.join(os.path.abspath('config.json')), "r") as config_file:
        config = json.load(config_file)
    return config

async def main(start: str, end: str):

    print(f"\n[{datetime.now()}] Starting backfilling process.\n")

    start, end = datetime.fromisoformat(start), datetime.fromisoformat(end)
    start_ts, start_block = DataFetcher.get_block(start)
    end_ts, end_block = DataFetcher.get_block(end)

    datahandler = DataHandler()
    token_metadata = datahandler.get_token_metadata()
    pool_metadata = datahandler.get_pool_metadata()

    datafetcher = DataFetcher(token_metadata=token_metadata)

    config = load_config()

    try:
        # Fetch and insert pool data
        # for pool in pool_metadata.keys():

        #     print(f"[{datetime.now()}] Backfilling pool {pool_metadata[pool]['name']}.")

        #     if pool_metadata[pool]['creationDate'] < start_ts:
        #         pool_start_ts, pool_start_block = start_ts, start_block
        #     elif start_ts < pool_metadata[pool]['creationDate'] < end_ts:
        #         pool_start_ts, pool_start_block = pool_metadata[pool]['creationDate'], pool_metadata[pool]['creationBlock']
        #     else:
        #         print(f"[{datetime.now()}] Pools {pool_metadata[pool]['name']} was created after the end date. Skipping...\n")
        #         continue

        #     print(f"[{datetime.now()}] Start time: {datetime.fromtimestamp(pool_start_ts)}")
        #     print(f"[{datetime.now()}] End time: {datetime.fromtimestamp(end_ts)}")

        #     pool_data = datafetcher.get_pool_data(pool_start_block, end_block, pool, step_size=1)
        #     datahandler.insert_pool_data(pool_data, start_ts, end_ts)

        #     print(f"[{datetime.now()}] Finished reserve data...")

        #     swaps_data = datafetcher.get_swaps_data(pool_start_block, end_block, pool, step_size=STEP_SIZE)
        #     datahandler.insert_swaps_data(swaps_data)

        #     print(f"[{datetime.now()}] Finished swap data...")
            
        #     lp_data = datafetcher.get_lp_data(pool_start_block, end_block, pool, step_size=STEP_SIZE)
        #     datahandler.insert_lp_data(lp_data)

        #     print(f"[{datetime.now()}] Finished lp event data...\n")

        #     snapshots = datafetcher.get_snapshots(pool_start_ts, end_ts, pool, step_size=STEP_SIZE)
        #     datahandler.insert_pool_snapshots(snapshots)

        #     print(f"[{datetime.now()}] Finished snapshots data...\n")

        # Fetch and insert token data
        # for token in token_metadata.keys():
        for token in [datahandler.token_ids[x] for x in ["frxETH", "cvxCRV"]]:

            token_start_ts, token_end_ts = start_ts, end_ts 

            if token_metadata[token]['symbol'] == "WETH":
                print(f"[{datetime.now()}] {token_metadata[token]['symbol']} assumed to be = ETH. Skipping...\n")
                continue

            elif token_metadata[token]['symbol'] == "3Crv":
                print(f"[{datetime.now()}] {token_metadata[token]['symbol']} uses pool virtual price. Skipping...\n")
                continue

            elif token_metadata[token]['symbol'] == "USDN":
                if token_end_ts < 1635284389:
                    print(f"[{datetime.now()}] USDN only indexed from 1635284389 2021-10-26. Skipping...")
                    continue
                elif token_start_ts < 1635284389 < token_end_ts:
                    print(f"[{datetime.now()}] USDN only indexed from 1635284389 2021-10-26 21:39:49. Setting start ts to 1635284389.")
                    token_start_ts = 1635284389

            elif token_metadata[token]['symbol'] == "UST":
                if token_end_ts > 1653667380:
                    token_end_ts = 1653667380
                    print(f"[{datetime.now()}] UST only indexed up to 1653667380 (2022-05-27 12:03:00 PM GMT-04:00 DST). Setting end time to {datetime.fromtimestamp(end_ts)}.")
                
            elif token_metadata[token]['symbol'] == "cbETH":
                if token_start_ts < 1661444040 < token_end_ts:
                    print(f"[{datetime.now()}] cbETH only indexed from 1661444040 (2022 12:14:00 PM GMT-04:00 DST). Setting start ts to 1661444040.")
                    token_start_ts = 1661444040
                elif token_end_ts < 1661444040:
                    print(f"[{datetime.now()}] cbETH only indexed from 1661444040 (2022 12:14:00 PM GMT-04:00 DST). Skipping...")
                    continue

            token_config = config['token_exchange_map'][token_metadata[token]['symbol']]
            if len(token_config) == 2:
                api, source = token_config
            else:
                api, source, numeraire = token_config

            print(f"[{datetime.now()}] Backfilling token {token_metadata[token]['symbol']} OHLCV using {api}.")

            if api == "ccxt":
                token_data = datafetcher.get_ohlcv(token_start_ts, token_end_ts, token, default_exchange=source)
            elif api == "chainlink":
                token_data = datafetcher.get_chainlink_prices(token, source, token_start_ts, token_end_ts)
            elif api == "curveswaps":
                token_data = datahandler.get_curve_price(token, source, start_ts, end_ts, numeraire=datahandler.token_ids[numeraire])

            if token_data:
                datahandler.insert_token_data(token_data)
            else:
                print(f'[{datetime.now()}] No price data for token {token_metadata[token]["symbol"]} at this time.\n')

        print(f"[{datetime.now()}] Done :)")
    
    except Exception as e:
        print(f"\n[{datetime.now()}] An error occurred during raw database backfilling: {traceback.print_exc()}\n")
    
    finally:
        datahandler.close()
        await datafetcher.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backfill pool and token data in SQL table.')
    parser.add_argument('start', type=str, help='Start date in format YYYY-MM-DD HH:MM:SS')
    parser.add_argument('end', type=str, help='end date in format YYYY-MM-DD HH:MM:SS')
    args = parser.parse_args()
    asyncio.run(main(args.start, args.end))