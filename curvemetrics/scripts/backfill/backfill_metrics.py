"""
Backfill metrics for given time period into SQL database.
"""
import argparse
import asyncio

from datetime import datetime

from ...src.classes.metricsprocessor import MetricsProcessor
from ...src.classes.datahandler import DataHandler

def main(start: str, end: str):

    print(f"\n[{datetime.now()}] Processing metrics...\n")

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
            elif start_ts < pool_metadata[pool]['creationDate'] < start_ts:
                pool_start_ts = pool_metadata[pool]['creationDate']
            else:
                print(f"[{datetime.now()}] Pools {pool_metadata[pool]['name']} was created after the end date. Skipping...")
                continue

            print(f"[{datetime.now()}] Start time: {datetime.fromtimestamp(pool_start_ts)}")
            print(f"[{datetime.now()}] End time: {datetime.fromtimestamp(end_ts)}")

            pool_data = datahandler.get_pool_data(pool, pool_start_ts, end_ts)
            swaps_data = datahandler.get_swaps_data(pool, pool_start_ts, end_ts)
            lp_data = datahandler.get_lp_data(pool, pool_start_ts, end_ts)

            ohlcvs = {}
            for token in set(swaps_data['tokenBought']):
                ohlcv = datahandler.get_ohlcv_data(token, start=start_ts, end=end_ts)
                ohlcvs[token] = ohlcv

            pool_metrics = metricsprocessor.process_metrics_for_pool(pool, pool_data, swaps_data, lp_data, ohlcvs)
            datahandler.insert_pool_metrics(pool_metrics, pool)

            print(f"[{datetime.now()}] Finished processing pool {pool_metadata[pool]['name']}.\n")

        for token in token_metadata.keys():

            token_start_ts, token_end_ts = start_ts, end_ts 

            print(f"[{datetime.now()}] Processing token {token_metadata[token]['symbol']}.")


            if token_metadata[token]['symbol'] == "WETH":
                print(f"[{datetime.now()}] {token_metadata[token]['symbol']} assumed to be = ETH. Skipping...\n")
                continue

            elif token_metadata[token]['symbol'] in ["3Crv", "frxETH", "cvxCRV"]:
                print(f"[{datetime.now()}] TODO: Add support for {token_metadata[token]['symbol']}. Skipping...\n")
                continue

            elif token_metadata[token]['symbol'] == "USDN":
                if token_start_ts < 1635284389:
                    print(f"[{datetime.now()}] USDN only indexed from 1635284389 2021-10-26. Skipping...")
                    continue
                elif token_start_ts < 1635284389 < token_end_ts:
                    print(f"[{datetime.now()}] USDN only indexed from 1635284389 2021-10-26 21:39:49. Setting start ts to 1661444040.")
                    token_start_ts = 1635284389
                elif token_end_ts > 1675776755:
                    token_end_ts = 1675776755
                    print(f"[{datetime.now()}] USDN only indexed until 1675776755 2023-02-07 13:32:35. Skipping...")

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

            print(f"[{datetime.now()}] Start time: {datetime.fromtimestamp(token_start_ts)}")
            print(f"[{datetime.now()}] End time: {datetime.fromtimestamp(token_end_ts)}")

            token_ohlcv = datahandler.get_ohlcv_data(token, token_start_ts, token_end_ts)
            token_metrics = metricsprocessor.process_metrics_for_token(token, token_ohlcv)
            datahandler.insert_token_metrics(token_metrics)

            print(f"[{datetime.now()}] Finished processing token {token_metadata[token]['symbol']}.\n")

        print(f"[{datetime.now()}] Done :)")
    
    except Exception as e:
        print(f"\n[{datetime.now()}] An error occurred during raw database backfilling: {e}\n")
    
    finally:
        datahandler.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backfill pool and token metrics in SQL tables.')
    parser.add_argument('start', type=str, help='Start date in format YYYY-MM-DD HH:MM:SS')
    parser.add_argument('end', type=str, help='end date in format YYYY-MM-DD HH:MM:SS')
    args = parser.parse_args()
    asyncio.run(main(args.start, args.end))