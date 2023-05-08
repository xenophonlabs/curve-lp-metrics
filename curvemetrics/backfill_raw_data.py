from .datafetcher import DataFetcher
from .datahandler import RawDataHandler
from datetime import datetime
import argparse
import asyncio

STEP_SIZE = 10 # NOTE: increasing this risks losing txs, 10 is probably safe

async def main(start: str, end: str):

    print()

    start, end = datetime.fromisoformat(start), datetime.fromisoformat(end)
    start_ts, start_block = DataFetcher.get_block(start)
    end_ts, end_block = DataFetcher.get_block(end)

    datahandler = RawDataHandler()
    token_metadata = datahandler.get_token_metadata()
    pool_metadata = datahandler.get_pool_metadata()

    datafetcher = DataFetcher(token_metadata=token_metadata)

    try:
        # Fetch and insert pool data
        for pool in pool_metadata.keys():

            print(f"Backfilling pool {pool_metadata[pool]['name']}.")

            if pool_metadata[pool]['creationDate'] < start_ts:
                pool_start_ts, pool_start_block = start_ts, start_block
            elif start_ts < pool_metadata[pool]['creationDate'] < start_ts:
                pool_start_ts, pool_start_block = pool_metadata[pool]['creationDate'], pool_metadata[pool]['creationBlock']
            else:
                print(f"Pools {pool_metadata[pool]['name']} was created after the end date. Skipping...")
                continue

            print(f"Start time: {datetime.fromtimestamp(pool_start_ts)}")
            print(f"End time: {datetime.fromtimestamp(end_ts)}")

            pool_data = datafetcher.get_pool_data(pool_start_block, end_block, pool, step_size=1)
            datahandler.insert_pool_data(pool_data, start_ts, end_ts)

            print("Finished reserve data...")

            swaps_data = datafetcher.get_swaps_data(pool_start_block, end_block, pool, step_size=STEP_SIZE)
            datahandler.insert_swaps_data(swaps_data)

            print("Finished swap data...")
            
            lp_data = datafetcher.get_lp_data(pool_start_block, end_block, pool, step_size=STEP_SIZE)
            datahandler.insert_lp_data(lp_data)

            print("Finished lp event data...\n")

        # Fetch and insert token data
        for token in token_metadata.keys():

            token_start_ts, token_end_ts = start_ts, end_ts # TODO: Check when token was created

            if token_metadata[token]['symbol'] in ["3Crv", "frxETH", "FRAX", "cvxCRV", "stETH", "sUSD", "WETH", "LUSD", "USDN"]:
                print(f"TODO: Add support for {token_metadata[token]['symbol']}. Skipping...\n")
                continue

            if token_metadata[token]['symbol'] == "UST":
                # TODO: We only index UST up to 1653667380 (2022-05-27 12:03:00 PM GMT-04:00 DST), when coinbasepro stocks indexing
                if end_ts > 1653667380:
                    token_end_ts = 1653667380
                    print(f"UST only indexed up to 1653667380 (2022-05-27 12:03:00 PM GMT-04:00 DST). Setting end time to {datetime.fromtimestamp(end_ts)}.")

            print(f"Backfilling token {token_metadata[token]['symbol']} OHLCV.")

            token_data = datafetcher.get_ohlcv(token_start_ts, token_end_ts, token)
            datahandler.insert_token_data(token_data)
        
        print("Done :)")
    
    except Exception as e:
        print(f"\nAn error occurred during raw database backfilling: {e}\n")
    
    finally:
        datahandler.close()
        await datafetcher.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backfill pool and token data in SQL table.')
    parser.add_argument('start', type=str, help='Start date in format YYYY-MM-DD HH:MM:SS')
    parser.add_argument('end', type=str, help='end date in format YYYY-MM-DD HH:MM:SS')
    args = parser.parse_args()
    asyncio.run(main(args.start, args.end))