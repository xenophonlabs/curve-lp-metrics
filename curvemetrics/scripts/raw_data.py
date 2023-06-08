"""
Populate raw data for given time period into SQL database.
"""
import asyncio
import os
import json
import traceback
import logging

from logging.handlers import RotatingFileHandler
from datetime import datetime

from ..src.classes.datafetcher import DataFetcher
from ..src.classes.datahandler import DataHandler

STEP_SIZE = 10 # NOTE: increasing this risks losing txs, 10 is probably safe

def load_config():
    # Load the configuration
    with open(os.path.join(os.path.abspath('config.json')), "r") as config_file:
        config = json.load(config_file)
    return config

config = load_config()

handler = RotatingFileHandler('../../logs/raw_data.log', maxBytes=10**6, backupCount=5) # 1MB file size, keep last 5 files
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

async def pools(start:int, end:int, start_block:int, end_block:int):

    datahandler = DataHandler()
    datafetcher = DataFetcher(token_metadata=datahandler.token_metadata)
    pool_metadata = datahandler.pool_metadata

    try:
        for pool in pool_metadata.keys():
            try:

                logger.info(f"Frontfilling pool {pool_metadata[pool]['name']}.")

                if pool_metadata[pool]['creationDate'] < start:
                    pool_start, pool_start_block = start, start_block
                elif start < pool_metadata[pool]['creationDate'] < end:
                    pool_start, pool_start_block = pool_metadata[pool]['creationDate'], pool_metadata[pool]['creationBlock']
                else:
                    logger.info(f"Pools {pool_metadata[pool]['name']} was created after the end date. Skipping...\n")
                    continue

                logger.info(f"Start time: {datetime.fromtimestamp(pool_start)}")
                logger.info(f"End time: {datetime.fromtimestamp(end)}")

                pool_data = datafetcher.get_pool_data(pool_start_block, end_block, pool, step_size=1)
                logger.info(f"Inserting pool data...")
                datahandler.insert_pool_data(pool_data)
                logger.info(f"Finished pool data...")

                swaps_data = datafetcher.get_swaps_data(pool_start_block, end_block, pool, step_size=STEP_SIZE)
                logger.info(f"Inserting swap data...")
                datahandler.insert_swaps_data(swaps_data)
                logger.info(f"Finished swap data...")
                
                lp_data = datafetcher.get_lp_data(pool_start_block, end_block, pool, step_size=STEP_SIZE)
                logger.info(f"Inserting lp event data...")
                datahandler.insert_lp_data(lp_data)
                logger.info(f"Finished lp event data...")

                snapshots = datafetcher.get_snapshots(pool_start, end, pool, step_size=60*60*24)
                logger.info(f"Inserting snapshots data...")
                datahandler.insert_pool_snapshots(snapshots)
                logger.info(f"Finished snapshots data...")

                logger.info(f"Finished pool {pool_metadata[pool]['name']}.\n")
            
            except Exception as e:
                logger.error(f"\nFailed to frontfill pool {pool}:\n{traceback.format_exc()}\n", exc_info=True)
                raise e

    except Exception as e:
        logger.error(f"\nAn error occurred during raw database frontfilling:\n{traceback.format_exc()}\n", exc_info=True)
        raise e
    
    finally:
        datahandler.close()
        await datafetcher.close()

async def tokens(start:int, end:int):

    datahandler = DataHandler()
    datafetcher = DataFetcher(token_metadata=datahandler.token_metadata)
    token_metadata = datahandler.token_metadata

    try:
        for token in token_metadata.keys():
            try:
                token_start, token_end = start, end 

                if token_metadata[token]['symbol'] == "WETH":
                    logger.info(f"{token_metadata[token]['symbol']} assumed to be = ETH. Skipping...\n")
                    continue

                elif token_metadata[token]['symbol'] == "3Crv":
                    logger.info(f"{token_metadata[token]['symbol']} uses pool virtual price. Skipping...\n")
                    continue

                elif token_metadata[token]['symbol'] == "USDN":
                    if token_end < 1635284389:
                        logger.info(f"USDN only indexed from 1635284389 2021-10-26. Skipping...")
                        continue
                    elif token_start < 1635284389 < token_end:
                        logger.info(f"USDN only indexed from 1635284389 2021-10-26 21:39:49. Setting start ts to 1635284389.")
                        token_start = 1635284389

                elif token_metadata[token]['symbol'] == "UST":
                    if token_end > 1653667380:
                        token_end = 1653667380
                        logger.info(f"UST only indexed up to 1653667380 (2022-05-27 12:03:00 PM GMT-04:00 DST). Setting end time to {datetime.fromtimestamp(end)}.")
                    
                elif token_metadata[token]['symbol'] == "cbETH":
                    if token_start < 1661444040 < token_end:
                        logger.info(f"cbETH only indexed from 1661444040 (2022 12:14:00 PM GMT-04:00 DST). Setting start ts to 1661444040.")
                        token_start = 1661444040
                    elif token_end < 1661444040:
                        logger.info(f"cbETH only indexed from 1661444040 (2022 12:14:00 PM GMT-04:00 DST). Skipping...")
                        continue

                token_config = config['token_exchange_map'][token_metadata[token]['symbol']]
                if len(token_config) == 2:
                    api, source = token_config
                else:
                    api, source, numeraire = token_config

                logger.info(f"Frontfilling token {token_metadata[token]['symbol']} OHLCV using {api}.")

                if api == "ccxt":
                    token_data = datafetcher.get_ohlcv(token_start, token_end, token, default_exchange=source)
                elif api == "chainlink":
                    token_data = datafetcher.get_chainlink_prices(token, source, token_start, token_end)
                elif api == "curveswaps":
                    token_data = datahandler.get_curve_price(token, source, start, end, numeraire=datahandler.token_ids[numeraire])

                logger.info(f"Inserting OHLCV.")
                if token_data:
                    datahandler.insert_token_data(token_data)
                else:
                    logger.info(f'No price data for token {token_metadata[token]["symbol"]} at this time.\n')

            except Exception as e:
                logger.error(f"\nFailed to frontfill token {token}:\n{traceback.format_exc()}\n", exc_info=True)
                raise e
            
    except Exception as e:
        logger.error(f"\nAn error occurred during raw database frontfilling: {traceback.format_exc()}\n", exc_info=True)
        raise e
    
    finally:
        datahandler.close()
        await datafetcher.close()

def main(start: int, end: int):

    logger.info(f"\nStarting frontfilling process.\n")

    start, start_block = DataFetcher.get_block(start)
    end, end_block = DataFetcher.get_block(end)

    asyncio.run(pools(start, end, start_block, end_block))
    asyncio.run(tokens(start, end))

    logger.info(f"Done :)")
