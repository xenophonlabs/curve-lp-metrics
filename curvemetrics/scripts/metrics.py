"""
Populate metrics for given time period into SQL database.
"""
import os
import json
import logging
import traceback

from logging.handlers import RotatingFileHandler
from datetime import datetime

from ..src.classes.metricsprocessor import MetricsProcessor
from ..src.classes.datahandler import DataHandler

def load_config():
    # Load the configuration
    with open(os.path.join(os.path.abspath('config.json')), "r") as config_file:
        config = json.load(config_file)
    return config

config = load_config()

handler = RotatingFileHandler('../../logs/metrics.log', maxBytes=10**6, backupCount=5) # 1MB file size, keep last 5 files
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def pools(start: int, end: int):

    datahandler = DataHandler()
    token_metadata = datahandler.token_metadata
    pool_metadata = datahandler.pool_metadata
    metricsprocessor = MetricsProcessor(pool_metadata, token_metadata)

    logger.info(f"\nProcessing pool metrics.\n")

    try:
        for pool in pool_metadata.keys():
            try:
                logger.info(f"Processing pool {pool_metadata[pool]['name']}.")

                if pool_metadata[pool]['creationDate'] < start:
                    pool_start = start
                    pool_end = end
                elif start < pool_metadata[pool]['creationDate'] < start:
                    pool_start = pool_metadata[pool]['creationDate']
                    pool_end = end
                else:
                    logger.info(f"Pools {pool_metadata[pool]['name']} was created after the end date. Skipping...\n")
                    continue

                # Edge cases

                if pool == "0xceaf7747579696a2f0bb206a14210e3c9e6fb269": # UST pool
                    if 1653667380 < start:
                        logger.info(f"Pool {pool_metadata[pool]['name']}, UST no longer indexed after {1653667380}. Skipping...\n")
                        continue
                    elif start < 1653667380 < end:
                        pool_start = start
                        pool_end = 1653667380
                
                elif pool == "0xdcef968d416a41cdac0ed8702fac8128a64241a2": # FRAX
                    if start < 1656399149 < end:
                        pool_start = 1656399149
                        pool_end = end
                    elif end < 1656399149:
                        logger.info(f"Pool {pool_metadata[pool]['name']}, no output token supply until {1653667380}. Skipping...\n")
                        continue
                
                elif pool == "0x0f9cb53ebe405d49a0bbdbd291a65ff571bc83e1": # USDN
                    if 1675776755 < start:
                        logger.info(f"Pool {pool_metadata[pool]['name']}, USDN no longer indexed after {1675776755}. Skipping...\n")
                        continue
                    elif start < 1675776755 < end:
                        pool_start = start
                        pool_end = 1675776755

                logger.info(f"Start time: {datetime.fromtimestamp(pool_start)}")
                logger.info(f"End time: {datetime.fromtimestamp(pool_end)}")

                pool_data = datahandler.get_pool_data(pool, pool_start, pool_end)
                swaps_data = datahandler.get_swaps_data(pool, pool_start, pool_end)
                lp_data = datahandler.get_lp_data(pool, pool_start, pool_end)

                ohlcvs = {}
                tokens = set(swaps_data['tokenBought']).union(set(swaps_data['tokenSold'])).union(set(pool_metadata[pool]['inputTokens']))
                for token in tokens:
                    ohlcv = datahandler.get_ohlcv_data(token, start=start, end=pool_end)
                    ohlcvs[token] = ohlcv

                logger.info(f"Inserting metrics.")

                pool_metrics = metricsprocessor.process_metrics_for_pool(pool, pool_data, swaps_data, lp_data, ohlcvs)
                datahandler.insert_pool_metrics(pool_metrics, pool)

                logger.info(f"Finished processing pool {pool_metadata[pool]['name']}.\n")

            except Exception as e:
                logger.error(f"\nFailed to compute metrics for pool {pool_metadata[pool]['name']}\n{traceback.format_exc()}\n", exc_info=True)
                raise e

    except Exception as e: # Just to be safe, but probably never reached
        logger.error(f"\nFailed to compute metrics for pools \n{traceback.format_exc()}\n", exc_info=True)
        raise e

    finally:
        datahandler.close()

def tokens(start: int, end: int):

    datahandler = DataHandler()
    token_metadata = datahandler.get_token_metadata()
    pool_metadata = datahandler.get_pool_metadata()
    metricsprocessor = MetricsProcessor(pool_metadata, token_metadata)

    logger.info(f"\nProcessing pool metrics.\n")

    try:
        for token in token_metadata.keys():
            try:
                logger.info(f"Processing token {token_metadata[token]['symbol']}.")

                token_start, token_end = start, end 

                # TODO: refactor these if statements for all scripts

                if token_metadata[token]['symbol'] == "WETH":
                    logger.info(f"{token_metadata[token]['symbol']} assumed to be = ETH. Skipping...\n")
                    continue

                elif token_metadata[token]['symbol'] in ["3Crv"]:
                    logger.info(f"We use virtual prices for {token_metadata[token]['symbol']}. Skipping...\n")
                    continue
                
                elif token_metadata[token]['symbol'] == "USDN":
                    if token_end < 1635284389:
                        logger.info(f"USDN only indexed from 1635284389 2021-10-26. Skipping...")
                        continue
                    elif token_start < 1635284389 < token_end:
                        logger.info(f"USDN only indexed from 1635284389 2021-10-26 21:39:49. Setting start ts to 1635284389.")
                    
                    if 1675776755 < start:
                        logger.info(f"USDN no longer indexed after {1675776755}. Skipping...\n")
                        continue
                    elif start < 1675776755 < end:
                        logger.info(f"USDN only indexed up to 1675776755 (2022-05-27 12:03:00 PM GMT-04:00 DST). Setting end time to {datetime.fromtimestamp(end)}.")
                        token_end = 1675776755

                elif token_metadata[token]['symbol'] == "UST":
                    if 1653667380 < start:
                        logger.info(f"UST only indexed up to 1653667380 (2022-05-27 12:03:00 PM GMT-04:00 DST). Skipping...")
                        continue
                    elif start < 1653667380 < end:
                        logger.info(f"UST only indexed up to 1653667380 (2022-05-27 12:03:00 PM GMT-04:00 DST). Setting end time to {datetime.fromtimestamp(end)}.")
                        token_end = 1653667380
                    
                elif token_metadata[token]['symbol'] == "cbETH":
                    if token_start < 1661444040 < token_end:
                        logger.info(f"cbETH only indexed from 1661444040 (2022 12:14:00 PM GMT-04:00 DST). Setting start ts to 1661444040.")
                        token_start = 1661444040
                    elif token_end < 1661444040:
                        logger.info(f"cbETH only indexed from 1661444040 (2022 12:14:00 PM GMT-04:00 DST). Skipping...")
                        continue
                
                elif token_metadata[token]['symbol'] in ["frxETH", "cvxCRV"]:
                    # depend on the pool creation date.
                    _, pool, _ = config['token_exchange_map'][datahandler.token_metadata[token]['symbol']]
                    if pool_metadata[pool]['creationDate'] < start:
                        token_start = start
                        token_end = end
                    elif start < pool_metadata[pool]['creationDate'] < start:
                        token_start = pool_metadata[pool]['creationDate']
                        token_end = end
                    else:
                        logger.info(f"Token {token_metadata[token]['symbol']} has no pricing data before {end}. Skipping...\n")
                        continue

                logger.info(f"Start time: {datetime.fromtimestamp(token_start)}")
                logger.info(f"End time: {datetime.fromtimestamp(token_end)}")

                token_ohlcv = datahandler.get_ohlcv_data(token, token_start, token_end)
                token_metrics = metricsprocessor.process_metrics_for_token(token, token_ohlcv)
                logger.info(f"Inserting metrics.")
                datahandler.insert_token_metrics(token_metrics, token)

                logger.info(f"Finished processing token {token_metadata[token]['symbol']}.\n")
    
            except Exception as e:
                logger.error(f"\nFailed to compute metrics for token {token_metadata[token]['name']}\n{traceback.format_exc()}\n", exc_info=True)
                raise e

    except Exception as e: # Just to be safe, but probably never reached
        logger.error(f"\nFailed to compute metrics for tokens \n{traceback.format_exc()}\n", exc_info=True)
        raise e

    finally:
        datahandler.close()

def main(start: int, end: int):

    logger.info(f"\nProcessing metrics...\n")

    pools(start, end)
    tokens(start, end)

    logger.info(f"Done :)")
