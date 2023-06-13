"""
Update takers and sharkflow metrics.
"""
import traceback
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from curvemetrics.src.classes.datahandler import DataHandler
from curvemetrics.src.classes.logger import Logger
from curvemetrics.src.classes.metricsprocessor import MetricsProcessor

def load_config():
    # Load the configuration
    with open(os.path.join(os.path.abspath('config.json')), "r") as config_file:
        config = json.load(config_file)
    return config

config = load_config()

def main(start: int, window: timedelta, sliding_window: timedelta, logger: Logger, takers: pd.DataFrame=pd.DataFrame(), top: float=0.9) -> pd.DataFrame:

    datahandler = DataHandler(logger=logger)
    metricsprocessor = MetricsProcessor(datahandler.pool_metadata, datahandler.token_metadata)

    try:

        # No takers means we are starting from scratch
        if takers.empty:
            takers = pd.DataFrame(columns=['buyer', 'amountBought', 'amountSold', 'cumulativeMarkout', 'meanMarkout', 'count', 'windowSize'])
            takers.set_index('buyer', inplace=True)

        # E.g. We just got Wednesday trading data.

        # E.g.: start = Monday 00:00:00, end = Wednesday 00:00:00
        # -> 1 day markouts means Monday swaps are marked out and takers table is updated
        swaps_start = start
        swaps_end = start + sliding_window.total_seconds()

        # E.g.: sharflow_start = Wednesday 00:00:00, sharkflow_end = Thursday 00:00:00
        # -> Using takers table (updated with Monday markouts) we calculate Wednesday sharkflow
        sharkflow_start = start + window.total_seconds()
        sharkflow_end = sharkflow_start + sliding_window.total_seconds() 

        # -> Every day, sharkflow is computed using 1d-Markout Sharks as of two days ago

        logger.info(f"Processing takers from {datetime.fromtimestamp(swaps_start)} to {datetime.fromtimestamp(swaps_end)}.")

        for pool in datahandler.pool_metadata:

            swaps_data = datahandler.get_swaps_data(pool, swaps_start, swaps_end)
            if swaps_data.empty:
                continue
            ohlcvs = {}
            for token in set(swaps_data['tokenBought']).union(set(swaps_data['tokenSold'])):
                ohlcvs[token] = datahandler.get_ohlcv_data(token, swaps_start, sharkflow_end)
            if any([x.empty for x in ohlcvs.values()]):
                continue
            pool_takers = metricsprocessor.takers(swaps_data, ohlcvs, window)

            for t, row in pool_takers.iterrows():
                # Update taker markouts
                if t in takers.index:
                    takers.loc[t, 'amountBought'] += row['amountBought']
                    takers.loc[t, 'amountSold'] += row['amountSold']
                    takers.loc[t, 'cumulativeMarkout'] += row['cumulativeMarkout']
                    takers.loc[t, 'count'] += row['count']
                    takers.loc[t, 'meanMarkout'] = takers.loc[t, 'cumulativeMarkout'] / takers.loc[t, 'count']
                # add new taker to takers
                else:
                    takers.loc[t] = row

        logger.info(f"Processing sharkflow from {datetime.fromtimestamp(sharkflow_start)} to {datetime.fromtimestamp(sharkflow_end)}.\n")

        for pool in datahandler.pool_metadata:
            
            swaps_data = datahandler.get_swaps_data(pool, sharkflow_start, sharkflow_end)
            if swaps_data.empty:
                continue
            for token in config['pool_tokens_map'][pool]:
                sharkflow = metricsprocessor.sharkflow(swaps_data, takers, token, datahandler.token_metadata[token]['symbol'], top=top)
                datahandler.insert_pool_metrics(pd.DataFrame(sharkflow), pool)
            
        takers['windowSize'] = window.total_seconds()
        takers = takers.astype({'amountBought':float, 'amountSold':float, 'cumulativeMarkout':float, 'meanMarkout':float, 'count':int, 'windowSize':int})
        datahandler.insert_takers(takers)

        return takers

    except Exception as e:
        logger.error(f'Failed to populate takers and sharkflow: {traceback.format_exc()}', exc_info=True)
        raise e

    finally:
        datahandler.close()