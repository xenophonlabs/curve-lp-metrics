"""
Update takers and sharkflow metrics.
"""
import logging
import traceback
import pandas as pd

from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta

from curvemetrics.src.classes.datahandler import DataHandler
from curvemetrics.src.classes.metricsprocessor import MetricsProcessor

handler = RotatingFileHandler('../../logs/takers.log', maxBytes=10**6, backupCount=5) # 1MB file size, keep last 5 files
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def main(start: int, window: timedelta, sliding_window: timedelta, takers: pd.DataFrame=pd.DataFrame(), top: float=0.9) -> pd.DataFrame:

    datahandler = DataHandler()
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
        swaps_end = start + window.total_seconds() + sliding_window.total_seconds()

        # E.g.: sharflow_start = Wednesday 00:00:00, sharkflow_end = Thursday 00:00:00
        # -> Using takers table (updated with Monday markouts) we calculate Wednesday sharkflow
        sharkflow_start = swaps_end
        sharkflow_end = sharkflow_start + sliding_window.total_seconds() 

        # -> Every day, sharkflow is computed using 1d-Markout Sharks as of two days ago

        logger.info(f"Processing takers from {datetime.fromtimestamp(swaps_start)} to {datetime.fromtimestamp(swaps_end)}.")

        for pool in datahandler.pool_metadata:

            swaps_data = datahandler.get_swaps_data(pool, swaps_start, swaps_end)
            if swaps_data.empty:
                continue
            ohlcvs = {}
            for token in set(swaps_data['tokenBought']).union(set(swaps_data['tokenSold'])):
                ohlcvs[token] = datahandler.get_ohlcv_data(token, swaps_start, swaps_end)
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
            
            mask = pd.DataFrame(index=pd.date_range(start=datetime.fromtimestamp(sharkflow_start), end=datetime.fromtimestamp(sharkflow_end), freq='T'))

            swaps_data = datahandler.get_swaps_data(pool, sharkflow_start, sharkflow_end)
            if swaps_data.empty:
                continue
            tokens = set(swaps_data['tokenBought']).union(set(swaps_data['tokenSold']))
            for token in tokens:
                sharkflow = metricsprocessor.sharkflow(swaps_data, takers, token, datahandler.token_metadata[token]['symbol'], top=top)
                sharkflow = mask.merge(sharkflow, left_index=True, right_index=True, how='left').fillna(0) # Ensure it includes all minutes
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