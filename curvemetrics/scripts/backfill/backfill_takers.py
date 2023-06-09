from datetime import datetime, timedelta

import pandas as pd

# Local imports
from curvemetrics.src.classes.datahandler import DataHandler
from curvemetrics.src.classes.metricsprocessor import MetricsProcessor

datahandler = DataHandler()
token_metadata = datahandler.get_token_metadata()
pool_metadata = datahandler.get_pool_metadata()
metricsprocessor = MetricsProcessor(pool_metadata, token_metadata)

start = datetime(2022, 1, 1)
end = datetime(2023, 5, 1)
start_ts = datetime.timestamp(start)
end_ts = datetime.timestamp(end)
window = timedelta(days=1)

# Read from SQL
takers = pd.DataFrame(columns=['buyer', 'amountBought', 'amountSold', 'cumulativeMarkout', 'meanMarkout', 'count', 'windowSize'])
takers.set_index('buyer', inplace=True)

sliding_window = timedelta(days=1)

# 1. Compute 1 day of 1-day markouts (t_0 : t_60) using 
# 2. Update takers table
# 3. Compute sharkflow for (t_3660 : t_3720) using latest takers data (t_0 : t_60) 
#    since there is a 3600min delay in computing 1d markouts.
# 4. Slide windows by 1 day

swaps_start_ts = start_ts
swaps_end_ts = start_ts + sliding_window.total_seconds()  + window.total_seconds()
sharkflow_start_ts = swaps_start_ts + sliding_window.total_seconds()  + window.total_seconds()
sharkflow_end_ts = sharkflow_start_ts + sliding_window.total_seconds() 

while swaps_end_ts <= end_ts:
    print(f"[{datetime.now()}] Processing takers from {datetime.fromtimestamp(swaps_start_ts)} to {datetime.fromtimestamp(swaps_end_ts)}.")

    for pool in datahandler.pool_metadata:

        swaps_data = datahandler.get_swaps_data(pool, swaps_start_ts, swaps_end_ts)
        if swaps_data.empty:
            continue
        ohlcvs = {}
        for token in set(swaps_data['tokenBought']).union(set(swaps_data['tokenSold'])):
            ohlcvs[token] = datahandler.get_ohlcv_data(token, swaps_start_ts, swaps_end_ts)
        if any([x.empty for x in ohlcvs.values()]):
            continue
        pool_takers = metricsprocessor.takers(swaps_data, ohlcvs, window)

        for t, row in pool_takers.iterrows():
            if t in takers.index:
                takers.loc[t, 'amountBought'] += row['amountBought']
                takers.loc[t, 'amountSold'] += row['amountSold']
                takers.loc[t, 'cumulativeMarkout'] += row['cumulativeMarkout']
                takers.loc[t, 'count'] += row['count']
                takers.loc[t, 'meanMarkout'] = takers.loc[t, 'cumulativeMarkout'] / takers.loc[t, 'count']
            else:
                # add new taker to takers
                takers.loc[t] = row

    print(f"[{datetime.now()}] Processing sharkflow from {datetime.fromtimestamp(sharkflow_start_ts)} to {datetime.fromtimestamp(sharkflow_end_ts)}.\n")

    for pool in datahandler.pool_metadata:
        
        mask = pd.DataFrame(index=pd.date_range(start=datetime.fromtimestamp(sharkflow_start_ts), end=datetime.fromtimestamp(sharkflow_end_ts), freq='T'))

        swaps_data = datahandler.get_swaps_data(pool, sharkflow_start_ts, sharkflow_end_ts)
        if swaps_data.empty:
            continue
        tokens = set(swaps_data['tokenBought']).union(set(swaps_data['tokenSold']))
        for token in tokens:
            sharkflow = metricsprocessor.sharkflow(swaps_data, takers, token, token_metadata[token]['symbol'], top=0.9)
            sharkflow = mask.merge(sharkflow, left_index=True, right_index=True, how='left').fillna(0) # Ensure it includes all minutes
            datahandler.insert_pool_metrics(pd.DataFrame(sharkflow), pool)
    
    swaps_start_ts += sliding_window.total_seconds()
    swaps_end_ts += sliding_window.total_seconds()
    sharkflow_start_ts += sliding_window.total_seconds()
    sharkflow_end_ts += sliding_window.total_seconds()

takers['windowSize'] = window.total_seconds()
takers = takers.astype({'amountBought':float, 'amountSold':float, 'cumulativeMarkout':float, 'meanMarkout':float, 'count':int, 'windowSize':int})
datahandler.insert_takers(takers)