"""
Had an issue with approxTimestamps for pool reserves. This script fixes them
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging

from .datafetcher import DataFetcher
from .datahandler import DataHandler

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler())

datahandler = DataHandler()
token_metadata = datahandler.get_token_metadata()
pool_metadata = datahandler.get_pool_metadata()

def get_pool_data(pool, start, end):
    start_ts = datetime.timestamp(start - timedelta(days=1))
    end_ts = datetime.timestamp(end + timedelta(days=1))
    df = datahandler.get_pool_data(pool, start_ts, end_ts)
    df = df.sort_values(by='block')
    return df

def get_updated_timestamps(df, start, end):
    min_ts, min_block = DataFetcher.get_block(start)
    max_ts, max_block = DataFetcher.get_block(end)

    blocknr = min_block
    blocks = set(df['block'])
    missing = []
    while blocknr <= max_block:
        if blocknr not in blocks:
            missing.append(blocknr)
        blocknr += 1
    assert len(missing) == 0, LOGGER.info(f"Missing blocks: {len(missing)}\n{missing}")

    ndf = df[(df['block'] >= min_block) & (df['block'] <= max_block)].copy()
    ndf['approxTimestamp'] = np.linspace(min_ts, max_ts, ndf.shape[0], dtype=int)
    ndf.index = ndf['approxTimestamp'].apply(datetime.fromtimestamp)

    return ndf

def update_timestamps(ndf):
    for _, row in ndf.iterrows():
        sql = f"""
            UPDATE pool_data
            SET approxTimestamp = {row['approxTimestamp']}
            WHERE id = {row['id']};
        """
        datahandler.conn.execute(sql)
    datahandler.conn.commit()

def update_chunk(start, end):
    LOGGER.info(f"\n[{datetime.now()}] Executing {start} to {end}")
    LOGGER.info(f"\n[{datetime.now()}] Getting data")
    df = get_pool_data(pool, start, end)
    LOGGER.info(f"\n[{datetime.now()}] Updating timestamps")
    ndf = get_updated_timestamps(df, start, end)
    assert ndf['block'].is_monotonic_increasing
    assert list(ndf['block'].diff().value_counts().keys()) == [1]
    LOGGER.info(f"\n[{datetime.now()}] Inserting")
    update_timestamps(ndf)
    return df, ndf

start = datetime(2022, 1, 1)
end = datetime(2023, 5, 1)

PROCESSED = ['0xdcef968d416a41cdac0ed8702fac8128a64241a2',
'0xceaf7747579696a2f0bb206a14210e3c9e6fb269',
'0x0f9cb53ebe405d49a0bbdbd291a65ff571bc83e1',
'0x5a6a4d54456819380173272a5e8e9b9904bdf41b',
'0xa5407eae9ba41422680e2e00537571bcc53efbfd',
"0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7", 
"0xdc24316b9ae028f1497c275eb9192a3ea0f67022",
'0xa1f8a6807c402e4a15ef4eba36528a3fed24e577',
'0xed279fdd11ca84beef15af5d39bb4d4bee23f0ca',
'0x4807862aa8b2bf68830e4c8dc86d0e9a998e085a',
'0x828b154032950c8ff7cf8085d841723db2696056']

for pool in pool_metadata:
    curr = start
    if pool in PROCESSED:
        LOGGER.info(f"\n[{datetime.now()}] Skipping {pool_metadata[pool]['name']}")
        continue
    LOGGER.info(f"\n[{datetime.now()}] Processing {pool_metadata[pool]['name']}")
    pool_created = datetime.fromtimestamp(pool_metadata[pool]['creationDate'])
    while curr + relativedelta(months=1) <= end:
        curr_end = curr+relativedelta(months=1)
        if pool_created < curr:
            actual = curr
        elif curr < pool_created < curr_end:
            actual = pool_created
            LOGGER.info(f"[{datetime.now()}] Pool {pool_metadata[pool]['name']} was created mid-month. Using {actual}")
        else:
            LOGGER.info(f"[{datetime.now()}] Pool {pool_metadata[pool]['name']} was created after the end date. Skipping...")
            curr += relativedelta(months=1)
            continue
        update_chunk(actual, curr_end)
        curr += relativedelta(months=1)

datahandler.close()