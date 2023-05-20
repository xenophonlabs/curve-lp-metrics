"""
Had an issue with timestamps for pool reserves. This script fixes them
    Didn't know how to get timestamp for block without spamming INFURA key,
    got .csv for timestamps for blocks, use to backfill
    Front fill by spamming INFURA key
Also renamed them to just "timestamp"
"""
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging

from ...src.classes.datahandler import DataHandler

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler())

datahandler = DataHandler()
token_metadata = datahandler.get_token_metadata()
pool_metadata = datahandler.get_pool_metadata()

start = datetime(2022, 1, 1)
end = datetime(2023, 5, 1)

PROCESSED = []

def get_pool_data(pool, start, end):
    start_ts = datetime.timestamp(start - timedelta(days=1))
    end_ts = datetime.timestamp(end + timedelta(days=1))
    df = datahandler.get_pool_data(pool, start_ts, end_ts)
    return df

def get_updated_timestamps(df):
    df['timestamp'] = df['block'].apply(lambda x: datahandler.get_block_timestamp(x)[0]['timestamp'])
    df = df.sort_values(by='block', ascending=True)
    assert df['timestamp'].is_monotonic_increasing
    return df

def update_timestamps(ndf):
    for _, row in ndf.iterrows():
        sql = f"""
            UPDATE pool_data
            SET timestamp = {row['timestamp']}
            WHERE id = {row['id']};
        """
        datahandler.conn.execute(sql)
    datahandler.conn.commit()

def update_chunk(start, end):
    LOGGER.info(f"\n[{datetime.now()}] Executing {start} to {end}")
    LOGGER.info(f"\n[{datetime.now()}] Getting data")
    df = get_pool_data(pool, start, end)
    LOGGER.info(f"\n[{datetime.now()}] Updating timestamps")
    ndf = get_updated_timestamps(df)
    LOGGER.info(f"\n[{datetime.now()}] Inserting")
    update_timestamps(ndf)
    return df, ndf

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