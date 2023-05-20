"""
Need to add column to pool_data table.
Used to add outputTokenSupply and inputTokens

NOTE: Modify this script to add new column to SQL table.
"""
import pandas as pd
import logging
import asyncio
import json

from datetime import datetime
from dateutil.relativedelta import relativedelta

from ...src.classes.datafetcher import DataFetcher
from ...src.classes.datahandler import DataHandler

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler())

datahandler = DataHandler()
token_metadata = datahandler.get_token_metadata()
pool_metadata = datahandler.get_pool_metadata()
datafetcher = DataFetcher(token_metadata=token_metadata)

start = datetime(2022, 1, 1)
end = datetime(2023, 5, 1)

PROCESSED = []
COL = 'inputTokens'

def get(pool, start, end):

    _, min_block = DataFetcher.get_block(start)
    _, max_block = DataFetcher.get_block(end)

    data = datafetcher.get_pool_data(min_block, max_block, pool, step_size=1)
    df = pd.DataFrame([x for y in data for x in y])
    df = df[['block', COL]]
    df['block'] = df['block'].astype(int)
    df[COL] = df[COL].apply(lambda x: json.dumps([y['id'] for y in x]))

    return df

def insert(df, pool):
    for _, row in df.iterrows():
        sql = f"""
            UPDATE pool_data
            SET {COL} = '{row[COL]}'
            WHERE block = {row['block']} AND pool_id = '{pool}';
        """
        datahandler.conn.execute(sql)
    datahandler.conn.commit()

def update_chunk(pool, start, end):
    LOGGER.info(f"\n[{datetime.now()}] Executing {start} to {end}")
    LOGGER.info(f"\n[{datetime.now()}] Fetching new data")
    df = get(pool, start, end)
    LOGGER.info(f"\n[{datetime.now()}] Inserting")
    insert(df, pool)
    LOGGER.info(f"\n[{datetime.now()}] Finished")

async def main():
    rel = relativedelta(days=1)
    try:
        for pool in pool_metadata:
            curr = start
            if pool in PROCESSED:
                LOGGER.info(f"\n[{datetime.now()}] Skipping {pool_metadata[pool]['name']}")
                continue
            LOGGER.info(f"\n[{datetime.now()}] Processing {pool_metadata[pool]['name']}")
            pool_created = datetime.fromtimestamp(pool_metadata[pool]['creationDate'])
            while curr + rel <= end:
                curr_end = curr+rel
                if pool_created < curr:
                    actual = curr
                elif curr < pool_created < curr_end:
                    actual = pool_created
                    LOGGER.info(f"[{datetime.now()}] Pool {pool_metadata[pool]['name']} was created mid-month. Using {actual}")
                else:
                    LOGGER.info(f"[{datetime.now()}] Pool {pool_metadata[pool]['name']} was created after the end date. Skipping...")
                    curr += rel
                    continue
                update_chunk(pool, actual, curr_end)
                curr += rel
    except Exception as e:
        LOGGER.exception(f"\n[{datetime.now()}] Error {e}")
    finally:
        await datafetcher.close()
        datahandler.close()

if __name__ == "__main__":
    asyncio.run(main())