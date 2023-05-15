"""
Need to add outputTokenSupply to table
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
import asyncio

from .datafetcher import DataFetcher
from .datahandler import DataHandler

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler())

datahandler = DataHandler()
token_metadata = datahandler.get_token_metadata()
pool_metadata = datahandler.get_pool_metadata()

def get_output_token_supply(pool, start, end):

    _, min_block = DataFetcher.get_block(start)
    _, max_block = DataFetcher.get_block(end)

    data = datafetcher.get_pool_data(min_block, max_block, pool, step_size=1)
    df = pd.DataFrame([x for y in data for x in y])
    df = df[['block', 'outputTokenSupply']]
    df = df.astype(int)
    print(df)

    return df

def update_output_token_supply(df):
    for _, row in df.iterrows():
        sql = f"""
            UPDATE pool_data
            SET outputTokenSupply = {row['outputTokenSupply']}
            WHERE block = {row['block']};
        """
        datahandler.conn.execute(sql)
    datahandler.conn.commit()

def update_chunk(pool, start, end):
    LOGGER.info(f"\n[{datetime.now()}] Executing {start} to {end}")
    LOGGER.info(f"\n[{datetime.now()}] Fetching Output Token Supply")
    df = get_output_token_supply(pool, start, end)
    LOGGER.info(f"\n[{datetime.now()}] Inserting")
    # update_output_token_supply(df)
    LOGGER.info(f"\n[{datetime.now()}] Finished")

start = datetime(2022, 1, 1)
end = datetime(2022, 1, 4)
# end = datetime(2023, 5, 1)

PROCESSED = []

# for pool in pool_metadata:
async def main():
    rel = relativedelta(days=1)
    try:
        global datafetcher
        datafetcher = DataFetcher(token_metadata=token_metadata)

        for pool in ["0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7"]:
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