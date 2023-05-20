"""
Frontfills the SQL database with raw data from theGraph.
"""
import asyncio
import schedule
import time

from ...src.classes.datafetcher import DataFetcher

async def main():
    async with DataFetcher() as datafetcher:
        # Call the methods to fetch the data
        # TODO: step_size should be 1 here for all of them
        swaps = await datafetcher.get_swaps()
        pool_data = await datafetcher.get_pool_data()
        lp_data = await datafetcher.get_lp_data()
        ohlcv_data = await datafetcher.get_ohlcv_data()

        # Process and store the fetched data using a DatabaseHandler (or other storage solution)
        # ...

def run_main_every_minute():
    print("Running scheduled job...")
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred during the scheduled job: {e}")
    print("Scheduled job finished")

# Schedule the scheduled job to run every minute
schedule.every(1).minutes.do(run_main_every_minute)

# Run the scheduled jobs in an infinite loop
while True:
    schedule.run_pending()
    asyncio.sleep(1)
