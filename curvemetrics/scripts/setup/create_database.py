"""
Create the SQL database tables.
"""
import os
import json
import re
from datetime import datetime

from ...src.classes.datafetcher import DataFetcher
from ...src.classes.datahandler import DataHandler

def load_config():
    # Load the configuration
    s = os.path.join(os.path.abspath('config.json'))
    s = re.sub(r'(/root/curve-lp-metrics/).*', r'\1', s) + 'config.json'
    with open(s, "r") as config_file:
        config = json.load(config_file)
    return config

def main():

    config = load_config()
    supported_pools = config["supported_pools"]

    pool_metadata = DataFetcher.get_pools_metadata(supported_pools)
    token_metadata = DataFetcher.get_tokens_metadata(pool_metadata)

    datahandler = DataHandler()

    print(f'[{datetime.now()}] Creating database tables...')

    try:
        datahandler.create_tables()
        datahandler.insert_pool_metadata(pool_metadata)
        datahandler.insert_token_metadata(token_metadata)
        # datahandler.insert_block_timestamps() # Very slow
    except Exception as e:
        print(f"An error occurred during raw database creation: {e}")
    finally:
        print(f'[{datetime.now()}] Done.')
        datahandler.close()

if __name__ == "__main__":
    main()