"""
Create the SQL database tables.
"""
import os
import json

from ...src.classes.datafetcher import DataFetcher
from ...src.classes.datahandler import DataHandler

def load_config():
    # Load the configuration
    with open(os.path.join(os.path.abspath('config.json')), "r") as config_file:
        config = json.load(config_file)
    return config

def main():

    config = load_config()
    supported_pools = config["supported_pools"]

    pool_metadata = DataFetcher.get_pools_metadata(supported_pools)
    token_metadata = DataFetcher.get_tokens_metadata(pool_metadata)

    datahandler = DataHandler()

    try:
        datahandler.create_tables()
        datahandler.insert_pool_metadata(pool_metadata)
        datahandler.insert_token_metadata(token_metadata)
        datahandler.insert_pool_tokens_metadata(pool_metadata)
        # datahandler.insert_block_timestamps()
    except Exception as e:
        print(f"An error occurred during raw database creation: {e}")
    finally:
        datahandler.close()

if __name__ == "__main__":
    main()