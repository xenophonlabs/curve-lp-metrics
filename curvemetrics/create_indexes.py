from .datahandler import DataHandler
import os
import json

def main():

    datahandler = DataHandler()

    try:
        datahandler.create_indexes()
    except Exception as e:
        print(f"An error occurred during index creation: {e}")
    finally:
        datahandler.close()

if __name__ == "__main__":
    main()