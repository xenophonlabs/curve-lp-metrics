"""
Create indexes for the SQL tables.
NOTE: Don't do this too before backfilling, it will slow down the process.
"""

from ...src.classes.datahandler import DataHandler

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