"""
This script resets the database by dropping all tables.
"""
from ...src.classes.datahandler import DataHandler

if __name__ == "__main__":
    datahandler = DataHandler()
    datahandler.reset()
