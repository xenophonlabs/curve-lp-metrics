import argparse
import logging
import pandas as pd
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from curvemetrics.scripts.takers import main
from curvemetrics.src.classes.logger import Logger

WINDOW = timedelta(days=1)
# SLIDING_WINDOW = timedelta(days=1) # NOTE: use this for most backfilling
SLIDING_WINDOW = timedelta(hours=1) # NOTE: use this for last few hours

if __name__ == "__main__":
    logger = Logger('./logs/backfill/takers.log').logger

    parser = argparse.ArgumentParser(description='Backfill pool and token metrics in SQL tables.')
    parser.add_argument('start', type=str, help='Start date in format YYYY-MM-DD HH:MM:SS')
    parser.add_argument('end', type=str, help='end date in format YYYY-MM-DD HH:MM:SS')
    args = parser.parse_args()

    curr = datetime.timestamp(datetime.fromisoformat(args.start))
    end = datetime.timestamp(datetime.fromisoformat(args.end))
    takers = pd.DataFrame()
    while curr + WINDOW.total_seconds() + SLIDING_WINDOW.total_seconds() <= end:
        takers = main(curr, WINDOW, SLIDING_WINDOW, logger, takers=takers)
        curr += SLIDING_WINDOW.total_seconds()