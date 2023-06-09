import argparse
from datetime import datetime
from curvemetrics.scripts.raw_data import main
from curvemetrics.src.classes.logger import Logger

BUFFER = 60*10 # 10 minutes

if __name__ == "__main__":
    logger = Logger('./logs/backfill/raw_data.log').logger
    parser = argparse.ArgumentParser(description='Backfill pool and token data in SQL tables.')
    parser.add_argument('start', type=str, help='Start date in format YYYY-MM-DD HH:MM:SS')
    parser.add_argument('end', type=str, help='end date in format YYYY-MM-DD HH:MM:SS')
    args = parser.parse_args()
    start = datetime.timestamp(datetime.fromisoformat(args.start)) - BUFFER
    end = datetime.timestamp(datetime.fromisoformat(args.end))
    main(start, end, logger)
