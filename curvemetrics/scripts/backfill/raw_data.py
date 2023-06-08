import argparse
from datetime import datetime
from ..raw_data import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backfill pool and token data in SQL tables.')
    parser.add_argument('start', type=str, help='Start date in format YYYY-MM-DD HH:MM:SS')
    parser.add_argument('end', type=str, help='end date in format YYYY-MM-DD HH:MM:SS')
    args = parser.parse_args()
    start = datetime.timestamp(datetime.fromisoformat(args.start))
    end = datetime.timestamp(datetime.fromisoformat(args.end))
    main(start, end)
