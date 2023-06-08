import argparse
from datetime import datetime
from curvemetrics.scripts.metrics import main
from curvemetrics.src.classes.logger import Logger

if __name__ == "__main__":
    logger = Logger('./logs/backfill/metrics.log')
    parser = argparse.ArgumentParser(description='Backfill pool and token metrics in SQL tables.')
    parser.add_argument('start', type=str, help='Start date in format YYYY-MM-DD HH:MM:SS')
    parser.add_argument('end', type=str, help='end date in format YYYY-MM-DD HH:MM:SS')
    args = parser.parse_args()
    start = datetime.timestamp(datetime.fromisoformat(args.start))
    end = datetime.timestamp(datetime.fromisoformat(args.end))
    main(start, end, logger)
