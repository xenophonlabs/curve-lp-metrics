import math
import time

from curvemetrics.scripts.raw_data import main
from curvemetrics.src.classes.logger import Logger

PERIOD = 60*60 # 1 hour
BUFFER = 60*10 # 10 minutes

if __name__ == "__main__":
    start = math.floor(time.time()) # UTC timestamp
    end = start - PERIOD - BUFFER
    logger = Logger('./logs/frontfill/raw_data.log')
    main(start, end, logger)
