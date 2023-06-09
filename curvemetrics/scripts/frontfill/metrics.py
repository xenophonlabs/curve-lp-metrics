import math
import time

from curvemetrics.scripts.metrics import main
from curvemetrics.src.classes.logger import Logger

PERIOD = 60*60 # 1 hour
BUFFER = 60*10 # 10 minutes

if __name__ == "__main__":
    start = math.floor(time.time()) # UTC timestamp
    end = start - PERIOD - BUFFER
    logger = Logger('./logs/frontfill/metrics.log').logger
    main(start, end, logger)