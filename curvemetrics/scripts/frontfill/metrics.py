import math
import time

from ..metrics import main

PERIOD = 60*60 # 1 hour
BUFFER = 60*10 # 10 minutes

if __name__ == "__main__":
    start = math.floor(time.time()) # UTC timestamp
    end = start - PERIOD - BUFFER
    main(start, end)
