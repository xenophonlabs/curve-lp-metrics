import math
import time
from datetime import timedelta

from ..takers import main
from curvemetrics.src.classes.datahandler import DataHandler

WINDOW = timedelta(days=1)
SLIDING_WINDOW = timedelta(hours=1)

if __name__ == "__main__":
    start = math.floor(time.time()) - SLIDING_WINDOW.total_seconds() - WINDOW.total_seconds()
    
    datahandler = DataHandler()
    takers = datahandler.get_takers()
    datahandler.close()

    main(start, WINDOW, SLIDING_WINDOW, takers=takers)