# Trim the database to prevent disk from filling up
# Tables to Trim: pool_data, swaps, lp_events, snapshots, token_ohlcv
# KEEP METRICS!
# We ran out of disk space with ~20 months of data. 
# I will keep 6 months of running raw data, and keep all metrics data until we have a better solution.

from datetime import timedelta
from curvemetrics.src.classes.datahandler import DataHandler
from curvemetrics.src.classes.entities import PoolData, TokenOHLCV, LPEvents, Swaps, Snapshots

WINDOW = timedelta(days=180)

datahandler = DataHandler()

ENTITIES = PoolData, 