# Trim the database to prevent disk from filling up
# Tables to Trim: pool_data, swaps, lp_events, snapshots, token_ohlcv
# KEEP METRICS!
# We ran out of disk space with ~20 months of data. 
# I will keep 6 months of running raw data, and keep all metrics data until we have a better solution.

from datetime import timedelta, datetime
from curvemetrics.src.classes.datahandler import DataHandler
from curvemetrics.src.classes.entities import PoolData, TokenOHLCV, LPEvents, Swaps, Snapshots
from curvemetrics.src.classes.logger import Logger

WINDOW = timedelta(days=180)
NOW = datetime.now()
CUTOFF = int(datetime.timestamp(NOW - WINDOW))

LOGGER = Logger('./logs/trim.log').logger

datahandler = DataHandler()

ENTITIES = [PoolData, TokenOHLCV, LPEvents, Swaps, Snapshots]

LOGGER.info(f"Trimming tables until: {datetime.fromtimestamp(CUTOFF)}.")

for entity in ENTITIES:
    LOGGER.info(f"Trimming: {entity.__tablename__}.")
    datahandler.trim(entity, CUTOFF)

LOGGER.info(f"Vacuuming Database.")
datahandler.vacuum_full()

LOGGER.info(f"Finished trimming.")
