
import logging
from logging.handlers import RotatingFileHandler

class Logger():

    def __init__(self, fn):
        handler = RotatingFileHandler(fn, maxBytes=10**6, backupCount=5) # 1MB file size, keep last 5 files
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger(__file__)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        self.logger = logger