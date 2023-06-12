import logging
from logging.handlers import RotatingFileHandler

class Logger():

    def __init__(self, fn):
        handler = RotatingFileHandler(fn, maxBytes=10**6, backupCount=5) # 1MB file size, keep last 5 files
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger = logging.getLogger(fn)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            logger.addHandler(handler)

        self.logger = logger
        self.fn = fn