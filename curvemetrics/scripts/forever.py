import math
import time
from datetime import datetime, timedelta
import smtplib
import os
from dotenv import load_dotenv
import pytz
import json
from typing import Dict
import tweepy

from curvemetrics.scripts.raw_data import main as raw
from curvemetrics.scripts.metrics import main as metrics
from curvemetrics.scripts.takers import main as takers

from curvemetrics.src.detection.scorer import early_weight
from curvemetrics.src.classes.model import BOCD
from curvemetrics.src.classes.logger import Logger
from curvemetrics.src.classes.datahandler import DataHandler
from curvemetrics.src.classes.welford import Welford

PERIOD = 60*60 # 1 hour
BUFFER = 60*10 # 10 minutes
WINDOW = timedelta(days=1) # Markout window for Sharks (5 minutes for markout metric)
SLIDING_WINDOW = timedelta(seconds=PERIOD)
RETRIES = 3

load_dotenv()

# Define your SMTP email server details
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587  # for starttls
STMP_LOGIN = "thomas@xenophonlabs.com"
SMTP_APP_PWD = os.getenv('SMTP_APP_PWD')

POOL_METRICS = {"shannonsEntropy", "netSwapFlow", "300.Markout"}
MODELED_POOLS = [
    "0xdc24316b9ae028f1497c275eb9192a3ea0f67022", # ETH/stETH
    "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7", # 3pool
    "0xdcef968d416a41cdac0ed8702fac8128a64241a2", # FRAX/USDC
    "0xa1f8a6807c402e4a15ef4eba36528a3fed24e577", # ETH/frxETH
    "0x828b154032950c8ff7cf8085d841723db2696056", # stETH Concentrated
]

def load_config():
    # Load the configuration
    with open(os.path.join(os.path.abspath('config.json')), "r") as config_file:
        config = json.load(config_file)
    return config

config = load_config()

def send_email_on_error(exc, start, end):
    # Establish server connection
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(STMP_LOGIN, SMTP_APP_PWD)

    # Define message parameters
    subject = f'Curvemetrics Frontfilling Failure {datetime.fromtimestamp(start)} : {datetime.fromtimestamp(end)}'
    body = f"An exception was raised during frontfilling:\n\n{str(exc)}"
    msg = f'Subject: {subject}\n\n{body}'

    # Send the email
    # server.sendmail(STMP_LOGIN, ["thomas@xenophonlabs.com", "max@xenophonlabs.com"], msg)
    server.sendmail(STMP_LOGIN, ["thomas@xenophonlabs.com"], msg)
    server.quit()

def tweet():
    # Authenticate to Twitter
    auth = tweepy.OAuthHandler("CONSUMER_KEY", "CONSUMER_SECRET")
    auth.set_access_token("ACCESS_TOKEN", "ACCESS_TOKEN_SECRET")

    # Create API object
    api = tweepy.API(auth)

    # Create a tweet
    api.update_status("Hello Tweepy")

# def main(models: Dict[BOCD]):
def main():
    """
    This script does the following, in order, every hour, forever:

    1. Frontfill raw data
    2. Computes and frontfills metrics
    3. Computes and frontfills new takers stats and sharkflow metric
    4. Runs one inference step on all models and frontfills the results
    5. Sleeps for 1 hour

    Fault tolerance:

    If steps (1-3) fail, the script will retry them 3 times, with a 10 second delay between each retry.
    If they still fail, the script will:
    1. Log the error
    2. Send an email to the Xenophon Labs team
    3. Tweet about the error
    4. Exit

    If step (4) fails for any model, the script will retry that model 3 times.
    If it still fails, the script will:
    1. Log the error
    2. Send an email to the Xenophon Labs team
    3. Tweet about the error
    4. Retire the failing model
    5. Continue to the next model
    """
    end = math.floor(now) # UTC timestamp
    start = end - PERIOD - BUFFER

    ### Frontfill Raw Data
    logger = Logger('./logs/frontfill/raw_data.log').logger
    for attempts in range(RETRIES):
        try:
            raw(start, end, logger)
            break
        except Exception as e:
            logger.error(f'Failed to frontfill raw data: {e}')
            if attempts == RETRIES - 1:
                send_email_on_error(e, start, end)
                raise e
            time.sleep(10)
    logger.info('Successfully frontfilled raw data.')

    del logger

    ### Frontfill Metrics
    logger = Logger('./logs/frontfill/metrics.log').logger
    for attempts in range(RETRIES):
        try:
            metrics(start, end, logger)
            break
        except Exception as e:
            logger.error(f'Failed to frontfill metrics: {e}')
            if attempts == RETRIES - 1:
                send_email_on_error(e)
                raise e
            time.sleep(10)
    logger.info('Successfully frontfilled metrics.')

    del logger
    
    # ### Frontfill Takers
    # start = math.floor(now) - SLIDING_WINDOW.total_seconds() - WINDOW.total_seconds()
    # logger = Logger('./logs/frontfill/takers.log').logger
    # for attempts in range(RETRIES):
    #     try:
    #         datahandler = DataHandler()
    #         t = datahandler.get_takers()
    #         datahandler.close()
    #         takers(start, WINDOW, SLIDING_WINDOW, logger, takers=t)
    #         break
    #     except Exception as e:
    #         logger.error(f'Failed to frontfill takers: {e}')
    #         if attempts == RETRIES - 1:
    #             send_email_on_error(e)
    #             raise e
    #         time.sleep(10)
    # logger.info('Successfully frontfilled takers.')

    # ### Model Inference
    # # TODO: Need to add CP detection with baseline model, Tweets!
    # Use Welford
    # for pool in MODELED_POOLS:
    #     for metric in POOL_METRICS:
    #         logger.info(f'Running inference for {pool} with {metric}.')
    #         model = models[pool][metric]
    #         old_rt = model.model.rt
    #         model.model.update(x) # DON'T STANDARDIZE
    #         new_rt = model.model.rt
    #         if new_rt != old_rt + 1:
    #             # Changepoint! 
    #             cp = end
    #             datahandler.insert_changepoints([cp], pool, 'bocd', metric)

# Run FOREVER!
if __name__ == "__main__":
    global now
    now = time.time()

    # Initialize models
    # margin = config['model_configs']['base']['margin']
    # IMPORT PICLE OBJECT
    # alpha = config['model_configs']['base']['alpha']
    # models = {}
    # for pool in MODELED_POOLS:
    #     models[pool] = {}
    #     for metric in POOL_METRICS:
    #         logger = Logger(f'./logs/frontfill/inference_{pool}_{metric}.log').logger
    #         params = config['model_configs'][metric]
    #         model = BOCD(margin=margin, alpha=alpha, weight_func=early_weight, logger=logger)
    #         model.update({'alpha':params['alpha'], 'beta':params['beta'], 'kappa':params['kappa']})
    #         models[pool][metric] = model

    # main(models)
    main()