import math
import time
from datetime import datetime, timedelta
import smtplib
import os
from dotenv import load_dotenv
import json
from typing import Dict
import tweepy
from crontab import CronTab
import pickle
import traceback
import re

from curvemetrics.scripts.raw_data import main as raw
from curvemetrics.scripts.metrics import main as metrics
from curvemetrics.scripts.takers import main as takers

from curvemetrics.src.classes.model import BOCD
from curvemetrics.src.classes.logger import Logger
from curvemetrics.src.classes.datahandler import DataHandler
from curvemetrics.src.classes.modelsetup import ModelSetup

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

TWEEPY_API_KEY = os.getenv('TWEEPY_API_KEY')
TWEEPY_API_SECRET = os.getenv('TWEEPY_API_SECRET')
TWEEPY_API_BEARER_TOKEN = os.getenv('TWEEPY_API_BEARER_TOKEN')
TWEEPY_API_ACCESS_TOKEN = os.getenv('TWEEPY_API_ACCESS_TOKEN')
TWEEPY_API_ACCESS_TOKEN_SECRET = os.getenv('TWEEPY_API_ACCESS_TOKEN_SECRET')

POOL_METRICS = {"shannonsEntropy", "netSwapFlow", "300.Markout"}
MODELED_POOLS = [
    "0xdc24316b9ae028f1497c275eb9192a3ea0f67022", # ETH/stETH
    "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7", # 3pool
    "0xdcef968d416a41cdac0ed8702fac8128a64241a2", # FRAX/USDC
    "0xa1f8a6807c402e4a15ef4eba36528a3fed24e577", # ETH/frxETH
    # "0x828b154032950c8ff7cf8085d841723db2696056", # stETH Concentrated
]

def load_config():
    # Load the configuration
    s = os.path.join(os.path.abspath('config.json'))
    s = re.sub(r'(/root/curve-lp-metrics/).*', r'\1', s) + 'config.json'
    with open(s, "r") as config_file:
        config = json.load(config_file)
    return config

config = load_config()

def send_email(msg):
    # Establish server connection
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(STMP_LOGIN, SMTP_APP_PWD)
    # server.sendmail(STMP_LOGIN, ["thomas@xenophonlabs.com", "max@xenophonlabs.com"], msg)
    server.sendmail(STMP_LOGIN, ["thomas@xenophonlabs.com"], msg)
    server.quit()

def send_email_on_error(exc, start, end):
    subject = f'Curvemetrics Frontfilling Failure {datetime.fromtimestamp(start)} : {datetime.fromtimestamp(end)}'
    body = f"An exception was raised during frontfilling {exc}:\n\n{traceback.print_exc()}\n\nThe Cron job has been removed. Re-add it as:\n\n* * * * * cd /root/curve-lp-metrics/ && /root/curve-lp-metrics/venv/bin/python3 -m curvemetrics.scripts.forever >> /root/curve-lp-metrics/logs/cron.log 2>&1 #curvemetrics_forever"
    msg = f'Subject: {subject}\n\n{body}'
    send_email(msg)

def send_email_on_changepoint(pool_name, metric, cp):
    subject = f'Curvemetrics Changepoint Detected'
    body = f"A changepoint was detected.\nPool: {pool_name}\nMetric: {metric}\Time: {cp}"
    msg = f'Subject: {subject}\n\n{body}'
    send_email(msg)

def tweet(pool_name, metric, cp, lp_share_price, virtual_price):
    client = tweepy.Client(bearer_token=TWEEPY_API_BEARER_TOKEN, 
                           consumer_key=TWEEPY_API_KEY, 
                           consumer_secret=TWEEPY_API_SECRET, 
                           access_token=TWEEPY_API_ACCESS_TOKEN, 
                           access_token_secret=TWEEPY_API_ACCESS_TOKEN_SECRET, 
                           wait_on_rate_limit=True
    )
    text = f'This is an example!\n\nA potential depeg has been detected.\nPool: {pool_name}\nMetric: {metric}\nTime: {datetime.fromtimestamp(cp)}\nThe current LP token price is: {round(lp_share_price, 3)}\nCompared to the virtual price: {round(virtual_price, 3)}.'
    response = client.create_tweet(text=text)
    return response

def delete_cronjob():
    cron = CronTab(user=True)
    for job in cron:
        if job.comment == "curvemetrics_forever":
            cron.remove_all(comment="curvemetrics_forever")
            cron.write()

def get_model_start(dt):
    return int(datetime.timestamp(datetime(dt.year, dt.month, dt.day, dt.hour) - timedelta(hours=2)))

def get_model_end(dt):
    return int(datetime.timestamp(datetime(dt.year, dt.month, dt.day, dt.hour)))

# def main(models: Dict[BOCD]):
def main():
    """
    This script does the following, in order, every hour, forever:

    1. Frontfill raw data
    2. Computes and frontfills metrics
    3. Computes and frontfills new takers stats and sharkflow metric
    4. Runs one inference step on all models and frontfills the results

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
                delete_cronjob()
                raise e
            time.sleep(10)
    logger.info('Successfully frontfilled raw data.')

    ### Frontfill Metrics
    logger = Logger('./logs/frontfill/metrics.log').logger
    for attempts in range(RETRIES):
        try:
            metrics(start, end, logger)
            break
        except Exception as e:
            logger.error(f'Failed to frontfill metrics: {e}')
            if attempts == RETRIES - 1:
                send_email_on_error(e, start, end)
                delete_cronjob()
                raise e
            time.sleep(10)
    logger.info('Successfully frontfilled metrics.')

    ### Frontfill Takers
    start = math.floor(now) - SLIDING_WINDOW.total_seconds() - WINDOW.total_seconds()
    logger = Logger('./logs/frontfill/takers.log').logger
    for attempts in range(RETRIES):
        try:
            datahandler = DataHandler()
            t = datahandler.get_takers()
            takers(start, WINDOW, SLIDING_WINDOW, logger, takers=t)
            break
        except Exception as e:
            logger.error(f'Failed to frontfill takers: {e}')
            if attempts == RETRIES - 1:
                send_email_on_error(e, start, end)
                delete_cronjob()
                raise e
            time.sleep(10)
        finally:
            datahandler.close()
    logger.info('Successfully frontfilled takers.')

    # try:
    #     dt = datetime.fromtimestamp(now)
    #     model_start = get_model_start(dt)
    #     model_end = get_model_end(dt)

    #     datahandler = DataHandler()
    #     tuner = ModelSetup(datahandler, logger=logger)

    #     ### Model Inference
    #     for pool in MODELED_POOLS:

    #         name = datahandler.pool_metadata[pool]['name']
            
    #         baseline = model[pool]['baseline']

    #         lp_share_price = datahandler.get_pool_metric_last(pool, 'lpSharePrice')[0]
    #         if pool in tuner.ETH_POOLS:
    #             eth_price = datahandler.get_ohlcv_data_last(datahandler.token_ids['ETH'])['close'][0]
    #             lp_share_price /= eth_price
    #         elif pool in tuner.CRV_POOLS:
    #             crv_price = datahandler.get_ohlcv_data_last(datahandler.token_ids['CRV'])['close'][0]
    #             lp_share_price /= crv_price
    #         virtual_price = datahandler.get_pool_snapshots_last(pool)['virtualPrice'][0] / 10**18

    #         is_true_cp = baseline.update(virtual_price, lp_share_price, model_start)
    #         if is_true_cp:
    #             true_cp = baseline.last_cp
    #             # datahandler.insert_changepoints([baseline.last_ts], pool, 'baseline', baseline)
    #             logger.info(f'Changepoint detected for {name} with baseline model at {datetime.fromtimestamp(true_cp)}.')
    #             tweet(name, 'baseline', true_cp, lp_share_price, virtual_price)
    #             send_email_on_changepoint(name, 'baseline', true_cp)

    #         for metric in POOL_METRICS:
    #             logger.info(f'Running inference for {pool} with {metric}.')
    #             model = models[pool][metric]
    #             X = datahandler.get_pool_X(metric, pool, model_start, model_end, '1h')
    #             x, ts = X[0], datetime.timestamp(X.index[0])
    #             # Ensure we are getting complete, non-overlapping data
    #             assert ts == model.last_ts + PERIOD 
    #             is_cp = model.predict(x, ts)
    #             if is_cp:
    #                 cp = now
    #                 # datahandler.insert_changepoints([cp], pool, 'bocd', metric)
    #                 logger.info(f'Changepoint detected for {name} with {metric} at {cp}.')
    #                 tweet(name, metric, cp, lp_share_price, virtual_price)
    #                 send_email_on_changepoint(name, metric, cp)

    # except Exception as e:
    #     logger.error(f'Failed to run inference: {e}')
    #     send_email_on_error(e, start, end)
    #     delete_cronjob()
    #     raise e

    # finally:
    #     datahandler.close()

# Run FOREVER!
if __name__ == "__main__":
    global now
    now = time.time()

    # Initialize models
    # margin = config['model_configs']['base']['margin']
    # alpha = config['model_configs']['base']['alpha']
    # models = {}
    # for pool in MODELED_POOLS:
    #     with open(f'./model_configs/baseline/{pool}.pkl', 'rb') as f:
    #         baseline = pickle.load(f)
    #     models[pool] = {}
    #     models[pool]['baseline'] = baseline
    #     for metric in POOL_METRICS:
    #         with open(f'../model_configs/{metric}/{pool}.pkl', 'rb') as f:
    #             model = pickle.load(f)
    #         model.logger = Logger(f'./logs/frontfill/inference_{pool}_{metric}.log').logger
    #         models[pool][metric] = model

    # main(models)
    main()