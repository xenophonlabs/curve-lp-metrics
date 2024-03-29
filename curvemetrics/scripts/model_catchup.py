import math
import time
from datetime import datetime, timedelta
import smtplib
import re
import os
from dotenv import load_dotenv
import json
from typing import Dict
import tweepy
from crontab import CronTab
import pickle
import traceback

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
    subject = f'Curvemetrics Model Catchup Failure {datetime.fromtimestamp(start)} : {datetime.fromtimestamp(end)}'
    body = f"An exception was raised during model catchup {exc}:\n\n{traceback.print_exc()}"
    msg = f'Subject: {subject}\n\n{body}'
    send_email(msg)

def send_email_on_changepoint(pool_name, metric, cp):
    subject = f'Curvemetrics Changepoint Detected'
    body = f"A changepoint was detected.\nPool: {pool_name}\nMetric: {metric}\nTime: {datetime.fromtimestamp(cp)}"
    msg = f'Subject: {subject}\n\n{body}'
    send_email(msg)

def get_model_start(dt):
    return int(datetime.timestamp(datetime(dt.year, dt.month, dt.day, dt.hour) - timedelta(hours=2)))

def get_model_end(dt):
    return int(datetime.timestamp(datetime(dt.year, dt.month, dt.day, dt.hour)))

def main(models, now):
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
    logger = Logger('./logs/frontfill/inference.log').logger
    try:
        dt = datetime.fromtimestamp(now)
        model_start = get_model_start(dt)
        model_end = get_model_end(dt) - 0.00001

        datahandler = DataHandler()
        tuner = ModelSetup(datahandler, logger=logger, freq=timedelta(seconds=PERIOD))

        ### Model Inference
        for pool in MODELED_POOLS:

            name = datahandler.pool_metadata[pool]['name']
            
            baseline = models[pool]['baseline']

            lp_share_price = datahandler.get_pool_metric_last(pool, 'lpSharePrice')[0]
            if pool in tuner.ETH_POOLS:
                eth_price = datahandler.get_ohlcv_data_last(datahandler.token_ids['ETH'])['close'][0]
                lp_share_price /= eth_price
            elif pool in tuner.CRV_POOLS:
                crv_price = datahandler.get_ohlcv_data_last(datahandler.token_ids['CRV'])['close'][0]
                lp_share_price /= crv_price
            virtual_price = datahandler.get_pool_snapshots_last(pool)['virtualPrice'][0] / 10**18

            is_true_cp = baseline.update(virtual_price, lp_share_price, model_start)
            if is_true_cp:
                true_cp = baseline.last_cp
                datahandler.insert_changepoints([datetime.fromtimestamp(true_cp)], pool, 'baseline', 'baseline', tuner.freq_str)
                logger.info(f'Changepoint detected for {name} with baseline model at {datetime.fromtimestamp(true_cp)}.')

            for metric in POOL_METRICS:
                model = models[pool][metric]
                X = datahandler.get_pool_X(metric, pool, model_start, model_end, '1h')
                x, ts = X[-1], datetime.timestamp(X.index[-1])
                logger.info(f'Running inference for {pool} with {metric} at {datetime.fromtimestamp(ts)}.')
                # Ensure we are getting complete, non-overlapping data
                assert ts == model.last_ts + PERIOD 
                is_cp = model.predict(x, ts)
                if is_cp:
                    cp = ts
                    datahandler.insert_changepoints([datetime.fromtimestamp(cp)], pool, 'bocd', metric, tuner.freq_str)
                    logger.info(f'Changepoint detected for {name} with {metric} at {datetime.fromtimestamp(cp)}.')
                    send_email_on_changepoint(name, metric, cp)

    except Exception as e:
        logger.error(f'Failed to run inference: {e}')
        send_email_on_error(e, model_start, model_end)
        raise e

    finally:
        datahandler.close()

    return models

# Run FOREVER!
if __name__ == "__main__":
    # NOTE: Edit these
    now = datetime.timestamp(datetime(2023, 6, 13, 20, 0, 0, 1))
    end = datetime.timestamp(datetime.now())

    # Read models
    models = {}
    for pool in MODELED_POOLS:
        with open(f'./model_configs/baseline/{pool}.pkl', 'rb') as f:
            baseline = pickle.load(f)
        models[pool] = {}
        models[pool]['baseline'] = baseline
        for metric in POOL_METRICS:
            with open(f'./model_configs/{metric}/{pool}.pkl', 'rb') as f:
                model = pickle.load(f)
            model.logger = Logger(f'./logs/frontfill/inference_{pool}_{metric}.log').logger
            models[pool][metric] = model

    while now <= end:
        models = main(models, now)
        now += PERIOD
        end = datetime.timestamp(datetime.now())

    for pool in MODELED_POOLS:
        baseline = models[pool]['baseline']
        with open(f'./model_configs/baseline/{pool}.pkl', 'wb') as f:
            pickle.dump(baseline, f)
        for metric in POOL_METRICS:
            model = models[pool][metric]
            with open(f'./model_configs/{metric}/{pool}.pkl', 'wb') as f:
                pickle.dump(model, f)
