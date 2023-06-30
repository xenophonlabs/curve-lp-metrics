import time
import math
from datetime import datetime, timedelta
import smtplib
import os
from dotenv import load_dotenv
import json
import tweepy
from crontab import CronTab
import pickle
import traceback
import re
import asyncio

from curvemetrics.scripts.raw_data import main as raw
from curvemetrics.scripts.metrics import main as metrics
from curvemetrics.scripts.takers import main as takers

from curvemetrics.src.classes.logger import Logger
from curvemetrics.src.classes.datahandler import DataHandler
from curvemetrics.src.classes.datafetcher import DataFetcher
from curvemetrics.src.classes.modelsetup import ModelSetup

INFERENCE_PERIOD = 60*60 # 1 hour
WINDOW = timedelta(days=1) # Markout window for Sharks (5 minutes for markout metric)
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

POOL_METRICS = [
    "shannonsEntropy", 
    "netSwapFlow", 
    # "300.Markout"
]

LINKS = {
    "Curve.fi DAI/USDC/USDT":"https://curve.fi/#/ethereum/pools/3pool/deposit",
    "Curve.fi ETH/stETH":"https://curve.fi/#/ethereum/pools/steth/deposit",
    "Curve.fi FRAX/USDC":"https://curve.fi/#/ethereum/pools/fraxusdc/deposit",
}

MODELED_POOLS = [
    "0xdc24316b9ae028f1497c275eb9192a3ea0f67022", # ETH/stETH
    "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7", # 3pool
    "0xdcef968d416a41cdac0ed8702fac8128a64241a2", # FRAX/USDC
    # "0xa1f8a6807c402e4a15ef4eba36528a3fed24e577", # ETH/frxETH
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
    body = f"A changepoint was detected.\nPool: {pool_name}\nMetric: {metric}\nTime: {datetime.fromtimestamp(cp)}"
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
    text = f'A potential depeg has been detected.\nPool: {pool_name}\nMetric: {metric}\nTime: {datetime.fromtimestamp(cp)} UTC\nPool Link: {LINKS[pool_name]}.\n\nDISCLAIMER: CurveMetrics is prone to reporting false positives; NFA.'
    response = client.create_tweet(text=text)
    return response

def delete_cronjob():
    cron = CronTab(user=True)
    for job in cron:
        if job.comment == "curvemetrics_forever":
            cron.remove_all(comment="curvemetrics_forever")
            cron.write()

def insert_cronjob():
    cron = CronTab(user=True)
    job = cron.new(command='cd /root/curve-lp-metrics/ && /root/curve-lp-metrics/venv/bin/python3 -m curvemetrics.scripts.forever >> /root/curve-lp-metrics/logs/cron.log 2>&1')
    job.set_comment("curvemetrics_forever")
    job.setall('0 * * * *')
    cron.write()

def get_model_start(ts):
    dt = datetime.fromtimestamp(ts)
    return int(datetime.timestamp(datetime(dt.year, dt.month, dt.day, dt.hour) - timedelta(seconds=INFERENCE_PERIOD)))

def get_model_end(ts):
    dt = datetime.fromtimestamp(ts)
    return int(datetime.timestamp(datetime(dt.year, dt.month, dt.day, dt.hour))) - 0.00001

async def check_availability():
    datafetcher = DataFetcher()
    block, ts = datafetcher.get_latest_availability()
    await datafetcher.close()
    return block, ts

def latest_run():
    # Load the configuration
    s = os.path.join(os.path.abspath('subgraphs.json'))
    s = re.sub(r'(/root/curve-lp-metrics/).*', r'\1', s) + 'subgraphs.json'
    with open(s, "r") as f:
        config = json.load(f)
    latest_block, latest_ts = config['latest_block'], config['latest_ts']
    return latest_block, latest_ts

def update_latest_run(block, ts):
    # Load the configuration
    s = os.path.join(os.path.abspath('subgraphs.json'))
    s = re.sub(r'(/root/curve-lp-metrics/).*', r'\1', s) + 'subgraphs.json'
    config = {'latest_block': block, 'latest_ts': ts}
    with open(s, "w") as f:
        json.dump(config, f)

def get_latest_inference_ts(models):
    last_ts = math.inf
    for pool in MODELED_POOLS:
        for metric in POOL_METRICS:
            model = models[pool][metric]
            last_ts = min(last_ts, model.last_ts)
    return last_ts

def main(models):
    """
    This script does the following, in order, every hour, forever:

    IF THERE IS NEW SUBGRAPH DATA PROCEED, ELSE EXIT

    1. Frontfill new raw data
    2. Compute and frontfill new metrics
    3. Compute and frontfill new takers stats and sharkflow metric
    4. Runs new inference step[s] on all models

    Fault tolerance:

    If any step fails, the script will retry them 3 times, with a 60 second delay between each retry.
    If they still fail, the script will:
    1. Log the error
    2. Delete the cron job
    3. Send an email to the Xenophon Labs team
    """

    ### Check data availability
    logger = Logger('./logs/frontfill/master.log').logger
    new_block, new_ts = asyncio.run(check_availability())
    latest_block, latest_ts = latest_run()
    if new_block <= latest_block:
        logger.info(f'[NO NEW DATA AVAILABLE] Available block: {new_block}, latest block filled: {latest_block}. Skipping...')
        ### No new data available
        return models
    else:
        logger.info(f'[NEW DATA AVAILABLE] Available block: {new_block}, latest block filled: {latest_block}. Filling...')

    start = latest_ts + 1 # Non-overlapping
    end = new_ts

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
            time.sleep(60)
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
                raise e
            time.sleep(60)
    logger.info('Successfully frontfilled metrics.')

    ### Frontfill Takers
    sliding_window = timedelta(seconds=int(end-start))
    logger = Logger('./logs/frontfill/takers.log').logger
    for attempts in range(RETRIES):
        try:
            datahandler = DataHandler()
            t = datahandler.get_takers()
            takers(end - sliding_window.total_seconds() - WINDOW.total_seconds(), WINDOW, sliding_window, logger, takers=t)
            break
        except Exception as e:
            logger.error(f'Failed to frontfill takers: {e}')
            if attempts == RETRIES - 1:
                send_email_on_error(e, start, end)
                raise e
            time.sleep(60)
        finally:
            datahandler.close()
    logger.info('Successfully frontfilled takers.')

    update_latest_run(new_block, new_ts)

    ### Run Inference
    logger = Logger('./logs/frontfill/inference.log').logger
    try:
        model_start = get_model_start(get_latest_inference_ts(models))
        model_end = get_model_end(end) - 0.00001

        datahandler = DataHandler()
        tuner = ModelSetup(datahandler, logger=logger, freq=timedelta(seconds=INFERENCE_PERIOD))

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

            # Check if lp token price < vp (if it was already true, returns false)
            is_true_cp = baseline.update(virtual_price, lp_share_price, model_start)
            if is_true_cp:
                true_cp = baseline.last_cp
                datahandler.insert_changepoints([datetime.fromtimestamp(true_cp)], pool, 'baseline', 'baseline', tuner.freq_str)
                logger.info(f'Changepoint detected for {name} with baseline model at {datetime.fromtimestamp(true_cp)}.')
                send_email_on_changepoint(name, 'baseline', true_cp)
                tweet(name, 'baseline', true_cp, lp_share_price, virtual_price)

            for metric in POOL_METRICS:
                model = models[pool][metric]
                X = datahandler.get_pool_X(metric, pool, model_start, model_end, '1h')
                # Check that there is enough new data for 1 hour of inference
                if X.empty:
                    logger.info(f'No new data available for inference for {name}, {metric}. Last model inference: {datetime.fromtimestamp(model.last_ts)}')
                    continue
                for idx, x in X.items(): 
                    ts = datetime.timestamp(idx)
                    if ts < model.last_ts + INFERENCE_PERIOD:
                        logger.info(f'Inference already performed for {name} with {metric} at {idx}.')
                        continue
                    logger.info(f'Running inference for {name} with {metric} at {idx}.')
                    is_cp = model.predict(x, ts)
                    if is_cp:
                        cp = ts 
                        datahandler.insert_changepoints([datetime.fromtimestamp(cp)], pool, 'bocd', metric, tuner.freq_str)
                        logger.info(f'Changepoint detected for {name} with {metric} at {datetime.fromtimestamp(cp)}.')
                        send_email_on_changepoint(name, metric, cp)
                        tweet(name, metric, cp, lp_share_price, virtual_price)

    except Exception as e:
        logger.error(f'Failed to run inference: {e}')
        send_email_on_error(e, model_start, model_end)
        raise e

    finally:
        datahandler.close()

    return models

# Run FOREVER!
if __name__ == "__main__":
    # NOTE: delete cron job to prevent multiple instances running at once
    delete_cronjob()

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

    models = main(models)

    # Dump models
    for pool in MODELED_POOLS:
        baseline = models[pool]['baseline']
        with open(f'./model_configs/baseline/{pool}.pkl', 'wb') as f:
            pickle.dump(baseline, f)
        for metric in POOL_METRICS:
            model = models[pool][metric]
            with open(f'./model_configs/{metric}/{pool}.pkl', 'wb') as f:
                pickle.dump(model, f)

    # NOTE: re-add cron job
    insert_cronjob()