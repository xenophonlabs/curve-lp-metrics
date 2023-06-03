"""
Train hyperparameters for BOCD models. Write them to hyperparameters.json.
"""
import os
import json
from datetime import datetime, timedelta
import numpy as np
import warnings

from ...src.classes.model import BOCD
from ...src.classes.datahandler import DataHandler
from ...src.classes.metricsprocessor import MetricsProcessor
from ...src.detection.scorer import f_measure, early_weight
from ...src.plotting.plot import bocd_plot_comp

# Suppress all runtime warnings - they pertain to large integers and divisions by 0 in tuning.
warnings.filterwarnings("ignore")

ALPHA = [10**i for i in range(-5, 5)]
BETA = [10**i for i in range(-5, 5)]
KAPPA = [10**i for i in range(-5, 5)]
GRID = [(a, b, k) for a in ALPHA for b in BETA for k in KAPPA]

FREQ = timedelta(hours=1)
MARGIN = timedelta(hours=24)
WEIGHT_FUNC = early_weight # linearly weighs early predictions over later ones up to MARGIN
ALPHA = 1/2
THRESH = 0.05

# Pre-processing
NORMALIZE = True
STANDARDIZE = False

datahandler = DataHandler()

def load_config():
    # Load the configuration
    with open(os.path.join(os.path.abspath('hyperparameters.json')), "r") as f:
        config = json.load(f)
    return config

ETH_POOLS = [
    '0xdc24316b9ae028f1497c275eb9192a3ea0f67022',
    '0x5fae7e604fc3e24fd43a72867cebac94c65b404a',
    '0x828b154032950c8ff7cf8085d841723db2696056',
    '0xa1f8a6807c402e4a15ef4eba36528a3fed24e577',
]

ETH_TOKENS = [
    '0xae7ab96520de3a18e5e111b5eaab095312d7fe84',
    '0xbe9895146f7af43049ca1c1ae358b0541ea49704',
    '0x5e8422345238f34275888049021821e8e08caa1f',
]

def setup_pool(pool, metric, start, end):
    """
    Retrieve the relevant data for modeling a pool metric.

    :param pool: (str) The pool address
    :param metric: (str) The metric to model
    :param start: (str) The start date (ISO8601)
    :param end: (str) The end date (ISO8601)
    """
    metricsprocessor = MetricsProcessor(datahandler.pool_metadata, datahandler.token_metadata)
    name = datahandler.pool_metadata[pool]['name']

    start_ts = datetime.timestamp(datetime.fromisoformat(start))
    end_ts = datetime.timestamp(datetime.fromisoformat(end))

    lp_share_price = datahandler.get_pool_metric(pool, 'lpSharePrice', start_ts, end_ts) 

    if pool in ETH_POOLS:
        eth = '0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'
        eth_price = datahandler.get_ohlcv_data(eth, start=start_ts, end=end_ts)['close']
        lp_share_price /= eth_price
    
    snapshots = datahandler.get_pool_snapshots(pool, start_ts, end_ts)
    virtual_price = snapshots['virtualPrice'] / 10**18
    y_true = metricsprocessor.true_cps(lp_share_price, virtual_price, freq=FREQ, thresh=THRESH)
    X = datahandler.get_pool_X(metric, pool, start_ts, end_ts, FREQ, normalize=NORMALIZE, standardize=STANDARDIZE)

    return X, y_true, lp_share_price.resample(FREQ).last(), virtual_price, name

def setup_token(token, metric, start, end):
    """
    Retrieve the relevant data for modeling a token metric.

    :param pool: (str) The pool address
    :param metric: (str) The metric to model
    :param start: (str) The start date (ISO8601)
    :param end: (str) The end date (ISO8601)
    """
    metricsprocessor = MetricsProcessor(datahandler.pool_metadata, datahandler.token_metadata)
    symbol = datahandler.token_metadata[token]['symbol']

    start_ts = datetime.timestamp(datetime.fromisoformat(start))
    end_ts = datetime.timestamp(datetime.fromisoformat(end))

    ohlcv = datahandler.get_ohlcv_data(token, start_ts, end_ts)
    ohlcv['peg'] = 1
    peg = ohlcv['peg']
    price = ohlcv['close']

    if token in ETH_TOKENS:
        eth = '0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'
        eth_price = datahandler.get_ohlcv_data(eth, start=start_ts, end=end_ts)['close']
        price /= eth_price # Get numeraire price, all ohlcv prices in dollars
    
    y_true = metricsprocessor.true_cps(price, peg, freq=FREQ, thresh=THRESH)
    if metric == 'logReturns':
        metric = f'{datahandler.token_metadata[token]["symbol"]}.{metric}'
    X = datahandler.get_token_X(metric, token, start_ts, end_ts, FREQ, normalize=NORMALIZE, standardize=STANDARDIZE)

    return X, y_true, price.resample(FREQ).last(), peg, symbol

def test(pool, metric, start, end, params, pool_token):    
    """
    Run model.predict() with the trained hyperparameters.

    :param pool: (str) The pool address
    :param metric: (str) The metric to model
    :param start: (str) The start date (ISO8601)
    :param end: (str) The end date (ISO8601)
    :param params: (dict) The hyperparameters to use
    :param pool_token: (str) Whether to model a pool or token metric
    """
    if pool_token == 'pool':
        X, y_true, price, peg, name = setup_pool(pool, metric, start, end)
    elif pool_token == 'token':
        X, y_true, price, peg, name = setup_token(pool, metric, start, end)

    print(f"[{datetime.now()}] Testing {metric} on {name} from {start} to {end}")

    model = BOCD(margin=MARGIN, alpha=ALPHA, verbose=True, weight_func=WEIGHT_FUNC)
    model.update({'alpha':params['alpha'], 'beta':params['beta'], 'kappa':params['kappa']})
    y_pred, y_ps = model.predict(X)
    print(f'[{datetime.now()}] True CPs: {y_true}')
    print(f'[{datetime.now()}] Predicted CPs: {y_pred}')
    print(f'[{datetime.now()}] Predicted CPs probabilities: {y_ps}')
    F, P, R = f_measure(y_true, y_pred, margin=MARGIN, alpha=ALPHA, return_PR=True, weight_func=early_weight)
    print(f'[{datetime.now()}] FPR: {F}, Precision: {P}, Recall: {R}')

    # datahandler.insert_changepoints(model.y_pred, pool, 'bocd', metric)

    bocd_plot_comp(X, price, peg, y_true, y_pred, save=True, file=f'./figs/testing/{pool_token}/{metric}/{pool}.png', metric=metric, pool=name)
    print(f"\n[{datetime.now()}] Finished testing {name}\n")

    return params, (F, P, R)

def train(pool, metric, start, end, pool_token):
    """
    Run model.tune() and return best performing hyperparameters.

    :param pool: (str) The pool address
    :param metric: (str) The metric to model
    :param start: (str) The start date (ISO8601)
    :param end: (str) The end date (ISO8601)
    :param pool_token: (str) Whether to model a pool or token metric
    """
    if pool_token == 'pool':
        X, y_true, price, peg, name = setup_pool(pool, metric, start, end)
    elif pool_token == 'token':
        X, y_true, price, peg, name = setup_token(pool, metric, start, end)

    if not len(y_true):
        raise Exception("No CPs in training set")

    print(f"[{datetime.now()}] Training {metric} on {name} from {start} to {end}")

    model = BOCD(margin=MARGIN, alpha=ALPHA, verbose=True, weight_func=WEIGHT_FUNC)
    model.tune(GRID, X, y_true)
    y_pred = model.y_pred

    # datahandler.insert_changepoints(model.y_pred, pool, 'bocd', metric)

    bocd_plot_comp(X, price, peg, y_true, y_pred, save=True, file=f'./figs/training/{pool_token}/{metric}/{pool}.png', metric=metric, pool=name)
    print(f"\n[{datetime.now()}] Finished training {name}\n")

    return model.best_params_dict, model.best_results

def main():
    """
    Train and test hyperparameters for each combination of pool/token and metric. 
    Save corresponding results in hyperparameters.json and plots in ./figs directory.
    """

    print(f"[{datetime.now()}] Tuning hyperparameters...\n")
    print(f"[{datetime.now()}] Using margin: {MARGIN}")
    print(f"[{datetime.now()}] Using alpha: {ALPHA}")
    print(f"[{datetime.now()}] Testing grid of length {len(GRID)}\n")

    config = load_config()

    pool_metrics = ['shannonsEntropy', 'giniCoefficient', 'netSwapFlow', 'netLPFlow', '300.Markout']
    for metric in pool_metrics:

        ### TRAINING

        train_pool = "0xceaf7747579696a2f0bb206a14210e3c9e6fb269"
        train_start = "2022-01-01"
        train_end = "2022-06-01"

        params, _ = train(train_pool, metric, train_start, train_end, 'pool')
        
        ### TESTING

        test_end = "2023-05-01"

        for pool in datahandler.pool_metadata:
            if pool == train_pool:
                continue
            try:
                test_start = min('2022-01-01', datetime.fromtimestamp(datahandler.pool_metadata[pool]['creationDate']).__str__())
                _, results = test(pool, metric, test_start, test_end, params, 'pool')
            except Exception as e:
                print(f"\n[{datetime.now()}] Failed for {datahandler.pool_metadata[pool]['name']}: {e}\n")
                continue
    
    token_metrics = ['logReturns']
    for metric in token_metrics:
        
        ### TRAINING

        train_token = '0xa693b19d2931d498c5b318df961919bb4aee87a5'
        train_start = "2022-01-01"
        train_end = "2022-06-01"

        params, _ = train(train_token, metric, train_start, train_end, 'token')

        ### TESTING

        test_end = "2023-05-01"

        for token in datahandler.token_metadata:
            if token in ['0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee', '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2']:
                continue
            if token == train_token:
                continue
            try:
                test_start = '2022-01-01'
                _, results = test(token, metric, test_start, test_end, params, 'token')
            except Exception as e:
                print(f"\n[{datetime.now()}] Failed for {datahandler.token_metadata[token]['symbol']}: {e}\n")
                continue

    # Token metrics

    #         # Track results
    #         if pool not in config['hyperparameters']['bocd']:
    #             config['hyperparameters']['bocd'][pool] = {}
    #         if metric not in config['hyperparameters']['bocd'][pool]:
    #             config['hyperparameters']['bocd'][pool][metric] = {}
    #         config['hyperparameters']['bocd'][pool][metric] = model.params

    #         if pool not in config['results']:
    #             config['results'][pool] = {}
    #         if metric not in config['results'][pool]:
    #             config['results'][pool][metric] = {}
    #         config['results'][pool][metric] = [F, P, R]

    #         print(f'[{datetime.now()}]Successfully tuned hyperparameters for {metric} in {pool}\n')

    # with open(os.path.join(os.path.abspath('hyperparameters.json')), "w") as f:
    #     json.dump(config, f, indent=4)
    
if __name__ == "__main__":
    main()
    datahandler.close()