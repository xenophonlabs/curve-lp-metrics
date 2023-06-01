"""
Train hyperparameters for BOCD models. Write them to config.json.
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

ALPHA = [10**i for i in range(-1, 4)]
BETA = [10**i for i in range(-3, 1)]
KAPPA = [10**i for i in range(-3, 1)]
GRID = [(a, b, k) for a in ALPHA for b in BETA for k in KAPPA]

FREQ = '1h'
MARGIN = timedelta(hours=24)
ALPHA = 1/5
THRESH = 0.01

def load_config():
    # Load the configuration
    with open(os.path.join(os.path.abspath('hyperparameters.json')), "r") as f:
        config = json.load(f)
    return config

ETH_POOLS = [
    '0xdc24316b9ae028f1497c275eb9192a3ea0f67022'
]

def test(pool, metric, start, end, params):
    datahandler = DataHandler()
    pool_name = datahandler.pool_metadata[pool]['name']

    start_ts = datetime.timestamp(datetime.fromisoformat(start))
    end_ts = datetime.timestamp(datetime.fromisoformat(end))

    print(f"[{datetime.now()}] Testing {metric} on {pool_name} from {start} to {end}")
    
    port = datahandler.get_pool_metric(pool, 'lpSharePrice', start_ts, end_ts).resample(FREQ).last()
    y_true = datahandler.get_changepoints(pool, 'baseline', 'baseline', start_ts, end_ts, freq=FREQ).index
    if not len(y_true):
        raise Exception("No CPs in dataset")
    X = datahandler.get_X(metric, pool, start_ts, end_ts, FREQ)

    model = BOCD(margin=MARGIN, alpha=ALPHA, verbose=True)
    model.update({'alpha':params['alpha'], 'beta':params['beta'], 'kappa':params['kappa']})
    y_pred = model.predict(X)
    print(f'Predicted CPs: {y_pred}')
    F, P, R = f_measure({1: y_true}, y_pred, margin=MARGIN, alpha=ALPHA, return_PR=True, weight_func=early_weight)
    print(f'[{datetime.now()}]FPR: {F}, Precision: {P}, Recall: {R}')

    # datahandler.insert_changepoints(model.y_pred, pool, 'bocd', metric)

    bocd_plot_comp(X, port, y_true, y_pred, save=True, file=f'./figs/testing/{metric}_{pool_name}.png', metric=metric)
    print(f"[{datetime.now()}] Finished testing {pool_name}")

    return params, (F, P, R)

def train(pool, metric, start, end):
    datahandler = DataHandler()
    metricsprocessor = MetricsProcessor(datahandler.pool_metadata, datahandler.token_metadata)
    pool_name = datahandler.pool_metadata[pool]['name']

    lp_share_price = datahandler.get_pool_metric(pool, 'lpSharePrice', start_ts, end_ts).resample(FREQ).mean() 

    if pool in ETH_POOLS:
        eth = '0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'
        eth_price = datahandler.get_ohlcv_data(eth, start=start_ts, end=end_ts)['close'].resample(FREQ).mean()
        lp_share_price /= eth_price

    start_ts = datetime.timestamp(datetime.fromisoformat(start))
    end_ts = datetime.timestamp(datetime.fromisoformat(end))

    print(f"[{datetime.now()}] Training {metric} on {pool_name} from {start} to {end}")
    
    snapshots = datahandler.get_pool_snapshots(pool, start_ts, end_ts)
    virtual_price = snapshots['virtualPrice'].resample(FREQ).mean() / 10**18
    y_true = metricsprocessor.true_cps(lp_share_price, virtual_price, thresh=THRESH)
    if not len(y_true):
        raise Exception("No CPs in dataset")
    X = datahandler.get_X(metric, pool, start_ts, end_ts, FREQ)

    model = BOCD(margin=MARGIN, alpha=ALPHA, verbose=True)
    model.tune(GRID, X, y_true)
    y_pred = model.y_pred

    # datahandler.insert_changepoints(model.y_pred, pool, 'bocd', metric)

    bocd_plot_comp(X, lp_share_price, y_true, y_pred, save=True, file=f'./figs/training/{metric}_{pool_name}.png', metric=metric)
    print(f"[{datetime.now()}] Finished training {pool_name}")

    datahandler.close()

    return model.best_params_dict, model.best_results

def main():

    print(f"\n[{datetime.now()}] Tuning hyperparameters...\n")

    print(f"Using margin: {MARGIN}")
    print(f"Using alpha: {ALPHA}")
    print(f"Testing grid of length {len(GRID)}: {GRID}\n")

    config = load_config()

    metric = 'shannonsEntropy'

    ### TRAINING

    train_pool = "0xceaf7747579696a2f0bb206a14210e3c9e6fb269"
    train_start = "2022-01-01"
    train_end = "2022-06-01"

    params, _ = train(train_pool, metric, train_start, train_end)
    
    ### TESTING

    # test_pool = "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7"
    # test_start = "2023-01-01"
    # test_end = "2023-06-01"

    # params, results = test(test_pool, metric, test_start, test_end, params=params)

    test_pool = "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7"
    test_start = "2023-01-01"
    test_end = "2023-06-01"

    params, results = test(test_pool, metric, test_start, test_end, params=params)

    test_pool = "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7"
    test_start = "2023-01-01"
    test_end = "2023-06-01"

    params, results = test(test_pool, metric, test_start, test_end, params=params)

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