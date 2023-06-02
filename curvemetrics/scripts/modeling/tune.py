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

ALPHA = [10**i for i in range(-1, 5)]
BETA = [10**i for i in range(-5, 2)]
KAPPA = [10**i for i in range(-5, 2)]
GRID = [(a, b, k) for a in ALPHA for b in BETA for k in KAPPA]

FREQ = timedelta(hours=1)
MARGIN = timedelta(hours=24)
ALPHA = 1/5
THRESH = 0.05

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

def setup(pool, metric, start, end):
    metricsprocessor = MetricsProcessor(datahandler.pool_metadata, datahandler.token_metadata)
    pool_name = datahandler.pool_metadata[pool]['name']

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
    if not len(y_true):
        raise Exception("No CPs in dataset")
    X = datahandler.get_X(metric, pool, start_ts, end_ts, FREQ)

    return X, y_true, lp_share_price.resample(FREQ).last(), pool_name

def test(pool, metric, start, end, params):

    X, y_true, lp_share_price, pool_name = setup(pool, metric, start, end)

    print(f"[{datetime.now()}] Testing {metric} on {pool_name} from {start} to {end}")

    model = BOCD(margin=MARGIN, alpha=ALPHA, verbose=True)
    model.update({'alpha':params['alpha'], 'beta':params['beta'], 'kappa':params['kappa']})
    y_pred = model.predict(X)
    print(f'[{datetime.now()}] True CPs: {y_true}')
    print(f'[{datetime.now()}] Predicted CPs: {y_pred}')
    F, P, R = f_measure(y_true, y_pred, margin=MARGIN, alpha=ALPHA, return_PR=True, weight_func=early_weight)
    print(f'[{datetime.now()}]FPR: {F}, Precision: {P}, Recall: {R}')

    # datahandler.insert_changepoints(model.y_pred, pool, 'bocd', metric)

    bocd_plot_comp(X, lp_share_price, y_true, y_pred, save=True, file=f'./figs/testing/{metric}_{pool}.png', metric=metric, pool=pool_name)
    print(f"\n[{datetime.now()}] Finished testing {pool_name}\n")

    return params, (F, P, R)

def train(pool, metric, start, end):

    X, y_true, lp_share_price, pool_name = setup(pool, metric, start, end)

    print(f"[{datetime.now()}] Training {metric} on {pool_name} from {start} to {end}")

    model = BOCD(margin=MARGIN, alpha=ALPHA, verbose=True)
    model.tune(GRID, X, y_true)
    y_pred = model.y_pred

    # datahandler.insert_changepoints(model.y_pred, pool, 'bocd', metric)

    bocd_plot_comp(X, lp_share_price, y_true, y_pred, save=True, file=f'./figs/training/{metric}_{pool}.png', metric=metric, pool=pool_name)
    print(f"\n[{datetime.now()}] Finished training {pool_name}\n")

    return model.best_params_dict, model.best_results

def main():

    print(f"\n[{datetime.now()}] Tuning hyperparameters...\n")

    print(f"[{datetime.now()}] Using margin: {MARGIN}")
    print(f"[{datetime.now()}] Using alpha: {ALPHA}")
    print(f"[{datetime.now()}] Testing grid of length {len(GRID)}\n")

    config = load_config()

    metric = 'shannonsEntropy'

    ### TRAINING

    train_pool = "0xceaf7747579696a2f0bb206a14210e3c9e6fb269"
    train_start = "2022-01-01"
    train_end = "2022-06-01"

    params, _ = train(train_pool, metric, train_start, train_end)
    
    ### TESTING

    test_end = "2023-05-01"

    for pool in datahandler.pool_metadata:
        try:
            test_start = min('2022-01-01', datetime.fromtimestamp(datahandler.pool_metadata[pool]['creationDate']).__str__())
            _, results = test(pool, metric, test_start, test_end, params=params)
        except Exception as e:
            print(f"\n[{datetime.now()}] Failed for {datahandler.pool_metadata[pool]['name']}: {e}\n")
            continue

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