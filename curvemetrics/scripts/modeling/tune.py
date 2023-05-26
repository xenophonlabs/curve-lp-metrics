"""
Train hyperparameters for BOCD models. Write them to config.json.
"""
import os
import json
from datetime import datetime, timedelta
import numpy as np

from ...src.classes.model import BOCD
from ...src.classes.datahandler import DataHandler
from ...src.detection.scorer import f_measure

SPLIT = 0.75
FREQ = '1min'
GRID = [
    [10**i for i in range(1, 2)], # alpha
    [10**i for i in range(-3, -2)], # beta
    [10**i for i in range(-3, -2)], # kappa
]
MARGIN = timedelta(hours=24)
ALPHA = 1/5

def load_config():
    # Load the configuration
    with open(os.path.join(os.path.abspath('hyperparameters.json')), "r") as f:
        config = json.load(f)
    return config

def main():

    print(f"\n[{datetime.now()}] Tuning hyperparameters...\n")

    datahandler = DataHandler()
    config = load_config()

    for (pool, (train, test)) in config['splits'].items():

        train_start_ts = datetime.timestamp(datetime.fromisoformat(train['start']))
        train_end_ts = datetime.timestamp(datetime.fromisoformat(train['end']))
        test_start_ts = datetime.timestamp(datetime.fromisoformat(test['start']))
        test_end_ts = datetime.timestamp(datetime.fromisoformat(test['end']))

        print(f"[{datetime.now()}] Training {datahandler.pool_metadata[pool]['name']} from {train['start']} to {train['end']}")
        print(f"[{datetime.now()}] Testing {datahandler.pool_metadata[pool]['name']} from {test['start']} to {test['end']}")

        for metric in ['shannonsEntropy']:

            print(f"\nTraining {metric}\n")
            
            data = datahandler.get_pool_metric(pool, metric, train_start_ts, train_end_ts)

            X = np.log1p(data.pct_change()).dropna().resample(FREQ).mean()
            y_true = datahandler.get_changepoints(pool, 'baseline', 'baseline', train_start_ts, train_end_ts)

            model = BOCD()
            model.tune(GRID, X, y_true)

            print(f'Results: {model.results}\n')
            print(f'Best Params: {model.best_params}')
            print(f'FPR: {model.best_results}')

            # TODO: insert predicted cps from best_params here? Need to track them in the tune method
            # datahandler.insert_changepoints(y_pred, pool, 'bocd', metric)

            a, b, k = model.best_params

            print(f"\nTesting {metric}\n")

            model = BOCD()
            model.update({'alpha':a, 'beta':b, 'kappa':k})

            data = datahandler.get_pool_metric(pool, metric, test_start_ts, test_end_ts)
            X = np.log1p(data.pct_change()).dropna().resample(FREQ).mean()

            y_true = datahandler.get_changepoints(pool, 'baseline', 'baseline', train_start_ts, train_end_ts)
            y_pred = model.predict(X)
            datahandler.insert_changepoints(y_pred, pool, 'bocd', metric)

            if len(y_pred) == 0:
                F, P, R = 0, 0, 0
            else:
                F, P, R = f_measure({1: y_true}, y_pred, margin=MARGIN, alpha=ALPHA, return_PR=True)

            print(f'FPR: {F}, Precision: {P}, Recall: {R}')

            # Track results
            if pool not in config['hyperparameters']['bocd']:
                config['hyperparameters']['bocd'][pool] = {}
            if metric not in config['hyperparameters']['bocd'][pool]:
                config['hyperparameters']['bocd'][pool][metric] = {}
            config['hyperparameters']['bocd'][pool][metric] = model.params

            if pool not in config['results']:
                config['results'][pool] = {}
            if metric not in config['results'][pool]:
                config['results'][pool][metric] = {}
            config['results'][pool][metric] = [F, P, R]

    with open(os.path.join(os.path.abspath('hyperparameters.json')), "w") as f:
        json.dump(config, f, indent=4)
    
if __name__ == "__main__":
    main()