import numpy as np
from datetime import timedelta
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import cpu_count
import logging

# Local imports
from curvemetrics.src.detection.bocd_stream.bocd.bocd import BayesianOnlineChangePointDetection
from curvemetrics.src.detection.bocd_stream.bocd.distribution import StudentT
from curvemetrics.src.detection.bocd_stream.bocd.hazard import ConstantHazard
from curvemetrics.src.detection.scorer import f_measure, early_weight

class BOCD():

    default_params = {
        'lambda': 100,
        'alpha': 1,
        'beta':0.1,
        'kappa': 0.1,
        'mu': 0
    }

    def __init__(self, margin=timedelta(hours=24), alpha=1/5, verbose=False, weight_func=None, logger=None):
        self.model = BayesianOnlineChangePointDetection(
            ConstantHazard(self.default_params['lambda']), 
            StudentT(mu=self.default_params['mu'], 
            kappa=self.default_params['kappa'], 
            alpha=self.default_params['alpha'], 
            beta=self.default_params['beta'])
        )
        self.margin = margin
        self.alpha = alpha

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                self.logger.addHandler(logging.StreamHandler())

        self.results = {}
        self.params = self.default_params
        self.y_pred = None
        self.y_ps = None
        self.verbose = verbose
        self.weight_func = weight_func

        self.last_ts = None
    
    def update(self, params):
        new_params = {key: params.get(key) or self.default_params.get(key) for key in self.default_params.keys()}
        self.model.reset_params(
            hazard=ConstantHazard(new_params['lambda']),
            distribution=StudentT(
                mu=new_params['mu'], 
                kappa=new_params['kappa'], 
                alpha=new_params['alpha'], 
                beta=new_params['beta']
            )
        )
        self.params = new_params

    def predict(self, X):
        rt_mle = np.empty(X.shape)
        p = np.empty(X.shape)
        for i, x in enumerate(X):
            self.model.update(x)
            rt_mle[i] = self.model.rt
            p[i] = self.model.p
        cps = np.where(np.diff(rt_mle)!=1)[0]+1
        ps = p[cps] # probabilities
        return X.index[cps], ps

    def _tune(self, chunk, X, y_true):
        results = {}
        y_pred = []
        y_ps = []
        score = -1
        for a, b, k in chunk:
            self.update({'alpha': a, 'beta': b, 'kappa': k})
            pred, ps = self.predict(X)
            results[(a, b, k)] = f_measure(y_true, pred, margin=self.margin, alpha=self.alpha, return_PR=True, weight_func=self.weight_func)
            if results[(a, b, k)][0] > score:
                y_pred = pred
                y_ps = ps
                score = results[(a, b, k)][0]
        return results, y_pred, score, y_ps

    def tune(self, grid, X, y_true):
        num_cpus = cpu_count()
        if len(grid) <= num_cpus:
            num_cpus = len(grid)
        chunk_size = len(grid) // num_cpus
        chunks = [grid[i:i + chunk_size] for i in range(0, len(grid), chunk_size)]

        if self.verbose:
            self.logger.info(f'Processing {len(chunks)} chunks of length {len(chunks[0])}; {cpu_count()} cpus.\n')

        with Pool(processes=num_cpus) as pool:
            results = pool.map(lambda args: self._tune(*args), [(chunk, X, y_true) for chunk in chunks])
        for result in results:
            self.results.update(result[0])

        self.y_pred = max(results, key=lambda x: x[2])[1]
        self.y_ps = max(results, key=lambda x: x[2])[3]

        if self.verbose:
            self.logger.info('\nFinished tuning hyperparameters\n')
            # self.logger.info(f'Results: {self.results}')
            self.logger.info(f'Best Params: {self.best_params}')
            self.logger.info(f'FPR: {self.best_results}')
            self.logger.info(f'True CPs: {y_true}')
            self.logger.info(f'Predicted CPs: {self.y_pred}')
            self.logger.info(f'Predicted CPs Probabilities: {self.y_ps}\n')

    @property
    def best_params_dict(self):
        a, b, k = self.best_params
        params = {}
        params['alpha'] = a
        params['beta'] = b
        params['kappa'] = k
        return params

    @property
    def best_params(self):
        return max(self.results, key=lambda x: self.results[x][0])
    
    @property
    def best_results(self):
        return self.results[self.best_params]
