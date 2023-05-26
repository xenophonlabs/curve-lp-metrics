import numpy as np
from datetime import timedelta
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import cpu_count
import logging

# Local imports
from ..detection.bocd_stream.bocd.bocd import BayesianOnlineChangePointDetection
from ..detection.bocd_stream.bocd.distribution import StudentT
from ..detection.bocd_stream.bocd.hazard import ConstantHazard
from ..detection.scorer import f_measure

class BOCD():

    default_params = {
        'lambda': 100,
        'alpha': 1,
        'beta':0.1,
        'kappa': 0.1,
        'mu': 0
    }

    def __init__(self, margin=timedelta(hours=24), alpha=1/5):

        self.model = BayesianOnlineChangePointDetection(
            ConstantHazard(self.default_params['lambda']), 
            StudentT(mu=self.default_params['mu'], 
            kappa=self.default_params['kappa'], 
            alpha=self.default_params['alpha'], 
            beta=self.default_params['beta'])
        )

        self.margin = margin
        self.alpha = alpha

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())

        self.results = {}
        self.params = self.default_params

        self.cps = None
    
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
        for i, x in enumerate(X):
            self.model.update(x)
            rt_mle[i] = self.model.rt

        return X.index[np.where(np.diff(rt_mle)!=1)[0]+1]
    
    def _tune(self, chunk, X, cps):
        results = {}
        cps = []
        score = 0
        for a, b, k in chunk:
            self.update({'alpha': a, 'beta': b, 'kappa': k})
            pred = self.predict(X)
            if len(pred) == 0:
                results[(a, b, k)] = (0, 0, 0)
            else:
                results[(a, b, k)] = f_measure({1: cps}, pred, margin=self.margin, alpha=self.alpha, return_PR=True)
            if results[(a, b, k)][0] > score:
                cps = pred
        self.logger.info('Finished processing chunk with alpha=%s, beta=%s, kappa=%s', a, b, k)
        return results, (cps, score)

    def tune(self, grid, X, cps):
        num_cpus = cpu_count()
        if len(grid) <= num_cpus:
            num_cpus = len(grid)
        chunk_size = len(grid) // num_cpus
        chunks = [grid[i:i + chunk_size] for i in range(0, len(grid), chunk_size)]

        with Pool(processes=num_cpus) as pool:
            result_list, (cps, score) = pool.map(lambda args: self._tune(*args), [(chunk, X, cps) for chunk in chunks])

        for result_dict in result_list:
            self.results.update(result_dict)

        self.cps = max(cps, key=lambda x: x[1])[0]
    
    @property
    def best_params(self):
        return max(self.results, key=lambda x: self.results[x][0])
    
    @property
    def best_results(self):
        return self.results[self.best_params]