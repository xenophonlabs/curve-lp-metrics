from datetime import datetime, timedelta
import logging
from curvemetrics.src.classes.model import BOCD
from curvemetrics.src.classes.metricsprocessor import MetricsProcessor
from curvemetrics.src.detection.scorer import f_measure, early_weight
from curvemetrics.src.plotting.plot import bocd_plot_comp

class ModelSetup():

    ETH_POOLS = [
        '0xdc24316b9ae028f1497c275eb9192a3ea0f67022',
        '0x5fae7e604fc3e24fd43a72867cebac94c65b404a',
        '0x828b154032950c8ff7cf8085d841723db2696056',
        '0xa1f8a6807c402e4a15ef4eba36528a3fed24e577',
    ]

    CRV_POOLS = [
        '0x971add32ea87f10bd192671630be3be8a11b8623'
    ]

    ALPHA = [10**i for i in range(-5, 5)]
    BETA = [10**i for i in range(-5, 5)]
    KAPPA = [10**i for i in range(-5, 5)]

    def __init__(self,
                 datahandler,
                 normalize=False, 
                 standardize=True, 
                 alpha=1/2, 
                 thresh=0.05, 
                 weight_func=early_weight, 
                 freq=timedelta(hours=1), 
                 margin=timedelta(hours=24),
                 plotting=False,
                 logger=None
    ):
        self.datahandler = datahandler
        self.metricsprocessor = MetricsProcessor(self.datahandler.pool_metadata, self.datahandler.token_metadata)

        self.normalize = normalize
        self.standardize = standardize
        self.alpha = alpha
        self.thresh = thresh
        self.weight_func = weight_func
        self.freq = freq
        self.margin = margin
        self.plotting = plotting

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())

        self.GRID = [(a, b, k) for a in self.ALPHA for b in self.BETA for k in self.KAPPA]

    def setup_pool(self, pool, start, end):
        """
        Retrieve the LP Share Price and Virtual Price.
        Check whether a changepoint occured.

        :param pool: (str) The pool address
        :param metric: (str) The metric to model
        :param start: (str) The start date (ISO8601)

        :returns y_true: (pd.Series) The true changepoints
        :returns lp_share_price: (pd.Series) The LP Share Price
        :returns virtual_price: (pd.Series) The Virtual Price
        :returns name: (str) The pool name
        """
        name = self.datahandler.pool_metadata[pool]['name']
        lp_share_price = self.datahandler.get_pool_metric(pool, 'lpSharePrice', start, end) 
        if pool in self.ETH_POOLS:
            eth_price = self.datahandler.get_ohlcv_data(self.datahandler.token_ids['ETH'], start=start, end=end)['close']
            lp_share_price /= eth_price
        elif pool in self.CRV_POOLS:
            crv_price = self.datahandler.get_ohlcv_data(self.datahandler.token_ids['CRV'], start=start, end=end)['close']
            lp_share_price /= crv_price
        snapshots = self.datahandler.get_pool_snapshots(pool, start, end)
        virtual_price = snapshots['virtualPrice'] / 10**18
        y_true = self.metricsprocessor.true_cps(lp_share_price, virtual_price, freq=self.freq, thresh=self.thresh)
        # datahandler.insert_changepoints(model.y_pred, address, 'bocd', metric)
        return y_true, lp_share_price.resample(self.freq).last(), virtual_price, name

    def setup_token(self, token, metric, start, end):
        """
        Retrieve the relevant data for modeling a token metric.

        :param pool: (str) The pool address
        :param metric: (str) The metric to model
        :param start: (str) The start date (ISO8601)
        :param end: (str) The end date (ISO8601)
        """
        symbol = self.datahandler.token_metadata[token]['symbol']
        ohlcv = self.datahandler.get_ohlcv_data(token, start, end)
        ohlcv['peg'] = 1
        peg = ohlcv['peg']
        price = ohlcv['close']
        numeraire = ohlcv['symbol'].dropna().unique()[0].split('/')[1]
        if numeraire != 'USD':
            numeraire_price = self.datahandler.get_ohlcv_data(self.datahandler.token_ids[numeraire], start=start, end=end)['close']
            price /= numeraire_price # Get numeraire price, all ohlcv prices in dollars
        y_true = self.metricsprocessor.true_cps(price, peg, freq=self.freq, thresh=self.thresh)
        if metric == 'logReturns':
            metric = f'{self.datahandler.token_metadata[token]["symbol"]}.{metric}'
        # datahandler.insert_changepoints(model.y_pred, address, 'bocd', metric)
        return y_true, price.resample(self.freq).last(), peg, symbol

    def test(self, address, metrics, start, end, params, pool_token):    
        """
        Run model.predict() with the trained hyperparameters for each metric.

        :param pool: (str) The pool address
        :param metric: (list) The metrics to model
        :param start: (str) The start date (ISO8601)
        :param end: (str) The end date (ISO8601)
        :param params: (dict) The hyperparameters to use
        :param pool_token: (str) Whether to model a pool or token metric

        :returns results: (list) The best performing results for each metric
        """
        if pool_token == 'pool':
            y_true, price, peg, name = self.setup_pool(address, metric, start, end)
        elif pool_token == 'token':
            y_true, price, peg, name = self.setup_token(address, metric, start, end)

        self.logger.info(f"Testing {name} from {datetime.fromtimestamp(start)} to {datetime.fromtimestamp(end)}.\n")

        results = []

        for metric in metrics:
            self.logger.info(f" Testing {metric}.")

            if pool_token == 'pool':
                X = self.datahandler.get_pool_X(metric, address, start, end, self.freq, normalize=self.normalize, standardize=self.standardize)
            elif pool_token == 'token':
                X = self.datahandler.get_token_X(metric, address, start, end, self.freq, normalize=self.normalize, standardize=self.standardize)

            metric_params = params[metric]
            model = BOCD(margin=self.margin, alpha=self.alpha, verbose=True, weight_func=self.weight_func)
            model.update({'alpha':metric_params['alpha'], 'beta':metric_params['beta'], 'kappa':metric_params['kappa']})

            y_pred, y_ps = model.predict(X)
            self.logger.info(f'True CPs: {y_true}')
            self.logger.info(f'Predicted CPs: {y_pred}')
            self.logger.info(f'Predicted CPs probabilities: {y_ps}')
            F, P, R = f_measure(y_true, y_pred, margin=self.margin, alpha=self.alpha, return_PR=True, weight_func=self.weight_func)
            self.logger.info(f'F-score: {F}, Precision: {P}, Recall: {R}')

            # datahandler.insert_changepoints(model.y_pred, address, 'bocd', metric)

            if self.plotting:
                bocd_plot_comp(X, price, peg, y_true, y_pred, save=True, file=f'./figs/testing/{pool_token}/{metric}/{address}.png', metric=metric, pool=name)

            results.append([address, metric, F, P, R, metric_params])

            self.logger.info(f"Finished testing {name}\n")

        return results

    def train(self, address, metrics, start, end, pool_token):
        """
        Run model.tune() and return best performing hyperparameters 
        and results for each metrics in a dictionary.

        :param pool: (str) The pool address
        :param metric: (list) The metrics to model
        :param start: (str) The start date (ISO8601)
        :param end: (str) The end date (ISO8601)
        :param pool_token: (str) Whether to model a pool or token metric

        :returns params: (dict) The best performing hyperparameters for each metric
        :returns results: (dict) The best performing results for each metric
        """
        if pool_token == 'pool':
            y_true, price, peg, name = self.setup_pool(address, start, end)
        elif pool_token == 'token':
            y_true, price, peg, name = self.setup_token(address, start, end)
        
        if not len(y_true):
            self.logger.error("No CPs in training set")
            raise Exception("No CPs in training set")
        
        self.logger.info(f"Training {name} from {datetime.fromtimestamp(start)} to {datetime.fromtimestamp(end)}.\n")

        params = {}

        for metric in metrics:
            self.logger.info(f" Training {metric}.")

            if pool_token == 'pool':
                X = self.datahandler.get_pool_X(metric, address, start, end, self.freq, normalize=self.normalize, standardize=self.standardize)
            elif pool_token == 'token':
                X = self.datahandler.get_token_X(metric, address, start, end, self.freq, normalize=self.normalize, standardize=self.standardize)

            model = BOCD(margin=self.margin, alpha=self.alpha, verbose=True, weight_func=self.weight_func)
            model.tune(self.GRID, X, y_true)

            # datahandler.insert_changepoints(model.y_pred, address, 'bocd', metric)

            if self.plotting:
                bocd_plot_comp(X, price, peg, y_true, model.y_pred, save=True, file=f'./figs/training/{pool_token}/{metric}/{address}.png', metric=metric, pool=name)

            self.logger.info(f"Finished training {name}\n")

            params[metric] = model.best_params_dict

        return params
