import pandas as pd
import numpy as np
from typing import List
import json

class MetricsProcessor:

    def __init__(self, pool_metadata, token_metadata, freq='1min'):
        """
        Initialize MetricsProcessor with a datahandler instance.

        :param datahandler: Instance of a datahandler.
        :param max_workers: Maximum number of workers to use for concurrent tasks.
        """
        self.pool_metadata = pool_metadata
        self.token_metadata = token_metadata
        self.freq = freq
    
    def process_metrics_for_pool(self, pool_id, pool_data, swaps_data, lp_data):
        metrics = []

        metrics.append(MetricsProcessor.gini(pool_data))
        metrics.append(MetricsProcessor.shannons_entropy(pool_data))

        for token_idx, token_id in enumerate(self.pool_metadata[pool_id]['coins']):
            metrics.append(MetricsProcessor.net_swap_flow(swaps_data, token_id, self.token_metadata[token_id]['symbol'], freq=self.freq))
            metrics.append(MetricsProcessor.net_lp_flow(lp_data, token_idx, self.token_metadata[token_id]['symbol'], freq=self.freq))
        
        metrics_df = pd.concat(metrics, axis=1)
        metrics_df = metrics_df.fillna(0)
        
        return metrics_df

    def process_metrics_for_token(self, token_id, token_ohlcv):
        metrics = []
        metrics.append(MetricsProcessor.log_returns(token_ohlcv, self.token_metadata[token_id]['symbol'], freq=self.freq))
    
        metrics_df = pd.concat(metrics, axis=1)
        metrics_df = metrics_df.fillna(0)

        return metrics_df
    
    @staticmethod
    def _gini(x: List) -> int:
        """
        Gini coefficient measures the inequality in the pool. 

        @Params
            x : Array 
                Token weights or balances.
        
        @Returns
            coef : Double 
                Gini coefficient 
        """
        x = np.array(x)
        x.sort()
        n = len(x)
        index = np.arange(1, n + 1)
        coef = (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))
        return coef

    @staticmethod
    def gini(df, freq='1min'):
        metric = df['inputTokenWeights'].apply(MetricsProcessor._gini).resample(freq).mean()
        metric.name = 'giniCoefficient'
        return metric

    @staticmethod
    def _shannons_entropy(x: List) -> int:
        """
        Imagine a pool is a basket and each unit of each asset is a ball with that asset's color.
        Shannon entropy [loosely] measures how easy it is to predict the color of a ball picked at random.

        @Params
            x : Array
                Token weights or balances
        
        @Returns
            entropy : Double
                Shannon's Entropy measurement
        """
        proportions = x / np.sum(x)
        entropy = -np.sum(proportions * np.log2(proportions))
        return entropy

    @staticmethod
    def shannons_entropy(df, freq='1min'):
        metric = df['inputTokenWeights'].apply(MetricsProcessor._shannons_entropy).resample(freq).mean()
        metric.name = 'shannonsEntropy'
        return metric

    @staticmethod
    def net_swap_flow(df, token_id, symbol, freq='1min'):
        # TODO: Need to ensure we get zeroes between start and end time
        """
        Calculate the net swap flow of a token in a pool in the discretized frequencies.

        @Params:    
            df : pd.DataFrame
                DataFrame of swap events
            token : str
                token_id

        @Returns
            flow['netSwapFlow'] : pd.Series
                net swap flow of the token in the pool
        """
        if len(df) == 0:
            return pd.Series([], name=f'{symbol}.netSwapFlow')
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        swap_in = df[df['tokenBought']==token_id]['amountBought']
        swap_out = -1*df[df['tokenSold']==token_id]['amountSold']
        flow = pd.merge(swap_in, swap_out, left_index=True, right_index=True, how='outer').fillna(0)
        metric = (flow['amountBought'] + flow['amountSold']).resample(freq).sum()
        metric.name = f'{symbol}.netSwapFlow'
        return metric

    @staticmethod
    def net_lp_flow(df, token_idx, symbol, freq='1min'):
        # TODO: Need to ensure we get zeroes between start and end time
        if len(df) == 0:
            return pd.Series([], name=f'{symbol}.netLPFlow')
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        deposits = df[df['removal']==False]
        deposits = deposits['tokenAmounts'].apply(lambda x: json.loads(x)[token_idx])
        deposits.name = "deposits"
        withdraws = df[df['removal']==True]
        withdraws = withdraws['tokenAmounts'].apply(lambda x: -1*json.loads(x)[token_idx])
        withdraws.name = "withdraws"
        flow = pd.merge(deposits, withdraws, left_index=True, right_index=True, how='outer').fillna(0)
        metric = (flow['deposits'] + flow['withdraws']).resample(freq).sum()
        metric.name = f'{symbol}.netLPFlow'
        return metric
        
    @staticmethod
    def log_returns(df, symbol, freq='1min'):
        """
        Calculate the log returns of a token in a pool in the discretized frequencies.

        @Params:    
            df : pd.DataFrame
                DataFrame of swap events
            symbol : str
                token symbol

        @Returns
            
        """
        metric = np.log(df['close']/df['close'].shift()).resample(freq).sum()
        metric.name = f'{symbol}.logReturns'
        return metric
