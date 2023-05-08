import numpy as np
import pandas as pd
import json
from typing import List

#TODO: Dumb that I resample in all of these funcs

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

def gini(df, freq='1min'):
    metric = df['inputTokenWeights'].apply(_gini).resample(freq).mean()
    metric.name = 'giniCoefficient'
    return metric

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

def shannons_entropy(df, freq='1min'):
    metric = df['inputTokenWeights'].apply(_shannons_entropy).resample(freq).mean()
    metric.name = 'shannonsEntropy'
    return metric

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
    flow = pd.merge(swap_in, swap_out, left_index=True, right_index=True, how='outer')
    metric = (flow['amountBought'] + flow['amountSold']).resample(freq).sum()
    metric.name = f'{symbol}.netSwapFlow'
    return metric

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
    flow = pd.merge(deposits, withdraws, left_index=True, right_index=True, how='outer')
    metric = (flow['deposits'] + flow['withdraws']).resample(freq).sum()
    metric.name = f'{symbol}.netLPFlow'
    return metric
    
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

def metrics_for_token(token_id, datahandler, token_metadata):
    # TODO: For now, just one token metric
    # TODO: This should probably be in the MetricsDataHandler class
    token_ohlcv = datahandler.get_ohlcv_data(token_id)

    metrics = []
    metrics.append(log_returns(token_ohlcv, token_metadata[token_id]['symbol']))
    metrics_df = pd.concat(metrics, axis=1)
    metrics_df = metrics_df.fillna(0)

    name = token_metadata[token_id]['symbol'].replace("/", "-")

    metrics_df.to_csv(f"./tmpdata/{name}.csv")

    return metrics_df

def metrics_for_pool(pool, datahandler, pool_metadata, token_metadata, freq='1min'):
    # TODO: Need to append token OHLCV data to these metrics
    # TODO: This should probably be in the MetricsDataHandler class
    pool_data = datahandler.get_pool_data(pool)
    swaps_data = datahandler.get_swaps_data(pool)
    lp_data = datahandler.get_lp_data(pool)

    metrics = []

    metrics.append(gini(pool_data))
    metrics.append(shannons_entropy(pool_data))

    for token_idx, token_id in enumerate(pool_metadata[pool]['coins']):
        metrics.append(net_swap_flow(swaps_data, token_id, token_metadata[token_id]['symbol'], freq=freq))
        metrics.append(net_lp_flow(lp_data, token_idx, token_metadata[token_id]['symbol'], freq=freq))

    metrics_df = pd.concat(metrics, axis=1)
    metrics_df = metrics_df.fillna(0)

    name = pool_metadata[pool]['symbol'].replace("/", "-")

    metrics_df.to_csv(f"./data/metrics/{name}.csv")

    return metrics_df
