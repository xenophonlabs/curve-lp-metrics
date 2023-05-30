import json
import warnings

from scipy.optimize import minimize
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import numpy as np

class MetricsProcessor:

    __metrics__ = [
        'giniCoefficient',
        'shannonsEntropy',
        'netSwapFlow',
        'absSwapFlow',
        'netLPFlow',
        'absLPFlow',
        'logReturns',
        'PIN',
        'markout',
        'sharks'
    ]

    def __init__(self, pool_metadata, token_metadata, freq='1min'):
        """
        Initialize MetricsProcessor with a datahandler instance.

        :param datahandler: Instance of a datahandler.
        :param max_workers: Maximum number of workers to use for concurrent tasks.
        """
        self.pool_metadata = pool_metadata
        self.token_metadata = token_metadata
        self.freq = freq
    
    @staticmethod
    def round_date(x):
        mydate = datetime.fromtimestamp(x)
        rounded_minute = mydate.minute + round(mydate.second/60)
        minute_difference = rounded_minute - mydate.minute
        mydate = mydate.replace(second=0) + timedelta(minutes=minute_difference)
        return mydate

    def process_metrics_for_pool(self, pool_id, pool_data, swaps_data, lp_data, ohlcvs):

        metrics = []

        metrics.extend([
            MetricsProcessor.gini(pool_data, freq=self.freq),
            MetricsProcessor.shannons_entropy(pool_data, freq=self.freq)
        ])

        tokens = set(swaps_data['tokenBought']).union(set(swaps_data['tokenSold']))

        for token_id in tokens:
            metrics.extend([
                MetricsProcessor.net_swap_flow(swaps_data, token_id, self.token_metadata[token_id]['symbol'], freq=self.freq),
                # MetricsProcessor.abs_swap_flow(swaps_data, token_id, self.token_metadata[token_id]['symbol'], freq=self.freq),
                # MetricsProcessor.rolling_pin(swaps_data, token_id, self.token_metadata[token_id]['symbol'], window=timedelta(days=7), freq=timedelta(days=1)),
                # MetricsProcessor.markout(swaps_data, ohlcvs, window=timedelta(minutes=5), who='lp', freq=self.freq),
            ])

        for token_idx, token_id in enumerate(self.pool_metadata[pool_id]['coins']):
            metrics.extend([
                MetricsProcessor.net_lp_flow(lp_data, token_idx, self.token_metadata[token_id]['symbol'], freq=self.freq),
                # MetricsProcessor.abs_lp_flow(lp_data, token_idx, self.token_metadata[token_id]['symbol'], freq=self.freq)
            ])
        
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
    def gini(df, freq='1min') -> pd.Series:
        metric = df['inputTokenBalances'].apply(MetricsProcessor._gini).resample(freq).last().fillna(method='ffill')
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
        x = np.array(x)
        proportions = x / np.sum(x)
        entropy = -np.sum(proportions * np.log2(proportions))
        return entropy

    @staticmethod
    def shannons_entropy(df, freq='1min') -> pd.Series:
        metric = df['inputTokenBalances'].apply(MetricsProcessor._shannons_entropy).resample(freq).last().fillna(method='ffill')
        metric.name = 'shannonsEntropy'
        return metric

    @staticmethod
    def net_swap_flow(df, token_id, symbol, freq='1min') -> pd.Series:
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
        swap_in = df[df['tokenBought']==token_id]['amountBought'].groupby(level=0).sum()
        swap_out = -1*df[df['tokenSold']==token_id]['amountSold'].groupby(level=0).sum()
        flow = pd.merge(swap_in, swap_out, left_index=True, right_index=True, how='outer').fillna(0)
        metric = (flow['amountBought'] + flow['amountSold']).resample(freq).sum()
        metric.name = f'{symbol}.netSwapFlow'
        return metric

    @staticmethod
    def net_lp_flow(df, token_idx, symbol, freq='1min') -> pd.Series:
        if len(df) == 0:
            return pd.Series([], name=f'{symbol}.netLPFlow')
        deposits = df[df['removal']==False]
        deposits = deposits['tokenAmounts'].apply(lambda x: json.loads(x)[token_idx]).groupby(level=0).sum().groupby(level=0).sum()
        deposits.name = "deposits"
        withdraws = df[df['removal']==True]
        withdraws = withdraws['tokenAmounts'].apply(lambda x: -1*json.loads(x)[token_idx]).groupby(level=0).sum().groupby(level=0).sum()
        withdraws.name = "withdraws"
        flow = pd.merge(deposits, withdraws, left_index=True, right_index=True, how='outer').fillna(0)
        metric = (flow['deposits'] + flow['withdraws']).resample(freq).sum()
        metric.name = f'{symbol}.netLPFlow'
        return metric
        
    @staticmethod
    def log_returns(df, symbol, freq='1min') -> pd.Series:
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

    @staticmethod
    def abs_swap_flow(df, token_id, symbol, freq='1min') -> pd.Series:
        """
        Calculate the absolute swap flow of a token in a pool in the discretized frequencies.

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
        swap_in = df[df['tokenBought']==token_id]['amountBought'].groupby(level=0).sum()
        swap_out = df[df['tokenSold']==token_id]['amountSold'].groupby(level=0).sum()
        flow = pd.merge(swap_in, swap_out, left_index=True, right_index=True, how='outer').fillna(0)
        metric = (flow['amountBought'] + flow['amountSold']).resample(freq).sum()
        metric.name = f'{symbol}.absSwapFlow'
        return metric

    @staticmethod
    def abs_lp_flow(df, token_idx, symbol, freq='1min') -> pd.Series:
        if len(df) == 0:
            return pd.Series([], name=f'{symbol}.netLPFlow')
        deposits = df[df['removal']==False]
        deposits = deposits['tokenAmounts'].apply(lambda x: json.loads(x)[token_idx]).groupby(level=0).sum()
        deposits.name = "deposits"
        withdraws = df[df['removal']==True]
        withdraws = withdraws['tokenAmounts'].apply(lambda x: json.loads(x)[token_idx]).groupby(level=0).sum()
        withdraws.name = "withdraws"
        flow = pd.merge(deposits, withdraws, left_index=True, right_index=True, how='outer').fillna(0)
        metric = (flow['deposits'] + flow['withdraws']).resample(freq).sum()
        metric.name = f'{symbol}.absLPFlow'
        return metric
    
    ### PIN ###

    @staticmethod
    def init_pin_params(df):
        """
        PIN MLE initial params.
        """
        avg_B = df['Buy'].mean()
        avg_S = df['Sell'].mean()
        alph = 0.1
        delt = 0.3
        gamm = 0.5
        epsiB = gamm*avg_B
        miu = (avg_B-epsiB)/(alph*(1-delt))
        epsiS = avg_S-alph*delt*miu
        return [alph,delt,miu,epsiB,epsiS]

    @staticmethod
    def pin_likelihood_EHO(params, df):
        """
        Estimate joint likelihood function using factorization from Easley, Hvidkjaer, and O’Hara (2010)
        
        @Params
            params (tuple): \alpha, \delta \mu, \epsilon_B, \epsilon_S 
            df (pd.DataFrame): timeseries DataFrame with Buy and Sell columns
        
        @Returns
            likelihood (float): joint likelihood function, the likelihood of the params given the buy and sell flow
        
        @Credit: modified from https://github.com/shuangology/Probability-of-Informed-Trading/tree/master 
            by shuangology
        """
        #initialize parameter values
        alph, delt, mu, epsiB, epsiS = params

        # Constraints from the model
        if any(x < 0 for x in params) or alph > 1 or delt > 1:
            return np.inf

        #number of buy- and sell-trades for the trading day
        buy_s = df['Buy'].values
        sell_s = df['Sell'].values

        #compute values of interest for the log-likelihood function
        M  = np.minimum(buy_s,sell_s) + np.maximum(buy_s,sell_s)/2
        Xs = epsiS/(mu+epsiS)
        Xb = epsiB/(mu+epsiB)

        a1 = np.exp(-mu)
        a2 = np.where(sell_s-M >= 0, Xs**(sell_s-M), 0)
        a3 = np.where(buy_s-M >= 0, Xb**(buy_s-M), 0)
        a4 = np.where(M >= 0, Xs**(-M), 0)
        a5 = np.where(M >= 0, Xb**(-M), 0)

        part1 = -epsiB-epsiS+M*(np.log(Xb)+np.log(Xs))+buy_s*np.log(mu+epsiB)+sell_s*np.log(mu+epsiS)
        part2 = np.log(alph*(1-delt)*a1*a2*a5+alph*delt*a1*a3*a4+(1-alph)*a2*a3)

        # Replace Inf and NaN values with a very large number
        inf_mask = (Xb <= 0) | (Xs <= 0) | (mu+epsiB <= 0) | (mu+epsiS <= 0) | (a1*a2*a5 <= 0) | (a1*a3*a4 <= 0) | (a2*a3 <= 0)
        part1 = np.where(inf_mask, np.inf, part1)
        part2 = np.where(inf_mask, np.inf, part2)

        return -1 * np.nansum(part1 + part2)

    @staticmethod
    def _pin(params) -> float:
        """
        Calculate PIN value given the params.

        @Params
            params (tuple): \alpha, \delta \mu, \epsilon_B, \epsilon_S
        
        @Returns
            PIN (float): PIN value
        """
        alph,delt,mu,epsiB,epsiS = params
        return (alph*mu)/(alph*mu+epsiB+epsiS)

    @staticmethod
    def pin(df) -> float:
        """
        Given a df corresponding to a window of data, calculate the PIN.

        @Params
            df (pd.DataFrame): timeseries DataFrame with Buy and Sell columns
        
        @Returns
            PIN (float): PIN value for the window
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning) # log runtime warnings don't matter
            initial_params = MetricsProcessor.init_pin_params(df)
            opt_params = minimize(MetricsProcessor.pin_likelihood_EHO, initial_params, args = (df),method = 'Nelder-Mead').x
            return MetricsProcessor._pin(opt_params)
    
    # NOTE: 1Inch executor contract is considered a "buyer" (i.e. not the actual address that
    # submitted the 1Inch transaction). This is okay when looking at sharks: we can assume
    # sharks are less likely to be going through 1Inch.

    @staticmethod
    def rolling_pin(df, token, symbol, window=timedelta(days=1), freq=timedelta(hours=1)) -> pd.Series:
        """
        Calculate PIN values for each resampled period. We take a swaps_df, convert it into 
        a timeseries dataframe of "Buy" and "Sell" columns (which count swaps for a token).
        Then we partition our DF into windows of size `window` and calculate the PIN value
        on that window. This PIN corresponds to the value of the last timestamp in the window.

        @Params
            df (pd.DataFrame): swaps_df
            window (int): window size (e.g. 100 minutes, or 100 hours, etc..)
            freq (str): resample frequency
        
        @Returns
            PIN (pd.Series): PIN values for each resampled period
        """
        token_df = df.copy()
        token_df['Buy'] = (token_df['tokenBought'] == token)
        token_df['Sell'] = (token_df['tokenSold'] == token)
        token_df['timestamp'] = token_df['timestamp'].apply(datetime.fromtimestamp)
        token_df = token_df.set_index('timestamp')
        token_df = token_df.resample(freq).agg({'Buy': 'sum', 'Sell': 'sum'})

        curr = token_df.index[0]
        end = token_df.index[-1]
        while curr <= end - window:
            token_df.loc[curr+window, f'{symbol}.PIN'] = MetricsProcessor.pin(token_df.loc[curr:curr+window])
            curr += freq
        
        return token_df[f'{symbol}.PIN']

    ### Markout

    @staticmethod
    def get_markout(df, ohlcvs, window, who) -> pd.DataFrame:
        """
        We define markout as the difference in value between a portfolio
        at the markout time vs at the current time. That is, we subtract the value
        of the tokens sold from the value of the tokens bought. This is different
        from other markout definition, which take the difference between the 
        markout value and the value at execution.

        This gives us markout info per swap to index the sharks.

        NOTE: Could alternatively be using curve `candles` data instead of CEX/Chainlink prices
        """

        tmp = df.copy()

        markout_col = f'{int(window.total_seconds())}.Markout'

        tmp['roundedDate'] = tmp['timestamp'].apply(MetricsProcessor.round_date)
        last = tmp['roundedDate'].iloc[-1]
        tmp = tmp[tmp['roundedDate'] <= last - window]
        tmp['markoutBoughtPrice'] = tmp.apply(lambda x: ohlcvs[x['tokenBought']].loc[x['roundedDate'] + window]['close'], axis=1)
        tmp['markoutSoldPrice'] = tmp.apply(lambda x: ohlcvs[x['tokenSold']].loc[x['roundedDate'] + window]['close'], axis=1)
        tmp[markout_col] = tmp['amountBought']*tmp['markoutBoughtPrice'] - tmp['amountSold']*tmp['markoutSoldPrice']

        if who == 'swapper':
            return tmp
        elif who == 'lp':
            tmp[markout_col] *= -1
            return tmp
        else:
            raise ValueError(f"who must be 'swapper' or 'lp', was {who}.")
    
    @staticmethod
    def markout(df, ohlcvs, window=timedelta(days=1), who='lp', freq='1min') -> pd.Series:
        """
        Convenience wrapper for get_markout(.)

        1. extract pd.Series
        2. resample to required frequency
        3. shift forward by window (so markout at time t is the markout of trades at time t-window with markout prices at t)
        """
        markout_col = f'{int(window.total_seconds())}.Markout'
        markouts = MetricsProcessor.get_markout(df, ohlcvs, window, who)[markout_col]
        markouts = markouts.resample(freq).sum()
        markouts.index += pd.Timedelta(window)
        markouts.name = markout_col
        return markouts

    ### Sharks

    def lp_share_price(self, pool, pool_data, ohlcvs) -> pd.Series:
        """
        Computes the LP Share Price for the given pool and the changepoints.
        """

        df = pool_data[['inputTokenBalances', 'outputTokenSupply']].resample('1min').last().fillna(method='ffill')

        tokens = self.pool_metadata[pool]['inputTokens']

        closes = pd.DataFrame()
        for token in tokens:
            closes[self.token_metadata[token]['symbol']] = ohlcvs[token]['close']
        
        closes['prices'] = closes.apply(lambda row: np.array(row.tolist()), axis=1)
        closes = closes[['prices']]

        df = df.join(closes)

        df['lpSharePrice'] = (df['inputTokenBalances'] * df['prices']) / (df['outputTokenSupply'] / 10**18)
        df['lpSharePrice'] = df['lpSharePrice'].apply(lambda x: np.sum(x))

        return df['lpSharePrice']

    def true_cps(self, lp_share_price, snapshots, freq = timedelta(minutes=1), thresh = 0.05):
        """
        Our baseline model
        """
        vp = (snapshots['virtualPrice']/10**18).resample(freq).mean().fillna(method='ffill')
        rp = lp_share_price.resample(freq).mean().fillna(method='ffill')

        error = abs((vp - rp) / vp)

        cps = error[error > thresh].index
        cps = np.array([cp for i, cp in enumerate(cps) if cp != cps[i-1] + freq])
        
        return cps