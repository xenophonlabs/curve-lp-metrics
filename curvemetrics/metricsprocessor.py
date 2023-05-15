import pandas as pd
import numpy as np
from typing import List
import json
from scipy.optimize import minimize
from datetime import datetime, timedelta

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
    
    @staticmethod
    def round_date(x):
        mydate = datetime.fromtimestamp(x)
        rounded_minute = mydate.minute + round(mydate.second/60)
        minute_difference = rounded_minute - mydate.minute
        mydate = mydate.replace(second=0) + timedelta(minutes=minute_difference)
        return mydate
    
    def process_metrics_for_pool(self, pool_id, pool_data, swaps_data, lp_data):
        metrics = []

        metrics.append(MetricsProcessor.gini(pool_data))
        metrics.append(MetricsProcessor.shannons_entropy(pool_data))

        for token_idx, token_id in enumerate(self.pool_metadata[pool_id]['coins']):
            metrics.append(MetricsProcessor.net_swap_flow(swaps_data, token_id, self.token_metadata[token_id]['symbol'], freq=self.freq))
            metrics.append(MetricsProcessor.net_lp_flow(lp_data, token_idx, self.token_metadata[token_id]['symbol'], freq=self.freq))
            metrics.append(MetricsProcessor.abs_swap_flow(swaps_data, token_id, self.token_metadata[token_id]['symbol'], freq=self.freq))
            metrics.append(MetricsProcessor.abs_lp_flow(lp_data, token_idx, self.token_metadata[token_id]['symbol'], freq=self.freq))
        
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

    @staticmethod
    def abs_swap_flow(df, token_id, symbol, freq='1min'):
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
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        swap_in = df[df['tokenBought']==token_id]['amountBought']
        swap_out = df[df['tokenSold']==token_id]['amountSold']
        flow = pd.merge(swap_in, swap_out, left_index=True, right_index=True, how='outer').fillna(0)
        metric = (flow['amountBought'] + flow['amountSold']).resample(freq).sum()
        metric.name = f'{symbol}.absSwapFlow'
        return metric

    @staticmethod
    def abs_lp_flow(df, token_idx, symbol, freq='1min'):
        if len(df) == 0:
            return pd.Series([], name=f'{symbol}.netLPFlow')
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        deposits = df[df['removal']==False]
        deposits = deposits['tokenAmounts'].apply(lambda x: json.loads(x)[token_idx])
        deposits.name = "deposits"
        withdraws = df[df['removal']==True]
        withdraws = withdraws['tokenAmounts'].apply(lambda x: json.loads(x)[token_idx])
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
        B_bar = avg_B
        epsiB = gamm*B_bar
        miu = (B_bar-epsiB)/(alph*(1-delt))
        epsiS = avg_S-alph*delt*miu
        return [alph,delt,miu,epsiB,epsiS]

    @staticmethod
    def pin_likelihood_LK(params, df):
        """
        Estimate joint likelihood function using factorization from Lin and Ke(2011)
        
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

        def likelihood(row):
            buy_s = row['Buy']
            sell_s = row['Sell']
            
            #compute values of interest for the log-likelihood function
            e1 = -mu-sell_s*np.log(1+mu/epsiS)
            e2 = -mu-buy_s*np.log(1+mu/epsiB)
            e3 = -buy_s*np.log(1+mu/epsiB)-sell_s*np.log(1+mu/epsiS)
            e_m = max(e1,e2,e3)
            
            part1 = -epsiB-epsiS+buy_s*np.log(mu+epsiB)+sell_s*np.log(mu+epsiS)+e_m
            part2 = np.log(alph*(1-delt)*np.exp(e1-e_m)+alph*delt*np.exp(e2-e_m)+(1-alph)*np.exp(e3-e_m))
        
            return part1+part2

        return -1 * df.apply(likelihood, axis=1).sum()

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

        def likelihood(row):
            #number of buy- and sell-trades for the trading day
            buy_s = row['Buy']
            sell_s = row['Sell']

            # Avoid invalid computations
            if mu <= 0 or epsiB <= 0 or epsiS <= 0:
                return np.inf

            #compute values of interest for the log-likelihood function
            M  = int(min(buy_s,sell_s)+max(buy_s,sell_s)/2)
            Xs = epsiS/(mu+epsiS)
            Xb = epsiB/(mu+epsiB)

            a1 = np.exp(-mu)
            a2 = Xs**(sell_s-M) if sell_s-M >= 0 else 0
            a3 = Xb**(buy_s-M) if buy_s-M >= 0 else 0
            a4 = Xs**(-M) if M >= 0 else 0
            a5 = Xb**(-M) if M >= 0 else 0

            # Avoid NaN and Inf in logarithmic operations
            if any(x <= 0 for x in [Xb, Xs, mu+epsiB, mu+epsiS, a1*a2*a5, a1*a3*a4, a2*a3]):
                return np.inf

            part1 = -epsiB-epsiS+M*(np.log(Xb)+np.log(Xs))+buy_s*np.log(mu+epsiB)+sell_s*np.log(mu+epsiS)
            part2 = np.log(alph*(1-delt)*a1*a2*a5+alph*delt*a1*a3*a4+(1-alph)*a2*a3)

            return part1+part2

        return -1 * df.apply(likelihood, axis=1).sum()

    @staticmethod
    def _pin(params):
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
    def pin_period(df):
        """
        Given a df corresponding to a window of data, calculate the PIN.

        @Params
            df (pd.DataFrame): timeseries DataFrame with Buy and Sell columns
        
        @Returns
            PIN (float): PIN value for the window
        """
        initial_params = MetricsProcessor.init_pin_params(df)
        opt_params = minimize(MetricsProcessor.pin_likelihood_EHO, initial_params, args = (df),method = 'Nelder-Mead').x
        return MetricsProcessor._pin(opt_params)
    
    ### Markout

    @staticmethod
    def markout(df, ohlcvs, window=timedelta(days=1), who='swapper'):
        """
        markout price = price at t0 + window
        current price = price at t0
        execution price = converts token sold to token bought units

        Add "{window}Markout" column to df, window in seconds
        """
        markout_col = f'{int(window.total_seconds())}.Markout'
        cols = list(df.columns) + [markout_col]

        df['executionPrice'] = df['amountBought'] / df['amountSold']
        df['roundedDate'] = df['timestamp'].apply(MetricsProcessor.round_date)
        last = df['roundedDate'].iloc[-1]
        df = df[df['roundedDate'] <= last - window]
        df['markoutBoughtPrice'] = df.apply(lambda x: ohlcvs[x['tokenBought']].loc[x['roundedDate'] + window]['close'], axis=1)
        df['currentSoldPrice'] = df.apply(lambda x: ohlcvs[x['tokenSold']].loc[x['roundedDate']]['close'], axis=1)
        df[markout_col] = df['amountSold'] * (df['executionPrice'] * df['markoutBoughtPrice'] - df['currentSoldPrice'])
        df = df.set_index('roundedDate')

        if who == 'swapper':
            # TODO: Include gas fees
            return df[cols]
        elif who == 'lp':
            df[markout_col] = df[markout_col] * -1
            return df[cols]
        else:
            raise ValueError(f"who must be 'swapper' or 'lp', was {who}.")
    
    ### Sharks

    # NOTE: 1Inch executor contract is considered a "buyer" (i.e. not the actual address that
    # submitted the 1Inch transaction). This is okay when looking at sharks: we can assume
    # sharks are less likely to be going through 1Inch.

    # @staticmethod
    # def pin(df, token, window=100, freq='1h'):
    #     """
    #     Calculate PIN values for each resampled period. We take a swaps_df, convert it into 
    #     a timeseries dataframe of "Buy" and "Sell" columns (which count swaps for a token).
    #     Then we partition our DF into windows of size `window` and calculate the PIN value
    #     on that window. This PIN corresponds to the value of the last timestamp in the window.

    #     @Params
    #         df (pd.DataFrame): swaps_df
    #         window (int): window size (e.g. 100 minutes, or 100 hours, etc..)
    #         freq (str): resample frequency
        
    #     @Returns
    #         PIN (pd.Series): PIN values for each resampled period
    #     """
    #     token_df = df.copy()
    #     token_df['Buy'] = (token_df['tokenBought'] == token)
    #     token_df['Sell'] = (token_df['tokenSold'] == token)
    #     token_df['timestamp'] = token_df['timestamp'].apply(datetime.fromtimestamp)
    #     token_df = token_df.set_index('timestamp')
    #     token_df = token_df.resample(freq).agg({'Buy': 'sum', 'Sell': 'sum'})

    #     def rolling_apply(df, delta=timedelta(days=1)):
    #         curr = df.index[0]
    #         end = df.index[-1]
    #         while curr <= end - delta:
    #             print(curr, curr+delta)
    #             df.loc[curr+delta, "PIN"] = pin(df.loc[curr:curr+delta])
    #             curr += timedelta(hours=6) # TODO: Fix this with the actual step size  
    #         return df