import json
import os
import logging
import re
from typing import Dict, List
from datetime import datetime, timedelta

from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .entities import *

from dotenv import load_dotenv
load_dotenv()

PSQL_USER = os.getenv("PSQL_USER")
PSQL_PASSWORD = os.getenv("PSQL_PASSWORD")

PATH = os.path.abspath(__file__).replace(os.path.basename(__file__), '')

class DataHandler():

    def __init__(self, db=f'postgresql://{PSQL_USER}:{PSQL_PASSWORD}@localhost:5432/{PSQL_USER}'):
        self.engine = create_engine(db)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
    
    def create_tables(self):
        Entity.metadata.create_all(self.engine)
        self.logger.info("Tables created in the database.")
    
    def create_indexes(self):
        with open(PATH+'../../../config/schemas/indexes.sql', 'r') as f:
            sql = f.read()
        
        with self.engine.connect() as connection:
            connection.execute(text(sql))
        
        self.logger.info("Indexes created in the database.")

    def close(self):
        self.session.close()
        self.logger.info("Database session closed.")
    
    def reset(self):
        self.logger.warning(f"WARNING: DROPPING ALL TABLES...")

        try:
            Entity.metadata.drop_all(self.engine)
        except Exception as e:
            self.logger.warning(f"An error occurred: {e.args}")
        
        self.logger.warning(f"Dropped all tables.")

    @staticmethod
    def pg_array_to_list(arr):
        return re.findall(r'{(.+)}', ''.join(arr))[0].split(',')

    def insert_pool_metadata(self, data):
        df = DataHandler.format_pool_metadata(data)
        self.insert(df, Pools, replace=True, index_elements=[Pools.id])
        self.logger.info(f"Pool Metadata has been inserted.")

    def insert_token_metadata(self, data):
        df = DataHandler.format_token_metadata(data)
        self.insert(df, Tokens, replace=True, index_elements=[Tokens.id])
        self.logger.info(f"Token Metadata has been inserted.")
    
    def insert_block_timestamps(self):
        """
        Used to backfill blocks with timestamps.
        """
        for i in range(5):
            fn = PATH+f'../../../data/timestamps_{i}.csv'
            df = pd.read_csv(fn)
            df['block'] = df['block'].astype(int)
            df['timestamp'] = df['unixtime'].apply(lambda x: int(datetime.timestamp(datetime.fromisoformat(x.replace("Z", "+00:00")))))
            df = df[['block', 'timestamp']]
            self.insert(df, BlockTimestamps, replace=True, index_elements=[BlockTimestamps.block])

        self.logger.info(f"Old blocktimestamps have been inserted.")

    def insert(self, 
               df: pd.DataFrame, 
               entity: object,
               replace: bool=False,
               index_elements: List[str]=[],):
        """
        Helper method to insert df into the database.
        """
        if len(df) == 0:
            return

        data = df.to_dict(orient='records') # convert dataframe to list of dictionaries

        stmt = insert(entity.__table__).values(data)

        if replace:
            stmt = stmt.on_conflict_do_update(index_elements=index_elements, set_=dict(df.iloc[-1]))
        else:
            stmt = stmt.on_conflict_do_nothing()

        try:
            self.session.execute(stmt)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e

    def insert_pool_data(self, data):
        df = DataHandler.format_pool_data(data)
        self.insert(df, PoolData)

    def insert_token_data(self, data):
        df = DataHandler.format_token_data(data)
        self.insert(df, TokenOHLCV)

    def insert_swaps_data(self, data):
        df = DataHandler.format_swaps_data(data)
        self.insert(df, Swaps)
    
    def insert_lp_data(self, data):
        df = DataHandler.format_lp_data(data)
        self.insert(df, LPEvents)

    def insert_pool_snapshots(self, data):
        df = DataHandler.format_pool_snapshots(data)
        self.insert(df, Snapshots)
    
    def insert_pool_metrics(self, data, pool_id):
        df = DataHandler.format_pool_metrics(data, pool_id)
        self.insert(df, PoolMetrics)

    def insert_token_metrics(self, data, token_id):
        df = DataHandler.format_token_metrics(data, token_id)
        self.insert(df, TokenMetrics)

    def insert_changepoints(self, data, pool_id, model, metric):
        df = DataHandler.format_changepoints(data, pool_id, model, metric)
        self.insert(df, Changepoints)
    
    def insert_takers(self, takers):
        df = DataHandler.format_takers(takers)
        self.insert(df, Takers, replace=True, index_elements=[Takers.buyer, Takers.windowSize])

    @staticmethod
    def format_takers(data):
        df = data.reset_index()
        df = df[['buyer', 'amountBought', 'amountSold', 'cumulativeMarkout', 'meanMarkout', 'count', 'windowSize']]
        return df

    @staticmethod
    def format_pool_metadata(data):
        df = pd.DataFrame.from_dict(data).T.reset_index(drop=True)
        return df

    @staticmethod
    def format_token_metadata(data):
        df = pd.DataFrame.from_dict(data).T.reset_index(drop=True)
        return df

    @staticmethod
    def format_pool_data(data):
        hack = DataHandler()
        df = pd.DataFrame([x for y in data for x in y])
        if len(df) == 0:
            return df
        # for col in ['totalValueLockedUSD']:
        #     df[col] = df[col].astype(float)
        # for col in ['block']:
        #     df[col] = df[col].astype(int)
        df['timestamp'] = df['block'].apply(lambda x: hack.get_block_timestamp(x))
        df = df[['pool_id', 'block', 'totalValueLockedUSD', 'inputTokenBalances', 'inputTokenWeights', 'timestamp', 'outputTokenSupply']]
        hack.close()
        return df

    @staticmethod
    def format_token_data(data):
        df = pd.DataFrame(data, columns=['token_id', 'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if len(df) == 0:
            return df
        df['timestamp'] = (df['timestamp'] / 1000).astype(int)
        # In case there are two prices observed in the same timestamp, e.g. from 2 curve swaps
        df = df.groupby(['token_id', 'symbol', 'timestamp']).agg({
            'open': 'mean', 
            'high': 'max', 
            'low': 'min', 
            'close': 'mean', # simple mean, not perfect weighted average but good enough. Doesn't make sense to take `last` since they coincided in timestamp
            'volume': 'sum'
        }).reset_index() 
        return df
    
    @staticmethod
    def format_swaps_data(data):
        df = pd.DataFrame([x for y in data for x in y])
        if len(df) == 0:
            return df
        for col in ['amountBought', 'amountSold', 'isUnderlying']:
            df[col] = df[col].astype(float)
        for col in ['timestamp', 'block', 'gasLimit', 'gasUsed']:
            df[col] = df[col].astype(int)
        df = df[['id', 'timestamp', 'tx', 'pool_id', 'amountBought', 'amountSold', 'tokenBought', 'tokenSold', 'buyer', 'gasLimit', 'gasUsed', 'isUnderlying', 'block_gte', 'block_lt', 'block']]
        return df
    
    @staticmethod
    def format_lp_data(data):
        df = pd.DataFrame([x for y in data for x in y])
        if len(df) == 0:
            return df
        for col in ['timestamp', 'block', 'removal']:
            df[col] = df[col].astype(int)
        df['totalSupply'] = df['totalSupply'].astype(float)
        df['tokenAmounts'] = df['tokenAmounts'].apply(lambda x: json.dumps(list(map(int, x))))
        df = df[['id', 'block', 'liquidityProvider', 'removal', 'timestamp', 'tokenAmounts', 'totalSupply', 'tx', 'pool_id', 'block_gte', 'block_lt']]
        return df

    @staticmethod
    def format_changepoints(data, pool_id, model, metric):
        df = pd.DataFrame(data, columns=['timestamp'])
        df['timestamp'] = df['timestamp'].apply(lambda x: int(datetime.timestamp(x)))
        df['pool_id'] = pool_id
        df['model'] = model
        df['metric'] = metric
        df = df[['pool_id', 'model', 'metric', 'timestamp']]
        return df
    
    @staticmethod
    def format_pool_metrics(df, pool_id):
        df = df.melt(var_name='metric', value_name='value', ignore_index=False)
        df = df.reset_index(names='timestamp')
        df['timestamp'] = df['timestamp'].apply(lambda x: int(datetime.timestamp(x)))
        df['pool_id'] = pool_id
        df = df[['timestamp', 'pool_id', 'metric', 'value']]
        return df

    def format_pool_snapshots(data):
        df = pd.DataFrame.from_dict([x for y in data for x in y])
        for col in ['A', 'offPegFeeMultiplier', 'timestamp', 'virtualPrice', 'lastPricesTimestamp', 'block_gte', 'block_lt']:
            df[col] = df[col].astype(int)
        for col in ['adminFee', 'fee', 'lpPriceUSD', 'tvl', 'totalDailyFeesUSD', 'lpFeesUSD']:
            df[col] = df[col].astype(float)
        for col in ['normalizedReserves', 'reserves']:
            df[col] = df[col].apply(lambda x: json.dumps(list(map(int, x))))
        for col in ['reservesUSD']:
            df[col] = df[col].apply(lambda x: json.dumps(list(map(float, x))))
        df = df[['id', 'A', 'adminFee', 'fee', 'timestamp', 'normalizedReserves', 'offPegFeeMultiplier', 'reserves', 'virtualPrice', 'lpPriceUSD', 'tvl', 'totalDailyFeesUSD', 'reservesUSD', 'lpFeesUSD', 'lastPricesTimestamp', 'lastPrices', 'pool_id', 'block_gte', 'block_lt']]
        return df

    @staticmethod
    def format_token_metrics(df, token_id):
        df = df.melt(var_name='metric', value_name='value', ignore_index=False)
        df = df.reset_index(names='timestamp')
        df['timestamp'] = df['timestamp'].apply(lambda x: int(datetime.timestamp(x)))
        df['token_id'] = token_id
        df = df[['timestamp', 'token_id', 'metric', 'value']]
        return df

    def get_pool_metadata(self) -> Dict:
        query = self.session.query(Pools)
        results = query.all()
        results = {d['id']:d for d in [row.as_dict() for row in results]}
        for k, v in results.items():
            v['coins'] = self.pg_array_to_list(v['coins'])
            v['inputTokens'] = self.pg_array_to_list(v['inputTokens'])
            results[k] = v
        return results
    
    def get_token_metadata(self) -> Dict:
        query = self.session.query(Tokens)
        results = query.all()
        results = {d['id']:d for d in [row.as_dict() for row in results]}
        return results

    def get_pool_data(self, 
                      pool_id: str, 
                      start: int, 
                      end: int, 
                      cols: List=['inputTokenBalances', 'timestamp', 'outputTokenSupply']
        ) -> pd.DataFrame:
        query = self.session.query([getattr(PoolData, col) for col in cols])
        query = query.filter(
            PoolData.pool_id == pool_id,
            PoolData.timestamp >= start,
            PoolData.timestamp <= end
        )
        query = query.order_by(PoolData.timestamp.asc())
        results = query.all()
        df = pd.DataFrame(results, columns=cols)
        # self.pg_array_to_list(v['inputTokens'])
        return df
        query = f'SELECT  FROM pool_data WHERE pool_id = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC'
        results = self._execute_query(query, params=[pool_id, start, end])
        if not len(results):
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(results)

        tokens = self.pool_metadata[pool_id]['inputTokens']
        decimals = np.array([self.token_metadata[token]['decimals'] for token in tokens])

        # convert JSON strings to lists and then to numpy arrays, then normalize balances
        # df['inputTokenWeights'] = np.array(df['inputTokenWeights'].map(json.loads).to_list())
        df['inputTokenBalances'] = (np.array(df['inputTokenBalances'].map(json.loads).to_list()) / 10**decimals).tolist()

        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        return df
    
    def get_swaps_data(self, 
                       pool_id: str, 
                       start: int=None, 
                       end: int=None
        ) -> pd.DataFrame:
        query = self.session.query(Swaps)
        query = query.filter(
            Swaps.pool_id == pool_id,
            Swaps.timestamp >= start,
            Swaps.timestamp <= end
        )
        query = query.order_by(Swaps.timestamp.asc())
        results = query.all()
        return results
        query = f'SELECT * FROM swaps WHERE pool_id = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC'
        results = self._execute_query(query, params=[pool_id, start, end])
        if not len(results):
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(results)
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        return df
    
    def get_lp_data(self, 
                    pool_id: str, 
                    start: int=None, 
                    end: int=None
        ) -> pd.DataFrame:
        query = self.session.query(LPEvents)
        query = query.filter(
            LPEvents.pool_id == pool_id,
            LPEvents.timestamp >= start,
            LPEvents.timestamp <= end
        )
        query = query.order_by(LPEvents.timestamp.asc())
        results = query.all()
        return results
        query = f'SELECT * FROM lp_events WHERE pool_id = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC'
        results = self._execute_query(query, params=[pool_id, start, end])
        if not len(results):
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(results)

        tokens = self.pool_metadata[pool_id]['coins']
        decimals = np.array([self.token_metadata[token]['decimals'] for token in tokens])

        df['tokenAmounts'] = (np.array(df['tokenAmounts'].map(json.loads).to_list()) / 10**decimals).tolist()
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        return df

    def get_pool_snapshots(self, 
                           pool_id: str, 
                           start: int=None, 
                           end: int=None,
                           cols: List=['timestamp', 'normalizedReserves', 'reserves', 'virtualPrice', 'lpPriceUSD', 'tvl', 'reservesUSD']
        ) -> pd.DataFrame:
        query = self.session.query([getattr(Snapshots, col) for col in cols])
        query = query.filter(
            Snapshots.pool_id == pool_id,
            Snapshots.timestamp >= start,
            Snapshots.timestamp <= end
        )
        query = query.order_by(Snapshots.timestamp.asc())
        results = query.all()
        return results
        query = f'SELECT timestamp, normalizedReserves, reserves, virtualPrice, lpPriceUSD, tvl, reservesUSD FROM snapshots WHERE pool_id = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC'
        results = self._execute_query(query, params=[pool_id, start, end])
        if not len(results):
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(results)
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        df = df.sort_index()
        df = df.loc[~(df['virtualPrice']==0)].dropna() # Drop rows where virtualPrice is 0
        return df

    def get_ohlcv_data(self, 
                       token_id: str, 
                       start: int=None, 
                       end: int=None
        ) -> pd.DataFrame:

        delta = int(timedelta(days=2).total_seconds()) # Hack to avoid missing data

        if self.token_metadata[token_id]['symbol'] == "3Crv":
            threepool = "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7"
            query = self.session.query([getattr(Snapshots, col) for col in ['timestamp', 'lpPriceUSD']])
            query = query.filter(
                Snapshots.pool_id == threepool,
                Snapshots.timestamp >= start - delta,
                Snapshots.timestamp <= end + delta
            )
            query = query.order_by(Snapshots.timestamp.asc())
            results = query.all()

        else:
            if self.token_metadata[token_id]['symbol'] == "WETH":
                token_id = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" # get ETH instead of WETH
            query = self.session.query([getattr(TokenOHLCV, col) for col in ['timestamp', 'symbol', 'close']])
            query = query.filter(
                TokenOHLCV.token_id == token_id,
                TokenOHLCV.timestamp >= start - delta,
                TokenOHLCV.timestamp <= end + delta
            )
            query = query.order_by(TokenOHLCV.timestamp.asc())
            results = query.all()

        return results
        delta = int(timedelta(days=2).total_seconds()) # Hack to avoid missing data for chainlink
        if self.token_metadata[token_id]['symbol'] == "3Crv":
            query = "SELECT timestamp, lpPriceUSD FROM snapshots WHERE pool_id == ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC"
            params = ["0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7", start - delta, end + delta] # 3Crv pool LP token price
        else:
            if self.token_metadata[token_id]['symbol'] == "WETH":
                token_id = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" # get ETH instead of WETH
            query = f'SELECT timestamp, symbol, close FROM token_ohlcv WHERE token_id = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC'
            params = [token_id, start - delta, end + delta]
        results = self._execute_query(query, params=params)
        if not len(results):
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(results)
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        df = df.resample('1min').ffill()
        df = df.loc[datetime.fromtimestamp(start):datetime.fromtimestamp(end)]
        if 'lpPriceUSD' in df.columns:
            df = df.rename(columns={'lpPriceUSD': 'close'})
            df['symbol'] = f'{self.token_metadata[token_id]["symbol"]}/VP'
        return df
    
    def get_pool_metric(self, 
                        pool_id: str, 
                        metric: str, 
                        start: int=None, 
                        end: int=None,
                        cols: List=['timestamp', 'value']
        ) -> pd.Series:
        query = self.session.query([getattr(PoolMetrics, col) for col in cols])
        query = query.filter(
            PoolMetrics.pool_id == pool_id,
            PoolMetrics.timestamp >= start,
            PoolMetrics.timestamp <= end
        )
        if metric in ['netSwapFlow', 'netLPFlow', 'sharkFlow']:
            query = query.filter(PoolMetrics.metric.like(f'%{metric}'))
        else:
            query = query.filter(PoolMetrics.metric == metric)
        query = query.order_by(PoolMetrics.timestamp.asc())
        results = query.all()
        return results
        if metric in ['netSwapFlow', 'netLPFlow', 'sharkFlow']:
            query = f'SELECT timestamp, value FROM pool_metrics WHERE pool_id = ? AND metric LIKE ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC'
            results = self._execute_query(query, params=[pool_id, '%'+metric, start, end])
        else:
            query = f'SELECT timestamp, value FROM pool_metrics WHERE pool_id = ? AND metric = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC'
            results = self._execute_query(query, params=[pool_id, metric, start, end])
        if not len(results):
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(results)
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        series = df['value']
        series.name = metric
        if metric in ['shannonsEntropy', 'giniCoefficient']:
            series.replace(0, method='ffill', inplace=True)
        elif metric in ['netSwapFlow', 'netLPFlow', 'sharkFlow']:
            series.groupby(series.index).sum()            
        if metric == 'lpSharePrice':
            series = series.ffill()
        return series

    def get_token_metric(self, 
                         token_id: str, 
                         metric: str, 
                         start: int=None, 
                         end: int=None
        ) -> pd.Series:
        query = self.session.query([getattr(TokenMetrics, col) for col in ['timestamp, value']])
        query = query.filter(
            TokenMetrics.token_id == token_id,
            TokenMetrics.metric == metric,
            TokenMetrics.timestamp >= start,
            TokenMetrics.timestamp <= end
        )
        query = query.order_by(TokenMetrics.timestamp.asc())
        results = query.all()
        return results
        query = f'SELECT timestamp, value FROM token_metrics WHERE token_id = ? AND metric = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC'
        results = self._execute_query(query, params=[token_id, metric, start, end])
        if not len(results):
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(results)
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        series = df['value']
        series.name = metric
        return series

    def get_changepoints(self, 
                         pool_id: str, 
                         model: str, 
                         metric: str, 
                         start: int=None, 
                         end: int=None
        ) -> pd.Series:
        query = self.session.query([getattr(Changepoints, col) for col in ['timestamp']])
        query = query.filter(
            Changepoints.pool_id == pool_id,
            Changepoints.model == model,
            Changepoints.metric == metric,
            Changepoints.timestamp >= start,
            Changepoints.timestamp <= end
        )
        query = query.order_by(Changepoints.timestamp.asc())
        results = query.all()
        return results
        results = self.query(Changepoints, pool_id, start, end, cols=['timestamp'], metric=metric)
        query = f'SELECT timestamp FROM changepoints WHERE pool_id = ? AND model = ? AND metric = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC'
        results = self._execute_query(query, params=[pool_id, model, metric, start, end])
        if not len(results):
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(results)
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        series = df['timestamp']
        series.name = 'changepoints'
        return series

    def get_takers(self) -> pd.DataFrame:
        query = self.session.query(Takers)
        query = query.order_by(Takers.cumulativeMarkout.desc())
        results = query.all()
        df = pd.DataFrame(results, columns=Takers.__table__.columns.keys())
        return df
        query = f'SELECT * FROM takers ORDER BY cumulativeMarkout DESC'
        results = self._execute_query(query)
        df = pd.DataFrame.from_dict(results)
        df.set_index('buyer', inplace=True)
        return df

    def get_sharks(self, top: float=0.9) -> np.array:
        takers = self.get_takers()
        sharks = takers[takers['cumulativeMarkout'] > takers['cumulativeMarkout'].quantile(top)]
        return np.array(sharks.index)

    def get_block_timestamp(self, block: int):
        query = self.session.query(BlockTimestamps)
        query = query.filter(BlockTimestamps.block == block)
        result = query.first()
        return result.timestamp

    def get_pool_X(self, metric, pool, start_ts, end_ts, freq, normalize=False, standardize=False):
        data = self.get_pool_metric(pool, metric, start_ts, end_ts)

        if metric in ['giniCoefficient', 'shannonsEntropy']:
            X = np.log1p(data.resample(freq).last().pct_change()).dropna()
        elif 'netSwapFlow' in metric \
            or 'netLPFlow' in metric \
            or 'sharkFlow' in metric \
            or 'Markout' in metric:
            X = data.resample(freq).sum()
        else:
            raise Exception('Not Implemented Error')
        
        if normalize:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X = pd.Series(scaler.fit_transform(X.values.reshape(-1, 1)).flatten(), index=X.index)
        elif standardize:
            scaler = StandardScaler()
            X = pd.Series(scaler.fit_transform(X.values.reshape(-1, 1)).flatten(), index=X.index)

        return X

    def get_token_X(self, metric, token, start_ts, end_ts, freq, normalize=False, standardize=False):
        data = self.get_token_metric(token, metric, start_ts, end_ts)

        if 'logReturns' in metric:
            X = data.resample(freq).sum()
        else:
            raise Exception('Not Implemented Error')

        if normalize:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X = pd.Series(scaler.fit_transform(X.values.reshape(-1, 1)).flatten(), index=X.index)
        elif standardize:
            scaler = StandardScaler()
            X = pd.Series(scaler.fit_transform(X.values.reshape(-1, 1)).flatten(), index=X.index)
        return X

    def get_fees(self, pool):
        fees = self._execute_query('SELECT timestamp, fee, adminFee FROM snapshots WHERE pool_id = ?', params=[pool])
        fees = pd.DataFrame(fees)
        fees.sort_values('timestamp', inplace=True, ascending=True)
        return fees

    def get_curve_price(self, token, pool, start_ts, end_ts, numeraire) -> List:
        """
        Get numeraire price for a token in a StableSwap pool.
        Assumes that the numeraire is ETH, and that ETH or WETH are included
        in the pool. Accounts for fees and adminFees.

        Relies on swaps and snapshots being filled in first.
        """
        symbol = f"{self.token_metadata[token]['symbol']}/{self.token_metadata[numeraire]['symbol']}"

        df = self.get_swaps_data(pool, start_ts, end_ts).reset_index(drop=True)
        if df.shape[0] == 0:
            return []
        df = (df.groupby(['timestamp', 'tokenBought', 'tokenSold'], as_index=False)
                        .agg({'amountBought': 'sum', 'amountSold': 'sum'}))
        
        assert df['tokenBought'].nunique() == 2 and df['tokenSold'].nunique() == 2, 'Too many tokens in pool'
        
        fees = self.get_fees(pool)
        ohlcv = []
        for i, row in df.iterrows():
            fee, admin_fee = fees[fees['timestamp'] >= row['timestamp']].iloc[0][['fee', 'adminFee']]
            if row['amountBought'] == 0 or row['amountSold'] == 0:
                continue
            price = row['amountSold'] / (row['amountBought'] * (1 + fee + admin_fee))
            if row['tokenSold'] != numeraire:
                price = 1 / price
            ohlcv.append([token, symbol, row['timestamp']*1000, None, None, None, price, None])

        return ohlcv
    
    @property
    def pool_metadata(self):
        return self.get_pool_metadata()

    @property
    def token_metadata(self):
        return self.get_token_metadata()

    @property
    def token_ids(self):
        return {v['symbol']:k for k, v in self.token_metadata.items()}