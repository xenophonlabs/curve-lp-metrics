import os
import logging
import re
from typing import Dict, List
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from web3 import Web3

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .entities import *

from dotenv import load_dotenv
load_dotenv()

INFURA_KEY = os.getenv("INFURA_KEY")
ALCHEMY_KEY = os.getenv("ALCHEMY_KEY")
PSQL_USER = os.getenv("PSQL_USER")
PSQL_PASSWORD = os.getenv("PSQL_PASSWORD")

# WEB3_ENDPOINT = f"https://mainnet.infura.io/v3/{INFURA_KEY}"
WEB3_ENDPOINT = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}"

PATH = os.path.abspath(__file__).replace(os.path.basename(__file__), '')

class DataHandler():

    def __init__(self, db=f'postgresql://{PSQL_USER}:{PSQL_PASSWORD}@localhost:5432/{PSQL_USER}', logger=None):
        self.engine = create_engine(db)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())
    
    def create_tables(self):
        Entity.metadata.create_all(self.engine)
        self.logger.info("Tables created in the database.")
    
    def close(self):
        self.session.close()
    
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
            self.logger.info(f'[{datetime.now()}] Inserting block_timestamps {i}...')
            fn = PATH+f'../../../data/timestamps_{i}.csv'
            df = pd.read_csv(fn)
            df['timestamp'] = df['unixtime'].apply(lambda x: int(datetime.timestamp(datetime.fromisoformat(x.replace("Z", "+00:00")))))
            df = df[['block', 'timestamp']]
            df['timestamp'] = df['timestamp'].astype(object)
            df['block'] = df['block'].astype(object)
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
        
        for _, row in df.iterrows():
            stmt = insert(entity.__table__).values(**row)
            if replace:
                stmt = stmt.on_conflict_do_update(index_elements=index_elements, set_=dict(row))
            else:
                stmt = stmt.on_conflict_do_nothing()
            try:
                self.session.execute(stmt)
            except Exception as e:
                self.session.rollback()
                raise e

        self.session.commit()

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

    def insert_changepoints(self, data, address, model, metric, freq):
        df = DataHandler.format_changepoints(data, address, model, metric, freq)
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
        df = pd.DataFrame([x for y in data for x in y])

        if len(df) == 0:
            return df

        hack = DataHandler()        
        df['timestamp'] = df['block'].apply(lambda x: hack.get_block_timestamp(x))
        hack.close()

        df = df[['pool_id', 'block', 'totalValueLockedUSD', 'inputTokenBalances', 'inputTokenWeights', 'timestamp', 'outputTokenSupply']]
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
        df = df[['id', 'timestamp', 'tx', 'pool_id', 'amountBought', 'amountSold', 'tokenBought', 'tokenSold', 'buyer', 'gasLimit', 'gasUsed', 'isUnderlying', 'block_gte', 'block_lt', 'block']]
        return df
    
    @staticmethod
    def format_lp_data(data):
        df = pd.DataFrame([x for y in data for x in y])
        if len(df) == 0:
            return df
        df = df[['id', 'block', 'liquidityProvider', 'removal', 'timestamp', 'tokenAmounts', 'totalSupply', 'tx', 'pool_id', 'block_gte', 'block_lt']]
        return df

    @staticmethod
    def format_changepoints(data, address, model, metric, freq):
        df = pd.DataFrame(data, columns=['timestamp'])
        if len(df) == 0:
            return df
        df['timestamp'] = df['timestamp'].apply(lambda x: int(datetime.timestamp(x)))
        df['address'] = address
        df['model'] = model
        df['metric'] = metric
        df['freq'] = freq
        df = df[['address', 'model', 'metric', 'freq', 'timestamp']]
        return df
    
    @staticmethod
    def format_pool_metrics(df, pool_id):
        df = df.melt(var_name='metric', value_name='value', ignore_index=False)
        df = df.reset_index(names='timestamp')
        df['timestamp'] = df['timestamp'].apply(datetime.timestamp)
        df['pool_id'] = pool_id
        df = df[['timestamp', 'pool_id', 'metric', 'value']]
        return df

    def format_pool_snapshots(data):
        df = pd.DataFrame.from_dict([x for y in data for x in y])
        if len(df) == 0:
            return df
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
    
    @staticmethod
    def process(df):
        for col in df:
            if type(df[col].iloc[0]).__name__ == 'Decimal':
                df[col] = pd.to_numeric(df[col])
        return df

    def get_pool_metadata(self) -> Dict:
        query = self.session.query(Pools)
        results = query.all()
        results = {d['id']:d for d in [row.as_dict() for row in results]}
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
        query = self.session.query(*[getattr(PoolData, col) for col in cols])
        query = query.filter(
            PoolData.pool_id == pool_id,
            PoolData.timestamp >= start,
            PoolData.timestamp <= end
        )
        query = query.order_by(PoolData.timestamp.asc())
        results = query.all()
        if not len(results):
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(results)
        tokens = self.pool_metadata[pool_id]['inputTokens']
        decimals = np.array([self.token_metadata[token]['decimals'] for token in tokens])
        df['inputTokenBalances'] = df['inputTokenBalances'].apply(lambda x: np.array(x, dtype=float))
        df['inputTokenBalances'] = (df['inputTokenBalances'].tolist() / 10**decimals).tolist()
        df['outputTokenSupply'] = df['outputTokenSupply'].astype(float)
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
        if not len(results):
            return pd.DataFrame()
        df = pd.DataFrame.from_dict([row.as_dict() for row in results])
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
        if not len(results):
            return pd.DataFrame()
        df = pd.DataFrame.from_dict([row.as_dict() for row in results])
        tokens = self.pool_metadata[pool_id]['coins']
        decimals = np.array([self.token_metadata[token]['decimals'] for token in tokens])
        df['tokenAmounts'] = df['tokenAmounts'].apply(lambda x: np.array(x, dtype=float))
        df['tokenAmounts'] = (df['tokenAmounts'].tolist() / 10**decimals).tolist()
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        return df

    def get_pool_snapshots(self, 
                           pool_id: str, 
                           start: int=None, 
                           end: int=None,
                           cols: List=['timestamp', 'normalizedReserves', 'reserves', 'virtualPrice', 'lpPriceUSD', 'tvl', 'reservesUSD']
        ) -> pd.DataFrame:
        query = self.session.query(*[getattr(Snapshots, col) for col in cols])
        query = query.filter(
            Snapshots.pool_id == pool_id,
            Snapshots.timestamp >= start,
            Snapshots.timestamp <= end
        )
        query = query.order_by(Snapshots.timestamp.asc())
        results = query.all()
        if not len(results):
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(results)
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        df = df.loc[~(df['virtualPrice']==0)].dropna() # Drop rows where virtualPrice is 0
        df = DataHandler.process(df)
        return df
    
    def get_pool_snapshots_last(self, 
                                pool_id: str, 
                                cols: List=['timestamp', 'normalizedReserves', 'reserves', 'virtualPrice', 'lpPriceUSD', 'tvl', 'reservesUSD']
    ) -> pd.DataFrame:
        query = self.session.query(*[getattr(Snapshots, col) for col in cols])
        query = query.filter(
            Snapshots.pool_id == pool_id,
        )
        query = query.order_by(Snapshots.timestamp.desc())
        results = query.first()
        if not len(results):
            return pd.DataFrame()
        df = pd.DataFrame.from_dict([results])
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        df = DataHandler.process(df)
        return df
    
    def get_ohlcv_data(self, 
                       token_id: str, 
                       start: int=None, 
                       end: int=None
        ) -> pd.DataFrame:

        mask = pd.date_range(start=datetime.fromtimestamp(start), end=datetime.fromtimestamp(end), freq='1min').round('1min')

        delta = int(timedelta(days=7).total_seconds()) # Hack to avoid missing data: get latest available price

        if self.token_metadata[token_id]['symbol'] == "3Crv":
            threepool = "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7"
            query = self.session.query(*[getattr(Snapshots, col) for col in ['timestamp', 'lpPriceUSD']])
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
            query = self.session.query(*[getattr(TokenOHLCV, col) for col in ['timestamp', 'symbol', 'close']])
            query = query.filter(
                TokenOHLCV.token_id == token_id,
                TokenOHLCV.timestamp >= start - delta,
                TokenOHLCV.timestamp <= end + delta
            )
            query = query.order_by(TokenOHLCV.timestamp.asc())
            results = query.all()

        if not len(results):
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(results)
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        df = df.resample('1min').ffill()
        df = df.reindex(mask, method='ffill')
        df = df.loc[datetime.fromtimestamp(start):datetime.fromtimestamp(end)]
        if 'lpPriceUSD' in df.columns:
            df = df.rename(columns={'lpPriceUSD': 'close'})
            df['symbol'] = f'{self.token_metadata[token_id]["symbol"]}/VP'
        return df   

    def get_ohlcv_data_last(self, 
                            token_id: str, 
        ) -> pd.DataFrame:
        if self.token_metadata[token_id]['symbol'] == "3Crv":
            threepool = "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7"
            query = self.session.query(*[getattr(Snapshots, col) for col in ['timestamp', 'lpPriceUSD']])
            query = query.filter(
                Snapshots.pool_id == threepool,
            )
            query = query.order_by(Snapshots.timestamp.desc())
            results = query.first()

        else:
            if self.token_metadata[token_id]['symbol'] == "WETH":
                token_id = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" # get ETH instead of WETH
            query = self.session.query(*[getattr(TokenOHLCV, col) for col in ['timestamp', 'symbol', 'close']])
            query = query.filter(
                TokenOHLCV.token_id == token_id,
            )
            query = query.order_by(TokenOHLCV.timestamp.desc())
            results = query.first()

        if not len(results):
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict([results])
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        df = df.resample('1min').ffill()
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
        query = self.session.query(*[getattr(PoolMetrics, col) for col in cols])
        query = query.filter(
            PoolMetrics.pool_id == pool_id,
            PoolMetrics.timestamp >= start,
            PoolMetrics.timestamp <= end
        )
        if metric in ['netSwapFlow', 'netLPFlow', 'sharkflow']:
            query = query.filter(PoolMetrics.metric.like(f'%{metric}'))
        else:
            query = query.filter(PoolMetrics.metric == metric)
        query = query.order_by(PoolMetrics.timestamp.asc())
        results = query.all()
        if not len(results):
            return pd.Series()
        df = pd.DataFrame.from_dict(results)
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        series = df['value']
        series.name = metric
        if metric in ['shannonsEntropy', 'giniCoefficient']:
            series.replace(0, method='ffill', inplace=True)
            series.fillna(method='ffill', inplace=True)
        elif metric in ['netSwapFlow', 'netLPFlow', 'sharkflow']:
            series.groupby(series.index).sum()            
        elif metric == 'lpSharePrice':
            series = series.ffill()
        series.fillna(0, inplace=True)
        return series

    def get_pool_metric_last(self, 
                             pool_id: str, 
                             metric: str, 
                             cols: List=['timestamp', 'value']
        ) -> pd.Series:
        query = self.session.query(*[getattr(PoolMetrics, col) for col in cols])
        query = query.filter(
            PoolMetrics.pool_id == pool_id,
        )
        if metric in ['netSwapFlow', 'netLPFlow', 'sharkflow']:
            query = query.filter(PoolMetrics.metric.like(f'%{metric}'))
        else:
            query = query.filter(PoolMetrics.metric == metric)
        if metric in ['lpSharePrice']:
            query = query.filter(PoolMetrics.value != 0)
            query = query.filter(PoolMetrics.value.isnot(None))
        query = query.order_by(PoolMetrics.timestamp.desc())
        results = query.first()
        if not len(results):
            return pd.Series()
        df = pd.DataFrame.from_dict([results])
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        series = df['value']
        series.name = metric
        if metric in ['shannonsEntropy', 'giniCoefficient']:
            series.replace(0, method='ffill', inplace=True)
        elif metric in ['netSwapFlow', 'netLPFlow', 'sharkflow']:
            series.groupby(series.index).sum()            
        elif metric == 'lpSharePrice':
            series = series.ffill()
        series.fillna(0, inplace=True)
        return series

    def get_token_metric(self, 
                         token_id: str, 
                         metric: str, 
                         start: int=None, 
                         end: int=None,
                         cols: List=['timestamp', 'value']
        ) -> pd.Series:
        query = self.session.query(*[getattr(TokenMetrics, col) for col in cols])
        query = query.filter(
            TokenMetrics.token_id == token_id,
            TokenMetrics.timestamp >= start,
            TokenMetrics.timestamp <= end
        )
        if metric in ['logReturns']:
            query = query.filter(TokenMetrics.metric.like(f'%{metric}'))
        else:
            query = query.filter(TokenMetrics.metric == metric)
        query = query.order_by(TokenMetrics.timestamp.asc())
        results = query.all()
        if not len(results):
            return pd.Series()
        df = pd.DataFrame.from_dict(results)
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        series = df['value']
        series.name = metric
        series.fillna(0, inplace=True)
        return series

    def get_changepoints(self, 
                         address: str, 
                         model: str, 
                         metric: str, 
                         start: int=None, 
                         end: int=None,
                         freq: str='1h',
                         cols: List=['timestamp']
        ) -> pd.Series:
        query = self.session.query(*[getattr(Changepoints, col) for col in cols])
        query = query.filter(
            Changepoints.address == address,
            Changepoints.model == model,
            Changepoints.metric == metric,
            Changepoints.timestamp >= start,
            Changepoints.timestamp <= end,
            Changepoints.freq == freq
        )
        query = query.order_by(Changepoints.timestamp.asc())
        results = query.all()
        if not len(results):
            return pd.Series()
        df = pd.DataFrame.from_dict(results)
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        series = df['timestamp']
        series.name = 'changepoints'
        return series

    def get_changepoints_last(self, 
                              address: str, 
                              model: str, 
                              metric: str,
                              freq: str='1h',
                              cols: List=['timestamp']
        ) -> pd.Series:
        query = self.session.query(*[getattr(Changepoints, col) for col in cols])
        query = query.filter(
            Changepoints.address == address,
            Changepoints.model == model,
            Changepoints.metric == metric,
            Changepoints.freq == freq
        )
        query = query.order_by(Changepoints.timestamp.desc())
        results = query.first()
        if not len(results):
            return pd.Series()
        df = pd.DataFrame.from_dict([results])
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        series = df['timestamp']
        series.name = 'changepoints'
        return series

    def get_takers(self) -> pd.DataFrame:
        query = self.session.query(Takers)
        query = query.order_by(Takers.cumulativeMarkout.desc())
        results = query.all()
        df = pd.DataFrame.from_dict([row.as_dict() for row in results])
        df.set_index('buyer', inplace=True)
        return df

    def get_sharks(self, top: float=0.9) -> List[str]:
        takers = self.get_takers()
        sharks = takers[takers['cumulativeMarkout'] > takers['cumulativeMarkout'].quantile(top)]
        return list(sharks.index)

    def get_block_timestamp(self, block: int):
        query = self.session.query(BlockTimestamps)
        query = query.filter(BlockTimestamps.block == block)
        result = query.first()
        if not result:
            client = Web3(Web3.HTTPProvider(WEB3_ENDPOINT))
            result = client.eth.get_block(block)
            try:
                stmt = insert(BlockTimestamps.__table__).values(block=block, timestamp=result.timestamp)
                stmt = stmt.on_conflict_do_nothing()
                self.session.execute(stmt)
            except Exception as e:
                self.logger.exception(f'Failed to insert block timestamp: {e}')
                self.session.rollback()
            self.session.commit()
        return result.timestamp

    def get_pool_X(self, 
                   metric: str, 
                   pool: str, 
                   start_ts: int, 
                   end_ts: int, 
                   freq: timedelta, 
                   normalize: bool=False, 
                   standardize: bool=False
        ) -> pd.Series:
        
        data = self.get_pool_metric(pool, metric, start_ts, end_ts)

        if data.empty:
            return data

        if metric in ['giniCoefficient', 'shannonsEntropy']:
            X = np.log1p(data.resample(freq).last().pct_change()).dropna()
            X = X[~np.isinf(X)] # if balances were 0, they increase by inf
        elif 'netSwapFlow' in metric \
            or 'netLPFlow' in metric \
            or 'sharkflow' in metric \
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

    def get_fees(self, 
                 pool_id: str,
                 cols: List=['timestamp', 'fee', 'adminFee'],
        ) -> pd.DataFrame:
        query = self.session.query(*[getattr(Snapshots, col) for col in cols])
        query = query.filter(
            Snapshots.pool_id == pool_id,
        )
        query = query.order_by(Snapshots.timestamp.asc())
        results = query.all()
        df = pd.DataFrame.from_dict(results)
        return df

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
        
        assert df['tokenBought'].nunique() <= 2 and df['tokenSold'].nunique() <= 2, 'Too many tokens in pool'
        
        fees = self.get_fees(pool)
        ohlcv = []
        for i, row in df.iterrows():
            tmp = fees[fees['timestamp'] >= row['timestamp']]
            if tmp.empty:
                fee, admin_fee = fees.iloc[-1][['fee', 'adminFee']]
            else:
                fee, admin_fee = tmp.iloc[0][['fee', 'adminFee']]
            if admin_fee == 0.5:
                # HACK!!!!!!!!!!!!!, why is this happening?
                admin_fee = 0
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