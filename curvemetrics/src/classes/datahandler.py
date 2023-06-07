import sqlite3
import json
import os
import logging

import numpy as np
import pandas as pd

from typing import Dict, List
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler

PATH = os.path.abspath(__file__).replace(os.path.basename(__file__), '')

class DataHandler():

    """
    Formats raw data and inserts it into the rawadata.db database.
    """

    def __init__(self, db_name=PATH+'../../../database/database.db'):
        self.conn = sqlite3.connect(db_name)
        self.conn.row_factory = self.dict_factory
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = cursor.fetchall()
        if len(existing_tables) != 0:
            self.logger.info("Tables already exist in the database: {}".format(existing_tables))

        with open(PATH+'../../../config/schemas/schema.sql', 'r') as f:
            create_tables_sql = f.read()
        
        self.conn.executescript(create_tables_sql)
        self.logger.info("Tables created in the database.")
    
    def create_indexes(self):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='index';")
        existing_indexes = cursor.fetchall()
        if len(existing_indexes) != 0:
            self.logger.info("Indexes already exist in the database: {}".format(existing_indexes))

        with open(PATH+'../../../config/schemas/indexes.sql', 'r') as f:
            create_indexes_sql = f.read()
        
        self.conn.executescript(create_indexes_sql)
        self.logger.info("Indexes created in the database.")

    @staticmethod
    def dict_factory(cursor, row):
        return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

    def close(self):
        self.conn.close()
    
    def reset(self):
        self.logger.warning(f"WARNING: DROPPING ALL TABLES...")
        cursor = self.conn.cursor()

        # get the list of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        # delete each table
        for table in tables:
            table_name = table['name']
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
                self.logger.warning(f"Dropped table {table_name}")
            except sqlite3.Error as e:
                self.logger.warning(f"An error occurred: {e.args}")

        cursor.execute("VACUUM;")

        # commit the transaction and close the cursor
        self.conn.commit()
        cursor.close()

    def insert_pool_metadata(self, data):
        df = DataHandler.format_pool_metadata(data)
        df.to_sql("pools", self.conn, if_exists="replace", index=False)
        self.conn.commit()
    
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
            df.to_sql("block_timestamps", self.conn, if_exists="append", index=False)
            self.conn.commit()

    def insert_token_metadata(self, data):
        df = DataHandler.format_token_metadata(data)
        df.to_sql("tokens", self.conn, if_exists="replace", index=False)
        self.conn.commit()
    
    def insert_pool_tokens_metadata(self, data):
        df = DataHandler.format_pool_tokens_metadata(data)
        df.to_sql("pool_tokens", self.conn, if_exists="replace", index=False)
        self.conn.commit()

    def insert_pool_tokens_messari_metadata(self, data):
        df = DataHandler.format_pool_tokens_messari_metadata(data)
        df.to_sql("pool_tokens", self.conn, if_exists="replace", index=False)
        self.conn.commit()

    def insert_pool_data(self, data, start_timestamp, end_timestamp):
        df = DataHandler.format_pool_data(data, start_timestamp, end_timestamp)

        if len(df) == 0:
            return

        # Insert the DataFrame into the `pool_data` table
        def insert_pool_row(row):
            # Create an SQL INSERT OR IGNORE statement
            sql = """
            INSERT OR IGNORE INTO pool_data (
                pool_id,
                block,
                totalValueLockedUSD,
                inputTokenBalances,
                inputTokenWeights,
                timestamp,
                outputTokenSupply
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            self.conn.execute(sql, row)
        
        df.apply(insert_pool_row, axis=1)
        self.conn.commit()

    def insert_token_data(self, data):
        df = DataHandler.format_token_data(data)

        if len(df) == 0:
            return

        # Insert the DataFrame into the `pool_data` table
        def insert_token_row(row):
            # Create an SQL INSERT OR IGNORE statement
            sql = """
            INSERT OR IGNORE INTO token_ohlcv (
                token_id,
                symbol,
                timestamp,
                open,
                high,
                low,
                close,
                volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            self.conn.execute(sql, row)
        
        df.apply(insert_token_row, axis=1)
        self.conn.commit()

    def insert_swaps_data(self, data):
        df = DataHandler.format_swaps_data(data)

        if len(df) == 0:
            return

        # Insert the DataFrame into the `swaps` table
        def insert_swap_row(row):
            # Create an SQL INSERT OR IGNORE statement
            sql = """
            INSERT OR IGNORE INTO swaps (
                id,
                timestamp,
                tx,
                pool_id,
                amountBought,
                amountSold,
                tokenBought,
                tokenSold,
                buyer,
                gasLimit,
                gasUsed,
                isUnderlying,
                block_gte,
                block_lt,
                block
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            # Insert the row into the `swaps` table
            self.conn.execute(sql, row)

        df.apply(insert_swap_row, axis=1)
        self.conn.commit()
    
    def insert_lp_data(self, data):
        # Convert JSON data to a pandas DataFrame
        df = DataHandler.format_lp_data(data)

        if len(df) == 0:
            return

        # Insert the DataFrame into the `lp_events` table
        def insert_lp_row(row):
            # Create an SQL INSERT OR IGNORE statement
            sql = """
            INSERT OR IGNORE INTO lp_events (
                id,
                block,
                liquidityProvider,
                removal,
                timestamp,
                tokenAmounts,
                totalSupply,
                tx,
                pool_id,
                block_gte,
                block_lt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            # Insert the row into the `lp_events` table
            self.conn.execute(sql, row)

        # Apply the custom function to each row in the DataFrame
        df.apply(insert_lp_row, axis=1)
        # Commit the changes
        self.conn.commit()

    def insert_pool_snapshots(self, data):
        # Convert JSON data to a pandas DataFrame
        df = DataHandler.format_pool_snapshots(data)

        if len(df) == 0:
            return

        # Insert the DataFrame into the `lp_events` table
        def insert_snapshots_row(row):
            # Create an SQL INSERT OR IGNORE statement
            sql = """
            INSERT OR IGNORE INTO snapshots (
                id,
                A,
                adminFee,
                fee,
                timestamp,
                normalizedReserves,
                offPegFeeMultiplier,
                reserves,
                virtualPrice,
                lpPriceUSD,
                tvl,
                totalDailyFeesUSD,
                reservesUSD,
                lpFeesUSD,
                lastPricesTimestamp,
                lastPrices,
                pool_id,
                block_gte,
                block_lt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            # Insert the row into the `lp_events` table
            self.conn.execute(sql, row)

        # Apply the custom function to each row in the DataFrame
        df.apply(insert_snapshots_row, axis=1)
        # Commit the changes
        self.conn.commit()
    
    def insert_pool_metrics(self, data, pool_id):
        # Convert JSON data to a pandas DataFrame
        df = DataHandler.format_pool_metrics(data, pool_id)

        if len(df) == 0:
            return

        # Insert the DataFrame into the `pool_metrics` table
        def insert_metrics_row(row):
            # Create an SQL INSERT OR IGNORE statement
            sql = """
            INSERT OR IGNORE INTO pool_metrics (
                timestamp,
                pool_id,
                metric,
                value
            ) VALUES (?, ?, ?, ?)
            """
            # Insert the row into the `pool_metrics` table
            self.conn.execute(sql, row)

        # Apply the custom function to each row in the DataFrame
        df.apply(insert_metrics_row, axis=1)
        # Commit the changes
        self.conn.commit()

    def insert_token_metrics(self, data, token_id):
        # Convert JSON data to a pandas DataFrame
        df = DataHandler.format_token_metrics(data, token_id)

        if len(df) == 0:
            return

        # Insert the DataFrame into the `lp_events` table
        def insert_metrics_row(row):
            # Create an SQL INSERT OR IGNORE statement
            sql = """
            INSERT OR IGNORE INTO token_metrics (
                timestamp,
                token_id,
                metric,
                value
            ) VALUES (?, ?, ?, ?)
            """
            # Insert the row into the `lp_events` table
            self.conn.execute(sql, row)

        # Apply the custom function to each row in the DataFrame
        df.apply(insert_metrics_row, axis=1)
        # Commit the changes
        self.conn.commit()

    def insert_changepoints(self, data, pool_id, model, metric):
        # Convert JSON data to a pandas DataFrame
        df = DataHandler.format_changepoints(data, pool_id, model, metric)

        if len(df) == 0:
            return

        # Insert the DataFrame into the `lp_events` table
        def insert_changepoints_row(row):
            # Create an SQL INSERT OR IGNORE statement
            sql = """
            INSERT OR IGNORE INTO changepoints (
                pool_id,
                model,
                metric,
                timestamp
            ) VALUES (?, ?, ?, ?)
            """
            # Insert the row into the `lp_events` table
            self.conn.execute(sql, row)

        # Apply the custom function to each row in the DataFrame
        df.apply(insert_changepoints_row, axis=1)
        # Commit the changes
        self.conn.commit()
    
    def insert_takers(self, takers):
        # Convert JSON data to a pandas DataFrame
        df = DataHandler.format_takers(takers)

        if len(df) == 0:
            return

        # Insert the DataFrame into the `lp_events` table
        def insert_takers_row(row):
            # Create an SQL INSERT OR IGNORE statement
            sql = """
            INSERT OR REPLACE INTO takers (
                buyer,
                amountBought,
                amountSold,
                cumulativeMarkout,
                meanMarkout,
                count,
                windowSize
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            # Insert the row into the `lp_events` table
            self.conn.execute(sql, row)

        # Apply the custom function to each row in the DataFrame
        df.apply(insert_takers_row, axis=1)
        # Commit the changes
        self.conn.commit()
    
    @staticmethod
    def format_takers(data):
        df = data.reset_index()
        df = df[['buyer', 'amountBought', 'amountSold', 'cumulativeMarkout', 'meanMarkout', 'count', 'windowSize']]
        return df

    @staticmethod
    def format_pool_metadata(data):
        df = pd.DataFrame.from_dict(data).T.reset_index(drop=True)
        for col in ['assetType', 'creationBlock', 'creationDate', 'c128', 'isRebasing', 'isV2', 'virtualPrice']:
            df[col] = df[col].astype(int)
        for col in ['baseApr']:
            df[col] = df[col].astype(float)
        df['coins'] = df['coins'].apply(lambda x: json.dumps(x))
        df['inputTokens'] = df['inputTokens'].apply(lambda x: json.dumps(x))
        return df

    @staticmethod
    def format_token_metadata(data):
        df = pd.DataFrame.from_dict(data).T.reset_index(drop=True)
        for col in ['decimals']:
            df[col] = df[col].astype(int)
        return df
    
    @staticmethod
    def format_pool_tokens_metadata(data):
        df = pd.DataFrame([[k, coin] for k,v in data.items() for coin in v['coins']], columns=['pool_id', 'token_id'])
        return df
    
    @staticmethod
    def format_pool_tokens_messari_metadata(data):
        df = pd.DataFrame([[k, coin] for k,v in data.items() for coin in v['coins']], columns=['pool_id', 'token_id'])
        return df
    
    @staticmethod
    def format_pool_data(data):
        hack = DataHandler()
        df = pd.DataFrame([x for y in data for x in y])
        if len(df) == 0:
            return df
        for col in ['totalValueLockedUSD']:
            df[col] = df[col].astype(float)
        for col in ['block']:
            df[col] = df[col].astype(int)
        df['inputTokenWeights'] = df['inputTokenWeights'].apply(lambda x: json.dumps(list(map(float, x))))
        df['inputTokenBalances'] = df['inputTokenBalances'].apply(lambda x: json.dumps(list(map(int, x))))
        if 'timestamp' not in df.columns:
            df['timestamp'] = df['block'].apply(lambda x: hack.get_block_timestamp(x)[0]['timestamp'])
        # NOTE: order must match the order in the INSERT statement. For convenience, ensure everything matches the schema.
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
        for col in ['amountBought', 'amountSold']:
            df[col] = df[col].astype(float)
        for col in ['timestamp', 'block', 'gasLimit', 'gasUsed', 'isUnderlying']:
            df[col] = df[col].astype(int)
        # NOTE: order must match the order in the INSERT statement. For convenience, ensure everything matches the schema.
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
        # NOTE: order must match the order in the INSERT statement. For convenience, ensure everything matches the schema.
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

    @staticmethod
    def format_token_aggregate_metrics(data):
        pass

    def _execute_query(
            self, 
            query: str,
            params: List=[],
        ) -> Dict:
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params)
            results = cursor.fetchall()
        except sqlite3.Error as e:
            raise sqlite3.Error(f'Error executing query: {e}')
        finally:
            cursor.close()
        return results
    
    def get_token_metadata(self) -> Dict:
        query = f'SELECT * FROM tokens'
        results = self._execute_query(query)
        metadata = {row["id"]: row for row in results}
        return metadata

    @property
    def token_metadata(self):
        return self.get_token_metadata()

    @property
    def token_ids(self):
        return {v['symbol']:k for k, v in self.token_metadata.items()}
    
    def get_pool_metadata(self) -> Dict:
        query = f'SELECT * FROM pools'
        results = self._execute_query(query)
        metadata = {row["id"]: row for row in results}
        for data in metadata.values():
            data['coins'] = json.loads(data['coins'])
            data['inputTokens'] = json.loads(data['inputTokens'])
        return metadata

    @property
    def pool_metadata(self):
        return self.get_pool_metadata()

    def get_pool_data(self, pool_id: str, start: int=None, end: int=None) -> pd.DataFrame:
        query = f'SELECT inputTokenBalances, timestamp, outputTokenSupply FROM pool_data WHERE pool_id = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC'
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
    
    def get_swaps_data(self, pool_id: str, start: int=None, end: int=None) -> pd.DataFrame:
        query = f'SELECT * FROM swaps WHERE pool_id = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC'
        results = self._execute_query(query, params=[pool_id, start, end])
        if not len(results):
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(results)
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        return df
    
    def get_lp_data(self, pool_id: str, start: int=None, end: int=None) -> pd.DataFrame:
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

    def get_pool_snapshots(self, pool_id: str, start: int=None, end: int=None) -> pd.DataFrame:
        query = f'SELECT timestamp, normalizedReserves, reserves, virtualPrice, lpPriceUSD, tvl, reservesUSD FROM snapshots WHERE pool_id = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC'
        results = self._execute_query(query, params=[pool_id, start, end])
        if not len(results):
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(results)
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        df = df.sort_index()
        df = df.loc[~(df['virtualPrice']==0)].dropna() # Drop rows where virtualPrice is 0
        return df

    def get_ohlcv_data(self, token_id: str, start: int=None, end: int=None) -> pd.DataFrame:
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
    
    def get_pool_metric(self, pool_id: str, metric: str, start: int=None, end: int=None) -> pd.Series:
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

    def get_token_metric(self, token_id: str, metric: str, start: int=None, end: int=None) -> pd.Series:
        query = f'SELECT timestamp, value FROM token_metrics WHERE token_id = ? AND metric = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC'
        results = self._execute_query(query, params=[token_id, metric, start, end])
        if not len(results):
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(results)
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        series = df['value']
        series.name = metric
        return series

    def get_changepoints(self, pool_id, model, metric, start: int=None, end: int=None) -> pd.Series:
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
        query = f'SELECT * FROM takers ORDER BY cumulativeMarkout DESC'
        results = self._execute_query(query)
        df = pd.DataFrame.from_dict(results)
        df.set_index('buyer', inplace=True)
        return df

    def get_sharks(self, top=0.9) -> np.array:
        takers = self.get_takers()
        sharks = takers[takers['cumulativeMarkout'] > takers['cumulativeMarkout'].quantile(top)]
        return np.array(sharks.index)

    def get_block_timestamp(self, block: int):
        cursor = self.conn.cursor()
        
        to_execute = f'SELECT * FROM block_timestamps WHERE block == ?'
        params = [block]
        
        try:
            cursor.execute(to_execute, params)
            results = cursor.fetchall()
        except sqlite3.Error as e:
            raise sqlite3.Error(f'Error executing query: {e}')
        finally:
            cursor.close()

        return results

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