import sqlite3
import pandas as pd
import json
import numpy as np
import os
import logging
from typing import Dict, List
from datetime import datetime

PATH = os.path.abspath(__file__).replace(os.path.basename(__file__), '')

class DataHandler():

    """
    Formats raw data and inserts it into the rawadata.db database.
    """

    def __init__(self, db_name=PATH+'../database/database.db'):
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

        with open(PATH+'../config/schemas/schema.sql', 'r') as f:
            create_tables_sql = f.read()
        
        self.conn.executescript(create_tables_sql)
        self.logger.info("Tables created in the database.")
    
    def create_indexes(self):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='index';")
        existing_indexes = cursor.fetchall()
        if len(existing_indexes) != 0:
            self.logger.info("Indexes already exist in the database: {}".format(existing_indexes))

        with open(PATH+'../config/schemas/indexes.sql', 'r') as f:
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

    def insert_token_metadata(self, data):
        df = DataHandler.format_token_metadata(data)
        df.to_sql("tokens", self.conn, if_exists="replace", index=False)
        self.conn.commit()
    
    def insert_pool_tokens_metadata(self, data):
        df = DataHandler.format_pool_tokens_metadata(data)
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
                approxTimestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
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
    
    def insert_pool_metrics(self, data):
        pass

    def insert_pool_aggregate_metrics(self, data):
        pass

    def insert_token_metrics(self, data):
        pass

    def insert_token_aggregate_metrics(self, data):
        pass

    @staticmethod
    def format_pool_metadata(data):
        df = pd.DataFrame.from_dict(data).T.reset_index(drop=True)
        for col in ['assetType', 'creationBlock', 'creationDate', 'c128', 'isRebasing', 'isV2', 'virtualPrice']:
            df[col] = df[col].astype(int)
        for col in ['baseApr']:
            df[col] = df[col].astype(float)
        df['coins'] = df['coins'].apply(lambda x: json.dumps(x))
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
    def format_pool_data(data, start_timestamp, end_timestamp):
        df = pd.DataFrame([x for y in data for x in y])
        if len(df) == 0:
            return df
        for col in ['totalValueLockedUSD']:
            df[col] = df[col].astype(float)
        for col in ['block']:
            df[col] = df[col].astype(int)
        df['inputTokenWeights'] = df['inputTokenWeights'].apply(lambda x: json.dumps(list(map(float, x))))
        df['inputTokenBalances'] = df['inputTokenBalances'].apply(lambda x: json.dumps(list(map(int, x))))
        df = df.sort_values(by='block')
        df['approxTimestamp'] = np.linspace(start_timestamp, end_timestamp, len(df), dtype=int)
        # NOTE: order must match the order in the INSERT statement. For convenience, ensure everything matches the schema.
        df = df[['pool_id', 'block', 'totalValueLockedUSD', 'inputTokenBalances', 'inputTokenWeights', 'approxTimestamp']]
        return df

    @staticmethod
    def format_token_data(data):
        df = pd.DataFrame(data, columns=['token_id', 'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if len(df) == 0:
            return df
        df['timestamp'] = (df['timestamp'] / 1000).astype(int)
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
    def format_pool_metrics(df, pool_id):
        df = df.melt(var_name='metric', value_name='value', ignore_index=False)
        df = df.reset_index(names='timestamp')
        df['timestamp'] = df['timestamp'].apply(lambda x: int(datetime.timestamp(x)))
        df['pool_id'] = pool_id
        df = df[['timestamp', 'pool_id', 'metric', 'value']]
        return df

    @staticmethod
    def format_pool_aggregate_metrics(data):
        pass

    @staticmethod
    def format_token_metrics(data):
        pass

    @staticmethod
    def format_token_aggregate_metrics(data):
        pass

    def _execute_query(
            self, 
            table: str, 
            pool_id: str=None,
            token_id: str=None,
            start: int=None,
            end: int=None,
            timecol: str='timestamp'
        ) -> Dict:
        cursor = self.conn.cursor()
        
        to_execute = f'SELECT * FROM {table}'
        params = []

        if pool_id or token_id:
            to_execute += ' WHERE'
            if pool_id:
                to_execute += ' pool_id=?'
                params.append(pool_id)
            elif token_id:
                to_execute += ' token_id=?'
                params.append(token_id)
            if start:
                to_execute += f' AND {timecol} >= ?'
                params.append(start)
            if end:
                to_execute += f' AND {timecol} <= ?'
                params.append(end)
            to_execute += f' ORDER BY {timecol} ASC'
        
        try:
            cursor.execute(to_execute, params)
            results = cursor.fetchall()
        except sqlite3.Error as e:
            raise sqlite3.Error(f'Error executing query: {e}')
        finally:
            cursor.close()

        return results
    
    def get_token_metadata(self) -> Dict:
        results = self._execute_query("tokens")
        return {row["id"]: row for row in results}
    
    def get_pool_metadata(self) -> Dict:
        results = self._execute_query("pools")
        metadata = {row["id"]: row for row in results}
        for data in metadata.values():
            data['coins'] = json.loads(data['coins'])
        return metadata

    def get_pool_data(self, pool_id: str, start: int=None, end: int=None) -> pd.DataFrame:
        results = self._execute_query('pool_data', pool_id=pool_id, start=start, end=end, timecol='approxTimestamp')
        df = pd.DataFrame.from_dict(results)
        df['inputTokenWeights'] = df['inputTokenWeights'].apply(json.loads)
        df['inputTokenBalances'] = df['inputTokenBalances'].apply(json.loads)
        df = df.set_index(pd.to_datetime(df['approxTimestamp'], unit='s'))
        return df
    
    def get_swaps_data(self, pool_id: str, start: int=None, end: int=None) -> pd.DataFrame:
        results = self._execute_query('swaps', pool_id=pool_id, start=start, end=end)
        return pd.DataFrame.from_dict(results)
    
    def get_lp_data(self, pool_id: str, start: int=None, end: int=None) -> pd.DataFrame:
        results = self._execute_query('lp_events', pool_id=pool_id, start=start, end=end)
        return pd.DataFrame.from_dict(results)
    
    def get_ohlcv_data(self, token_id: str, start: int=None, end: int=None) -> pd.DataFrame:
        results = self._execute_query('token_ohlcv', token_id=token_id, start=start, end=end)
        df = pd.DataFrame.from_dict(results)
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s'))
        df = df.sort_index()
        return df

    
