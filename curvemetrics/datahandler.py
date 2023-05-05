import sqlite3
import pandas as pd
import json
from datetime import datetime
import numpy as np

class RawDataHandler:

    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def create_tables(self):
        # Should this also populate the pools.json? That should be sent to the SQL table?
        # TODO: This should insert the pool <-> tokens relationship into the `pool_tokens` table
        pass

    def insert_pool_metadata(self, data):
        df = RawDataHandler.format_pool_metadata(data)

        # Insert the DataFrame into the `pools` table
        def insert_pool_metadata_row(row):
            sql = """
            INSERT OR IGNORE INTO pools (
                id,
                assetType,
                baseApr,
                basePool,
                c128,
                creationBlock,
                creationDate,
                creationTx,
                address,
                isRebasing,
                isV2,
                lpToken,
                metapool,
                name,
                poolType,
                virtualPrice,
                symbol
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            # Insert the row into the `pool_metadata` table
            self.conn.execute(sql, row)

        df.apply(insert_pool_metadata_row, axis=1)
        self.conn.commit()

    def insert_token_metadata(self, data):
        df = RawDataHandler.format_token_metadata(data)

        # Insert the DataFrame into the `pools` table
        def insert_token_metadata_row(row):
            sql = """
            INSERT OR IGNORE INTO tokens (
                id,
                name,
                symbol,
                decimals
            ) VALUES (?, ?, ?, ?)
            """
            # Insert the row into the `pool_metadata` table
            self.conn.execute(sql, row)

        df.apply(insert_token_metadata_row, axis=1)
        self.conn.commit()

    def insert_pool_data(self, data, start_timestamp, end_timestamp):
        df = RawDataHandler.format_pool_data(data, start_timestamp, end_timestamp)

        # Insert the DataFrame into the `pool_data` table
        def insert_pool_row(row):
            # Create an SQL INSERT OR IGNORE statement
            sql = """
            INSERT OR IGNORE INTO pool_data (
                block,
                inputTokenWeights,
                inputTokenBalances,
                totalValueLockedUSD,
                approxTimestamp,
                pool_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """
            self.conn.execute(sql, row)
        
        df.apply(insert_pool_row, axis=1)
        self.conn.commit()

    def insert_token_data(self, data):
        df = RawDataHandler.format_token_data(data)

        # Insert the DataFrame into the `pool_data` table
        def insert_token_row(row):
            # Create an SQL INSERT OR IGNORE statement
            sql = """
            INSERT OR IGNORE INTO token_ohlcv (
                token_id,
                symbol,
                open,
                high,
                low,
                close,
                volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            self.conn.execute(sql, row)
        
        df.apply(insert_token_row, axis=1)
        self.conn.commit()

    def insert_swaps_data(self, data):
        df = RawDataHandler.format_swaps_data(data)

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
                block,
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            # Insert the row into the `swaps` table
            self.conn.execute(sql, row)

        df.apply(insert_swap_row, axis=1)
        self.conn.commit()
    
    def insert_lp_data(self, data):
        # Convert JSON data to a pandas DataFrame
        df = RawDataHandler.format_lp_data(data)

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

    @staticmethod
    def format_pool_metadata(data):
        df = pd.DataFrame.from_dict(data).T.reset_index(drop=True)
        for col in ['assetType', 'creationBlock', 'creationDate', 'c128', 'isRebasing', 'isV2', 'virtualPrice']:
            df[col] = df[col].astype(int)
        for col in ['baseApr']:
            df[col] = df[col].astype(float)
        return df

    @staticmethod
    def format_token_metadata(data):
        df = pd.DataFrame.from_dict(data).T.reset_index(drop=True)
        for col in ['decimals']:
            df[col] = df[col].astype(int)
        return df
    
    @staticmethod
    def format_pool_data(data, start_timestamp, end_timestamp):
        df = pd.DataFrame([x for y in data for x in y])
        for col in ['totalValueLockedUSD']:
            df[col] = df[col].astype(float)
        for col in ['block']:
            df[col] = df[col].astype(int)
        df['inputTokenWeights'] = df['inputTokenWeights'].apply(lambda x: json.dumps(list(map(float, x))))
        df['inputTokenBalances'] = df['inputTokenBalances'].apply(lambda x: json.dumps(list(map(int, x))))
        df['approxTimestamp'] = np.linspace(start_timestamp, end_timestamp, len(df), dtype=int)
        return df

    @staticmethod
    def format_token_data(data):
        df = pd.DataFrame(data, columns=['token_id', 'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = (df['timestamp'] / 1000).astype(int)
        return df
    
    @staticmethod
    def format_swaps_data(data):
        df = pd.DataFrame([x for y in data for x in y])
        for col in ['amountBought', 'amountSold']:
            df[col] = df[col].astype(float)
        for col in ['timestamp', 'block', 'gasLimit', 'gasUsed', 'isUnderlying']:
            df[col] = df[col].astype(int)
        return df

    @staticmethod
    def format_lp_data(data):
        df = pd.DataFrame([x for y in data for x in y])
        for col in ['timestamp', 'block', 'removal']:
            df[col] = df[col].astype(int)
        df['totalSupply'] = df['totalSupply'].astype(float)
        df['tokenAmounts'] = df['tokenAmounts'].apply(lambda x: json.dumps(list(map(int, x))))
        return df
    