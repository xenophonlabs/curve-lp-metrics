import sqlite3
import pandas as pd
import json

class RawDatabaseHelper:

    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.conn.row_factory = sqlite3.Row

    def create_tables(self):
        # Create all tables as shown in previous answers
        ...

    def insert_swaps_data(self, data):
        # Convert JSON data to a pandas DataFrame
        df = RawDatabaseHelper.format_swaps_data(data)

        # Insert the DataFrame into the `swaps` table
        def insert_swap_row(row):
            # Create an SQL INSERT OR IGNORE statement
            sql = """
            INSERT OR IGNORE INTO swaps (
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
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            # Insert the row into the `swaps` table
            self.conn.execute(sql, row)

        # Apply the custom function to each row in the DataFrame
        df.apply(insert_swap_row, axis=1)

        # Commit the changes
        self.conn.commit()
    
    def close(self):
        self.conn.close()
    
    @staticmethod
    def format_swaps_data(data):
        df = pd.DataFrame([x for y in data for x in y])
        for col in ['amountBought', 'amountSold']:
            df[col] = df[col].astype(float)
        for col in ['timestamp', 'block', 'gasLimit', 'gasUsed', 'isUnderlying']:
            df[col] = df[col].astype(int)
        return df

    def format_lp_data(data):
        df = pd.DataFrame([x for y in data for x in y])
        for col in ['timestamp', 'block']:
            df[col] = df[col].astype(int)
        for col in ['tokenAmounts']:
            df[col] = df[col].apply(lambda x: json.dumps(x))
        return df
    