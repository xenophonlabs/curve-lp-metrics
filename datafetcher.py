import numpy as np
import requests as req
import pandas as pd
from datetime import date, datetime, timedelta
import nest_asyncio
import asyncio
import aiohttp
import json
from web3 import Web3
from dotenv import load_dotenv
import os

nest_asyncio.apply()
load_dotenv()

# TODO: Create a results class for each results output
# TODO: Step size should be specific to the query not to the class

FILE_PATH = os.path.abspath(__file__)
FIGS_PATH = FILE_PATH.replace(os.path.basename(__file__), 'figs/')
DATA_PATH = FILE_PATH.replace(os.path.basename(__file__), 'data/')

CURVE_SUBGRAPH_URL_CVX = 'https://api.thegraph.com/subgraphs/name/convex-community/curve-mainnet'
CURVE_SUBGRAPH_URL_MESSARI = 'https://api.thegraph.com/subgraphs/name/messari/curve-finance-ethereum'

LLAMA_BLOCK_GETTER = 'https://coins.llama.fi/block/ethereum/'
INFURA_KEY = os.getenv("INFURA_KEY")

class DataFetcher():

    def __init__(self, start, end, step_size=50):
        
        # Connect to web3 for any specific data we need
        self.web3 = Web3(Web3.HTTPProvider(f"https://mainnet.infura.io/v3/{INFURA_KEY}"))
        
        # Get start and end blocks from input date strings
        self.get_blocks(start, end)

        # Set step size for data fetcher
        self.step_size = step_size
        
        # Load pool mapping
        with open('pools.json', 'r') as f:
            self.pools = json.load(f)
        
        # Load token mapping
        with open('tokens.json', 'r') as f:
            self.tokens = json.load(f)
        
        self.erc20_abi = [{"constant":True,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"payable":False,"stateMutability":"view","type":"function"}]

    def get_blocks(self, start, end):
        """ 
        Get start and end blocks from input date strings using Defi Llama API.
        """
        start = datetime.fromisoformat(start)
        start_ts = int(datetime.timestamp(start))
        
        start_res = req.get(LLAMA_BLOCK_GETTER + str(start_ts)).json()

        self.start_timestamp = start_res['timestamp']
        self.start_block = start_res['height']

        end = datetime.fromisoformat(end)
        end_ts = int(datetime.timestamp(end))

        end_res = req.get(LLAMA_BLOCK_GETTER + str(end_ts)).json()

        self.end_timestamp = end_res['timestamp']
        self.end_block = end_res['height']

    async def fetch_data_one_block(self, session, query, key, url, **kwargs):
        """
        Fetch the data for one block.
        'block' should be a key in kwargs
        """
        query = query(**kwargs)
        async with session.post(url, json={'query': query}) as res:
            block_data_object = await res.json()
            block_data_object = block_data_object['data'][key]

            # Ensure all outputs are lists of dicts (some are just 1 dict)
            if type(block_data_object) != list:
                block_data_object = [block_data_object]
            
            # Add in kwargs to output for transparency
            for obj in block_data_object:
                for k in kwargs:
                    obj[k] = kwargs[k]

            return block_data_object

    async def fetch_data(self, query, key, url, full=False, **kwargs):
        """
        Fetch data for a sequence of blocks async.
        @Params
            Full : bool.    If True: get ALL the data using block_lt and block_gte.
                            If False: get just the data at each step block
        """
        tasks = []
        async with aiohttp.ClientSession() as session:
            for b in range(self.start_block, self.end_block, self.step_size):
                block_kwargs = kwargs.copy()
                if full:
                    block_kwargs['block_gte'] = b
                    block_kwargs['block_lt'] = b + self.step_size
                else:
                    block_kwargs['block'] = b
                task = asyncio.ensure_future(self.fetch_data_one_block(session, query, key, url, **block_kwargs))
                tasks.append(task)

            raw_data = await asyncio.gather(*tasks)
            return raw_data
    
    def get_pool_data(self, name, save=False):

        query = (
            lambda **kwargs: f"""
            {{ liquidityPool(
                id: "{kwargs['pool_id']}"
                block: {{number: {kwargs['block']}}}
            ) {{
                symbol
                name
                totalValueLockedUSD
                isSingleSided
                inputTokenBalances
                inputTokenWeights
                inputTokens {{
                    decimals
                    id
                    lastPriceBlockNumber
                    lastPriceUSD
                    name
                    symbol
                }}
                }}
            }}"""
        )

        loop = asyncio.get_event_loop()
        pool_id = self.get_pool_id(name)
        data = loop.run_until_complete(self.fetch_data(query, 'liquidityPool', CURVE_SUBGRAPH_URL_MESSARI, pool_id=pool_id))
        data = self.format_pool_data(data)

        if save:
            data.to_csv(f'{DATA_PATH}/{name}_pool.csv')

        return data
    
    def get_swaps_data(self, name, save=False):
        """
        Run our async data fetcher on a particular pool and return the results.
        """
        query = (
            lambda **kwargs: f"""
            {{ 
                swaps(
                    where: {{
                        blockNumber_gte: "{kwargs['block_gte']}", 
                        blockNumber_lt: "{kwargs['block_lt']}", 
                        pool: "{kwargs['pool_id']}"
                    }}
                ) {{
                    amountIn
                    amountInUSD
                    amountOut
                    amountOutUSD
                    blockNumber
                    from
                    hash
                    timestamp
                    to
                    tokenIn {{
                        lastPriceUSD
                        symbol
                        id
                        decimals
                    }}
                    tokenOut {{
                        decimals
                        id
                        lastPriceUSD
                        symbol
                    }}
                }}
            }}"""
        )

        loop = asyncio.get_event_loop()
        pool_id = self.get_pool_id(name)
        data = loop.run_until_complete(self.fetch_data(query, 'swaps', CURVE_SUBGRAPH_URL_MESSARI, full=True, pool_id=pool_id))
        data = self.format_swaps_data(data)

        if save:
            data.to_csv(f'{DATA_PATH}/{name}_swaps.csv')

        return data
    
    def get_lp_data(self, name, target):
        """
        Internal method to get either deposits or withdraws data
        """
        query = (
            lambda **kwargs: f"""
            {{ 
                {target}(
                    where: {{
                        blockNumber_gte: "{kwargs['block_gte']}", 
                        blockNumber_lt: "{kwargs['block_lt']}", 
                        pool: "{kwargs['pool_id']}"
                    }}
                ) {{
                    timestamp
                    amountUSD
                    blockNumber
                    from
                    to
                    hash
                    inputTokenAmounts
                    inputTokens {{
                        decimals
                        id
                        lastPriceBlockNumber
                        lastPriceUSD
                        name
                        symbol
                    }}
                }}
            }}"""
        )

        loop = asyncio.get_event_loop()
        pool_id = self.get_pool_id(name)
        data = loop.run_until_complete(self.fetch_data(query, target, CURVE_SUBGRAPH_URL_MESSARI, full=True, pool_id=pool_id))
        data = self.format_lp_data(data)

        return data
    
    def get_withdraws_data(self, name, save=False):
        """
        Run our async data fetcher on a particular pool and return the results.
        """
        data = self.get_lp_data(name, 'withdraws')

        if save:
            data.to_csv(f'{DATA_PATH}/{name}_withdraws.csv')

        return data
    
    def get_deposits_data(self, name, save=False):
        """
        Run our async data fetcher on a particular pool and return the results.
        """
        data = self.get_lp_data(name, 'deposits')

        if save:
            data.to_csv(f'{DATA_PATH}/{name}_deposits.csv')

        return data
    
    def format_swaps_data(self, data):
        for block in data:
            if len(block) == 0:
                continue
            for row in block:
                row['tokenIn.symbol'] = row['tokenIn']['symbol']
                row['tokenIn.price'] = float(row['tokenIn']['lastPriceUSD'])
                row['amountIn'] = float(row['amountIn']) / 10**row['tokenIn']['decimals']

                row['tokenOut.symbol'] = row['tokenOut']['symbol']
                row['tokenOut.price'] = float(row['tokenOut']['lastPriceUSD'])
                row['amountOut'] = float(row['amountOut']) / 10**row['tokenOut']['decimals']
        
        df = pd.DataFrame([x for y in data for x in y])

        for col in ['timestamp', 'blockNumber']:
            df[col] = df[col].astype(int)
        
        for col in ['amountInUSD', 'amountOutUSD']:
            df[col] = df[col].astype(float)

        # Approximate datetime index
        df.index = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))

        df = df.drop(columns=['tokenIn', 'tokenOut', 'pool_id', 'block_gte', 'block_lt', 'timestamp', 'blockNumber'], axis=1)

        df = df.round(5)

        return df

    def format_lp_data(self, data):
        for block in data:
            if len(block) == 0:
                continue
            for row in block:
                for i, info in enumerate(row['inputTokens']):
                    row[info['symbol']+'.amount'] = int(row['inputTokenAmounts'][i])  / 10**info['decimals']
        
        df = pd.DataFrame([x for y in data for x in y])

        for col in ['timestamp', 'blockNumber']:
            df[col] = df[col].astype(int)
        
        for col in ['amountUSD']:
            df[col] = df[col].astype(float)

        # Approximate datetime index
        df.index = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))

        df = df.drop(columns=['inputTokenAmounts', 'inputTokens', 'pool_id', 'block_gte', 'block_lt', 'timestamp', 'blockNumber'], axis=1)

        df = df.round(5)

        return df

    def format_pool_data(self, data):
        addresses = []
        for row in data:
            row = row[0]
            for i, info in enumerate(row['inputTokens']):
                addresses.append(info['id'])
                row[info['symbol']+'.price'] = float(info['lastPriceUSD'])
                row[info['symbol']+'.weight'] = float(row['inputTokenWeights'][i])
                row[info['symbol']+'.balance'] = int(row['inputTokenBalances'][i])  / 10**info['decimals']
        
        df = pd.DataFrame([x for y in data for x in y])
        df = df.drop(columns=['inputTokens', 'inputTokenWeights', 'inputTokenBalances', 'pool_id'], axis=1)

        for col in ['symbol', 'name']:
            df[col] = df[col].astype(str)
        
        for col in ['totalValueLockedUSD']:
            df[col] = df[col].astype(float)

        # Approximate datetime index
        df.index = pd.date_range(datetime.fromtimestamp(self.start_timestamp), end=datetime.fromtimestamp(self.end_timestamp), periods=df.shape[0], unit='s', name='datetime')
        
        self.cache_token_data(addresses)

        df = df.round(5)

        return df

    def cache_token_data(self, tokens):
        """
        Checks that we have the name and symbol of all token addresses
        """
        missing_addresses = set(tokens) - set(self.tokens)
        if len(missing_addresses) > 0:
            for addy in missing_addresses:
                _addy = Web3.to_checksum_address(addy)
                contract = self.web3.eth.contract(address=_addy, abi=self.erc20_abi)
                name = contract.functions.name().call()
                symbol = contract.functions.symbol().call()
                decimals = contract.functions.decimals().call()
                self.tokens[addy] = {'name': name, 'symbol': symbol, 'decimals': decimals}

            with open('tokens.json', 'w') as f:
                json.dump(self.tokens, f)

    def get_pool_id(self, name):
        """
        Get the pool id from the pool name.
        """
        # TODO: Add created timestamp and block
        try:
            pool_id = self.pools[name]
        except:
            raise(f'Pool {name} not found in pools.json')

        return pool_id
    
    def get_token_info(self, addy):
        """
        Get the token info from the token addresses.
        """
        return self.tokens[addy]


# NOTE: Using convex community subgraph
#         query = (
#             lambda **kwargs: f"""
#                 {{ pool(
#                     id: "{kwargs['pool_id']}"
#                     block: {{number: {kwargs['block']}}}
#                 ) {{
#                 address
#                 name
#                 assetType
#                 basePool
#                 coinDecimals
#                 coinNames
#                 coins
#                 creationBlock
#                 creationDate
#                 isRebasing
#                 isV2
#                 lpToken
#                 metapool
#                 poolType
#                 symbol
#                 virtualPrice
#                 baseApr
#                 c128
#                 creationTx
#                 platform {{
#                     id
#                 }}
#                 }}
#             }}"""
#         )

# query = (
#             lambda **kwargs: f"""
#                 {{  swapEvents(
#                         where: {{
#                             pool: "{kwargs['pool_id']}",
#                             block_gte: "{kwargs['block_gte']}"
#                             block_lt: "{kwargs['block_lt']}"
#                         }}
#                     ) {{
#                     amountBought
#                     amountSold
#                     block
#                     buyer
#                     gasLimit
#                     gasUsed
#                     isUnderlying
#                     timestamp
#                     tokenBought
#                     tokenSold
#                     tx
#                 }}
#             }}"""
#         )

# def format_pool_data(self, data):
#     # NOTE: the only thing that really changes is virtualPrice
#     df = pd.DataFrame([x for y in data for x in y])
#     df['coinNames'] = df['coinNames'].apply(lambda lst:','.join(lst))
#     df['coinDecimals'] = df['coinDecimals'].apply(lambda lst:','.join(lst))
#     df['coins'] = df['coins'].apply(lambda lst:','.join(lst))
#     df['coins'] = df['coins'].apply(lambda lst:','.join(lst))
#     df = df.drop(columns=['platform'])

#     return df

# def format_swaps_data(self, data):
#         df = pd.DataFrame([x for y in data for x in y])
#         self.cache_token_data(df['tokenBought'].unique().tolist() + df['tokenSold'].unique().tolist())
#         df['tokenBought'] = df['tokenBought'].apply(lambda addy: self.tokens[addy]['symbol'])
#         df['tokenSold'] = df['tokenSold'].apply(lambda addy: self.tokens[addy]['symbol'])
#         df['datetime'] = df['timestamp'].apply(lambda ts: datetime.fromtimestamp(int(ts)))

#         return df