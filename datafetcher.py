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
from queries import queries

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

    def __init__(self, start, end):
        
        # Connect to web3 for any specific data we need
        self.web3 = Web3(Web3.HTTPProvider(f"https://mainnet.infura.io/v3/{INFURA_KEY}"))
        
        # Get start and end blocks from input date strings
        self.get_blocks(start, end)
        
        # Load pool mapping
        with open('pools.json', 'r') as f:
            self.pools = json.load(f)
        
        # Load token mapping
        with open('tokens.json', 'r') as f:
            self.tokens = json.load(f)
        
        self.erc20_abi = [{"constant":True,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"payable":False,"stateMutability":"view","type":"function"}]

    ### Data fetching methods

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
    
    def fetch_data_simple(self, source, key, pool):
        """
        Make single API call at start block.
        """
        query, key = queries[source][key]['query'], queries[source][key]['key']
        url = self.get_url(source)
        pool_id = self.get_pool_id(pool)
        kwargs = {'pool_id':pool_id, 'block':self.start_block}
        query = query(**kwargs)
        res = req.post(url, json={'query':query}).json()['data']['pool']
        return res

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

    async def fetch_data(self, query, key, url, step_size, full=False, **kwargs):
        """
        Fetch data for a sequence of blocks async.
        @Params
            Full:       bool.       If True: get ALL the data using block_lt and block_gte.
                                    If False: get just the data at each step block
            step_size:  int.        Number of blocks to get data for at each step (if full=false, just the data at each step block)
        """
        tasks = []
        async with aiohttp.ClientSession() as session:
            for b in range(self.start_block, self.end_block, step_size):
                block_kwargs = kwargs.copy()
                if full:
                    block_kwargs['block_gte'] = b
                    block_kwargs['block_lt'] = b + step_size
                else:
                    block_kwargs['block'] = b
                task = asyncio.ensure_future(self.fetch_data_one_block(session, query, key, url, **block_kwargs))
                tasks.append(task)

            raw_data = await asyncio.gather(*tasks)
            return raw_data
    
    def get_pool_data(self, name, source="messari", save=False, step_size=50):
        """
        source = "messari" or "cvx"
        """
        query, key = queries[source]['pool']['query'], queries[source]['pool']['key']
        url = self.get_url(source)
        loop = asyncio.get_event_loop()
        pool_id = self.get_pool_id(name)
        data = loop.run_until_complete(self.fetch_data(query, key, url, step_size, pool_id=pool_id))
        data = self.format_pool_data(data, source)

        if save:
            data.to_csv(f'{DATA_PATH}/{name}_{source}_pool.csv')

        return data
    
    def get_swaps_data(self, name, source="messari", save=False, step_size=50):
        """
        Run our async data fetcher on a particular pool and return the results.
        source = "messari" or "cvx"
        """
        query, key = queries[source]['swaps']['query'], queries[source]['swaps']['key']
        url = self.get_url(source)
        loop = asyncio.get_event_loop()
        pool_id = self.get_pool_id(name)
        data = loop.run_until_complete(self.fetch_data(query, key, url, step_size, full=True, pool_id=pool_id))
        data = self.format_swaps_data(data, source)

        if save:
            data.to_csv(f'{DATA_PATH}/{name}_{source}_swaps.csv')

        return data
    
    def get_lp_data(self, name, source="messari", save=False, step_size=50):
        """
        source = "messari" or "cvx"
        """
        url = self.get_url(source)
        pool_id = self.get_pool_id(name)

        if source == "messari":
            # One query for deposits, one for withdraws
            dfs = []
            for type_ in ['deposits', 'withdraws']:
                query, key = queries[source][type_]['query'], queries[source][type_]['key']
                loop = asyncio.get_event_loop()
                data_ = loop.run_until_complete(self.fetch_data(query, key, url, step_size, full=True, pool_id=pool_id))
                data_ = self.format_lp_data(data_, source, type_=type_)
                dfs.append(data_)
            data = pd.concat(dfs)
            data = data.sort_index()

        elif source == "cvx":
            query, key = queries[source]['liquidityEvents']['query'], queries[source]['liquidityEvents']['key']
            loop = asyncio.get_event_loop()
            data = loop.run_until_complete(self.fetch_data(query, key, url, step_size, full=True, pool_id=pool_id))
            data = self.format_lp_data(data, source, pool=name)

        if save:
            data.to_csv(f'{DATA_PATH}/{name}_{source}_lp.csv')

        return data
    
    ### Formatting methods

    def format_pool_data(self, data, source):
        if source == "messari":
            return self.format_pool_data_messari(data)
        elif source == "cvx":
            return self.format_pool_data_cvx(data)

    def format_swaps_data(self, data, source):
        if source == "messari":
            return self.format_swaps_data_messari(data)
        elif source == "cvx":
            return self.format_swaps_data_cvx(data)
    
    def format_lp_data(self, data, source, type_=None, pool=None):
        if source == "messari":
            return self.format_lp_data_messari(data, type_)
        elif source == "cvx":
            return self.format_lp_data_cvx(data, pool)
    
    def format_swaps_data_messari(self, data):
        for block in data:
            if len(block) == 0:
                continue
            for row in block:
                row['tokenIn.symbol'] = row['tokenIn']['symbol']
                # row['tokenIn.price'] = float(row['tokenIn']['lastPriceUSD'])
                row['amountIn'] = float(row['amountIn']) / 10**row['tokenIn']['decimals']

                row['tokenOut.symbol'] = row['tokenOut']['symbol']
                # row['tokenOut.price'] = float(row['tokenOut']['lastPriceUSD'])
                row['amountOut'] = float(row['amountOut']) / 10**row['tokenOut']['decimals']
        
        df = pd.DataFrame([x for y in data for x in y])

        for col in ['timestamp', 'blockNumber']:
            df[col] = df[col].astype(int)
        
        for col in ['amountInUSD', 'amountOutUSD']:
            df[col] = df[col].astype(float)

        # Approximate datetime index
        df.index = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
        df = df.drop(columns=['tokenIn', 'tokenOut', 'pool_id', 'block_gte', 'block_lt', 'timestamp', 'blockNumber'], axis=1)
        df = df[['tokenIn.symbol', 'amountIn', 'amountInUSD', 'tokenOut.symbol', 'amountOut', 'amountOutUSD', 'from', 'hash']]
        df = df.round(5)

        return df

    def format_swaps_data_cvx(self, data):
        df = pd.DataFrame([x for y in data for x in y])
        df['tokenBought'] = df['tokenBought'].apply(lambda addy: self.tokens[addy]['symbol'])
        df['tokenSold'] = df['tokenSold'].apply(lambda addy: self.tokens[addy]['symbol'])
        df['datetime'] = df['timestamp'].apply(lambda ts: datetime.fromtimestamp(int(ts)))
        df = df.set_index('datetime')
        df = df.drop(['gasLimit', 'gasUsed', 'isUnderlying'], axis=1)
        # df = df.drop(['block', 'timestamp', 'pool_id', 'block_gte', 'block_lt', 'gasLimit', 'gasUsed', 'isUnderlying'], axis=1)
        df = df.rename(columns={
            'amountSold': 'amountIn',
            'amountBought': 'amountOut',
            'tx': 'hash',
            'buyer': 'from',
            'tokenSold': 'tokenIn.symbol',
            'tokenBought': 'tokenOut.symbol'
        })
        for col in ['amountIn', 'amountOut']:
            df[col] = df[col].astype(float)
        # TODO: pls fix
        df['amountInUSD'] = None
        df['amountOutUSD'] = None
        df = df[['tokenIn.symbol', 'amountIn', 'amountInUSD', 'tokenOut.symbol', 'amountOut', 'amountOutUSD', 'from', 'hash']]
        df = df.round(5)

        return df
    
    def format_lp_data_messari(self, data, type_):
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
        df['type'] = type_

        return df
    
    def format_lp_data_cvx(self, data, pool):
        pool_data = self.fetch_data_simple('cvx', 'pool', pool)

        for block in data:
            if len(block) == 0:
                continue
            for row in block:
                for i, amt in enumerate(row['tokenAmounts']):
                    token = pool_data['coinNames'][i]
                    decimals = int(pool_data['coinDecimals'][i])
                    row[f'{token}.amount'] = int(amt)  / 10**decimals
                row['totalSupply'] = int(row['totalSupply']) / 10**18

        df = pd.DataFrame([x for y in data for x in y])

        df['type'] = df['removal'].apply(lambda x: 'withdraws' if x==True else 'deposits')

        for col in ['timestamp', 'block']:
            df[col] = df[col].astype(int)

        df['amountUSD'] = None # TODO: what do here?
        df['to'] = None # TODO: convert from/to to just LP

        df.index = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
        df = df.rename(columns={'tx':'hash', 'liquidityProvider':'from'})
        df = df.drop(columns=['removal', 'timestamp', 'block', 'pool_id', 'block_gte', 'block_lt', 'tokenAmounts'], axis=1)
        df = df.round(5)
        
        return df

    def format_pool_data_messari(self, data):
        tokens = []
        for row in data:
            row = row[0]
            for i, info in enumerate(row['inputTokens']):
                tokens.append(info['id'])
                # row[info['symbol']+'.price'] = float(info['lastPriceUSD'])
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
        
        # Cache token data in tokens.json
        self.cache_token_data(tokens)

        df = df.round(5)

        return df

    def format_pool_data_cvx(self, data):
        tokens = []
        for row in data:
            row = row[0]
            for i, addy in enumerate(row['coins']):
                tokens.append(addy)
        
        df = pd.DataFrame([x for y in data for x in y])
        df = df.drop(columns=['coins', 'coinDecimals', 'coinNames'], axis=1)

        for col in ['virtualPrice']:
            df[col] = df[col].astype(float)
        
        df['virtualPrice'] = df['virtualPrice'] / 10**18

        # Approximate datetime index
        df.index = pd.date_range(datetime.fromtimestamp(self.start_timestamp), end=datetime.fromtimestamp(self.end_timestamp), periods=df.shape[0], unit='s', name='datetime')
        
        self.cache_token_data(tokens)

        df = df.round(5)

        return df
    
    ### Helper methods

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

    def get_url(self, source):
        if source == "messari": return CURVE_SUBGRAPH_URL_MESSARI
        elif source == "cvx": return CURVE_SUBGRAPH_URL_CVX
        else: raise(f'Invalid source {source}. Must be "messari" or "cvx"')
