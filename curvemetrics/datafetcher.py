import requests as req
import nest_asyncio
import asyncio
import aiohttp
import os
import json
import ccxt.async_support as ccxt
import logging
from datetime import datetime, timedelta
from typing import Any, List, Tuple, Callable, Dict
from .queries import queries
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

CURVE_SUBGRAPH_URL_CVX = 'https://api.thegraph.com/subgraphs/name/convex-community/curve-mainnet'
CURVE_SUBGRAPH_URL_MESSARI = 'https://api.thegraph.com/subgraphs/name/messari/curve-finance-ethereum'
LLAMA_BLOCK_GETTER = 'https://coins.llama.fi/block/ethereum/'
SUPPORTED_POOLS = ['0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7', '0xdc24316b9ae028f1497c275eb9192a3ea0f67022', '0xdcef968d416a41cdac0ed8702fac8128a64241a2', '0xceaf7747579696a2f0bb206a14210e3c9e6fb269', '0x0f9cb53ebe405d49a0bbdbd291a65ff571bc83e1', '0x5a6a4d54456819380173272a5e8e9b9904bdf41b', '0xa5407eae9ba41422680e2e00537571bcc53efbfd', '0xa1f8a6807c402e4a15ef4eba36528a3fed24e577', '0xed279fdd11ca84beef15af5d39bb4d4bee23f0ca', '0x4807862aa8b2bf68830e4c8dc86d0e9a998e085a', '0x828b154032950c8ff7cf8085d841723db2696056', '0x5fae7e604fc3e24fd43a72867cebac94c65b404a', '0x971add32ea87f10bd192671630be3be8a11b8623']
INFURA_KEY = os.getenv("INFURA_KEY")

PATH = os.path.abspath(__file__).replace(os.path.basename(__file__), '')

class DataFetcher():
    """
    A class to asynchronously fetch data from theGraph API using a specified query.
    """

    def __init__(self, start: datetime, end: datetime, exchanges: List[str] = ['binanceus', 'coinbasepro', 'kraken']) -> None:
        """
        Initialize the DataFetcher class.

        @Params
            start_block (int): the starting block to fetch data from.
            end_block (int): the ending block to fetch data up to.
            exchanges (list): list of ccxt-supported exchanges (we will try getting OHLCV 
                data from these exchanges in the given order)
        """
        self.start_timestamp, self.start_block = DataFetcher.get_block(start)
        self.end_timestamp, self.end_block = DataFetcher.get_block(end)
        self.logger = logging.getLogger(__name__)
        self.exchanges = exchanges
        self.session = aiohttp.ClientSession()
    
    async def close(self) -> None:
        """
        Close the aiohttp session.
        """
        await self.session.close()

    @staticmethod
    def get_pool_metadata(pool_id, datetime_):
        url = DataFetcher.get_url('cvx')
        _, block = DataFetcher.get_block(datetime_)
        query = queries['pool'](pool_id=pool_id, block=block)
        res = req.post(url, json={'query': query})
        return res.json()['data']['pool']
    
    @staticmethod
    def get_pools_metadata():
        datetime_ = datetime.now() - timedelta(1)
        data = {}
        for pool_id in SUPPORTED_POOLS:
            pool_data = DataFetcher.get_pool_metadata(pool_id, datetime_)
            data[pool_id] = pool_data

        # TODO: temp json, can delete
        with open(PATH+'./pools.json', 'w') as f:
            json.dump(data, f)

        return data

    @staticmethod
    def get_token_metadata(token, client, erc20_abi):
        if token == '0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee':
            # Dummy for ETH since not erc20
            return {'name':'Ethereum', 'symbol':'ETH', 'decimals':18, 'id':token}
        token_checksummed = Web3.to_checksum_address(token)
        contract = client.eth.contract(address=token_checksummed, abi=erc20_abi)
        name = contract.functions.name().call()
        symbol = contract.functions.symbol().call()
        decimals = contract.functions.decimals().call()
        return {'name':name, 'symbol':symbol, 'decimals':decimals, 'id':token}
    
    @staticmethod
    def get_tokens_metadata(pool_metadata):
        tokens = {coin for x in pool_metadata.values() for coin in x['coins']}
        client = Web3(Web3.HTTPProvider(f"https://mainnet.infura.io/v3/{INFURA_KEY}"))
        erc20_abi = [{"constant":True,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"payable":False,"stateMutability":"view","type":"function"}]
        data = {}
        for token in tokens:
            data[token] = DataFetcher.get_token_metadata(token, client, erc20_abi)

        # TODO: temp json, can delete
        with open('tokens.json', 'w') as f:
            json.dump(data, f)
        return data

    async def execute_query_async(
        self, 
        query: Callable[..., str], 
        key: str, 
        url: str, 
        **kwargs: Any
    ) -> Any:
        """
        Execute query with specified kwargs.
        
        @Params
            query (function): a function that generates the GraphQL query string.
            key (str): the key in the response JSON object to extract the data.
            url (str): the API URL to send requests to.
            kwargs: keyword arguments for the query function.
        
        @Returns
            block_data_object (list of dictionaries): the data for one block.
        """
        query = query(**kwargs)
        async with self.session.post(url, json={'query': query}) as res:
            block_data_object = await res.json()
            block_data_object = block_data_object['data'][key]

            # Ensure all outputs are lists of dicts (some are just 1 dict)
            if type(block_data_object) != list:
                block_data_object = [block_data_object]
            
            if len(block_data_object) == 100:
                self.logger.warning(f'theGraph rate limit hit (100 txs). Might be missing data for {key} with kwargs {kwargs}.')

            # Add in kwargs to output for transparency
            for obj in block_data_object:
                for k in kwargs:
                    obj[k] = kwargs[k]

            return block_data_object
    
    async def execute_queries_async(
        self,
        query: Callable[..., str],
        key: str,
        url: str,
        step_size: int,
        full: bool,
        **kwargs: Any
    ) -> Any:
        """
        Execute a sequence of queries

        @Params
            query (function): a function that generates the GraphQL query string
            key (str): the key in the response JSON object to extract the data
            url (str): the API URL to send requests to
            step_size (int): number of blocks to get data for at each step
            full (bool): whether to get all data using block_lt and block_gte (True) or just data at each step block (False)
            kwargs: keyword arguments for the query function
        @Returns
            raw_data (list of lists of dictionaries): the data for all specified blocks
        """
        tasks = set()
        for b in range(self.start_block, self.end_block+1, step_size):
            query_kwargs = kwargs.copy()
            if full:
                query_kwargs['block_gte'] = b
                lt = b + step_size
                if lt > self.end_block:
                    lt = self.end_block + 1 # Make sure to include the end_block
                query_kwargs['block_lt'] = lt
            else:
                query_kwargs['block'] = b
            task = asyncio.create_task(self.execute_query_async(query, key, url, **query_kwargs))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        raw_data = await asyncio.gather(*tasks)
        return raw_data

    def execute_queries(
        self,
        pool_id: str,
        source: str,
        key: str,
        step_size: int,
        full: bool = False
    ) -> Any:
        """
        Wrapper for execute_queries().
        
        @Params
            pool_id (str): The name of the pool for which data is fetched.
            source (str): The subgraph source.
            key (str): The entity which we are querying (e.g. swapEvents in Convex-community subgraph).
            step_size (int): The number of blocks to fetch data for at each step.
            full (bool): If True, fetch data for the entire block range. If False, fetch data for each step block only. Defaults to False.
        
        @Returns
            data (list): The fetched data.
        """
        query = queries[key]
        url = DataFetcher.get_url(source)
        data = asyncio.run(self.execute_queries_async(query, key, url, step_size, full, pool_id=pool_id))
        return data

    def get_pool_data(
        self,
        pool_id: str,
        step_size: int = 1
    ) -> Any:
        """
        Get pool reserves data from Messari subgraph.
        """
        return self.execute_queries(pool_id, 'messari', 'liquidityPool', step_size, False)
    
    def get_swaps_data(
        self,
        pool_id: str,
        step_size: int = 10 # NOTE: increasing step_size risks losing txs. This is a subgraph bug.
    ) -> Any:
        """
        Get swaps data from Convex-community subgraph.
        """
        return self.execute_queries(pool_id, 'cvx', 'swapEvents', step_size, True)
    
    def get_lp_data(
        self,
        pool_id: str,
        step_size: int = 10 # NOTE: increasing step_size risks losing txs. This is a subgraph bug.
    ) -> Any:
        """
        Get lp deposits and withdrawals data from Convex-community subgraph.
        """
        return self.execute_queries(pool_id, 'cvx', 'liquidityEvents', step_size, True)

    async def get_ohlcv_async(
        self,
        symbol: str,
        limit: int,
        timeframe: str
    ) -> Any:
        """
        Fetch OHLCV data for the given symbol, limit, and timeframe asynchronously.

        @Params
            symbol (str): The trading pair symbol.
            limit (int): The maximum number of data points to fetch. Defaults to 1000.
            timeframe (str): The timeframe for the OHLCV data. Defaults to '1m'.

        @Returns
            data (list): The fetched OHLCV data.
        """
        since = self.start_timestamp * 1000
        data = []

        for exchange_id in self.exchanges:
            exchange = getattr(ccxt, exchange_id)()
            self.logger.info(f'Fetching OHLCV for {symbol} using {exchange}...')
            try:
                while since < self.end_timestamp * 1000:
                    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
                    data.extend(ohlcv)
                    since = int(ohlcv[-1][0]) + 1
                    await asyncio.sleep(exchange.rateLimit / 1000)
                return data
            except Exception as e:
                self.logger.warning(f'Failed to fetch {symbol} using {exchange}: {e}')
            finally:
                await exchange.close()

        raise Exception(f"Couldn't fetch OHLCV for {symbol} from any of {self.exchanges}.")

    def get_ohlcv(
        self,
        token: str,
        limit: int = 1000,
        timeframe: str = '1m'
    ) -> Any:
        """
        Wrapper for get_ohlcv_async().
        """
        symbol = DataFetcher.get_symbol_for_token(token)
        data = asyncio.run(self.get_ohlcv_async(symbol, limit, timeframe))
        data = [[token, symbol] + sublist for sublist in data]
        return data

    @staticmethod
    def get_symbol_for_token(token: str) -> str:
        # Note: let's do just USD for now
        with open(PATH+'./tokens.json', 'r') as f:
            tokens = json.load(f)
        symbol = tokens[token]['symbol']
        return f'{symbol}/USD'

    ### Helper methods

    @staticmethod
    def get_block(datetime_: datetime) -> Tuple[int, int]:
        """
        Get the block number corresponding to a given timestamp.

        Args:
            date_str: A string representing a timestamp in ISO 8601 format.

        Returns:
            A tuple containing two integers: the Unix timestamp corresponding to the
            input timestamp, and the block number at which that timestamp was recorded.
        """
        ts = int(datetime.timestamp(datetime_))
        res = req.get(LLAMA_BLOCK_GETTER + str(ts)).json()
        ts = res['timestamp']
        block = res['height']

        return ts, block

    @staticmethod
    def get_url(source: str) -> str:
        if source == "messari": return CURVE_SUBGRAPH_URL_MESSARI
        elif source == "cvx": return CURVE_SUBGRAPH_URL_CVX
        else: raise Exception(f'Invalid source {source}. Must be "messari" or "cvx"')
