import requests as req
import nest_asyncio
import asyncio
import aiohttp
import os
import json
import ccxt.async_support as ccxt
import logging
from datetime import datetime
from typing import Any, List, Tuple, Callable
from .queries import queries

nest_asyncio.apply()

CURVE_SUBGRAPH_URL_CVX = 'https://api.thegraph.com/subgraphs/name/convex-community/curve-mainnet'
CURVE_SUBGRAPH_URL_MESSARI = 'https://api.thegraph.com/subgraphs/name/messari/curve-finance-ethereum'
LLAMA_BLOCK_GETTER = 'https://coins.llama.fi/block/ethereum/'
SUPPORTED_POOLS = ['3pool', 'steth', 'fraxusdc', 'UST wormhole', 'USDN', 'mim', 'susd', 'frxeth', 'lusd', 'busdv2', 'stETH concentrated', 'cbETH/ETH', 'cvxCRV/CRV']

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

    ### Data fetching methods
    @staticmethod
    def get_pool_metadata(pool_name, datetime_):
        url = DataFetcher.get_url('cvx')
        pool_id = DataFetcher.get_pool_id(pool_name)
        _, block = DataFetcher.get_block(datetime_)
        query = queries['pool'](pool_id=pool_id, block=block)
        res = req.post(url, json={'query': query}).json()['data']['pool']
        return res
    
    @staticmethod
    def cache_pool_metadata():
        datetime_ = datetime.now()
        data = {}
        for pool in SUPPORTED_POOLS:
            pool_data = DataFetcher.get_pool_metadata(pool, datetime_)
            data[pool] = pool_data
        with open(PATH+'./pools.json', 'w') as f:
            json.dump(data, f)

    @staticmethod
    async def execute_query_async(
        session: aiohttp.ClientSession, 
        query: Callable[..., str], 
        key: str, 
        url: str, 
        **kwargs: Any
    ) -> Any:
        """
        Execute query with specified kwargs.
        
        @Params
            session (aiohttp.ClientSession): HTTP session to post requests.
            query (function): a function that generates the GraphQL query string.
            key (str): the key in the response JSON object to extract the data.
            url (str): the API URL to send requests to.
            kwargs: keyword arguments for the query function.
        
        @Returns
            block_data_object (list of dictionaries): the data for one block.
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
        for b in range(self.start_block, self.end_block, step_size):
            query_kwargs = kwargs.copy()
            if full:
                query_kwargs['block_gte'] = b
                query_kwargs['block_lt'] = min(b + step_size, self.end_block)
            else:
                query_kwargs['block'] = b
            task = asyncio.create_task(self.execute_query_async(self.session, query, key, url, **query_kwargs))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        raw_data = await asyncio.gather(*tasks)
        return raw_data

    def execute_queries(
        self,
        pool_name: str,
        source: str,
        key: str,
        step_size: int,
        full: bool = False
    ) -> Any:
        """
        Wrapper for execute_queries().
        
        @Params
            pool_name (str): The name of the pool for which data is fetched.
            source (str): The subgraph source.
            key (str): The entity which we are querying (e.g. swapEvents in Convex-community subgraph).
            step_size (int): The number of blocks to fetch data for at each step.
            full (bool): If True, fetch data for the entire block range. If False, fetch data for each step block only. Defaults to False.
        
        @Returns
            data (list): The fetched data.
        """
        query = queries[key]
        url = DataFetcher.get_url(source)
        pool_id = DataFetcher.get_pool_id(pool_name)
        data = asyncio.run(self.execute_queries_async(query, key, url, step_size, full, pool_id=pool_id))
        return data

    def get_pool_data(
        self,
        pool_name: str,
        step_size: int = 1
    ) -> Any:
        """
        Get pool reserves data from Messari subgraph.
        """
        return self.execute_queries(pool_name, 'messari', 'liquidityPool', step_size, False)
    
    def get_swaps_data(
        self,
        pool_name: str,
        step_size: int = 100
    ) -> Any:
        """
        Get swaps data from Convex-community subgraph.
        """
        return self.execute_queries(pool_name, 'cvx', 'swapEvents', step_size, True)
    
    def get_lp_data(
        self,
        pool_name: str,
        step_size: int = 100
    ) -> Any:
        """
        Get lp deposits and withdrawals data from Convex-community subgraph.
        """
        return self.execute_queries(pool_name, 'cvx', 'liquidityEvents', step_size, True)

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

        raise Exception(f"Couldn't fetch OHLCV for {symbol} from any of {self.exchanges}.")

    def get_ohlcv(
        self,
        symbol: str,
        limit: int = 1000,
        timeframe: str = '1m'
    ) -> Any:
        """
        Wrapper for get_ohlcv_async().
        """
        return asyncio.run(self.get_ohlcv_async(symbol, limit, timeframe))
    
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
    def get_pool_id(name: str) -> str:
        """
        Get the pool id from the pool name.
        """
        try:
            with open(PATH+'./pools.json', 'r') as f:
                pools = json.load(f)
                return pools[name]['address']
        except Exception as e: 
            raise Exception(f'Pool {name} not found in pools.json. {e}')

    @staticmethod
    def get_url(source: str) -> str:
        if source == "messari": return CURVE_SUBGRAPH_URL_MESSARI
        elif source == "cvx": return CURVE_SUBGRAPH_URL_CVX
        else: raise Exception(f'Invalid source {source}. Must be "messari" or "cvx"')
