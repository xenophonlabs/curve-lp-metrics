import nest_asyncio
import asyncio
import aiohttp
import os
import json
import logging
import time

import ccxt.async_support as ccxt
import requests as req

from datetime import datetime, timedelta
from typing import Any, List, Tuple, Callable, Dict, Union
from web3 import Web3
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, after_log, wait_exponential
import math

from ..queries import queries

load_dotenv()
nest_asyncio.apply()

RETRY_AMOUNTS = 5
CURVE_SUBGRAPH_URL_CVX = 'https://api.thegraph.com/subgraphs/name/convex-community/curve-mainnet'
CURVE_SUBGRAPH_URL_VOLUME_CVX = "https://api.thegraph.com/subgraphs/name/convex-community/volume-mainnet"
CURVE_SUBGRAPH_URL_MESSARI = 'https://api.thegraph.com/subgraphs/name/messari/curve-finance-ethereum'
LLAMA_BLOCK_GETTER = 'https://coins.llama.fi/block/ethereum/'

INFURA_KEY = os.getenv("INFURA_KEY")
ALCHEMY_KEY = os.getenv("ALCHEMY_KEY")
API_KEYS = json.loads(os.getenv("API_KEYS"))
API_SECRETS = json.loads(os.getenv("API_SECRETS"))

# WEB3_ENDPOINT = f"https://mainnet.infura.io/v3/{INFURA_KEY}"
WEB3_ENDPOINT = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}"

PATH = os.path.abspath(__file__).replace(os.path.basename(__file__), '')

class DataFetcher():
    """
    A class to asynchronously fetch data from theGraph API using a specified query.
    """

    def __init__(
            self, 
            exchanges: List[str] = ['binanceus', 'coinbasepro', 'bitfinex2'], 
            token_metadata: Dict={},
            logger = None
        ) -> None:
        """
        Initialize the DataFetcher class.

        @Params
            start (datetime): the starting time to fetch data from.
            end (datetime): the ending time to fetch data up to.
            exchanges (list): list of ccxt-supported exchanges (we will try getting OHLCV 
                data from these exchanges in the given order)
            token_metadata (dict): dictionary of token metadata (name, symbol, decimals, id)
        """
        self.token_metadata = token_metadata

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())

        self.exchanges = exchanges

        self.session = aiohttp.ClientSession()
    
    async def close(self) -> None:
        """
        Close the aiohttp session.
        """
        await self.session.close()

    @staticmethod
    @retry(stop=stop_after_attempt(RETRY_AMOUNTS), after=after_log(logging.getLogger(__name__), logging.DEBUG), wait=wait_exponential(multiplier=2, min=1, max=60))
    def get_pool_metadata(pool_id, datetime_):
        cvx_url = DataFetcher.get_url('cvx')
        messari_url = DataFetcher.get_url('messari')
        _, block = DataFetcher.get_block(datetime_)
        cvx_query = queries['pool'](pool_id=pool_id, block=block)
        cvx_res = req.post(cvx_url, json={'query': cvx_query})
        messari_query = queries['messari_pool_tokens'](pool_id=pool_id, block=block)
        data = cvx_res.json()['data']['pool']
        data['inputTokens'] = [x['id'] for x in req.post(messari_url, json={'query': messari_query}).json()['data']['liquidityPool']['inputTokens']]
        return data

    @staticmethod
    @retry(stop=stop_after_attempt(RETRY_AMOUNTS), after=after_log(logging.getLogger(__name__), logging.DEBUG), wait=wait_exponential(multiplier=2, min=1, max=60))
    def get_latest_availability():
        latest_ts = math.inf
        result = {}
        for src in ["cvx", "cvx-volume", "messari"]:
            data = DataFetcher.get_meta(src)
            ts = data['block']['timestamp']
            if ts < latest_ts:
                result = data['block']
                latest_ts = ts
        return result['number'], result['timestamp']

    @staticmethod
    @retry(stop=stop_after_attempt(RETRY_AMOUNTS), after=after_log(logging.getLogger(__name__), logging.DEBUG), wait=wait_exponential(multiplier=2, min=1, max=60))
    def get_meta(source: str):
        url = DataFetcher.get_url(source)
        query = queries['_meta']()
        res = req.post(url, json={'query': query})
        data = res.json()['data']['_meta']
        return data
    
    @staticmethod
    def get_pools_metadata(pools):
        datetime_ = datetime.now() - timedelta(1)
        data = {}
        for pool_id in pools:
            pool_data = DataFetcher.get_pool_metadata(pool_id, datetime_)
            data[pool_id] = pool_data
        return data

    @staticmethod
    @retry(stop=stop_after_attempt(RETRY_AMOUNTS), after=after_log(logging.getLogger(__name__), logging.DEBUG), wait=wait_exponential(multiplier=2, min=1, max=60))
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
        client = Web3(Web3.HTTPProvider(WEB3_ENDPOINT))
        erc20_abi = [{"constant":True,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"payable":False,"stateMutability":"view","type":"function"}]
        data = {}
        for token in tokens:
            data[token] = DataFetcher.get_token_metadata(token, client, erc20_abi)
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
            if "block" in kwargs and \
                "errors" in block_data_object and \
                f'data for block number {kwargs["block"]} is therefore not yet available' in block_data_object['errors'][0]['message']:
                return [] # No data for this block yet
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
        start_block,
        end_block,
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
        for b in range(start_block, end_block+1, step_size):
            query_kwargs = kwargs.copy()
            if full:
                query_kwargs['block_gte'] = b
                lt = b + step_size
                if lt > end_block:
                    lt = end_block + 1 # Make sure to include the end_block
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
        start, # block or ts
        end, # block or ts
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
        data = asyncio.run(self.execute_queries_async(start, end, query, key, url, step_size, full, pool_id=pool_id))
        return data

    @retry(stop=stop_after_attempt(RETRY_AMOUNTS), after=after_log(logging.getLogger(__name__), logging.DEBUG), wait=wait_exponential(multiplier=2, min=1, max=60))
    def get_pool_data(
        self,
        start_block,
        end_block,
        pool_id: str,
        step_size: int = 1
    ) -> Any:
        return self.execute_queries(start_block, end_block, pool_id, 'messari', 'liquidityPool', step_size, False)
    
    @retry(stop=stop_after_attempt(RETRY_AMOUNTS), after=after_log(logging.getLogger(__name__), logging.DEBUG), wait=wait_exponential(multiplier=2, min=1, max=60))
    def get_swaps_data(
        self,
        start_block,
        end_block,
        pool_id: str,
        step_size: int = 10 # NOTE: increasing step_size risks losing txs, limits to 100 results per response
    ) -> Any:
        """
        Get swaps data from Convex-community subgraph.
        """
        return self.execute_queries(start_block, end_block, pool_id, 'cvx', 'swapEvents', step_size, True)
    
    @retry(stop=stop_after_attempt(RETRY_AMOUNTS), after=after_log(logging.getLogger(__name__), logging.DEBUG), wait=wait_exponential(multiplier=2, min=1, max=60))
    def get_lp_data(
        self,
        start_block,
        end_block,
        pool_id: str,
        step_size: int = 10 # NOTE: increasing step_size risks losing txs, limits to 100 results per response
    ) -> Any:
        """
        Get lp deposits and withdrawals data from Convex-community subgraph.
        """
        return self.execute_queries(start_block, end_block, pool_id, 'cvx', 'liquidityEvents', step_size, True)
    
    @retry(stop=stop_after_attempt(RETRY_AMOUNTS), after=after_log(logging.getLogger(__name__), logging.DEBUG), wait=wait_exponential(multiplier=2, min=1, max=60))
    def get_snapshots(
        self,
        start_ts: int,
        end_ts: int,
        pool_id: str,
        step_size: int = 60*60*24 # Expect 1 response per day, timestamps increment in seconds
    ) -> Any:
        """
        Get  daily pool snapshots from Convex-community subgraph.
        """
        return self.execute_queries(start_ts, end_ts, pool_id, 'cvx-volume', 'dailyPoolSnapshots', step_size, True)

    async def get_ohlcv_async(
        self,
        start_timestamp: int,
        end_timestamp: int,
        symbol: str,
        limit: int,
        timeframe: str,
        default_exchange: str='' # Default exchange to use
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
        since = start_timestamp * 1000
        data = []

        exchanges = self.exchanges
        if default_exchange != '':
            exchanges = [default_exchange] + [e for e in self.exchanges if e != default_exchange]

        for exchange_id in exchanges:
            exchange = getattr(ccxt, exchange_id)()
            # add API key and secrets if specified in .env
            if exchange_id in API_KEYS.keys():
                exchange.apiKey = API_KEYS[exchange_id]
            if exchange_id in API_SECRETS.keys():
                exchange.secret = API_SECRETS[exchange_id]
            try:
                while since < end_timestamp * 1000:
                    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
                    data.extend(ohlcv)
                    if len(ohlcv) == 0: # If no data is returned, then there is no more data.
                        break
                    since = int(ohlcv[-1][0]) + 1
                    await asyncio.sleep(exchange.rateLimit / 1000)
                self.logger.info(f'Using {exchange} for {symbol}.')
                return data
            except Exception as e:
                self.logger.warning(f'Failed to fetch {symbol} using {exchange}: {e}.')
            finally:
                await exchange.close()

        raise Exception(f"Couldn't fetch OHLCV for {symbol} from any of {self.exchanges}.")

    @retry(stop=stop_after_attempt(RETRY_AMOUNTS), after=after_log(logging.getLogger(__name__), logging.DEBUG), wait=wait_exponential(multiplier=2, min=1, max=60))
    def get_ohlcv(
        self,
        start_timestamp: int,
        end_timestamp: int,
        token: str,
        limit: int = 1000,
        timeframe: str = '1m',
        default_exchange: str='' # Default exchange to use
    ) -> Any:
        """
        Wrapper for get_ohlcv_async().
        """
        symbol = self.get_symbol_for_token(token)
        data = asyncio.run(self.get_ohlcv_async(start_timestamp, end_timestamp, symbol, limit, timeframe, default_exchange=default_exchange))
        data = [[token, symbol] + sublist for sublist in data]
        return data

    def get_symbol_for_token(self, token: str) -> str:
        if self.token_metadata == {}:
            raise Exception('Token metadata not loaded in constructor.')
        symbol = self.token_metadata[token]['symbol']
        return f'{symbol}/USD'.upper()

    def search_rounds(self, contract, target):
        # Let's start by determining the current phase and aggregator round.
        curr_round = contract.functions.latestRoundData().call()[0]
        curr_phase = curr_round >> 64

        # Reverse O(n) search for correct phase
        for phase in range(curr_phase, -1, -1):
            first_round_for_phase = (phase << 64) + 1
            first_ts = contract.functions.getRoundData(first_round_for_phase).call()[2]
            if target >= first_ts:
                break
        
        left = first_round_for_phase
        right = curr_round

        closest_round = -1
        closest_diff = float('inf')
        count = 0
        while left <= right:
            time.sleep(1)
            count += 1
            mid = (left + right) // 2
            try:
                round_data = contract.functions.getRoundData(mid).call()
                round_timestamp = round_data[3]

                diff = target - round_timestamp
                if 0 < diff < closest_diff:
                    closest_round = mid
                    closest_diff = diff
                    closest_timestamp = round_timestamp
                if round_timestamp < target:
                    left = mid + 1
                else:
                    right = mid - 1
            except Exception as e:
                self.logger.exception(e)
                right = mid - 1

        if closest_round != -1:
            self.logger.info(f"Found the closest round: {closest_round}, at {datetime.fromtimestamp(closest_timestamp)}.")
        else:
            self.logger.info("No round found.")

        self.logger.info(f"Number of iterations: {count}")

        return closest_round
    
    # @retry(stop=stop_after_attempt(RETRY_AMOUNTS), after=after_log(logging.getLogger(__name__), logging.DEBUG), wait=wait_exponential(multiplier=2, min=1, max=60))
    def get_chainlink_prices(self, token, chainlink_address, start_timestamp, end_timestamp):
        chainlink_address = Web3.to_checksum_address(chainlink_address)
        abi = abi = '[{"inputs":[],"name":"decimals","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"description","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint80","name":"_roundId","type":"uint80"}],"name":"getRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"latestRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"version","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]'
        client = Web3(Web3.HTTPProvider(WEB3_ENDPOINT))
        contract = client.eth.contract(address=chainlink_address, abi=abi)
        symbol = contract.functions.description().call().replace(" ", "")
        
        latest_ts = contract.functions.latestRoundData().call()[3]
        if latest_ts < start_timestamp:
            self.logger.exception(f"Latest Chainlink timestamp {latest_ts} is before start timestamp {start_timestamp}.")
            return []
        end_timestamp = min(end_timestamp, latest_ts)

        roundnr = self.search_rounds(contract, start_timestamp)

        decimals = contract.functions.decimals().call()
        current_timestamp = 0
        data = []
        while True:
            try:
                round_data = contract.functions.getRoundData(roundnr).call()
                data.append([token, symbol, round_data[2]*1000, None, None, None, round_data[1]/10**decimals, None])
                current_timestamp = round_data[3]
                if current_timestamp >= end_timestamp:
                    break
                roundnr += 1
            except Exception as e:
                self.logger.info(f"Chainlink exception: {e}")
                roundnr = (((roundnr >> 64) + 1) << 64) + 1 # first round of next aggregator
        return data

    @staticmethod
    @retry(stop=stop_after_attempt(RETRY_AMOUNTS), after=after_log(logging.getLogger(__name__), logging.DEBUG), wait=wait_exponential(multiplier=2, min=1, max=60))
    def get_block(datetime_: Union[int, datetime]) -> Tuple[int, int]:
        """
        Get the block number corresponding to a given timestamp.

        Args:
            date_str: A string representing a timestamp in ISO 8601 format.

        Returns:
            A tuple containing two integers: the Unix timestamp corresponding to the
            input timestamp, and the block number at which that timestamp was recorded.
        """
        if type(datetime_) == datetime:
            ts = int(datetime.timestamp(datetime_))
        else: ts = datetime_
        res = req.get(LLAMA_BLOCK_GETTER + str(ts)).json()
        ts = res['timestamp']
        block = res['height']

        return ts, block

    @staticmethod
    def get_url(source: str) -> str:
        if source == "messari": return CURVE_SUBGRAPH_URL_MESSARI
        elif source == "cvx": return CURVE_SUBGRAPH_URL_CVX
        elif source == "cvx-volume": return CURVE_SUBGRAPH_URL_VOLUME_CVX
        else: raise Exception(f'Invalid source {source}')
