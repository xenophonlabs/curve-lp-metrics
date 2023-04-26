import numpy as np
import requests as req
import pandas as pd
from datetime import date, datetime, timedelta
import nest_asyncio
import asyncio
import aiohttp

nest_asyncio.apply()

# TODO: Pool Data, Swap Data, Candle Data, LiquidityEvents Data, 

QUERY_POOL_DATA = (
    lambda **kwargs: f"""
        {{ pool(
            id: "{kwargs['pool_id']}"
            block: {{number: {kwargs['block']}}}
        ) {{
        id
        address
        name
        assetType
        basePool
        coinDecimals
        coinNames
        coins
        creationBlock
        creationDate
        isRebasing
        isV2
        lpToken
        metapool
        poolType
        symbol
        virtualPrice
        baseApr
        c128
        creationTx
        platform {{
        id
        }}
        }}
    }}"""
)

QUERY_SWAP_DATA = (
    lambda **kwargs: f"""
        {{ pool(
                id: "{kwargs['pool_id']}"
                block: {{number: {kwargs['block']}}}
            ) {{
            swapEvents(
                first: {kwargs['first']}
            ) {{
            amountBought
            amountSold
            block
            buyer
            gasLimit
            gasUsed
            id
            isUnderlying
            timestamp
            tokenBought
            tokenSold
            tx
            }}
        }}
    }}"""
)

class DataFetcher():

    # TODO: Is it pythonic to define these global constants inside the class?

    POOLS = {
        "3pool": "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7",
        "steth": "0xdc24316b9ae028f1497c275eb9192a3ea0f67022",
        "fraxusdc": "0xdcef968d416a41cdac0ed8702fac8128a64241a2",
        "UST wormhole": "0xceaf7747579696a2f0bb206a14210e3c9e6fb269",
        "USDN": "0x0f9cb53ebe405d49a0bbdbd291a65ff571bc83e1",
        "mim": "0x5a6a4d54456819380173272a5e8e9b9904bdf41b",
        "susd": "0xa5407eae9ba41422680e2e00537571bcc53efbfd",
        "frxeth": "0xa1f8a6807c402e4a15ef4eba36528a3fed24e577",
        "lusd": "0xed279fdd11ca84beef15af5d39bb4d4bee23f0ca",
        "busdv2": "0x4807862aa8b2bf68830e4c8dc86d0e9a998e085a",
        "stETH concentrated": "0x828b154032950c8ff7cf8085d841723db2696056",
        "cbETH/ETH": "0x5fae7e604fc3e24fd43a72867cebac94c65b404a",
        "cvxCRV/CRV": "0x971add32ea87f10bd192671630be3be8a11b8623"
    }

    CURVE_SUBGRAPH_URL = 'https://api.thegraph.com/subgraphs/name/convex-community/curve-mainnet'
    LLAMA_BLOCK_GETTER = 'https://coins.llama.fi/block/ethereum/'

    def __init__(self, start, end, step_size=50):
        
        self.get_blocks(start, end)
        self.step_size = step_size

    def get_blocks(self, start, end):
        """ 
        Get start and end blocks from input date strings using Defi Llama API.
        """
        start = datetime.fromisoformat(start)
        start_ts = int(datetime.timestamp(start))
        
        start_res = req.get(self.LLAMA_BLOCK_GETTER + str(start_ts)).json()

        self.start_timestamp = start_res['timestamp']
        self.start_block = start_res['height']

        end = datetime.fromisoformat(end)
        end_ts = int(datetime.timestamp(end))

        end_res = req.get(self.LLAMA_BLOCK_GETTER + str(end_ts)).json()

        self.end_timestamp = end_res['timestamp']
        self.end_block = end_res['height']

    async def fetch_data_one_block(self, session, query, url=CURVE_SUBGRAPH_URL, **kwargs):
        """
        Fetch the data for one block.
        'block' should be a key in kwargs
        """
        query = query(**kwargs)
        async with session.post(url, json={'query': query}) as res:
            block_data_object = await res.json()
            block_data_object = block_data_object['data']['pool']
            block_data_object['block'] = kwargs['block']
            return block_data_object # TODO: this isn't always the key

    async def fetch_data(self, query, url=CURVE_SUBGRAPH_URL, **kwargs):
        """
        Fetch data for a sequence of blocks async.
        """
        tasks = []
        async with aiohttp.ClientSession() as session:
            for b in range(self.start_block, self.end_block, self.step_size):  # Hourly
                block_kwargs = kwargs.copy()
                block_kwargs['block'] = b
                task = asyncio.ensure_future(self.fetch_data_one_block(session, query, url, **block_kwargs))
                tasks.append(task)

            raw_data = await asyncio.gather(*tasks)
            return raw_data
        
    def _get_pool_data(self, name):
        """
        Run our async data fetcher on a particular pool and return the results.
        """
        loop = asyncio.get_event_loop()
        pool_id = self.get_pool_id(name)
        pool_data = loop.run_until_complete(self.fetch_data(QUERY_POOL_DATA, pool_id=pool_id))
        return pool_data

    def get_pool_data(self, name):
        pool_data = self._get_pool_data(name)
        pool_data = self.format_pool_data(pool_data)
        return pool_data
    
    def format_pool_data(self, data):
        # TODO: convert block to timestamp and datetime.index the df to visual virtual price changes, fix the list->str.
        # TODO: save relevant pool data as attributes of the class so we can reference in other funcs
        # NOTE: the only thing that really changes is virtualPrice
        _df = pd.DataFrame.from_dict(data)
        _df['coinNames'] = _df['coinNames'].apply(lambda lst:','.join(lst))
        _df['coinDecimals'] = _df['coinDecimals'].apply(lambda lst:','.join(lst))
        _df['coins'] = _df['coins'].apply(lambda lst:','.join(lst))
        _df['coins'] = _df['coins'].apply(lambda lst:','.join(lst))
        _df = _df.drop(columns=['platform'])

        return _df
    
    def get_swaps_data(self, name, first=1):
        """
        Run our async data fetcher on a particular pool and return the results.
        """
        loop = asyncio.get_event_loop()
        pool_id = self.get_pool_id(name)
        data = loop.run_until_complete(self.fetch_data(QUERY_SWAP_DATA, pool_id=pool_id, first=first))
        return data

    def get_pool_id(self, name):
        """
        Get the pool id from the pool name.
        TODO: Maybe this shouldn't be a class method
        """
        return self.POOLS[name]