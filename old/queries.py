### CONVEX-COMMUNITY QUERIES ###

cvx_swaps_query = (
    lambda **kwargs: f"""
        {{  swapEvents(
                where: {{
                    pool: "{kwargs['pool_id']}",
                    block_gte: "{kwargs['block_gte']}"
                    block_lt: "{kwargs['block_lt']}"
                }}
            ) {{
            amountBought
            amountSold
            block
            buyer
            gasLimit
            gasUsed
            isUnderlying
            timestamp
            tokenBought
            tokenSold
            tx
        }}
    }}"""
)

cvx_liquidity_query = (
    lambda **kwargs: f"""
        {{  liquidityEvents(
                where: {{
                    pool: "{kwargs['pool_id']}",
                    block_gte: "{kwargs['block_gte']}"
                    block_lt: "{kwargs['block_lt']}"
                }}
            ) {{
            block
            liquidityProvider
            removal
            timestamp
            tokenAmounts
            totalSupply
            tx
        }}
    }}"""
)

cvx_pool_query = (
    lambda **kwargs: f"""
        {{  pool(
                id: "{kwargs['pool_id']}"
                block: {{number: {kwargs['block']}}}
            ) {{
            virtualPrice
            symbol
            poolType
            name
            metapool
            lpToken
            isV2
            isRebasing
            creationTx
            creationDate
            creationBlock
            coins
            coinNames
            coinDecimals
            c128
            basePool
            baseApr
            assetType
            address
            id
        }}
    }}"""
)

### MESSARI QUERIES ###

messari_pool_query = (
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

messari_swaps_query = (
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

messari_withdraws_query = (
    lambda **kwargs: f"""
    {{ 
        withdraws(
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

messari_deposits_query = (
    lambda **kwargs: f"""
    {{ 
        deposits(
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

queries = {
    'cvx': {
        'swaps': {
            'query' : cvx_swaps_query,
            'key' : 'swapEvents',
        },
        'liquidityEvents': {
            'query' : cvx_liquidity_query,
            'key' : 'liquidityEvents',
        },
        'pool': {
            'query' : cvx_pool_query,
            'key' : 'pool',
        },
    },
    'messari': {
        'pool': {
            'query' : messari_pool_query,
            'key' : 'liquidityPool',
        },
        'withdraws': {
            'query' : messari_withdraws_query,
            'key' : 'withdraws',
        },
        'deposits': {
            'query' : messari_deposits_query,
            'key' : 'deposits',
        },
        'swaps': {
            'query' : messari_swaps_query,
            'key' : 'swaps',
        },
    }
}