swaps_query = (
    lambda **kwargs: f"""
        {{  swapEvents(
                where: {{
                    pool: "{kwargs['pool_id']}",
                    block_gte: "{kwargs['block_gte']}"
                    block_lt: "{kwargs['block_lt']}"
                }}
            ) {{
            id
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

liquidity_query = (
    lambda **kwargs: f"""
        {{  liquidityEvents(
                where: {{
                    pool: "{kwargs['pool_id']}",
                    block_gte: "{kwargs['block_gte']}"
                    block_lt: "{kwargs['block_lt']}"
                }}
            ) {{
            id
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

liquidity_pool_query = (
    lambda **kwargs: f"""
    {{ liquidityPool(
        id: "{kwargs['pool_id']}"
        block: {{number: {kwargs['block']}}}
    ) {{
        totalValueLockedUSD
        inputTokenBalances
        inputTokenWeights
        outputTokenSupply
        }}
    }}"""
)

pool_query = (
    lambda **kwargs: f"""
    {{ pool(
        id: "{kwargs['pool_id']}"
        block: {{number: {kwargs['block']}}}
    ) {{
        address
        assetType
        baseApr
        basePool
        c128
        coins
        creationBlock
        creationDate
        creationTx
        id
        isRebasing
        isV2
        lpToken
        metapool
        name
        poolType
        virtualPrice
        symbol
        }}
    }}"""
)

# Fetch less data to make faster
virtual_price_query = (
    lambda **kwargs: f"""
    {{ pool(
        id: "{kwargs['pool_id']}"
        block: {{number: {kwargs['block']}}}
    ) {{
        virtualPrice
        }}
    }}"""
)

queries = {
    'swapEvents' : swaps_query,
    'liquidityEvents' : liquidity_query,
    'liquidityPool' : liquidity_pool_query,
    'pool' : pool_query,
    'virtualPrice' : virtual_price_query,
}