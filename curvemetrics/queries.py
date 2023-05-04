swaps_query = (
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

liquidity_query = (
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

liquidity_pool_query = (
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
        createdBlockNumber
        createdTimestamp
        outputToken
        symbol
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
        coinDecimals
        coinNames
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

queries = {
    'swapEvents' : swaps_query,
    'liquidityEvents' : liquidity_query,
    'liquidityPool' : liquidity_pool_query,
    'pool' : pool_query,
}