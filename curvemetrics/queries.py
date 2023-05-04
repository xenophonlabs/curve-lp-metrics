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

pool_query = (
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

queries = {
    'swapEvents' : swaps_query,
    'liquidityEvents' : liquidity_query,
    'liquidityPool' : pool_query,
}