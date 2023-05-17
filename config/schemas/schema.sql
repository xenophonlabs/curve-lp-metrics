-- TODO: I think the "id" column is useless for tables that have UNIQUE(...) columns
-- Storing pool metadata
CREATE TABLE IF NOT EXISTS pools (
    id TEXT PRIMARY KEY,
    assetType INTEGER,
    baseApr REAL,
    basePool TEXT,
    c128 INTEGER,
    coins TEXT,
    creationBlock INTEGER,
    creationDate INTEGER,
    creationTx TEXT,
    address TEXT UNIQUE,
    isRebasing INTEGER,
    isV2 INTEGER,
    lpToken TEXT,
    metapool TEXT,
    name TEXT,
    poolType TEXT,
    virtualPrice INTEGER,
    symbol TEXT
);

CREATE TABLE IF NOT EXISTS block_timestamps (
    block INTEGER PRIMARY KEY,
    timestamp INTEGER,
)

-- Storing pool <-> token one-to-many relationship. Ignore this table for now
CREATE TABLE IF NOT EXISTS pool_tokens (
    pool_id TEXT REFERENCES pools (id),
    token_id TEXT REFERENCES tokens (id),
    PRIMARY KEY (pool_id, token_id)
);

-- Storing token metadata
CREATE TABLE IF NOT EXISTS tokens (
    id TEXT PRIMARY KEY,
    name TEXT,
    symbol TEXT,
    decimals INTEGER
);

-- Storing token prices
CREATE TABLE IF NOT EXISTS token_ohlcv (
    id INTEGER PRIMARY KEY,
    token_id TEXT REFERENCES tokens (id),
    symbol TEXT,
    timestamp DATETIME, -- inconsistent type...
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    UNIQUE (token_id, timestamp)
);

-- Storing pool reserves
CREATE TABLE IF NOT EXISTS pool_data (
    id INTEGER PRIMARY KEY,
    pool_id TEXT REFERENCES pools (id),
    block INTEGER,
    totalValueLockedUSD REAL,
    inputTokenBalances TEXT,
    inputTokenWeights TEXT,
    approxTimestamp INTEGER,
    outputTokenSupply NUMERIC,
    UNIQUE (pool_id, block)
);

-- Storing lp events NOTE: we could avoid json TEXT repr for tokenAmounts by storing as a separate table
CREATE TABLE IF NOT EXISTS lp_events (
    id TEXT PRIMARY KEY,
    block INTEGER,
    liquidityProvider TEXT,
    removal INTEGER,
    timestamp INTEGER,
    tokenAmounts TEXT,
    totalSupply REAL,
    tx TEXT,
    pool_id TEXT REFERENCES pools (id),
    block_gte INTEGER,
    block_lt INTEGER
);

-- Storing swaps
CREATE TABLE IF NOT EXISTS swaps (
    id TEXT PRIMARY KEY,
    timestamp INTEGER,
    tx TEXT,
    pool_id TEXT REFERENCES pools (id),
    amountBought REAL,
    amountSold REAL,
    tokenBought TEXT REFERENCES tokens (id),
    tokenSold TEXT REFERENCES tokens (id),
    buyer TEXT,
    gasLimit INTEGER,
    gasUsed INTEGER,
    isUnderlying INTEGER,
    block_gte INTEGER,
    block_lt INTEGER,
    block INTEGER
);

-- storing raw pool metrics
CREATE TABLE IF NOT EXISTS pool_metrics (
    timestamp INTEGER,
    pool_id TEXT REFERENCES pools (id),
    metric TEXT,
    value REAL, 
    PRIMARY KEY (pool_id, metric, timestamp)
);

-- storing raw token metrics
CREATE TABLE IF NOT EXISTS token_metrics (
    timestamp INTEGER,
    token_id TEXT REFERENCES tokens (id),
    metric TEXT,
    value REAL,
    PRIMARY KEY (token_id, metric, timestamp)
);

-- storing windowed/aggregated pool metrics
CREATE TABLE IF NOT EXISTS pool_aggregates (
    timestamp INTEGER,
    pool_id TEXT REFERENCES pools (id),
    metric TEXT,
    type TEXT,
    window_size TEXT,
    value REAL,
    PRIMARY KEY (pool_id, metric, type, window_size, timestamp)
);

-- storing windowed/aggregated token metrics
CREATE TABLE IF NOT EXISTS token_aggregates (
    timestamp INTEGER,
    token_id TEST REFERENCES tokens (id),
    metric TEXT,
    type TEXT,
    window_size TEXT,
    value REAL,
    PRIMARY KEY (token_id, metric, type, window_size, timestamp)
);
