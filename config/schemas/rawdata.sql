-- Storing pool metadata
CREATE TABLE pools (
    id TEXT PRIMARY KEY,
    assetType INTEGER,
    baseApr REAL,
    basePool TEXT,
    c128 INTEGER,
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

-- Storing pool <-> token one-to-many relationship
CREATE TABLE pool_tokens (
    pool_id TEXT REFERENCES pools (id),
    token_id TEXT REFERENCES tokens (id),
    PRIMARY KEY (pool_id, token_id)
);

-- Storing token metadata
CREATE TABLE tokens (
    id TEXT PRIMARY KEY,
    name TEXT,
    symbol TEXT,
    decimals INTEGER
);

-- Storing token prices
CREATE TABLE token_ohlcv (
    id INTEGER PRIMARY KEY,
    token_id TEXT REFERENCES tokens (id),
    symbol TEXT,
    timestamp DATETIME,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL
);

-- Storing pool reserves
CREATE TABLE pool_data (
    id INTEGER PRIMARY KEY,
    pool_id TEXT REFERENCES pools (id),
    block INTEGER,
    totalValueLockedUSD REAL,
    inputTokenBalances TEXT,
    inputTokenWeights TEXT,
    approxTimestamp INTEGER,
    UNIQUE (pool_id, block)
);

-- Storing lp events NOTE: we could avoid json TEXT repr for tokenAmounts by storing as a separate table
CREATE TABLE lp_events (
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
CREATE TABLE swaps (
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
