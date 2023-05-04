-- Storing pool metadata
CREATE TABLE pools (
    id INTEGER PRIMARY KEY,
    block INTEGER UNIQUE,
    name TEXT UNIQUE,
    symbol TEXT,
    creation_date DATETIME,
    creation_tx TEXT,
    token_ids TEXT -- Storing as TEXT but will contain a JSON array of token IDs
);

-- Storing token metadata
CREATE TABLE tokens (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    symbol TEXT
);

-- Storing token prices
CREATE TABLE token_prices (
    id INTEGER PRIMARY KEY,
    token_id INTEGER REFERENCES tokens (id),
    timestamp DATETIME,
    price REAL,
    exchange TEXT,
    UNIQUE (token_id, timestamp, exchange)
);

-- Storing pool reserves
CREATE TABLE pool_data (
    id INTEGER PRIMARY KEY,
    pool_id INTEGER REFERENCES pools (id),
    timestamp DATETIME,
    token_ids TEXT, -- Storing as TEXT but will contain a JSON array of token IDs
    token_reserves TEXT, -- Storing as TEXT but will contain a JSON array of token reserves
    UNIQUE (pool_id, timestamp)
);

-- Storing lp events
CREATE TABLE lp_events (
    id INTEGER PRIMARY KEY,
    pool_id INTEGER REFERENCES pools (id),
    tx TEXT UNIQUE,
    timestamp DATETIME,
    token_ids TEXT, -- Storing as TEXT but will contain a JSON array of token IDs
    token_amounts TEXT, -- Storing as TEXT but will contain a JSON array of token amounts
    type TEXT
);

-- Storing swaps
CREATE TABLE swaps (
    tx TEXT PRIMARY KEY,
    timestamp INTEGER,
    pool_id TEXT,
    amountBought REAL,
    amountSold REAL,
    tokenBought TEXT,
    tokenSold TEXT,
    buyer TEXT,
    gasLimit INTEGER,
    gasUsed INTEGER,
    isUnderlying INTEGER,
    block_gte INTEGER,
    block_lt INTEGER,
    block INTEGER
);
