CREATE INDEX IF NOT EXISTS idx_token_ohlcv_token_id_timestamp ON token_ohlcv(token_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_pool_data_pool_id_timestamp ON pool_data(pool_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_swaps_pool_id_timestamp ON swaps(pool_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_lp_events_pool_id_timestamp ON lp_events(pool_id, timestamp);
CREATE INDEX IF NOT EXISTS ids_block_timestamp ON block_timestamps(block);
-- TODO: metrics indexes