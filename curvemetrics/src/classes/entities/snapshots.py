from sqlalchemy import Column, Integer, String, Float, ARRAY, ForeignKey, Index

from .entity import Entity

class Snapshots(Entity):
    __tablename__ = 'snapshots'

    id = Column(String, primary_key=True)
    A = Column(Integer)
    adminFee = Column(Float)
    fee = Column(Float)
    timestamp = Column(Integer)
    normalizedReserves = Column(ARRAY(String))
    offPegFeeMultiplier = Column(Float)
    reserves = Column(ARRAY(String))
    virtualPrice = Column(Integer)
    lpPriceUSD = Column(Float)
    tvl = Column(Float)
    totalDailyFeesUSD = Column(Float)
    reservesUSD = Column(ARRAY(String))
    lpFeesUSD = Column(Float)
    lastPricestimestamp = Column(Integer)
    lastPrices = Column(ARRAY(String))
    pool_id = Column(String, ForeignKey('pools.id'))
    block_gte = Column(Integer)
    block_lt = Column(Integer)

    idx = Index('idx_snapshots_pool_id_timestamp', 'pool_id', 'timestamp')

    def __init__(self, id, A, adminFee, fee, timestamp, normalizedReserves, offPegFeeMultiplier,
                 reserves, virtualPrice, lpPriceUSD, tvl, totalDailyFeesUSD, reservesUSD,
                 lpFeesUSD, lastPricestimestamp, lastPrices, pool_id, block_gte, block_lt):
                
        super().__init__()
        
        self.id = id
        self.A = A
        self.adminFee = adminFee
        self.fee = fee
        self.timestamp = timestamp
        self.normalizedReserves = normalizedReserves
        self.offPegFeeMultiplier = offPegFeeMultiplier
        self.reserves = reserves
        self.virtualPrice = virtualPrice
        self.lpPriceUSD = lpPriceUSD
        self.tvl = tvl
        self.totalDailyFeesUSD = totalDailyFeesUSD
        self.reservesUSD = reservesUSD
        self.lpFeesUSD = lpFeesUSD
        self.lastPricestimestamp = lastPricestimestamp
        self.lastPrices = lastPrices
        self.pool_id = pool_id
        self.block_gte = block_gte
        self.block_lt = block_lt
