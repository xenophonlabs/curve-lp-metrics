from sqlalchemy import Column, Integer, String, Boolean, Float, ARRAY, ForeignKey, Index

from .entity import Entity

class LPEvents(Entity):
    __tablename__ = 'lp_events'

    id = Column(String, primary_key=True)
    block = Column(Integer)
    liquidityProvider = Column(String)
    removal = Column(Boolean)
    timestamp = Column(Integer)
    tokenAmounts = Column(ARRAY(String))
    totalSupply = Column(Float)
    tx = Column(String)
    pool_id = Column(String, ForeignKey('pools.id'))
    block_gte = Column(Integer)
    block_lt = Column(Integer)

    idx = Index('idx_lp_events_pool_id_timestamp', 'pool_id', 'timestamp')

    def __init__(self, id, block, liquidityProvider, removal, timestamp, tokenAmounts,
                 totalSupply, tx, pool_id, block_gte, block_lt):
                
        super().__init__()
        
        self.id = id
        self.block = block
        self.liquidityProvider = liquidityProvider
        self.removal = removal
        self.timestamp = timestamp
        self.tokenAmounts = tokenAmounts
        self.totalSupply = totalSupply
        self.tx = tx
        self.pool_id = pool_id
        self.block_gte = block_gte
        self.block_lt = block_lt
