from sqlalchemy import Column, Integer, String, Float, ForeignKey, Boolean, Index

from .entity import Entity

class Swaps(Entity):
    __tablename__ = 'swaps'

    id = Column(String, primary_key=True)
    timestamp = Column(Integer)
    tx = Column(String)
    pool_id = Column(String, ForeignKey('pools.id'))
    amountBought = Column(Float)
    amountSold = Column(Float)
    tokenBought = Column(String, ForeignKey('tokens.id'))
    tokenSold = Column(String, ForeignKey('tokens.id'))
    buyer = Column(String)
    gasLimit = Column(Integer)
    gasUsed = Column(Integer)
    isUnderlying = Column(Boolean)
    block_gte = Column(Integer)
    block_lt = Column(Integer)
    block = Column(Integer)

    idx = Index('idx_swaps_pool_id_timestamp', 'pool_id', 'timestamp')

    def __init__(self, id, timestamp, tx, pool_id, amountBought, amountSold,
                 tokenBought, tokenSold, buyer, gasLimit, gasUsed, isUnderlying,
                 block_gte, block_lt, block):
                
        super().__init__()
        
        self.id = id
        self.timestamp = timestamp
        self.tx = tx
        self.pool_id = pool_id
        self.amountBought = amountBought
        self.amountSold = amountSold
        self.tokenBought = tokenBought
        self.tokenSold = tokenSold
        self.buyer = buyer
        self.gasLimit = gasLimit
        self.gasUsed = gasUsed
        self.isUnderlying = isUnderlying
        self.block_gte = block_gte
        self.block_lt = block_lt
        self.block = block
