from sqlalchemy import Column, String, Integer, Float, ARRAY, Numeric, ForeignKey, Index

from .entity import Entity

class PoolData(Entity):
    __tablename__ = 'pool_data'
    
    id = Column(Integer, primary_key=True)
    pool_id = Column(String, ForeignKey('pools.id'))
    block = Column(Integer)
    totalValueLockedUSD = Column(Float)
    inputTokenBalances = Column(ARRAY(String))
    inputTokenWeights = Column(ARRAY(String))
    timestamp = Column(Integer)
    outputTokenSupply = Column(Numeric)
    
    idx = Index('idx_pool_data_pool_id_timestamp', 'pool_id', 'timestamp')

    def __init__(self, id, pool_id, block, totalValueLockedUSD, inputTokenBalances, inputTokenWeights, timestamp, outputTokenSupply):
                
        super().__init__()
        
        self.id = id
        self.pool_id = pool_id
        self.block = block
        self.totalValueLockedUSD = totalValueLockedUSD
        self.inputTokenBalances = inputTokenBalances
        self.inputTokenWeights = inputTokenWeights
        self.timestamp = timestamp
        self.outputTokenSupply = outputTokenSupply
