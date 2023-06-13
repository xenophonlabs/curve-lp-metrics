from sqlalchemy import Column, String, Integer, Float, ARRAY, Numeric, ForeignKey, Index, DECIMAL

from .entity import Entity

class PoolData(Entity):
    __tablename__ = 'pool_data'
    
    pool_id = Column(String, ForeignKey('pools.id'), primary_key=True)
    block = Column(Integer)
    totalValueLockedUSD = Column(Float)
    inputTokenBalances = Column(ARRAY(Numeric))
    inputTokenWeights = Column(ARRAY(DECIMAL))
    timestamp = Column(Integer, primary_key=True)
    outputTokenSupply = Column(Numeric)
    
    idx = Index('idx_pool_data_pool_id_timestamp', 'pool_id', 'timestamp')

    def __init__(self, pool_id, block, totalValueLockedUSD, inputTokenBalances, inputTokenWeights, timestamp, outputTokenSupply):
                
        super().__init__()
        
        self.id = id
        self.pool_id = pool_id
        self.block = block
        self.totalValueLockedUSD = totalValueLockedUSD
        self.inputTokenBalances = inputTokenBalances
        self.inputTokenWeights = inputTokenWeights
        self.timestamp = timestamp
        self.outputTokenSupply = outputTokenSupply
