from sqlalchemy import Column, Integer, String, Float, ForeignKey, Index

from .entity import Entity

class PoolMetrics(Entity):
    __tablename__ = 'pool_metrics'

    timestamp = Column(Integer, primary_key=True)
    pool_id = Column(String, ForeignKey('pools.id'), primary_key=True)
    metric = Column(String, primary_key=True)
    value = Column(Float)

    idx = Index('idx_pool_metrics_pool_id_metric_timestamp', 'pool_id', 'metric', 'timestamp')

    def __init__(self, timestamp, pool_id, metric, value):
                
        super().__init__()
        
        self.timestamp = timestamp
        self.pool_id = pool_id
        self.metric = metric
        self.value = value
