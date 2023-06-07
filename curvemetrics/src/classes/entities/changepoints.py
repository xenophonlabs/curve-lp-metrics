from sqlalchemy import Column, String, Integer, ForeignKey

from .entity import Entity

class Changepoints(Entity):
    __tablename__ = 'changepoints'

    pool_id = Column(String, ForeignKey('pools.id'), primary_key=True)
    model = Column(String, primary_key=True)
    metric = Column(String, primary_key=True)
    freq = Column(String, primary_key=True)
    timestamp = Column(Integer, primary_key=True)

    def __init__(self, pool_id, model, metric, freq, timestamp):
                
        super().__init__()
        
        self.pool_id = pool_id
        self.model = model
        self.metric = metric
        self.freq = freq
        self.timestamp = timestamp
