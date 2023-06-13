from sqlalchemy import Column, Integer, String, Float, ForeignKey

from .entity import Entity

class TokenMetrics(Entity):
    __tablename__ = 'token_metrics'

    timestamp = Column(Integer, primary_key=True)
    token_id = Column(String, ForeignKey('tokens.id'), primary_key=True)
    metric = Column(String, primary_key=True)
    value = Column(Float)

    def __init__(self, timestamp, token_id, metric, value):
                
        super().__init__()
        
        self.timestamp = timestamp
        self.token_id = token_id
        self.metric = metric
        self.value = value
