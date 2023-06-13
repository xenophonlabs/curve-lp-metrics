from sqlalchemy import Column, Integer, String, Float, ForeignKey, Index

from .entity import Entity

class TokenOHLCV(Entity):
    __tablename__ = 'token_ohlcv'

    token_id = Column(String, ForeignKey('tokens.id'), primary_key=True)
    symbol = Column(String, primary_key=True)
    timestamp = Column(Integer, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    idx = Index('idx_token_ohlcv_token_id_timestamp', 'token_id', 'timestamp')

    def __init__(self, token_id, symbol, timestamp, open, high, low, close, volume):
                
        super().__init__()
        
        self.id = id
        self.token_id = token_id
        self.symbol = symbol
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
