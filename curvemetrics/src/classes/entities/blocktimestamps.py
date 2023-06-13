from sqlalchemy import Column, Integer, Index

from .entity import Entity

class BlockTimestamps(Entity):
    __tablename__ = 'blockTimestamps'

    block = Column(Integer, primary_key=True)
    timestamp = Column(Integer)

    idx = Index('idx_block_timestamp_block', 'block')

    def __init__(self, block, timestamp):
                
        super().__init__()
        
        self.block = block
        self.timestamp = timestamp
