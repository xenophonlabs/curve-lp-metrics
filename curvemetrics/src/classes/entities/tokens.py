from sqlalchemy import Column, Integer, String

from .entity import Entity

class Tokens(Entity):
    __tablename__ = 'tokens'

    id = Column(String, primary_key=True)
    name = Column(String)
    symbol = Column(String)
    decimals = Column(Integer)

    def __init__(self, id, name, symbol, decimals):
        
        super().__init__()

        self.id = id
        self.name = name
        self.symbol = symbol
        self.decimals = decimals
