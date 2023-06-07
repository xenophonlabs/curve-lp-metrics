from sqlalchemy import Column, String, Float, Integer, Index

from .entity import Entity

class Takers(Entity):
    __tablename__ = 'takers'

    buyer = Column(String, primary_key=True)
    amountBought = Column(Float)
    amountSold = Column(Float)
    cumulativeMarkout = Column(Float)
    meanMarkout = Column(Float)
    count = Column(Float)
    windowSize = Column(Integer, primary_key=True)

    idx = Index('idx_takers_buyer_windowSize', 'buyer', 'windowSize')

    def __init__(self, buyer, amountBought, amountSold, cumulativeMarkout,
                 meanMarkout, count, windowSize):
                
        super().__init__()
        
        self.buyer = buyer
        self.amountBought = amountBought
        self.amountSold = amountSold
        self.cumulativeMarkout = cumulativeMarkout
        self.meanMarkout = meanMarkout
        self.count = count
        self.windowSize = windowSize
