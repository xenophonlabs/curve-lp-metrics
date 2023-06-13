from sqlalchemy import Column, String, Integer, Float, ARRAY, Boolean, Numeric

from .entity import Entity

class Pools(Entity):
    __tablename__ = 'pools'
    
    id = Column(String, primary_key=True)
    assetType = Column(Integer)
    baseApr = Column(Float)
    basePool = Column(String)
    c128 = Column(Boolean)
    coins = Column(ARRAY(String))
    creationBlock = Column(Integer)
    creationDate = Column(Integer)
    creationTx = Column(String)
    address = Column(String, unique=True)
    isRebasing = Column(Boolean)
    isV2 = Column(Boolean)
    lpToken = Column(String)
    metapool = Column(String)
    name = Column(String)
    poolType = Column(String)
    virtualPrice = Column(Numeric)
    symbol = Column(String)
    inputTokens = Column(ARRAY(String))
    
    def __init__(self, id, assetType, baseApr, basePool, c128, coins,
                 creationBlock, creationDate, creationTx, address,
                 isRebasing, isV2, lpToken, metapool, name, poolType,
                 virtualPrice, symbol, inputTokens):
                         
        super().__init__()
        
        self.id = id
        self.assetType = assetType
        self.baseApr = baseApr
        self.basePool = basePool
        self.c128 = c128
        self.coins = coins
        self.creationBlock = creationBlock
        self.creationDate = creationDate
        self.creationTx = creationTx
        self.address = address
        self.isRebasing = isRebasing
        self.isV2 = isV2
        self.lpToken = lpToken
        self.metapool = metapool
        self.name = name
        self.poolType = poolType
        self.virtualPrice = virtualPrice
        self.symbol = symbol
        self.inputTokens = inputTokens