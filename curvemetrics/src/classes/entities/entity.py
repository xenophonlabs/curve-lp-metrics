from sqlalchemy.ext.declarative import declarative_base
import ast

base = declarative_base()

class Entity(base):

    __abstract__ = True
    
    def as_dict(self):
        data = self.__dict__.copy()
        data.pop('_sa_instance_state')
        if '_id' in data:
            data['id'] = data.pop('_id')
        return data