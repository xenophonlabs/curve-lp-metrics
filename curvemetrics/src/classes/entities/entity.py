from ....app import db

class Entity(db.Model):

    __abstract__ = True
    
    def as_dict(self):
        data = self.__dict__.copy()
        data.pop('_sa_instance_state')
        if '_id' in data:
            data['id'] = data.pop('_id')
        return data