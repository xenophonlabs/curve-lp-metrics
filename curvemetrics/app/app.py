from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table, Column, Integer, String, MetaData, DateTime
import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    height = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(DateTime, default=datetime.datetime.utcnow)

@app.route('/person', methods=['GET'])
def get_person():
    name = request.args.get('name')
    person = Person.query.filter_by(name=name).first()
    if person:
        return jsonify({'name': person.name, 'height': person.height, 'timestamp': person.timestamp.strftime('%Y-%m-%d %H:%M:%S')}), 200
    else:
        return jsonify({'error': 'Person not found'}), 404

if __name__ == '__main__':
    db.create_all()  # Creates the table based on the classes defined
    app.run(debug=True)
