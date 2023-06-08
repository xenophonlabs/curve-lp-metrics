import os
from datetime import datetime, timedelta

from flask import Flask, jsonify, request

from ..src.classes.datahandler import DataHandler
from ..src.classes.entities import *

from .db import db

from dotenv import load_dotenv
load_dotenv()

PSQL_USER = os.getenv("PSQL_USER")
PSQL_PASSWORD = os.getenv("PSQL_PASSWORD")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{PSQL_USER}:{PSQL_PASSWORD}@localhost:5432/{PSQL_USER}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

@app.route('/pools', methods=['GET'])
def get_pools():
    datahandler = DataHandler()
    try:
        return jsonify(datahandler.pool_metadata)
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        datahandler.close()

@app.route('/tokens', methods=['GET'])
def get_tokens():
    datahandler = DataHandler()
    try:
        return jsonify(datahandler.token_metadata)
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        datahandler.close()

@app.route('/pool_data', methods=['GET'])
def get_pool_data():
    pool_id = request.args.get('pool_id')
    start = int(request.args.get('start'))
    end = int(request.args.get('end'))
    cols = request.args.get('cols', ['inputTokenBalances', 'timestamp', 'outputTokenSupply'])
    datahandler = DataHandler()
    try:
        return jsonify(datahandler.get_pool_data(pool_id, start, end, cols=cols).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        datahandler.close()

@app.route('/swaps_data', methods=['GET'])
def get_swaps_data():
    pool_id = request.args.get('pool_id')
    start = int(request.args.get('start'))
    end = int(request.args.get('end'))
    datahandler = DataHandler()
    try:
        return jsonify(datahandler.get_swaps_data(pool_id, start, end).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        datahandler.close()

@app.route('/lp_data', methods=['GET'])
def get_lp_data():
    pool_id = request.args.get('pool_id')
    start = int(request.args.get('start'))
    end = int(request.args.get('end'))
    datahandler = DataHandler()
    try:
        return jsonify(datahandler.get_lp_data(pool_id, start, end).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        datahandler.close()

@app.route('/snapshots', methods=['GET'])
def get_snapshots_data():
    pool_id = request.args.get('pool_id')
    start = int(request.args.get('start'))
    end = int(request.args.get('end'))
    cols = request.args.get('cols', ['timestamp', 'normalizedReserves', 'reserves', 'virtualPrice', 'lpPriceUSD', 'tvl', 'reservesUSD'])
    datahandler = DataHandler()
    try:
        return jsonify(datahandler.get_pool_snapshots(pool_id, start, end, cols=cols).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        datahandler.close()

@app.route('/ohlcv', methods=['GET'])
def get_ohlcv_data():
    pool_id = request.args.get('token_id')
    start = int(request.args.get('start'))
    end = int(request.args.get('end'))
    datahandler = DataHandler()
    try:
        return jsonify(datahandler.get_ohlcv_data(pool_id, start, end).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        datahandler.close()

@app.route('/pool_metrics', methods=['GET'])
def get_pool_metrics():
    pool_id = request.args.get('pool_id')
    metric = request.args.get('metric')
    start = int(request.args.get('start'))
    end = int(request.args.get('end'))
    cols = request.args.get('cols', ['timestamp', 'value'])
    datahandler = DataHandler()
    try:
        df = datahandler.get_pool_metric(pool_id, metric, start, end, cols=cols)
        df.index = [int(datetime.timestamp(x)) for x in df.index]
        return jsonify(df.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        datahandler.close()

@app.route('/token_metrics', methods=['GET'])
def get_token_metrics():
    token_id = request.args.get('token_id')
    metric = request.args.get('metric')
    start = int(request.args.get('start'))
    end = int(request.args.get('end'))
    cols = request.args.get('cols', ['timestamp', 'value'])
    datahandler = DataHandler()
    try:
        df = datahandler.get_token_metric(token_id, metric, start, end, cols=cols)
        df.index = [int(datetime.timestamp(x)) for x in df.index]
        return jsonify(df.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        datahandler.close()

@app.route('/changepoints', methods=['GET'])
def get_changepoints():
    pool_id = request.args.get('pool_id')
    model = request.args.get('model')
    metric = request.args.get('metric')
    start = int(request.args.get('start'))
    end = int(request.args.get('end'))
    cols = request.args.get('cols', ['timestamp'])
    datahandler = DataHandler()
    try:
        df = datahandler.get_changepoints(pool_id, model, metric, start, end, cols=cols)
        df.index = [int(datetime.timestamp(x)) for x in df.index]
        return jsonify(df.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        datahandler.close()

@app.route('/takers', methods=['GET'])
def get_takers():
    datahandler = DataHandler()
    try:
        return jsonify(datahandler.get_takers().to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        datahandler.close()

@app.route('/sharks', methods=['GET'])
def get_sharks():
    top = request.args.get('top', 0.9)
    datahandler = DataHandler()
    try:
        return jsonify(datahandler.get_sharks(top))
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        datahandler.close()

@app.route('/pool_X', methods=['GET'])
def get_pool_X():
    pool_id = request.args.get('pool_id')
    metric = request.args.get('metric')
    start = int(request.args.get('start'))
    end = int(request.args.get('end'))
    freq = request.args.get('freq', timedelta(hours=1))
    normalize = request.args.get('normalize', 'false').lower() == 'true'
    standardize = request.args.get('standardize', 'true').lower() == 'true'
    datahandler = DataHandler()
    try:
        df = datahandler.get_pool_X(metric, pool_id, start, end, freq, normalize=normalize, standardize=standardize)
        df.index = [int(datetime.timestamp(x)) for x in df.index]
        return jsonify(df.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        datahandler.close()

@app.route('/token_X', methods=['GET'])
def get_token_X():
    token_id = request.args.get('token_id')
    metric = request.args.get('metric')
    start = int(request.args.get('start'))
    end = int(request.args.get('end'))
    freq = request.args.get('freq', timedelta(hours=1))
    normalize = request.args.get('normalize', 'false').lower() == 'true'    
    standardize = request.args.get('standardize', 'true').lower() == 'true'
    datahandler = DataHandler()
    try:
        df = datahandler.get_token_X(metric, token_id, start, end, freq, normalize=normalize, standardize=standardize)
        df.index = [int(datetime.timestamp(x)) for x in df.index]
        return jsonify(df.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        datahandler.close()

if __name__ == '__main__':
    app.run(debug=True)
