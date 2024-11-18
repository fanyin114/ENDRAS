from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'API is working'

@app.route('/api/test')
def test():
    return {'status': 'ok'}
