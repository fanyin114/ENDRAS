from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)

@app.route('/api/test')
def test():
    return jsonify({
        'status': 'ok',
        'message': 'API is working'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        return jsonify({
            'success': True,
            'received_data': data,
            'message': 'Prediction endpoint working'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run()
