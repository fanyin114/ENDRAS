from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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
        # 模拟预测结果
        risk_percentage = 15  # 示例风险值
        
        risk_level = "高风险" if risk_percentage >= 29 else "低风险"
        risk_description = "发生早期神经功能恶化的风险较高" if risk_percentage >= 29 else "发生早期神经功能恶化的风险较低"
        
        return jsonify({
            'success': True,
            'risk_probability': risk_percentage,
            'risk_level': risk_level,
            'risk_description': risk_description
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run()
