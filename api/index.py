
from flask import Flask, render_template, jsonify, request
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': '无效的请求数据'
            }), 400

        # 使用固定的模拟数据进行测试
        risk_prob = 15.5

        # 确定风险等级和描述
        if risk_prob >= 29:
            risk_level = "高风险"
            risk_description = "患者发生早期神经功能恶化的风险较高，建议密切监测病情变化，及时进行干预。"
        else:
            risk_level = "低风险"
            risk_description = "患者发生早期神经功能恶化的风险较低，建议按常规进行治疗和监测。"

        return jsonify({
            'success': True,
            'risk_probability': risk_prob,
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

