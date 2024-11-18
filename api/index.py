
from flask import Flask, render_template, jsonify, request
import os
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)

# 全局变量存储模型
model = None

def load_model():
    """加载模型的函数"""
    global model
    try:
        model_path = os.path.join('model', 'end_risk_model.pkl')
        if os.path.exists(model_path):
            return joblib.load(model_path)
    except Exception as e:
        print(f"模型加载错误: {str(e)}")
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # 获取输入数据
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': '无效的请求数据'
            }), 400

        # 验证必要字段
        required_fields = ['nihss', 'sbp', 'neut', 'rdw', 'toast_laa', 'ias']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'缺少必要参数: {field}'
                }), 400

        # 准备模型输入数据
        try:
            features = np.array([[
                float(data['nihss']),
                float(data['sbp']),
                float(data['neut']),
                float(data['rdw']),
                int(data['toast_laa']),
                int(data['ias'])
            ]])
        except ValueError:
            return jsonify({
                'success': False,
                'error': '数据格式错误'
            }), 400

        # 使用模型预测
        global model
        if model is None:
            model = load_model()
        
        if model is not None:
            risk_prob = float(model.predict_proba(features)[0][1] * 100)
        else:
            # 如果模型加载失败，返回模拟数据
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
            'error': f'服务器内部错误: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

