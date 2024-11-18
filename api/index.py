from flask import Flask, render_template, jsonify, request
import os
import joblib
import numpy as np

app = Flask(__name__,
            template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates')))

# 加载模型
try:
    model = joblib.load('model/end_risk_model.pkl')
except:
    print("警告：模型文件加载失败，将使用模拟数据")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # 验证输入数据
        required_fields = ['nihss', 'sbp', 'neut', 'rdw', 'toast_laa', 'ias']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'缺少必要参数: {field}'
                }), 400
                
        # 验证数值范围
        if not (0 <= data['nihss'] <= 42):
            return jsonify({
                'success': False,
                'error': 'NIHSS评分必须在0-42之间'
            }), 400
            
        if not (0 <= data['sbp'] <= 300):
            return jsonify({
                'success': False,
                'error': '收缩压必须在0-300之间'
            }), 400
            
        if not (0 <= data['neut'] <= 20):
            return jsonify({
                'success': False,
                'error': '中性粒细胞计数必须在0-20之间'
            }), 400
            
        if not (0 <= data['rdw'] <= 60):
            return jsonify({
                'success': False,
                'error': '红细胞分布宽度必须在0-60之间'
            }), 400

        if model is not None:
            # 准备模型输入数据
            features = np.array([[
                data['nihss'],
                data['sbp'],
                data['neut'],
                data['rdw'],
                data['toast_laa'],
                data['ias']
            ]])
            
            # 使用模型进行预测
            risk_prob = float(model.predict_proba(features)[0][1] * 100)
        else:
            # 如果模型未加载，使用模拟数据
            risk_prob = 15.5

        # 根据风险概率确定风险等级和描述
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
    app.run(debug=True, host='0.0.0.0', port=5000)
