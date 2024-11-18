from flask import Flask, request, jsonify, render_template
import os
import joblib
import logging
import pandas as pd

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# 加载模型
try:
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'XGBOOST_model1113.pkl')
    model = joblib.load(model_path)
    print(f"模型加载成功: {model_path}")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    model = None

MODEL_FEATURES = ['NIHSS', 'SBP', 'NEUT', 'RDW', 'TOAST-LAA_1', 'IAS_1']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': '模型未加载'}), 500
        
    try:
        data = request.get_json()
        
        # 创建特征数据框
        features = pd.DataFrame([[0] * len(MODEL_FEATURES)], columns=MODEL_FEATURES)
        
        # 填充数据
        features.loc[0, 'NIHSS'] = float(data.get('nihss', 0))
        features.loc[0, 'SBP'] = float(data.get('sbp', 0))
        features.loc[0, 'NEUT'] = float(data.get('neut', 0))
        features.loc[0, 'RDW'] = float(data.get('rdw', 0))
        features.loc[0, 'TOAST-LAA_1'] = int(data.get('toast_laa', 0))
        features.loc[0, 'IAS_1'] = int(data.get('ias', 0))

        # 预测
        risk_prob = float(model.predict_proba(features)[0][1])
        risk_percentage = risk_prob * 100
        
        # 风险评估
        risk_level = "高风险" if risk_percentage >= 29 else "低风险"
        risk_description = "发生早期神经功能恶化的风险较高" if risk_percentage >= 29 else "发生早期神经功能恶化的风险较低"

        return jsonify({
            'success': True,
            'risk_probability': risk_percentage,
            'risk_level': risk_level,
            'risk_description': risk_description
        })

    except Exception as e:
        print(f"预测错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True)
