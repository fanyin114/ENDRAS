from flask import Flask, request, jsonify, render_template
import xgboost as xgb
import numpy as np
import pandas as pd
import joblib
import logging
import os

app = Flask(__name__, template_folder='../templates')  # 更新模板文件夹路径

# 设置日志
logging.basicConfig(level=logging.DEBUG)

# 更新模型加载路径
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'XGBOOST_model1113.pkl')

# 加载模型
try:
    model = joblib.load(MODEL_PATH)
    logging.info("模型加载成功")
except Exception as e:
    logging.error(f"模型加载失败: {str(e)}")

# 模型特征定义
MODEL_FEATURES = ['NIHSS', 'SBP', 'NEUT', 'RDW',  'TOAST-LAA_1', 'IAS_1']

def get_risk_level(probability):
    prob_percentage = probability * 100
    if prob_percentage < 29:
        return "低风险", "发生早期神经功能恶化的风险较低"
    else:
        return "高风险", "发生早期神经功能恶化的风险较高"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logging.debug(f"接收到的数据: {data}")

        # 创建特征DataFrame
        features = pd.DataFrame(columns=MODEL_FEATURES)
        features.loc[0] = [0] * len(MODEL_FEATURES)

        # 填充基本特征
        basic_features = ['NIHSS', 'SBP', 'NEUT', 'RDW']
        for feature in basic_features:
            value = float(data.get(feature.lower(), 0))
            features.loc[0, feature] = value

        # 处理特殊特征
        features.loc[0, 'TOAST-LAA_1'] = int(data.get('toast_laa', 0))
        features.loc[0, 'IAS_1'] = int(data.get('ias', 0))

        logging.debug(f"处理后的特征数据: {features}")

        # 使用模型预测
        risk_prob = float(model.predict_proba(features)[0][1])
        logging.debug(f"预测概率: {risk_prob}")

        # 获取风险等级和描述
        risk_percentage = risk_prob * 100
        risk_level, risk_description = get_risk_level(risk_prob)

        response = {
            'success': True,
            'risk_probability': risk_percentage,
            'risk_level': risk_level,
            'risk_description': risk_description
        }
        
        logging.debug(f"返回结果: {response}")
        return jsonify(response)

    except Exception as e:
        logging.error(f"预测过程出错: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
