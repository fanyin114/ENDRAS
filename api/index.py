from flask import Flask, request, jsonify, render_template
import os
from pathlib import Path
import joblib
import logging
import pandas as pd

app = Flask(__name__)

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 获取当前文件的目录
current_dir = Path(__file__).parent.parent
model_path = current_dir / 'models' / 'XGBOOST_model1113.pkl'
template_path = current_dir / 'templates'

app = Flask(__name__, template_folder=str(template_path))

try:
    model = joblib.load(str(model_path))
    logger.info(f"模型成功加载，路径: {model_path}")
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    model = None

MODEL_FEATURES = ['NIHSS', 'SBP', 'NEUT', 'RDW', 'TOAST-LAA_1', 'IAS_1']

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"渲染模板失败: {str(e)}")
        return str(e), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'success': False, 'error': '模型未正确加载'}), 500

    try:
        data = request.get_json()
        logger.debug(f"接收到的数据: {data}")

        features = pd.DataFrame(columns=MODEL_FEATURES)
        features.loc[0] = [0] * len(MODEL_FEATURES)

        # 填充基本特征
        features.loc[0, 'NIHSS'] = float(data.get('nihss', 0))
        features.loc[0, 'SBP'] = float(data.get('sbp', 0))
        features.loc[0, 'NEUT'] = float(data.get('neut', 0))
        features.loc[0, 'RDW'] = float(data.get('rdw', 0))
        features.loc[0, 'TOAST-LAA_1'] = int(data.get('toast_laa', 0))
        features.loc[0, 'IAS_1'] = int(data.get('ias', 0))

        logger.debug(f"处理后的特征数据: {features}")

        risk_prob = float(model.predict_proba(features)[0][1])
        risk_percentage = risk_prob * 100

        # 确定风险等级
        if risk_percentage < 29:
            risk_level = "低风险"
            risk_description = "发生早期神经功能恶化的风险较低"
        else:
            risk_level = "高风险"
            risk_description = "发生早期神经功能恶化的风险较高"

        response = {
            'success': True,
            'risk_probability': risk_percentage,
            'risk_level': risk_level,
            'risk_description': risk_description
        }
        
        logger.debug(f"返回结果: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"预测过程出错: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
