from flask import Flask, request, jsonify, render_template
import os
from pathlib import Path
import joblib
import logging
import pandas as pd

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 获取当前文件的目录
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / 'templates'
MODEL_PATH = BASE_DIR / 'models' / 'XGBOOST_model1113.pkl'

# 初始化Flask应用
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# 加载模型
try:
    logger.info(f"尝试加载模型，路径: {MODEL_PATH}")
    model = joblib.load(str(MODEL_PATH))
    logger.info("模型加载成功")
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    model = None

# 模型特征
MODEL_FEATURES = ['NIHSS', 'SBP', 'NEUT', 'RDW', 'TOAST-LAA_1', 'IAS_1']

@app.route('/')
def home():
    try:
        logger.info("访问首页")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"渲染首页失败: {str(e)}")
        return str(e), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'success': False, 'error': '模型未正确加载'}), 500

    try:
        data = request.get_json()
        logger.info(f"收到预测请求: {data}")

        # 创建特征DataFrame
        features = pd.DataFrame(columns=MODEL_FEATURES)
        features.loc[0] = [0] * len(MODEL_FEATURES)

        # 填充特征值
        features.loc[0, 'NIHSS'] = float(data.get('nihss', 0))
        features.loc[0, 'SBP'] = float(data.get('sbp', 0))
        features.loc[0, 'NEUT'] = float(data.get('neut', 0))
        features.loc[0, 'RDW'] = float(data.get('rdw', 0))
        features.loc[0, 'TOAST-LAA_1'] = int(data.get('toast_laa', 0))
        features.loc[0, 'IAS_1'] = int(data.get('ias', 0))

        logger.debug(f"处理后的特征数据: {features}")

        # 预测
        risk_prob = float(model.predict_proba(features)[0][1])
        risk_percentage = risk_prob * 100

        # 风险评估
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
        
        logger.info(f"预测结果: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"预测过程出错: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True)
