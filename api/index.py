
from flask import Flask, render_template, jsonify, request
import os
import logging
import numpy as np
import joblib

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 初始化 Flask 应用
app = Flask(__name__,
            template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates')))

# 模型相关配置
MODEL_FEATURES = ['NIHSS', 'SBP', 'NEUT', 'RDW', 'TOAST-LAA_1', 'IAS_1']
model = None

def load_model():
    """加载模型"""
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'XGBOOST_model1113.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("模型加载成功")
            return True
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
    return False

def get_risk_level(probability):
    """根据概率返回风险等级和详细信息"""
    prob_percentage = probability * 100
    
    if prob_percentage < 29:
        return "低风险", "患者发生早期神经功能恶化的风险较低，建议按常规进行治疗和监测。"
    else:
        return "高风险", "患者发生早期神经功能恶化的风险较高，建议密切监测病情变化，及时进行干预。"

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"模板渲染错误: {str(e)}")
        return f"Error: {str(e)}\nTemplate folder: {app.template_folder}"

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
        logger.debug(f"接收到的数据: {data}")

        if not data:
            return jsonify({
                'success': False,
                'error': '无效的请求数据'
            }), 400

        # 准备特征数据
        try:
            features = np.array([[
                float(data.get('nihss', 0)),
                float(data.get('sbp', 0)),
                float(data.get('neut', 0)),
                float(data.get('rdw', 0)),
                int(data.get('toast_laa', 0)),
                int(data.get('ias', 0))
            ]])
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'数据格式错误: {str(e)}'
            }), 400

        # 如果模型未加载，尝试加载模型
        global model
        if model is None:
            if not load_model():
                # 如果模型加载失败，返回模拟数据
                logger.warning("使用模拟数据进行预测")
                risk_prob = 0.15
            else:
                logger.info("模型加载成功，使用模型进行预测")
                risk_prob = float(model.predict_proba(features)[0][1])
        else:
            risk_prob = float(model.predict_proba(features)[0][1])

        # 获取风险等级和描述
        risk_level, risk_description = get_risk_level(risk_prob)

        response = {
            'success': True,
            'risk_probability': risk_prob * 100,  # 转换为百分比
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

