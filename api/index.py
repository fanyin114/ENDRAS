
from flask import Flask, render_template, jsonify, request
import os
import joblib
import numpy as np
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建logs目录（如果不存在）
if not os.path.exists('logs'):
    os.makedirs('logs')

# 添加文件处理器
file_handler = logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

app = Flask(__name__,
            template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates')))

# 加载模型
try:
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'end_risk_model.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info("模型加载成功")
    else:
        logger.warning("模型文件不存在，将使用模拟数据")
        model = None
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    model = None

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"渲染主页失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': '服务器内部错误'
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # 记录请求开始
        logger.info("收到预测请求")
        
        # 获取并验证输入数据
        data = request.get_json()
        if not data:
            logger.warning("请求数据为空")
            return jsonify({
                'success': False,
                'error': '无效的请求数据'
            }), 400

        logger.info(f"请求数据: {data}")
        
        # 验证必要字段
        required_fields = ['nihss', 'sbp', 'neut', 'rdw', 'toast_laa', 'ias']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.warning(f"缺少必要字段: {missing_fields}")
            return jsonify({
                'success': False,
                'error': f'缺少必要参数: {", ".join(missing_fields)}'
            }), 400

        # 验证数值范围
        validations = [
            ('nihss', 0, 42, 'NIHSS评分必须在0-42之间'),
            ('sbp', 0, 300, '收缩压必须在0-300之间'),
            ('neut', 0, 20, '中性粒细胞计数必须在0-20之间'),
            ('rdw', 0, 60, '红细胞分布宽度必须在0-60之间')
        ]

        for field, min_val, max_val, error_msg in validations:
            value = float(data[field])
            if not (min_val <= value <= max_val):
                logger.warning(f"数值范围验证失败: {field}={value}")
                return jsonify({
                    'success': False,
                    'error': error_msg
                }), 400

        # 准备模型输入数据
        features = np.array([[
            float(data['nihss']),
            float(data['sbp']),
            float(data['neut']),
            float(data['rdw']),
            int(data['toast_laa']),
            int(data['ias'])
        ]])

        # 使用模型预测或返回模拟数据
        if model is not None:
            logger.info("使用模型进行预测")
            risk_prob = float(model.predict_proba(features)[0][1] * 100)
        else:
            logger.info("使用模拟数据")
            risk_prob = 15.5

        logger.info(f"预测风险概率: {risk_prob}%")

        # 确定风险等级和描述
        if risk_prob >= 29:
            risk_level = "高风险"
            risk_description = "患者发生早期神经功能恶化的风险较高，建议密切监测病情变化，及时进行干预。"
        else:
            risk_level = "低风险"
            risk_description = "患者发生早期神经功能恶化的风险较低，建议按常规进行治疗和监测。"

        response_data = {
            'success': True,
            'risk_probability': risk_prob,
            'risk_level': risk_level,
            'risk_description': risk_description
        }
        
        logger.info(f"返回结果: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"预测过程发生错误: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'服务器内部错误: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


