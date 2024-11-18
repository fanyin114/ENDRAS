from flask import Flask, request, jsonify, render_template  
import xgboost as xgb  
import numpy as np  
import pandas as pd  
import joblib  
import logging  

# 初始化 Flask 应用  
app = Flask(__name__)  

# 配置日志  
logging.basicConfig(  
    level=logging.DEBUG,  
    format='%(asctime)s - %(levelname)s - %(message)s'  
)  
logger = logging.getLogger(__name__)  

# 加载模型  
try:  
    model = joblib.load('XGBOOST_model1113.pkl')  
    logger.info("模型加载成功")  
except Exception as e:  
    logger.error(f"模型加载失败: {str(e)}")  
    model = None  

# 定义模型特征  
MODEL_FEATURES = ['NIHSS', 'SBP', 'NEUT', 'RDW', 'TOAST-LAA_1', 'IAS_1']  

def get_risk_level(probability):  
    """根据概率返回风险等级和详细信息"""  
    prob_percentage = probability * 100  
    
    if prob_percentage < 29:  
        return "低风险", "发生早期神经功能恶化的风险较低"  
    else:  
        return "高风险", "发生早期神经功能恶化的风险较高"  

def validate_input_data(data):  
    """验证输入数据"""  
    required_fields = ['nihss', 'sbp', 'neut', 'rdw', 'toast_laa', 'ias']  
    for field in required_fields:  
        if field not in data:  
            raise ValueError(f"缺少必要字段: {field}")  
        if data[field] is None:  
            raise ValueError(f"字段不能为空: {field}")  

@app.route('/')  
def home():  
    """渲染主页"""  
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])  
def predict():  
    """处理预测请求"""  
    try:  
        # 检查模型是否加载成功  
        if model is None:  
            raise RuntimeError("模型未成功加载，无法进行预测")  

        # 获取并验证输入数据  
        data = request.get_json()  
        if not data:  
            raise ValueError("未接收到有效的JSON数据")  
        
        logger.debug(f"接收到的数据: {data}")  
        validate_input_data(data)  

        # 创建特征DataFrame  
        features = pd.DataFrame(columns=MODEL_FEATURES)  
        features.loc[0] = [0] * len(MODEL_FEATURES)  

        # 填充基本特征  
        for feature in ['NIHSS', 'SBP', 'NEUT', 'RDW']:  
            features.loc[0, feature] = float(data[feature.lower()])  

        # 处理特殊特征  
        features.loc[0, 'TOAST-LAA_1'] = int(data['toast_laa'])  
        features.loc[0, 'IAS_1'] = int(data['ias'])  

        logger.debug(f"处理后的特征数据:\n{features}")  

        # 使用模型预测  
        risk_prob = float(model.predict_proba(features)[0][1])  
        logger.debug(f"预测概率: {risk_prob}")  

        # 获取风险等级和描述  
        risk_level, risk_description = get_risk_level(risk_prob)  

        # 构建响应  
        response = {  
            'success': True,  
            'risk_probability': risk_prob * 100,  
            'risk_level': risk_level,  
            'risk_description': risk_description  
        }  
        
        logger.debug(f"返回结果: {response}")  
        return jsonify(response)  

    except ValueError as e:  
        logger.error(f"输入数据错误: {str(e)}")  
        return jsonify({  
            'success': False,  
            'error': f"输入数据错误: {str(e)}"  
        }), 400  

    except Exception as e:  
        logger.error(f"预测过程出错: {str(e)}", exc_info=True)  
        return jsonify({  
            'success': False,  
            'error': f"服务器内部错误: {str(e)}"  
        }), 500  

@app.errorhandler(404)  
def page_not_found(e):  
    """处理404错误"""  
    return jsonify({  
        'success': False,  
        'error': '请求的页面不存在'  
    }), 404  

if __name__ == '__main__':  
    app.run(debug=True)
