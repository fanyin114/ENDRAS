from flask import Flask, request, jsonify, render_template  
import xgboost as xgb  
import numpy as np  
import pandas as pd  
import joblib  
import os  

app = Flask(__name__)  

# 获取当前文件目录  
current_dir = os.path.dirname(os.path.abspath(__file__))  

# 模型路径  
model_path = os.path.join(current_dir, 'XGBOOST_model1113.pkl')  

# 加载模型  
try:  
    model = joblib.load(model_path)  
except Exception as e:  
    print(f"模型加载错误: {str(e)}")  
    model = None  

# 模型特征  
MODEL_FEATURES = ['NIHSS', 'SBP', 'NEUT', 'RDW', 'TOAST-LAA_1', 'IAS_1']  

@app.route('/')  
def home():  
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])  
def predict():  
    try:  
        if model is None:  
            return jsonify({  
                'success': False,  
                'error': '模型未正确加载'  
            }), 500  

        data = request.get_json()  
        
        # 创建特征DataFrame  
        features = pd.DataFrame(columns=MODEL_FEATURES)  
        features.loc[0] = [0] * len(MODEL_FEATURES)  

        # 填充特征  
        features.loc[0, 'NIHSS'] = float(data.get('nihss', 0))  
        features.loc[0, 'SBP'] = float(data.get('sbp', 0))  
        features.loc[0, 'NEUT'] = float(data.get('neut', 0))  
        features.loc[0, 'RDW'] = float(data.get('rdw', 0))  
        features.loc[0, 'TOAST-LAA_1'] = int(data.get('toast_laa', 0))  
        features.loc[0, 'IAS_1'] = int(data.get('ias', 0))  

        # 预测  
        risk_prob = float(model.predict_proba(features)[0][1])  
        
        # 风险等级判断  
        risk_level = "高风险" if risk_prob >= 0.29 else "低风险"  
        risk_description = "发生早期神经功能恶化的风险较高" if risk_prob >= 0.29 else "发生早期神经功能恶化的风险较低"  

        return jsonify({  
            'success': True,  
            'risk_probability': risk_prob * 100,  
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
