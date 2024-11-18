from flask import Flask, request, jsonify, render_template  
import joblib  
import pandas as pd  
import os  

app = Flask(__name__)  

# 模型路径  
MODEL_PATH = 'XGBOOST_model1113.pkl'  

# 特征列表  
FEATURES = ['NIHSS', 'SBP', 'NEUT', 'RDW', 'TOAST-LAA_1', 'IAS_1']  

# 加载模型  
try:  
    model = joblib.load(MODEL_PATH)  
except Exception as e:  
    print(f"模型加载错误: {str(e)}")  
    model = None  

@app.route('/')  
def home():  
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])  
def predict():  
    try:  
        if model is None:  
            return jsonify({'error': '模型未加载'}), 500  

        data = request.get_json()  
        if not data:  
            return jsonify({'error': '无效的输入数据'}), 400  

        # 创建特征数据框  
        features = pd.DataFrame([[  
            float(data.get('nihss', 0)),  
            float(data.get('sbp', 0)),  
            float(data.get('neut', 0)),  
            float(data.get('rdw', 0)),  
            int(data.get('toast_laa', 0)),  
            int(data.get('ias', 0))  
        ]], columns=FEATURES)  

        # 预测  
        prob = float(model.predict_proba(features)[0][1])  
        risk_level = "高风险" if prob >= 0.29 else "低风险"  
        description = "发生早期神经功能恶化的风险较高" if prob >= 0.29 else "发生早期神经功能恶化的风险较低"  

        return jsonify({  
            'success': True,  
            'risk_probability': round(prob * 100, 2),  
            'risk_level': risk_level,  
            'risk_description': description  
        })  

    except Exception as e:  
        return jsonify({'error': str(e)}), 500  

if __name__ == '__main__':  
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
