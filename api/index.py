from flask import Flask, render_template, jsonify, request
import os

app = Flask(__name__,
            template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates')))

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        # 添加详细的错误信息以便调试
        return f"Error: {str(e)}\nTemplate folder: {app.template_folder}\nTemplates: {os.listdir(app.template_folder)}"

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
        # 模拟预测结果
        return jsonify({
            'success': True,
            'risk_probability': 15,
            'risk_level': '低风险',
            'risk_description': '发生早期神经功能恶化的风险较低'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
