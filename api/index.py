from flask import Flask, jsonify

app = Flask(name)

@app.route('/')
def home():
return "Hello, World!"

@app.route('/test')
def test():
return jsonify({
'status': 'ok',
'message': 'API is working'
})

if name == 'main':
app.run()
