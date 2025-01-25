
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response, predict_class
import json

app = Flask(__name__)
CORS(app)

# Load intents once when the application starts
with open('intents.json', 'r') as f:
    intents = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.json.get('message')
        if not text:
            return jsonify({"error": "No message provided"}), 400
        
        ints = predict_class(text)
        response = get_response(ints, intents)
        
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
