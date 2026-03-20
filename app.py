from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
else:
    print("Model or vectorizer not found. Please run train_model.py first.")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get text from request
            data = request.get_json()
            news_text = data.get('news_text', '')
            
            if not news_text.strip():
                return jsonify({'error': 'Please enter some news text.'}), 400
            
            # Vectorize the input
            transformed_input = vectorizer.transform([news_text])
            
            # Predict
            prediction = model.predict(transformed_input)
            
            # Calculate confidence (distance from decision boundary)
            # Higher absolute value means more confident
            decision_score = model.decision_function(transformed_input)[0]
            # Normalize decision score to a pseudo-confidence percentage
            # (Note: PAC doesn't give true probabilities, so we use a tanh-like normalization)
            import numpy as np
            confidence = float(np.tanh(abs(decision_score))) * 100
            
            return jsonify({
                'prediction': prediction[0],
                'confidence': round(confidence, 2)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
