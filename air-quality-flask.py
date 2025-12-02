from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__, template_folder="template")

# Load trained model
MODEL_PATH = 'models/rf_co_model.joblib'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file missing. Train it first!")

store = joblib.load(MODEL_PATH)
pipeline = store['pipeline']
FEATURES = store['features']

# Home route
@app.route('/')
def index():
    return render_template('index.html', features=FEATURES)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    # Fill missing datetime features
    data.setdefault("hour", 12)
    data.setdefault("dayofweek", 3)
    data.setdefault("month", 6)

    # Convert to DataFrame
    df = pd.DataFrame([data], columns=FEATURES)

    try:
        pred = float(pipeline.predict(df)[0])
        return jsonify({"result": pred})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
