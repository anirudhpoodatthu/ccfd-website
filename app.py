"""
=============================================================================
CREDIT CARD FRAUD DETECTION – FLASK WEB APPLICATION
=============================================================================
A professional web interface for real-time credit card fraud detection.
Loads the trained ML model and provides an interactive UI for predictions.

Run:  python app.py
=============================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ──────────────────────────────────────────────────────────────────────────────
# APP CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "fraud-detection-secret-key-2026"

MODEL_PATH = "model.pkl"


# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────────────────────────
def load_model():
    """Load the trained model from pickle file."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    return model_data


model_data = load_model()


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Render the main prediction page."""
    model_loaded = model_data is not None
    model_name = model_data.get("model_name", "N/A") if model_loaded else "Not loaded"
    feature_names = model_data.get("feature_names", []) if model_loaded else []
    return render_template(
        "index.html",
        model_loaded=model_loaded,
        model_name=model_name,
        feature_names=feature_names,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept transaction features via JSON POST and return fraud prediction.
    Expects: { "features": { "V1": 0.0, "V2": 0.0, ... } }
    Returns: { "prediction": 0|1, "probability": [p0, p1], "risk_level": "..." }
    """
    if model_data is None:
        return jsonify({"error": "Model not loaded. Run model_training.py first."}), 500

    try:
        data = request.get_json()
        feature_values = data.get("features", {})
        feature_names = model_data["feature_names"]
        model = model_data["model"]

        # Build feature array in the correct order
        features = np.array([[float(feature_values.get(fn, 0.0)) for fn in feature_names]])

        prediction = int(model.predict(features)[0])
        probability = model.predict_proba(features)[0].tolist()

        fraud_prob = probability[1]
        if fraud_prob < 0.3:
            risk_level = "Low"
        elif fraud_prob < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return jsonify({
            "prediction": prediction,
            "probability": probability,
            "fraud_probability_pct": round(fraud_prob * 100, 2),
            "legit_probability_pct": round(probability[0] * 100, 2),
            "risk_level": risk_level,
            "label": "Fraudulent" if prediction == 1 else "Legitimate",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """
    Accept a CSV file upload and return batch predictions.
    """
    if model_data is None:
        return jsonify({"error": "Model not loaded."}), 500

    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "No file uploaded."}), 400

        df = pd.read_csv(file)
        feature_names = model_data["feature_names"]
        model = model_data["model"]

        # Handle Time/Amount → scaled versions if present
        if "scaled_time" not in df.columns and "Time" in df.columns:
            df["scaled_time"] = df["Time"]
        if "scaled_amount" not in df.columns and "Amount" in df.columns:
            df["scaled_amount"] = df["Amount"]

        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {', '.join(missing)}"}), 400

        features = df[feature_names].values
        predictions = model.predict(features).tolist()
        probabilities = model.predict_proba(features)[:, 1].tolist()

        results = []
        for i in range(len(predictions)):
            prob = probabilities[i]
            results.append({
                "index": i + 1,
                "prediction": predictions[i],
                "label": "Fraudulent" if predictions[i] == 1 else "Legitimate",
                "fraud_probability": round(prob * 100, 2),
                "risk_level": "High" if prob >= 0.7 else "Medium" if prob >= 0.3 else "Low",
            })

        fraud_count = sum(1 for p in predictions if p == 1)
        legit_count = len(predictions) - fraud_count

        return jsonify({
            "total": len(predictions),
            "fraud_count": fraud_count,
            "legit_count": legit_count,
            "results": results,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/model-info")
def model_info():
    """Return model metadata."""
    if model_data is None:
        return jsonify({"loaded": False})

    return jsonify({
        "loaded": True,
        "model_name": model_data.get("model_name", "Unknown"),
        "feature_count": len(model_data.get("feature_names", [])),
        "feature_names": model_data.get("feature_names", []),
    })


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Credit Card Fraud Detection - Web Server")
    print("=" * 60)
    if model_data:
        print(f"  [OK] Model loaded: {model_data.get('model_name', 'Unknown')}")
    else:
        print("  [!!] Model not found. Run model_training.py first.")
    print(f"  URL: http://127.0.0.1:5000")
    print("=" * 60 + "\n")

    app.run(debug=True, host="127.0.0.1", port=5000)
