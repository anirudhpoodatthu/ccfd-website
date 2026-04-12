"""
Flask backend — Credit Card Fraud Detection Website
Run: python server.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify
)

app = Flask(__name__)
app.secret_key = b'ccfd_secret_key_2024'

# ── Simple in-memory user store (no DB needed) ───────────────────────────────
USERS = {"demo": {"password": "demo123", "email": "demo@ccfd.com"}}

# ── Load model once ──────────────────────────────────────────────────────────
MODEL_DATA = None

def load_model():
    global MODEL_DATA
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            MODEL_DATA = pickle.load(f)
        print(f"[INFO] Model loaded: {MODEL_DATA.get('model_name', 'Unknown')}")
    else:
        print("[WARN] model.pkl not found. Run model_training.py first.")

load_model()

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        # Quick demo login
        if username == "dummy" and password == "dummy":
            session["username"] = "Demo User"
            return redirect(url_for("dashboard"))

        user = USERS.get(username)
        if user and user["password"] == password:
            session["username"] = username
            return redirect(url_for("dashboard"))

        session["error"] = "Invalid username or password."
        return redirect(url_for("login"))

    error = session.pop("error", None)
    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        email    = request.form.get("email", "").strip()

        if username in USERS:
            session["reg_error"] = "Username already exists."
            return redirect(url_for("login"))

        USERS[username] = {"password": password, "email": email}
        session["username"] = username
        return redirect(url_for("dashboard"))

    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["username"])


@app.route("/model")
def model_page():
    if "username" not in session:
        return redirect(url_for("login"))
    model_name = MODEL_DATA.get("model_name", "Unknown") if MODEL_DATA else None
    feature_names = MODEL_DATA.get("feature_names", []) if MODEL_DATA else []
    return render_template(
        "model.html",
        username=session["username"],
        model_name=model_name,
        feature_names=feature_names,
        model_loaded=MODEL_DATA is not None,
    )


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("index"))


# ── API: single prediction ────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if MODEL_DATA is None:
        return jsonify({"error": "Model not loaded. Run model_training.py first."}), 503

    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing features."}), 400

    try:
        feature_names = MODEL_DATA["feature_names"]
        features = np.array([[data["features"].get(f, 0.0) for f in feature_names]])
        model = MODEL_DATA["model"]
        prediction  = int(model.predict(features)[0])
        probability = model.predict_proba(features)[0].tolist()
        risk = "Low" if probability[1] < 0.3 else "Medium" if probability[1] < 0.7 else "High"
        return jsonify({
            "prediction": prediction,
            "probability_legitimate": round(probability[0] * 100, 2),
            "probability_fraud":      round(probability[1] * 100, 2),
            "risk_level": risk,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: batch CSV prediction ─────────────────────────────────────────────────
@app.route("/api/predict-batch", methods=["POST"])
def api_predict_batch():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if MODEL_DATA is None:
        return jsonify({"error": "Model not loaded."}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    try:
        df = pd.read_csv(request.files["file"])
        feature_names = MODEL_DATA["feature_names"]

        if "scaled_time" not in df.columns and "Time" in df.columns:
            df["scaled_time"] = df["Time"]
        if "scaled_amount" not in df.columns and "Amount" in df.columns:
            df["scaled_amount"] = df["Amount"]

        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {', '.join(missing)}"}), 400

        model        = MODEL_DATA["model"]
        features     = df[feature_names].values
        predictions  = model.predict(features).tolist()
        probabilities = model.predict_proba(features)[:, 1].tolist()

        results = [
            {
                "index": i + 1,
                "status": "Fraudulent" if p == 1 else "Legitimate",
                "fraud_probability": round(prob * 100, 2),
            }
            for i, (p, prob) in enumerate(zip(predictions, probabilities))
        ]

        fraud_count = sum(1 for p in predictions if p == 1)
        return jsonify({
            "total":       len(predictions),
            "fraud_count": fraud_count,
            "legit_count": len(predictions) - fraud_count,
            "results":     results,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
