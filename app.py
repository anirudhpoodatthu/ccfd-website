import os
import pickle
import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify
)

# ──────────────────────────────────────────────────────────────────────────────
# APP CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = b'ccfd_secret_key_2024'  # Standardized key

MODEL_PATH = "model.pkl"
USERS = {"demo": {"password": "demo123", "email": "demo@ccfd.com"}}

# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────────────────────────
model_data = None

def load_model():
    """Load the trained model from pickle file."""
    global model_data
    if not os.path.exists(MODEL_PATH):
        print("[WARN] model.pkl not found. Run model_training.py first.")
        return None
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    print(f"[OK] Model loaded: {model_data.get('model_name', 'Unknown')}")
    return model_data

load_model()

# ──────────────────────────────────────────────────────────────────────────────
# ROUTES (Authentication & Pages)
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Landing page."""
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
    
    loaded = model_data is not None
    model_name = model_data.get("model_name", "N/A") if loaded else "Not loaded"
    feature_names = model_data.get("feature_names", []) if loaded else []
    
    return render_template(
        "model.html",
        username=session["username"],
        model_loaded=loaded,
        model_name=model_name,
        feature_names=feature_names,
    )


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("index"))


# ──────────────────────────────────────────────────────────────────────────────
# API ROUTES (Standardized under /api/)
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Single transaction prediction API."""
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if model_data is None:
        return jsonify({"error": "Model not loaded."}), 503

    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Missing features."}), 400

        feature_values = data["features"]
        feature_names = model_data["feature_names"]
        model = model_data["model"]

        # Build feature array in the correct order
        features = np.array([[float(feature_values.get(fn, 0.0)) for fn in feature_names]])

        prediction = int(model.predict(features)[0])
        probability = model.predict_proba(features)[0].tolist()

        fraud_prob = probability[1]
        risk_level = "High" if fraud_prob >= 0.7 else "Medium" if fraud_prob >= 0.3 else "Low"

        return jsonify({
            "prediction": prediction,
            "probability_legitimate": round(probability[0] * 100, 2),
            "probability_fraud":      round(fraud_prob * 100, 2),
            "risk_level": risk_level,
            "label": "Fraudulent" if prediction == 1 else "Legitimate",
        })
    except Exception as e:
        import traceback
        print(f"[ERROR] Predict: {traceback.format_exc()}")
        return jsonify({"error": f"Backend processing error: {str(e)}"}), 500


@app.route("/api/predict-batch", methods=["POST"])
def api_predict_batch():
    """Batch prediction API via CSV upload."""
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if model_data is None:
        return jsonify({"error": "Model not loaded."}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    try:
        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400
            
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
        for i, (p, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "index": i + 1,
                "prediction": p,
                "status": "Fraudulent" if p == 1 else "Legitimate",
                "fraud_probability": round(prob * 100, 2),
                "risk_level": "High" if prob >= 0.7 else "Medium" if prob >= 0.3 else "Low",
            })

        fraud_count = sum(1 for p in predictions if p == 1)
        return jsonify({
            "total": len(predictions),
            "fraud_count": fraud_count,
            "legit_count": len(predictions) - fraud_count,
            "results": results,
        })
    except Exception as e:
        import traceback
        print(f"[ERROR] Batch Predict: {traceback.format_exc()}")
        return jsonify({"error": f"File processing failed: {str(e)}"}), 500


@app.route("/api/model-info")
def model_info():
    """Return model metadata."""
    if model_data is None:
        return jsonify({"loaded": False}), 503

    return jsonify({
        "loaded": True,
        "model_name": model_data.get("model_name", "Unknown"),
        "feature_count": len(model_data.get("feature_names", [])),
        "feature_names": model_data.get("feature_names", []),
    })


@app.route("/api/health")
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "online",
        "model_loaded": model_data is not None,
        "username": session.get("username", "Guest"),
        "version": "1.1.0",
        "timestamp": "2026-04-15"
    })


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Credit Card Fraud Detection - Unified Web Server")
    print("=" * 60)
    print(f"  URL: http://127.0.0.1:5000")
    print("=" * 60 + "\n")

    app.run(debug=True, host="127.0.0.1", port=5000)

