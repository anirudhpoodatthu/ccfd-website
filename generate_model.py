"""
Generates a model.pkl trained on synthetic data that mirrors the
real creditcard.csv feature structure (V1-V28, scaled_time, scaled_amount).
Use this for deployment when the real dataset is unavailable.
"""
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

feature_names = [f"V{i}" for i in range(1, 29)] + ["scaled_time", "scaled_amount"]
n_features = len(feature_names)  # 30

# ── Generate synthetic training data ────────────────────────────────────────
# Legitimate transactions: tight normal distribution
n_legit = 9000
X_legit = np.random.randn(n_legit, n_features) * 0.8
y_legit = np.zeros(n_legit)

# Fraudulent transactions: shifted distribution on fraud-correlated features
n_fraud = 1000
X_fraud = np.random.randn(n_fraud, n_features) * 1.5
# Shift known fraud-correlated PCA components
for col, shift in [(13, -4.5), (11, -3.8), (9, -3.2), (3, -2.5), (16, 2.8)]:
    X_fraud[:, col] += shift
y_fraud = np.ones(n_fraud)

X = np.vstack([X_legit, X_fraud])
y = np.concatenate([y_legit, y_fraud])

# Shuffle
idx = np.random.permutation(len(y))
X, y = X[idx], y[idx]

# ── Train ────────────────────────────────────────────────────────────────────
print("[INFO] Training Random Forest on synthetic data...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1,
)
model.fit(X, y)
print("[INFO] Training complete.")

# ── Save ─────────────────────────────────────────────────────────────────────
payload = {
    "model":         model,
    "model_name":    "Random Forest (synthetic)",
    "feature_names": feature_names,
}
with open("model.pkl", "wb") as f:
    pickle.dump(payload, f)

print("[SAVED] model.pkl")
