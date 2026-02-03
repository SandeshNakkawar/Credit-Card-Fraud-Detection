from fastapi import FastAPI
import numpy as np
import joblib
import json
import os
import logging

# ==============================
# LOGGING
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("fraud-api")

# ==============================
# APP INIT
# ==============================
app = FastAPI(title="Credit Card Fraud Detection API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# ==============================
# LOAD ARTIFACTS
# ==============================
model = joblib.load(os.path.join(ARTIFACTS_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

with open(os.path.join(ARTIFACTS_DIR, "threshold.json")) as f:
    THRESHOLD = json.load(f)["threshold"]

with open(os.path.join(ARTIFACTS_DIR, "baseline_stats.json")) as f:
    BASELINE = json.load(f)

FEATURE_NAMES = list(BASELINE.keys())

# ==============================
# DRIFT CHECK FUNCTION
# ==============================
def check_feature_drift(feature_name, value, z_threshold=3.0):
    mean = BASELINE[feature_name]["mean"]
    std = BASELINE[feature_name]["std"]

    if std == 0:
        return False, 0.0

    z_score = abs((value - mean) / std)
    return z_score > z_threshold, z_score


# ==============================
# PREDICT ENDPOINT
# ==============================
@app.post("/predict")
def predict(transaction: dict):
    try:
        logger.info("Request received")

        features = transaction.get("features")
        if features is None:
            return {"error": "Missing features"}

        if len(features) != 30:
            return {"error": "Expected exactly 30 features"}

        features = np.array(features, dtype=float).reshape(1, -1)

        # ------------------------------
        # INPUT VALIDATION
        # ------------------------------
        pca_features = features[:, :28]
        amount = features[:, 28]
        time = features[:, 29]

        if np.any(np.abs(pca_features) > 50):
            return {"error": "PCA features out of expected range"}

        if amount < 0 or amount > 100000:
            return {"error": "Transaction amount out of expected range"}

        if time < 0:
            return {"error": "Invalid transaction time"}

        # ------------------------------
        # SCALE
        # ------------------------------
        features_scaled = scaler.transform(features)

        # ------------------------------
        # DRIFT DETECTION
        # ------------------------------
        drift_detected = False
        drift_summary = {}

        for i, fname in enumerate(FEATURE_NAMES):
            drifted, z = check_feature_drift(fname, features_scaled[0][i])
            drift_summary[fname] = round(z, 2)
            if drifted:
                drift_detected = True

        # ------------------------------
        # PREDICTION
        # ------------------------------
        prob = model.predict_proba(features_scaled)[0][1]
        decision = int(prob >= THRESHOLD)

        logger.info(
            f"prob={prob:.4f} | decision={decision} | drift={drift_detected}"
        )

        return {
            "fraud_probability": round(float(prob), 4),
            "is_fraud": decision,
            "threshold": round(float(THRESHOLD), 4),
            "drift_detected": drift_detected
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {"error": "Internal prediction error"}