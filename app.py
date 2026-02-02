from fastapi import FastAPI
import numpy as np
import joblib
import json
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("fraud-api")

app = FastAPI(title="Credit Card Fraud Detection API")

# Load artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

model = joblib.load(os.path.join(ARTIFACTS_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

with open(os.path.join(ARTIFACTS_DIR, "threshold.json")) as f:
    THRESHOLD = json.load(f)["threshold"]


@app.post("/predict")
def predict(transaction: dict):
    try:
        logger.info("Request received")

        # 1️⃣ Extract features
        features = transaction.get("features")
        if features is None:
            logger.warning("Missing features")
            return {"error": "Missing features"}

        # 2️⃣ Length check
        if len(features) != 30:
            logger.warning(f"Invalid feature length: {len(features)}")
            return {"error": "Expected exactly 30 features"}

        # 3️⃣ Convert to NumPy
        features = np.array(features, dtype=float).reshape(1, -1)

        # =================================================
        # 4️⃣ FEATURE-AWARE INPUT VALIDATION
        # =================================================

        pca_features = features[:, :28]
        amount = features[:, 28]
        time = features[:, 29]

        if np.any(np.abs(pca_features) > 50):
            logger.warning("PCA features out of expected range")
            return {"error": "PCA features out of expected range"}

        if amount < 0 or amount > 100000:
            logger.warning("Transaction amount out of expected range")
            return {"error": "Transaction amount out of expected range"}

        if time < 0:
            logger.warning("Invalid transaction time")
            return {"error": "Invalid transaction time"}

        # =================================================
        # 5️⃣ SCALE + PREDICT
        # =================================================

        features_scaled = scaler.transform(features)
        prob = model.predict_proba(features_scaled)[0][1]
        decision = int(prob >= THRESHOLD)

        logger.info(
            f"Prediction success | prob={prob:.4f} | decision={decision}"
        )

        return {
            "fraud_probability": round(float(prob), 4),
            "is_fraud": decision,
            "threshold": round(float(THRESHOLD), 4)
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {"error": "Internal prediction error"}