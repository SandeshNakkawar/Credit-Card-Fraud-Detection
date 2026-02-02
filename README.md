# Credit Card Fraud Detection System (Production-Oriented ML)

## 📌 Overview
This project implements a **production-aware credit card fraud detection system** using machine learning on **highly imbalanced payment transaction data**.  
Instead of optimizing for raw accuracy, the system focuses on **business-driven decision making**, balancing fraud prevention with customer experience — similar to real-world fintech systems.

---

## 🎯 Problem Statement
Credit card fraud detection presents several real-world challenges:

- Extreme class imbalance (fraud < 0.2%)
- High cost of false negatives (missed fraud)
- High customer friction from false positives
- Need for fast, reliable, real-time decisions
- Requirement for robustness and auditability

A naïve high-accuracy model fails to address these constraints.

---

## 🧠 Solution Approach
- Preserved real-world class imbalance (no full resampling)
- Used **class-weighted learning** to handle skewed data
- Evaluated multiple models and selected **XGBoost**
- Selected the decision threshold using a **cost-based optimization** instead of accuracy
- Exposed the trained model through a **FastAPI inference service**
- Added **input validation and logging** to improve reliability and debuggability

---

## 🗂 Dataset
- **Source:** Kaggle – Credit Card Fraud Detection Dataset  
- **Records:** 284,807 transactions  
- **Fraud Rate:** ~0.17%  
- **Features:**  
  - V1–V28: PCA-transformed numerical features  
  - Amount: Transaction amount  
  - Time: Time since first transaction  

---

## 🧪 Model & Training
- **Algorithm:** XGBoost
- **Imbalance Handling:** Class-weighted learning (`scale_pos_weight`)
- **Evaluation Metrics:**  
  - ROC-AUC  
  - Precision / Recall  
  - Cost-based analysis  

### Threshold Optimization
Instead of using a fixed threshold (e.g., 0.5), the system selects a threshold that minimizes:
Total Cost = (False Positives × Customer Friction Cost) + (False Negatives × Fraud Loss Cost)


This aligns model decisions with real business trade-offs.

---

## 🧩 System Architecture

credit-card-fraud-detection/
│
├── data/
│ └── creditcard.csv
│
├── training/
│ └── train_model.ipynb
│
├── artifacts/
│ ├── model.pkl
│ ├── scaler.pkl
│ └── threshold.json
│
├── app.py
├── requirements.txt
└── README.md


---

## 🚀 Inference API (FastAPI)
The trained model is served via a **FastAPI** application.

### Endpoint

---

## POST /predict

## Input Format

{
  "features": [30 numerical values]
}

## Output Format
{
  "fraud_probability": 0.0001,
  "is_fraud": 0,
  "threshold": 0.5831
}

---

## Reliability & Monitoring

To make the system robust and production-aware:

1) Feature-aware input validation

   PCA features validated separately

   Amount and time validated using business constraints

2) Inference-time logging

   Request received

   Validation failures

   Prediction outcomes

3) Prevents crashes from invalid or out-of-distribution inputs

4) Enables basic monitoring, auditability, and debugging

---

🧠 Key Design Decisions

❌ Did not optimize for accuracy
❌ Did not fully balance the dataset
❌ Did not retrain during inference

✅ Used cost-based threshold selection
✅ Preserved real-world data distribution
✅ Separated training from inference
✅ Added defensive checks and logging

---

## 📈 Real-World Relevance

This system mirrors how fraud detection is handled in fintech and payment platforms:

  1) Fast, low-latency transaction scoring

  2) Business-driven decision thresholds

  3) Robust APIs instead of notebooks

  4) Monitoring and failure handling

---

## 🔮 Future Improvements

  1) Cost tuning based on business policy

  2) Model monitoring and data drift detection

  3) Rule + ML hybrid fraud detection

  4) Analyst dashboard for reviewing flagged transactions

--- 

### 🏁 Summary

This project demonstrates not just machine learning, but how ML systems are designed, deployed, and protected in real-world payment environments, with a focus on reliability, cost-awareness, and explainability.


