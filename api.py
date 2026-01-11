from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
import pickle
from io import StringIO

# ---------------------------
# Initialize FastAPI
# ---------------------------
app = FastAPI(title="AI Fraud Detection API")

# ---------------------------
# Load Model & Scaler
# ---------------------------
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------------------
# Feature Engineering
# ---------------------------
def preprocess(df):
    df["transaction_time"] = pd.to_datetime(df["transaction_time"])
    df["hour"] = df["transaction_time"].dt.hour
    df["is_night"] = df["hour"].apply(lambda x: 1 if x < 6 or x > 22 else 0)
    df["is_weekend"] = df["transaction_time"].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)

    df["log_amount"] = np.log1p(df["amount"])
    dept_mean = df.groupby("department_id")["amount"].transform("mean")
    dept_std = df.groupby("department_id")["amount"].transform("std")
    df["amount_zscore_dept"] = (df["amount"] - dept_mean) / dept_std
    df["amount_vs_dept_mean"] = df["amount"] / dept_mean

    df["vendor_txn_count"] = df.groupby("vendor_id")["vendor_id"].transform("count")
    df["vendor_avg_amount"] = df.groupby("vendor_id")["amount"].transform("mean")
    df["vendor_amount_ratio"] = df["amount"] / df["vendor_avg_amount"]

    features = [
        "log_amount","amount_zscore_dept","amount_vs_dept_mean",
        "hour","is_night","is_weekend",
        "vendor_txn_count","vendor_avg_amount","vendor_amount_ratio"
    ]

    X = df[features].replace([np.inf, -np.inf], 0).fillna(0)
    return X, scaler.transform(X)

# ---------------------------
# Explainable AI
# ---------------------------
def explain(row):
    reasons = []
    if row["amount_zscore_dept"] > 3:
        reasons.append("Unusually high amount for department")
    if row["vendor_amount_ratio"] > 3:
        reasons.append("Vendor amount spike")
    if row["is_night"] == 1:
        reasons.append("Night-time transaction")
    if row["is_weekend"] == 1:
        reasons.append("Weekend transaction")
    if row["vendor_txn_count"] > 50:
        reasons.append("High frequency vendor")
    return ", ".join(reasons) if reasons else "Normal pattern"

# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/predict")
async def predict_fraud(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    # Check required columns
    required_cols = ["transaction_id","department_id","vendor_id","amount","transaction_time"]
    for col in required_cols:
        if col not in df.columns:
            return {"error": f"Missing column: {col}"}

    # Preprocess
    X_raw, X_scaled = preprocess(df)

    # Model prediction
    df["anomaly_score"] = model.decision_function(X_scaled)
    df["fraud_flag"] = model.predict(X_scaled)
    df["fraud_flag"] = df["fraud_flag"].apply(lambda x: 1 if x == -1 else 0)

    # Risk scoring
    df["risk_score"] = ((df["anomaly_score"].max() - df["anomaly_score"]) /
                        (df["anomaly_score"].max() - df["anomaly_score"].min())) * 100
    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[0,30,70,100],
        labels=["Low","Medium","High"]
    )

    # Explanations
    df["explanation"] = df.apply(explain, axis=1)

    # Summary metrics
    total = len(df)
    fraud_count = int(df["fraud_flag"].sum())

    # Return enriched results
    return {
        "total_transactions": total,
        "fraud_detected": fraud_count,
        "results": df[[
            "transaction_id",
            "department_id",
            "vendor_id",
            "amount",
            "fraud_flag",
            "anomaly_score",
            "risk_score",
            "risk_level",
            "explanation"
        ]].to_dict(orient="records")
    }

# ---------------------------
# Optional: Health check
# ---------------------------
@app.get("/")
def home():
    return {"status": "AI Fraud Detection API is running"}

