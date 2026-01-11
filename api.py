from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
import pickle
from io import StringIO

app = FastAPI(title="AI Fraud Detection API")

# ---------------------------
# Load Model & Scaler
# ---------------------------
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------------------
# Feature Engineering Function
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
    return scaler.transform(X)

# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/predict")
async def predict_fraud(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    X_scaled = preprocess(df)

    df["anomaly_score"] = model.decision_function(X_scaled)
    df["fraud_flag"] = model.predict(X_scaled)
    df["fraud_flag"] = df["fraud_flag"].apply(lambda x: 1 if x == -1 else 0)

    return {
        "total_transactions": len(df),
        "fraud_detected": int(df["fraud_flag"].sum()),
        "results": df[[
            "transaction_id",
            "amount",
            "fraud_flag",
            "anomaly_score"
        ]].to_dict(orient="records")
    }
