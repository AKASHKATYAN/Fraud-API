from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
from io import StringIO

# ---------------------------
# App Initialization
# ---------------------------
app = FastAPI(title="AI Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for hackathon, restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load Model & Scaler
# ---------------------------
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------------------
# Utility: Clean DF for JSON
# ---------------------------
def clean_df_for_json(df: pd.DataFrame):
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].astype(float)
    return df

# ---------------------------
# Feature Engineering
# ---------------------------
def preprocess(df):
    df["transaction_time"] = pd.to_datetime(
        df["transaction_time"],
        errors="coerce"
    )

    df["hour"] = df["transaction_time"].dt.hour.fillna(0)
    df["is_night"] = df["hour"].apply(lambda x: 1 if x < 6 or x > 22 else 0)
    df["is_weekend"] = df["transaction_time"].dt.weekday.fillna(0).apply(
        lambda x: 1 if x >= 5 else 0
    )

    df["log_amount"] = np.log1p(df["amount"].fillna(0))

    dept_mean = df.groupby("department_id")["amount"].transform("mean")
    dept_std = df.groupby("department_id")["amount"].transform("std")

    df["amount_zscore_dept"] = (df["amount"] - dept_mean) / dept_std
    df["amount_vs_dept_mean"] = df["amount"] / dept_mean

    df["vendor_txn_count"] = df.groupby("vendor_id")["vendor_id"].transform("count")
    df["vendor_avg_amount"] = df.groupby("vendor_id")["amount"].transform("mean")
    df["vendor_amount_ratio"] = df["amount"] / df["vendor_avg_amount"]

    features = [
        "log_amount",
        "amount_zscore_dept",
        "amount_vs_dept_mean",
        "hour",
        "is_night",
        "is_weekend",
        "vendor_txn_count",
        "vendor_avg_amount",
        "vendor_amount_ratio",
    ]

    X = df[features].replace([np.inf, -np.inf], 0).fillna(0)
    return scaler.transform(X)

# ---------------------------
# Health Check
# ---------------------------
@app.get("/")
def root():
    return {"status": "Fraud Detection API is running 🚀"}

# ---------------------------
# Prediction Logic
# ---------------------------
def run_prediction(df):
    X_scaled = preprocess(df)

    df["anomaly_score"] = model.decision_function(X_scaled)
    df["fraud_flag"] = model.predict(X_scaled)
    df["fraud_flag"] = df["fraud_flag"].apply(lambda x: 1 if x == -1 else 0)

    # Risk score & explanation (UI friendly)
    df["risk_score"] = (-df["anomaly_score"]).clip(0, None)
    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-1, 0.5, 1.5, 3, 10],
        labels=["Low", "Medium", "High", "Critical"]
    )

    df["explanation"] = df.apply(
        lambda r: "Unusual transaction pattern detected"
        if r["fraud_flag"] == 1 else "Normal transaction",
        axis=1
    )

    return clean_df_for_json(df)

# ---------------------------
# /predict Endpoint
# ---------------------------
@app.post("/predict")
async def predict_fraud(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    df = run_prediction(df)

    return {
        "total_transactions": len(df),
        "fraud_detected": int(df["fraud_flag"].sum()),
        "results": df[
            [
                "transaction_id",
                "amount",
                "fraud_flag",
                "risk_score",
                "risk_level",
                "explanation",
            ]
        ].to_dict(orient="records"),
    }

