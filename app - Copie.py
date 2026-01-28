import json
import joblib
import pandas as pd
import streamlit as st
from preprocessing import build_features

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_xgb_model.pkl")
    with open("feature_columns.json", "r") as f:
        feature_cols = json.load(f)
    return model, feature_cols

def align_features(df, feature_cols):
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]

st.title("Fraud Detection – Demo Analyste")

model, feature_cols = load_artifacts()

threshold = st.slider("Seuil de fraude", 0.05, 0.95, 0.70, 0.01)

tab1, tab2 = st.tabs(["Option A – CSV", "Option B – Formulaire"])

# -------- OPTION A --------
with tab1:
    file = st.file_uploader("Uploader un fichier CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        X = build_features(df)
        X = align_features(X, feature_cols)

        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)

        out = df.copy()
        out["fraud_proba"] = proba
        out["fraud_pred"] = pred

        st.dataframe(out.head(50))

        st.download_button(
            "Télécharger les résultats",
            out.to_csv(index=False),
            "fraud_predictions.csv",
        )

# -------- OPTION B --------
with tab2:
    amount = st.number_input("amount", value=100.0)
    country = st.text_input("country", "FR")
    bin_country = st.text_input("bin_country", "FR")
    channel = st.text_input("channel", "web")
    merchant_category = st.text_input("merchant_category", "electronics")
    promo_used = st.selectbox("promo_used", [0, 1])
    avs_match = st.selectbox("avs_match", [0, 1])
    cvv_result = st.selectbox("cvv_result", [0, 1])
    three_ds_flag = st.selectbox("three_ds_flag", [0, 1])
    shipping_distance_km = st.number_input("shipping_distance_km", value=10.0)

    if st.button("Prédire"):
        row = pd.DataFrame([{
            "transaction_id": 0,
            "user_id": 0,
            "account_age_days": 100,
            "total_transactions_user": 10,
            "avg_amount_user": 50,
            "amount": amount,
            "country": country,
            "bin_country": bin_country,
            "channel": channel,
            "merchant_category": merchant_category,
            "promo_used": promo_used,
            "avs_match": avs_match,
            "cvv_result": cvv_result,
            "three_ds_flag": three_ds_flag,
            "shipping_distance_km": shipping_distance_km,
            "transaction_time": pd.Timestamp.now()
        }])

        X = build_features(row)
        X = align_features(X, feature_cols)

        proba = model.predict_proba(X)[0, 1]

        st.metric("Probabilité de fraude", f"{proba:.2%}")
        st.metric("Décision", "FRAUDE" if proba >= threshold else "OK")

        # Feature importance XGBoost
        fi = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).head(20)

        st.subheader("Top features importantes")
        st.dataframe(fi)
