"""Streamlit demo: fraud probability for one transaction.

NOTE: Streamlit is NOT installed in the project venv used for training.
Install it first:  pip install streamlit
Then run:  streamlit run app.py

You can either pick a real row from the saved test set, or type in
Time / Amount / V1-V28 values by hand.
"""
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@st.cache_resource
def load_all():
    with open(os.path.join(BASE, "models", "random_forest.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE, "models", "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    X_test = pd.read_pickle(os.path.join(BASE, "results", "X_test.pkl"))
    y_test = np.load(os.path.join(BASE, "results", "y_test.npy"))
    return model, scaler, X_test, y_test


st.title("Credit Card Fraud Detection")
st.write("Random Forest model trained on 284,807 real European card "
         "transactions (only 0.17% fraud). Enter a transaction and see "
         "its fraud probability.")

model, scaler, X_test, y_test = load_all()

mode = st.radio("Input mode", ["Pick a real test transaction",
                               "Enter values by hand"])
if mode == "Pick a real test transaction":
    idx = st.number_input("Test-set row", min_value=0,
                          max_value=len(X_test) - 1, value=0, step=1)
    row = X_test.iloc[int(idx)]
    st.write(f"True label of this row: "
             f"{'FRAUD' if y_test[int(idx)] == 1 else 'legit'}")
    proba = float(model.predict_proba(row.to_frame().T)[0, 1])
else:
    time_v = st.number_input("Time (seconds since first transaction)",
                             value=0.0)
    amount = st.number_input("Amount", value=0.0)
    vals = {"Time": time_v, "Amount": amount}
    for i in range(1, 29):
        vals[f"V{i}"] = st.number_input(f"V{i}", value=0.0)
    row = pd.DataFrame([vals], columns=FEATURES)
    row[["Time", "Amount"]] = scaler.transform(row[["Time", "Amount"]])
    proba = float(model.predict_proba(row)[0, 1])

st.metric("Fraud probability", f"{proba:.4f}")
if proba >= 0.5:
    st.error("Predicted: FRAUD - this transaction would be flagged.")
else:
    st.success("Predicted: legit.")
