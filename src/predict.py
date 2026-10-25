"""Predict the fraud probability for ONE transaction.

Usage:
  python src/predict.py --model random_forest --row 0
      -> predicts test-set row 0 (a real transaction from results/X_test.pkl)
  python src/predict.py --model logistic_regression --time 0 --amount 149.62 \
      --v1 -1.36 --v2 -0.07 ... (all V1-V28, defaults 0)
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model(name):
    with open(os.path.join(BASE, "models", f"{name}.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE, "models", "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(BASE, "models", "meta.json")) as f:
        features = json.load(f)["features"]
    return model, scaler, features


def predict_one(model, scaler, features, values):
    row = pd.DataFrame([values], columns=features)
    row[["Time", "Amount"]] = scaler.transform(row[["Time", "Amount"]])
    proba = float(model.predict_proba(row)[0, 1])
    return int(proba >= 0.5), proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="random_forest")
    ap.add_argument("--row", type=int, default=None,
                    help="index into the saved test set")
    ap.add_argument("--time", type=float, default=0.0)
    ap.add_argument("--amount", type=float, default=0.0)
    for i in range(1, 29):
        ap.add_argument(f"--v{i}", type=float, default=0.0)
    args = ap.parse_args()

    model, scaler, features = load_model(args.model)
    if args.row is not None:
        X_test = pd.read_pickle(os.path.join(BASE, "results", "X_test.pkl"))
        y_test = np.load(os.path.join(BASE, "results", "y_test.npy"))
        row = X_test.iloc[args.row]
        # X_test is already scaled; predict directly.
        proba = float(model.predict_proba(row.to_frame().T)[0, 1])
        label = int(proba >= 0.5)
        print(f"Test row {args.row}: true label = {int(y_test[args.row])} "
              f"({'fraud' if y_test[args.row] == 1 else 'legit'})")
        print(f"Predicted label = {label}, fraud probability = {proba:.4f}")
        return
    values = {"Time": args.time, "Amount": args.amount}
    values.update({f"V{i}": getattr(args, f"v{i}") for i in range(1, 29)})
    label, proba = predict_one(model, scaler, features, values)
    print(f"Predicted label = {label} "
          f"({'fraud' if label == 1 else 'legit'}), "
          f"fraud probability = {proba:.4f}")


if __name__ == "__main__":
    main()
