"""Train the fraud models and save them to models/.

Models:
  - Logistic Regression baseline, class_weight='balanced' (full training set).
  - Random Forest, class_weight='balanced_subsample', CPU-friendly settings
    (100 trees, max_depth=20, n_jobs=2) on the full training set.

XGBoost and SMOTE were deliberately NOT used: XGBoost did not install in
the available environment and SMOTE on ~228k rows would have created
~228k synthetic fraud rows, which is too heavy for this 2-CPU machine.
class_weight handles the imbalance instead, and it is enough here.
"""
import json
import os
import pickle
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from data_prep import FEATURES, prepare

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    X_train, X_test, y_train, y_test, scaler = prepare(
        os.path.join(BASE, "data", "creditcard.csv")
    )
    os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "results"), exist_ok=True)

    models = {}

    t0 = time.time()
    print("Training Logistic Regression (class_weight=balanced)...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                            random_state=42)
    lr.fit(X_train, y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    models["logistic_regression"] = lr

    t0 = time.time()
    print("Training Random Forest (100 trees, max_depth=20, n_jobs=2)...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=2, random_state=42,
    )
    rf.fit(X_train, y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    models["random_forest"] = rf

    with open(os.path.join(BASE, "models", "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    for name, model in models.items():
        with open(os.path.join(BASE, "models", f"{name}.pkl"), "wb") as f:
            pickle.dump(model, f)

    # Save the test set so evaluate.py / predict.py demos use the real data.
    X_test.to_pickle(os.path.join(BASE, "results", "X_test.pkl"))
    import numpy as np
    np.save(os.path.join(BASE, "results", "y_test.npy"), y_test)
    meta = {"features": FEATURES, "models": list(models.keys())}
    with open(os.path.join(BASE, "models", "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved models to models/")


if __name__ == "__main__":
    main()
