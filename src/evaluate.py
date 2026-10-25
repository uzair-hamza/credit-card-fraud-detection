"""Evaluate saved models on the held-out test set.

Reports, for the FRAUD class (label 1): precision, recall, F1 and AUC-PR
(average precision). Accuracy is printed too, but it is NOT the metric
that matters here: predicting "all legit" already scores ~99.83%.
Also writes results/eval.log and results/confusion_matrix.png
(for the best model by fraud F1).
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_cm(cm, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit (0)", "Fraud (1)"])
    ax.set_yticklabels(["Legit (0)", "Fraud (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    X_test = __import__("pandas").read_pickle(
        os.path.join(BASE, "results", "X_test.pkl"))
    y_test = np.load(os.path.join(BASE, "results", "y_test.npy"))
    lines = []
    lines.append(f"Test set: {len(y_test):,} transactions, "
                 f"{int((y_test == 1).sum())} fraud "
                 f"({100 * y_test.mean():.4f}%)")
    # The "all legit" baseline, to show why accuracy is meaningless here.
    lines.append(f"All-legit baseline accuracy: "
                 f"{accuracy_score(y_test, np.zeros_like(y_test)):.4f} "
                 f"(it catches 0 frauds)")

    best_name, best_f1, best_cm = None, -1, None
    for name in ["logistic_regression", "random_forest"]:
        with open(os.path.join(BASE, "models", f"{name}.pkl"), "rb") as f:
            model = pickle.load(f)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)
        lines.append("")
        lines.append(f"== {name} ==")
        lines.append(f"Accuracy:  {accuracy_score(y_test, pred):.4f}")
        lines.append(f"Precision (fraud): "
                     f"{precision_score(y_test, pred, zero_division=0):.4f}")
        lines.append(f"Recall (fraud):    "
                     f"{recall_score(y_test, pred, zero_division=0):.4f}")
        lines.append(f"F1 (fraud):        {f1:.4f}")
        lines.append(f"AUC-PR (fraud):    "
                     f"{average_precision_score(y_test, proba):.4f}")
        lines.append("Confusion matrix [[TN, FP], [FN, TP]]:")
        lines.append(str(cm.tolist()))
        lines.append(classification_report(y_test, pred, zero_division=0))
        if f1 > best_f1:
            best_name, best_f1, best_cm = name, f1, cm

    plot_cm(best_cm, os.path.join(BASE, "results", "confusion_matrix.png"))
    lines.append(f"Confusion matrix plot: best model by fraud F1 "
                 f"({best_name}) -> results/confusion_matrix.png")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(BASE, "results", "eval.log"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
