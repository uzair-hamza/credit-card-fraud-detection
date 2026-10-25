"""Load and prepare the credit card fraud dataset.

The CSV is NOT shipped in this repo (see .gitignore). Download it from the
public mirror documented in the README and place it at data/creditcard.csv.

Only Time and Amount are scaled (V1-V28 are already PCA-transformed).
The split is stratified so the tiny fraud share is preserved in both sets.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
SCALE_COLS = ["Time", "Amount"]
RANDOM_STATE = 42


def load_data(path="data/creditcard.csv"):
    df = pd.read_csv(path)
    print(f"Rows: {len(df):,}")
    print(f"Fraud: {int((df['Class'] == 1).sum()):,} "
          f"({100 * df['Class'].mean():.4f}% of all transactions)")
    print(f"Legit: {int((df['Class'] == 0).sum()):,}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    return df


def prepare(path="data/creditcard.csv", test_size=0.2):
    df = load_data(path)
    X = df[FEATURES].copy()
    y = df["Class"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
    X_test[SCALE_COLS] = scaler.transform(X_test[SCALE_COLS])
    print(f"Train: {len(X_train):,} (fraud: {int((y_train == 1).sum())})")
    print(f"Test:  {len(X_test):,} (fraud: {int((y_test == 1).sum())})")
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    prepare()
