# Credit Card Fraud Detection

I built this project to learn how to handle a real, extremely imbalanced
classification problem. The model looks at a credit card transaction and
predicts the probability that it is fraud.

## What it does

Given one transaction (Time, Amount, and 28 anonymised features V1–V28),
it outputs a fraud probability between 0 and 1. Transactions at or above
0.5 are flagged as fraud. There is also a small Streamlit demo app
(`app.py`) where you can pick a real test transaction or type in values
by hand.

## Dataset

The data is the well-known "Credit Card Fraud Detection" dataset:
European cardholder transactions from September 2013, collected during a
research collaboration between Worldline and the Machine Learning Group
of ULB (Université Libre de Bruxelles).

- **284,807 transactions, 492 frauds — only 0.1727% fraud**
- 31 columns: Time, V1–V28 (PCA-transformed for privacy, so their real
  meaning is hidden), Amount, and the label Class (0 = legit, 1 = fraud)
- No missing values

I did not have Kaggle login for this, so I downloaded the data from a
public mirror instead — the copy hosted by TensorFlow's tutorial data:

`https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`

I verified the file after downloading: 284,807 rows, 492 frauds, 0
missing values — exactly matching the published dataset. The CSV itself
is **not** in this repo (see `.gitignore`); download it from the link
above into `data/creditcard.csv` to run the code.

## The imbalance problem (the main lesson)

This dataset taught me that **accuracy is meaningless here**. If a model
just predicts "legit" for every transaction, it scores **99.83%
accuracy** — and catches zero frauds. That is why I only judge the
models on the fraud class: precision, recall, F1, and AUC-PR (area under
the precision-recall curve).

How I handled the imbalance:

- **Stratified train/test split** (80/20, seed 42) so both sets keep the
  same tiny fraud share: 227,845 train (394 fraud), 56,962 test (98 fraud).
- **`class_weight='balanced'`** for Logistic Regression and
  **`class_weight='balanced_subsample'`** for Random Forest, so the
  models are penalised much more for missing a fraud than for a false
  alarm.
- Only Time and Amount were scaled (StandardScaler fitted on train);
  V1–V28 are already PCA-transformed, so I left them as they are.

## Models

| Model | Settings |
|---|---|
| Logistic Regression (baseline) | `class_weight='balanced'`, `max_iter=1000`, full training set |
| Random Forest | 100 trees, `max_depth=20`, `min_samples_leaf=2`, `class_weight='balanced_subsample'`, `n_jobs=2`, full training set |

## Results (measured on the real test set, 56,962 transactions)

### Logistic Regression (baseline)

- Accuracy: 0.9755 (do not trust this — see above)
- **Precision (fraud): 0.0610 — Recall (fraud): 0.9184 — F1 (fraud): 0.1144 — AUC-PR: 0.7159**
- Confusion matrix [[TN, FP], [FN, TP]]: [[55,478, 1,386], [8, 90]]
- It caught 90 of 98 frauds, but raised 1,386 false alarms. In a real
  bank that many false alarms would be unusable — this is why the
  baseline alone is not good enough.

### Random Forest (best model)

- Accuracy: 0.9995
- **Precision (fraud): 0.9259 — Recall (fraud): 0.7653 — F1 (fraud): 0.8380 — AUC-PR: 0.8533**
- Confusion matrix [[TN, FP], [FN, TP]]: [[56,858, 6], [23, 75]]
- It caught 75 of 98 frauds with only 6 false alarms. Much better
  precision, but it still misses 23 frauds — see
  `demo_predictions.md` for a real missed case (probability 0.4599,
  just under the threshold). Lowering the threshold would catch more
  frauds at the cost of more false alarms; that trade-off is a business
  decision, not just a modelling one.

Full output is in `results/eval.log`, and the confusion matrix plot for
the Random Forest is in `results/confusion_matrix.png`.

## How to run

```bash
pip install -r requirements.txt
# 1. Download the data into data/creditcard.csv (link above)
# 2. Train both models
python src/train.py
# 3. Evaluate on the test set
python src/evaluate.py
# 4. Predict one transaction (a real test-set row)
python src/predict.py --model random_forest --row 840
# 5. Demo app (needs streamlit installed first: pip install streamlit)
streamlit run app.py
```

Note: Streamlit was **not** installed in the venv I trained with, so
`app.py` is written but I could not run it here — install streamlit
first to use it.

## What I tried, failed at, and learned

- **What failed: XGBoost.** I tried to install it, but the download
  kept failing on network retries in my environment, so I skipped it
  rather than faking a result. The Random Forest result (F1 0.8380)
  was strong without it.
- **What I skipped on purpose: SMOTE.** Oversampling the fraud class
  to match ~227k legit rows would have created ~227k synthetic fraud
  rows on a 2-CPU machine with limited RAM. `class_weight` handled the
  imbalance well enough, and it keeps the training data 100% real.
- **What I learned:** on imbalanced data, the metric choice matters
  more than the model choice. Logistic Regression looked great on
  recall (91.8%) but its 6% precision made it useless in practice,
  while Random Forest balanced both. Also, a "99.8% accurate" model can
  be completely worthless — always look at the confusion matrix.
- **Environment note:** the shared venv I used
  (`image-classifier-cnn/.venv`, Python 3.12) had no pandas, so I
  installed pandas into it for this project. Training took ~2 seconds
  for Logistic Regression and ~109 seconds for the Random Forest on
  2 CPUs.

## Project structure

```
credit-card-fraud/
├── app.py                 # Streamlit demo (needs: pip install streamlit)
├── demo_predictions.md    # 5 real test transactions, predicted live
├── requirements.txt
├── .gitignore             # ignores data/, models/, test-set pickles
├── src/
│   ├── data_prep.py       # load, verify, stratified split, scaling
│   ├── train.py           # train Logistic Regression + Random Forest
│   ├── evaluate.py        # fraud-class metrics + confusion matrix plot
│   └── predict.py         # fraud probability for one transaction
├── results/
│   ├── eval.log           # real measured metrics
│   └── confusion_matrix.png
└── data/                  # creditcard.csv goes here (not in repo)
```
