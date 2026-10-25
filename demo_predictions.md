# Demo Predictions

These 5 transactions come from the real held-out test set (56,962
transactions, 98 fraud). They were predicted by actually running the
trained Random Forest model (`src/predict.py --model random_forest
--row <n>`). Time and Amount below are the original, unscaled values
(the model itself works on scaled Time/Amount). V1–V28 are the anonymised
PCA features from the dataset, so they are not listed here — the
prediction used all 30 features.

| # | Test row | Time (s) | Amount | True label | Predicted label | Fraud probability | Outcome |
|---|----------|----------|--------|------------|-----------------|-------------------|---------|
| 1 | 840 | 57007 | 0.01 | fraud (1) | fraud (1) | 0.9598 | Correct — fraud caught |
| 2 | 1146 | 68207 | 1.00 | fraud (1) | fraud (1) | 0.9900 | Correct — fraud caught |
| 3 | 4276 | 43494 | 24.90 | fraud (1) | legit (0) | 0.4599 | Missed — a real fraud the model did not catch |
| 4 | 165 | 128759 | 0.76 | legit (0) | fraud (1) | 0.6399 | False alarm — a legit transaction flagged |
| 5 | 0 | 160760 | 23.00 | legit (0) | legit (0) | 0.0000 | Correct — legit passed |

I picked these five on purpose to show the honest picture: the model
catches most frauds, but it still misses some (row 4276, probability
0.4599, just under the 0.5 threshold) and it sometimes flags a legit
transaction (row 165). A fraud system is judged on exactly these
trade-offs, not on accuracy.
