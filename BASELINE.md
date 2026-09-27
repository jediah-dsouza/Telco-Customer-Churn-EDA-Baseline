# Naive Majority-Class Baseline

## Method

`DummyClassifier(strategy="most_frequent")` was used as the naive baseline. Since **No Churn** is the majority class, the classifier predicts `No` for every customer.

## Results

| Metric | Score |
|---|---:|
| Accuracy | 0.7346 (73.46%) |
| Precision — Churn | 0.0000 |
| Recall — Churn | 0.0000 |
| F1 — Churn | 0.0000 |

## Interpretation

The baseline reaches **73.46% accuracy** without learning any relationship between the features and churn. It simply predicts the majority class for every record.

Its churn precision, recall, and F1 are all **0.00** because the classifier never predicts a churner. This demonstrates why accuracy alone is misleading for this dataset: the majority class is large enough to produce a seemingly respectable accuracy while the model completely fails at the minority class.

The majority-class classifier is therefore a useful reference point. A real churn model should be assessed on whether it can identify churners while managing false positives, not merely whether it reproduces the majority class.

## Evaluation scope

Task 2 specifies full-dataset EDA and does not introduce a train/test split. Therefore, this baseline is evaluated on the full dataset as a descriptive reference. Proper train/test evaluation begins with the later modeling tasks.
