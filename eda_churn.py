"""
Task 2 — Statistical Profiling, EDA & Naive Baseline
Dataset: Telco Customer Churn

Place this script and the CSV in the same directory, then run:
    python eda_churn.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

# Known TotalCharges data-type issue.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Blank TotalCharges values occur for zero-tenure customers.
df_eda = df.copy()
df_eda["TotalCharges"] = df_eda["TotalCharges"].fillna(0)

numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# 1. Numerical summary
summary = df_eda[numeric_cols].describe().T
summary["variance"] = df_eda[numeric_cols].var()
summary["std_dev"] = df_eda[numeric_cols].std()
summary["skewness"] = df_eda[numeric_cols].skew()
print(summary)
summary.to_csv("numeric_summary_statistics.csv")

# 2. Target distribution
class_counts = df_eda["Churn"].value_counts()
class_pct = df_eda["Churn"].value_counts(normalize=True) * 100
print("\nClass balance:\n", class_pct)

plt.figure(figsize=(8, 5))
sns.countplot(data=df_eda, x="Churn")
plt.title("Customer Churn Distribution")
plt.tight_layout()
plt.savefig("01_churn_distribution.png", dpi=180)
plt.show()

# 3a. Contract type vs churn
contract_rates = pd.crosstab(
    df_eda["Contract"], df_eda["Churn"], normalize="index"
) * 100
print("\nContract churn rates:\n", contract_rates)

plt.figure(figsize=(8, 5))
sns.barplot(x=contract_rates.index, y=contract_rates["Yes"].values)
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("02_contract_vs_churn.png", dpi=180)
plt.show()

# 3b. Tenure vs churn
tenure_bins = pd.cut(
    df_eda["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0–12", "13–24", "25–48", "49–72"]
)
tenure_churn = (
    df_eda.assign(TenureGroup=tenure_bins)
    .groupby("TenureGroup", observed=False)["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)
print("\nTenure churn rates:\n", tenure_churn)

plt.figure(figsize=(8, 5))
sns.barplot(x=tenure_churn.index.astype(str), y=tenure_churn.values)
plt.title("Churn Rate by Tenure Group")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig("03_tenure_vs_churn.png", dpi=180)
plt.show()

# 3c. Monthly charges vs churn
charge_bins = pd.qcut(df_eda["MonthlyCharges"], q=4, duplicates="drop")
charges_churn = (
    df_eda.assign(MonthlyChargeQuartile=charge_bins)
    .groupby("MonthlyChargeQuartile", observed=False)["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)
print("\nMonthly charge churn rates:\n", charges_churn)

plt.figure(figsize=(9, 5))
sns.barplot(x=charges_churn.index.astype(str), y=charges_churn.values)
plt.title("Churn Rate by Monthly Charges Quartile")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("04_monthly_charges_vs_churn.png", dpi=180)
plt.show()

# 4. Numerical distributions
for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_eda, x=col, kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(f"dist_{col}.png", dpi=180)
    plt.show()

# 5. Correlation heatmap
corr = df_eda[numeric_cols].corr()
print("\nCorrelation matrix:\n", corr)
corr.to_csv("numeric_correlation_matrix.csv")

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap — Numerical Features")
plt.tight_layout()
plt.savefig("05_correlation_heatmap.png", dpi=180)
plt.show()

# 6. Majority-class baseline
X = df_eda.drop(columns=["Churn"])
y = df_eda["Churn"]

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X, y)
pred = dummy.predict(X)

print("\nMajority-class baseline")
print(f"Accuracy:          {accuracy_score(y, pred):.4f}")
print(f"Precision (Churn): {precision_score(y, pred, pos_label='Yes', zero_division=0):.4f}")
print(f"Recall (Churn):    {recall_score(y, pred, pos_label='Yes', zero_division=0):.4f}")
print(f"F1 (Churn):        {f1_score(y, pred, pos_label='Yes', zero_division=0):.4f}")
