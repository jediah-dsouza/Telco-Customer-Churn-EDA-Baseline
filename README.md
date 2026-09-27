# Telco Customer Churn — EDA & Classification Baseline

## Overview

This project performs **statistical profiling, exploratory data analysis (EDA), and baseline classification evaluation** on the IBM Telco Customer Churn dataset.

The objective is to understand the characteristics of customers who churn, identify relationships between key features and churn behavior, assess class imbalance, and establish a simple **majority-class baseline** before developing more advanced machine learning models.

> **Key takeaway:** The majority-class baseline achieves **73.46% accuracy while identifying 0% of churned customers**, demonstrating why accuracy alone is not an appropriate metric for evaluating churn prediction models.

---

## Objectives

* Profile numerical variables using descriptive statistics.
* Identify skewed distributions and understand their implications.
* Analyze the distribution of the target variable, `Churn`.
* Investigate relationships between customer attributes and churn.
* Examine correlations among numerical features.
* Quantify class imbalance in the dataset.
* Establish a naive majority-class classification baseline.
* Demonstrate why metrics such as **precision, recall, and F1-score** are important for imbalanced classification problems.

---

## Dataset

The project uses the **IBM Telco Customer Churn** dataset, containing customer demographic information, account details, subscribed services, billing information, and churn status.

**Dataset size:**

* **7,043 customers**
* **21 columns**
* Mixed numerical and categorical features
* Binary target: `Churn`

### Target Distribution

| Churn Status | Percentage |
| ------------ | ---------: |
| No Churn     |     73.46% |
| Churn        |     26.54% |

The target is moderately imbalanced, with approximately one-quarter of customers represented by the churn class.

---

## Exploratory Data Analysis

### 1. Statistical Profiling

Descriptive statistics were calculated for all numerical variables, including:

* Count
* Mean
* Median
* Standard deviation
* Minimum and maximum
* Quartiles
* Skewness

The analysis identified several distributions with noticeable asymmetry.

For example:

* `SeniorCitizen` — skewness: **1.834**
* `TotalCharges` — skewness: **0.963**

Right-skewed variables can have their means influenced by larger observations, making the **median and standard deviation** useful complementary measures when interpreting the data.

---

### 2. Churn Distribution

The target variable shows:

* **73.46%** No Churn
* **26.54%** Churn

This imbalance is important when evaluating future classification models. A model can achieve relatively high accuracy by favoring the majority class while performing poorly at identifying actual churners.

Therefore, **precision, recall, and F1-score** should be considered alongside accuracy.

---

## Key Bivariate Findings

### Contract Type vs Churn

Customers on shorter-term contracts show substantially higher observed churn rates than customers on longer-term contracts.

| Contract       | Churn Rate |
| -------------- | ---------: |
| Month-to-month |     42.71% |
| One year       |     11.27% |
| Two year       |      2.83% |

This indicates a strong association between contract type and churn in this dataset.

---

### Tenure vs Churn

Churn is more prevalent among customers with shorter tenure.

| Tenure Group | Churn Rate |
| ------------ | ---------: |
| 0–12 months  |     47.44% |
| 13–24 months |     28.71% |
| 25–48 months |     20.39% |
| 49–72 months |      9.51% |

The observed pattern suggests that churn rates generally decrease as customer tenure increases.

---

### Monthly Charges vs Churn

Customers in higher monthly-charge ranges generally show higher observed churn rates than customers in the lowest charge range.

| Monthly Charges  | Churn Rate |
| ---------------- | ---------: |
| Lowest quartile  |     11.24% |
| Q2               |     24.58% |
| Q3               |     37.51% |
| Highest quartile |     32.88% |

This relationship is descriptive and does not establish that monthly charges directly cause churn.

---

## Correlation Analysis

A correlation matrix was calculated for the numerical features.

The strongest relationships observed were:

| Feature Pair                      | Correlation |
| --------------------------------- | ----------: |
| `tenure` ↔ `TotalCharges`         |   **0.826** |
| `MonthlyCharges` ↔ `TotalCharges` |   **0.651** |
| `tenure` ↔ `MonthlyCharges`       |   **0.248** |

The strong relationship between `tenure` and `TotalCharges` is expected because total charges accumulate over the customer's tenure.

This relationship should be considered during future feature engineering and model development because these variables contain overlapping information.

---

## Naive Classification Baseline

A `DummyClassifier` using the **most frequent class strategy** was used to establish a baseline.

The model always predicts:

> **No Churn**

### Baseline Performance

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        | **73.46%** |
| Churn Precision |  **0.00%** |
| Churn Recall    |  **0.00%** |
| Churn F1-Score  |  **0.00%** |

Although the baseline achieves **73.46% accuracy**, it completely fails to identify customers who churn.

This provides an important reference point for future machine learning models: a useful churn classifier should demonstrate meaningful improvement over this naive strategy, particularly in **churn recall and F1-score**, rather than simply maximizing accuracy.

---

## Visualizations

The project includes visualizations covering:

* Churn class distribution
* Contract type vs churn
* Tenure vs churn
* Monthly charges vs churn
* Numerical correlation heatmap
* Distributions of selected numerical variables

### Example Outputs

| Churn Distribution          | Contract vs Churn          |
| --------------------------- | -------------------------- |
| `01_churn_distribution.png` | `02_contract_vs_churn.png` |

| Tenure vs Churn          | Monthly Charges vs Churn          |
| ------------------------ | --------------------------------- |
| `03_tenure_vs_churn.png` | `04_monthly_charges_vs_churn.png` |

---

## Project Structure

```text
Telco-Customer-Churn-EDA-Baseline/
│
├── eda_churn.py
│
├── EDA_FINDINGS.md
├── BASELINE.md
├── README.md
│
├── numeric_summary_statistics.csv
├── numeric_correlation_matrix.csv
│
├── 01_churn_distribution.png
├── 02_contract_vs_churn.png
├── 03_tenure_vs_churn.png
├── 04_monthly_charges_vs_churn.png
├── 05_correlation_heatmap.png
│
├── dist_tenure.png
├── dist_MonthlyCharges.png
└── dist_TotalCharges.png
```

---

## Technologies Used

* **Python**
* **Pandas** — data manipulation and statistical profiling
* **Matplotlib** — data visualization
* **Seaborn** — exploratory visualization
* **Scikit-learn** — baseline classification

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Telco-Customer-Churn-EDA-Baseline.git
cd Telco-Customer-Churn-EDA-Baseline
```

### 2. Install dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Add the dataset

Place the Telco Customer Churn CSV file in the project directory:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 4. Run the Python script

```bash
python eda_churn.py
```
Run all cells to reproduce the analysis and visualizations.

---

## Data Quality Considerations

The dataset contains a few characteristics worth noting before modeling:

* `TotalCharges` is stored as text in the raw dataset and requires numeric conversion.
* Blank `TotalCharges` values correspond to customers with very short/zero tenure and require appropriate handling.
* Values such as `"No internet service"` represent valid categorical responses rather than missing values.
* `TotalCharges` has a strong relationship with `tenure` and `MonthlyCharges`, which should be considered when developing future models.

---

## Limitations

This project focuses on **EDA and baseline evaluation**, rather than building a production-ready predictive model.

It does not include:

* Train/test splitting
* Feature engineering for predictive modeling
* Hyperparameter tuning
* Cross-validation
* Advanced classification algorithms
* Model interpretability
* Deployment

These would be natural next steps for a subsequent churn prediction project.

## Project Outcome

This analysis establishes a clear statistical and modeling foundation for the Telco Customer Churn problem.

The EDA identifies meaningful differences in churn rates across **contract type, customer tenure, and monthly charges**, while the naive baseline demonstrates that **high accuracy can be misleading when the target classes are imbalanced**.

The resulting baseline provides a measurable benchmark against which future machine learning models can be evaluated.

---

## Dataset Reference

IBM Telco Customer Churn Dataset:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---
