# EDA Findings — Telco Customer Churn

## Dataset overview

The dataset contains **7,043 customers** and **21 columns**, with a binary `Churn` target and mixed categorical/numerical predictors.

`TotalCharges` is initially loaded as text because some records contain blank strings. These occur for customers with zero tenure. This is treated as the known dataset type/data-quality quirk rather than arbitrary random missingness. For numerical EDA, the accumulated charge is represented as 0 for those zero-tenure records.

## Numerical statistical profile

| Feature | Mean | Median | Std. Dev. | Variance | Skewness |
|---|---:|---:|---:|---:|---:|
| SeniorCitizen | 0.16 | 0.00 | 0.37 | 0.14 | 1.83 |
| tenure | 32.37 | 29.00 | 24.56 | 603.17 | 0.24 |
| MonthlyCharges | 64.76 | 70.35 | 30.09 | 905.41 | -0.22 |
| TotalCharges | 2279.73 | 1394.55 | 2266.79 | 5138357.17 | 0.96 |

Two distributions that deserve attention are **TotalCharges** and **MonthlyCharges**. TotalCharges is strongly right-skewed because it accumulates over customer tenure, while MonthlyCharges is also somewhat asymmetric. With skewed data, the mean can be pulled toward unusually large observations, so the median and standard deviation should be interpreted together rather than relying on the mean alone.

## Target distribution and class balance

- **No Churn:** 5,174 (73.46%)
- **Churn:** 1,869 (26.54%)

The positive class is the minority class. This means accuracy alone can give an overly optimistic impression of performance. Later modeling should report precision, recall, and F1 for churn, alongside accuracy.

## Bivariate analysis 1 — Contract type vs. churn

Observed churn rates:

Contract
Month-to-month    42.71
One year          11.27
Two year           2.83

The churn rate differs substantially by contract type. **Month-to-month** customers have the highest observed churn rate, while **Two year** customers have the lowest. This makes contract type a useful variable to investigate in later modeling. The relationship is descriptive and does not demonstrate that contract type itself causes churn; contract type can be associated with tenure, service configuration, and customer behavior.

## Bivariate analysis 2 — Tenure vs. churn

Observed churn rates by tenure group:

TenureGroup
0–12     47.44
13–24    28.71
25–48    20.39
49–72     9.51

Customers in the shortest-tenure group show the highest churn rate, while longer-tenure groups generally show lower churn. This suggests that churn is concentrated more heavily among newer customer relationships. This is an association rather than a causal claim, and tenure is also structurally related to accumulated charges.

## Bivariate analysis 3 — Monthly charges vs. churn

Observed churn rates by monthly-charge quartile:

MonthlyChargeQuartile
(18.249, 35.5]     11.24
(35.5, 70.35]      24.58
(70.35, 89.85]     37.51
(89.85, 118.75]    32.88

Churn rates vary across the distribution of monthly charges, with higher-charge groups showing a different churn profile from lower-charge groups. MonthlyCharges therefore warrants attention in later modeling. The analysis does not imply that changing a customer's monthly charge would necessarily cause churn, because pricing is connected to the services and contract characteristics selected by the customer.

## Numerical correlation analysis

The strongest numerical relationships are:

- **tenure vs. TotalCharges: 0.826**
- **MonthlyCharges vs. TotalCharges: 0.651**
- **tenure vs. MonthlyCharges: 0.248**
- **SeniorCitizen vs. MonthlyCharges: 0.220**

The especially strong relationship between **tenure and TotalCharges** is expected: TotalCharges represents accumulated spending over the customer relationship, so it naturally increases with tenure. This should be remembered during later feature-engineering decisions because the variables contain overlapping information.

Correlation measures linear association. It does not prove causation and may not capture nonlinear relationships.

## Main conclusions

1. Churn is meaningfully imbalanced at roughly 26–27%.
2. Contract type shows large differences in observed churn rates.
3. Shorter-tenure customers show higher observed churn than longer-tenure customers.
4. Monthly charges show different churn rates across their distribution.
5. Tenure and TotalCharges are strongly related due to the accumulated nature of TotalCharges.
6. The majority-class baseline is important because accuracy alone can hide complete failure to detect churners.
