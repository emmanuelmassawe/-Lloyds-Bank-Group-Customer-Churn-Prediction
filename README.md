# -Lloyds-Bank-Group-Customer-Churn-Prediction

# 🏦 Lloyds Bank Group — Customer Churn Prediction

A machine learning project that predicts whether a bank customer is likely to leave (churn), using demographic, behavioural, and account-level features from Lloyds Bank Group's customer records.

---

## 📋 Project Overview

Customer churn is one of the most costly challenges in retail banking. Acquiring a new customer is significantly more expensive than retaining an existing one. This project builds a classification model to identify **at-risk customers before they leave**, enabling the bank to intervene with targeted retention strategies.

**Core question:** Given a customer's profile, account activity, and satisfaction data, can we predict whether they will exit the bank?

---

## 🗂️ Files

```
bank_customer_churn_prediction_at_LLOYND_Bank_GROUP.ipynb   # Main ML notebook
Customer-Churn-Records.csv                                   # Input dataset
```

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| Source | `Customer-Churn-Records.csv` |
| Target variable | `Exited` (binary: 0 = stayed, 1 = churned) |
| Feature types | Numerical, binary, and categorical |

**Key columns:**

| Column | Description |
|--------|-------------|
| `CreditScore` | Customer credit score |
| `Age` | Customer age |
| `Tenure` | Years with the bank |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products held |
| `HasCrCard` | Whether the customer has a credit card |
| `IsActiveMember` | Whether the customer is active |
| `EstimatedSalary` | Estimated annual salary |
| `Satisfaction Score` | Customer satisfaction rating |
| `Point Earned` | Loyalty points earned |
| `Complain` | Whether a complaint was raised |
| `Geography` | Country of residence |
| `Gender` | Customer gender |
| `Card Type` | Type of card held |

---

## 🔍 Analysis Pipeline

| Step | Description |
|------|-------------|
| 1 | **Data Loading & Understanding** — Shape, dtypes, binary/categorical distributions |
| 2 | **Data Cleaning** — Missing values, duplicate check, IQR-based outlier detection |
| 3 | **Outlier Visualisation** — Box plots for all numerical features |
| 4 | **Univariate Analysis** — Distribution histograms (numerical), count plots (binary & categorical) |
| 5 | **Bivariate Analysis** — Feature vs. `Exited` relationship (box plots, count plots, correlation heatmap) |
| 6 | **Feature Engineering** — 4 new derived features |
| 7 | **Encoding** — One-hot encoding for Geography, Gender, Card Type |
| 8 | **Train/Test Split** — 80/20 split, random state 42 |
| 9 | **Scaling** — StandardScaler applied to all features |
| 10 | **Resampling** — SMOTE oversampling on training set to handle class imbalance |
| 11 | **Modelling** — 4 classifiers compared; GridSearchCV tuning for best two |

---

## 🧠 Engineered Features

| Feature | Description |
|---------|-------------|
| `balance_per_col` | Account balance divided by number of products (wealth concentration) |
| `balance_salary_ratio` | Balance relative to estimated salary (financial dependency) |
| `age_group` | Binned age: Young / Middle / Senior / Elder |
| `is_high_value` | 1 if balance > median AND customer is active |

---

## 🤖 Models Compared

| Model | Notes |
|-------|-------|
| **Logistic Regression** | Baseline linear classifier |
| **Random Forest** | Tuned via GridSearchCV (n_estimators, max_depth, min_samples) |
| **XGBoost** | Tuned via GridSearchCV (learning_rate, max_depth, subsample) |
| **Support Vector Machine (SVC)** | Default parameters, used as benchmark |

All models trained on SMOTE-resampled data to address class imbalance.

### Hyperparameter Tuning (GridSearchCV)
- **Random Forest:** searched over 6 × 3 × 2 × 2 = 72 parameter combinations, 5-fold CV
- **XGBoost:** searched over 2 × 3 × 2 × 2 = 24 combinations, 5-fold CV, scored on accuracy

---

## 🛠️ Tech Stack

```python
pandas / numpy              # Data wrangling
matplotlib / seaborn        # Visualisation
scikit-learn                # Logistic Regression, Random Forest, SVM, preprocessing, metrics
xgboost                     # Gradient boosting
imbalanced-learn            # SMOTE oversampling
```

Developed in **Google Colab**.

---

## 🚀 Getting Started

1. Clone or download this repository
2. Open the notebook in Jupyter or Google Colab
3. Upload `Customer-Churn-Records.csv` when prompted
4. Run all cells top to bottom

---

## 💼 Business Value

Identifying churners in advance allows Lloyds Bank Group to:

- **Prioritise retention calls** for high-risk, high-value customers
- **Offer personalised incentives** (rate adjustments, loyalty rewards) to at-risk segments
- **Reduce revenue loss** from unexpected customer exits
- **Focus complaints teams** — since `Complain` is a strong churn signal

---

## 👤 Author

**Emmanuel J. Lelo**
Data Science Project — Lloyds Bank Group Customer Churn Analysis
📧 leloemmanuel540@gmail.com
