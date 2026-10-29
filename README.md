# 📡 Customer Churn Prediction for SaaS Platforms

A machine learning project to predict customer churn using the **Telco Customer Churn dataset**, with actionable retention suggestions.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-orange?logo=scikit-learn)
![SMOTE](https://img.shields.io/badge/SMOTE-Class%20Balancing-purple)
![Gradio](https://img.shields.io/badge/Gradio-UI-blueviolet)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)

---

## 🔗 Live Demo

👉 **[Try the App Here](https://huggingface.co/spaces/Anshika005/customer-churn-prediction)**

---

## 📌 Project Overview

This project builds an end-to-end churn prediction pipeline:

- Load and inspect the Telco Customer Churn dataset
- Perform EDA — churn distribution, feature correlations, service usage patterns
- Clean and preprocess data (Label Encoding, missing value handling)
- Engineer features: `tenure_group`, `charge_ratio`, `num_services`
- Handle class imbalance using **SMOTE**
- Train and compare multiple models using 5-fold Stratified Cross-Validation
- Tune XGBoost with GridSearchCV
- Evaluate with ROC-AUC, Precision, Recall, F1, Confusion Matrix
- Deploy interactive churn risk prediction UI with retention suggestions

---

## 📊 Dataset

- **Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** 7,043 customers
- **Target:** Churn (Yes / No) — **26.5% churn rate**

| Feature | Description |
|---|---|
| `tenure` | Months with the company |
| `Contract` | Month-to-month / One year / Two year |
| `InternetService` | DSL / Fiber optic / No |
| `MonthlyCharges` | Monthly billing amount ($) |
| `TotalCharges` | Total amount charged ($) |
| `PaymentMethod` | Payment type |
| + 14 more service & demographic features | — |

---

## 🔧 Feature Engineering

| Feature | Description |
|---|---|
| `tenure_group` | 0–12 months / 13–36 / 36+ |
| `charge_ratio` | TotalCharges ÷ (MonthlyCharges + 1) |
| `num_services` | Count of active services |

---

## 🤖 Models Trained & Compared

| Model | ROC-AUC |
|---|---|
| Logistic Regression (Baseline) | 0.8398 |
| Random Forest (SMOTE) | ~0.91 |
| Gradient Boosting | ~0.91 |
| **XGBoost (GridSearchCV Tuned)** | **~0.93** |

✅ **Final Model:** Tuned XGBoost with `scale_pos_weight` for class imbalance

---

## 🏗️ Project Structure

```
📦 Customer-Churn-Prediction
 ┣ 📓 customer-churn-prediction-for-saas-platforms.ipynb  ← Main notebook
 ┗ 📄 README.md
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/anshikagautam162-blip/Customer-Churn-Prediction
cd Customer-Churn-Prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn gradio joblib
```

### 3. Open the notebook
Open `customer-churn-prediction-for-saas-platforms.ipynb` in Kaggle or Jupyter.

### 4. Dataset path
```
/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## 📈 Key Findings

- **Contract type** is the strongest churn predictor — month-to-month customers churn 3x more
- **Fiber optic** users without Online Security churn significantly more
- **New customers** (tenure ≤ 12 months) are highest risk
- **Electronic check** payment users churn more than auto-pay users
- Adding SMOTE improved minority class recall by ~12%

---

## ⚠️ Limitations

- Results apply to this specific Telco dataset and test split
- Real-world churn patterns vary by industry and customer base
- Not intended for production use without additional validation

---

## 👩‍💻 Author

**Anshika Gautam**  
B.Tech — Artificial Intelligence & Data Science

[GitHub](https://github.com/anshikagautam162-blip) | [Live Demo](https://huggingface.co/spaces/Anshika005/customer-churn-prediction)
