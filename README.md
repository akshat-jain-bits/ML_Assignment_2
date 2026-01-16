# ML Assignment 2 - Classification Models Comparison + Streamlit App

## a) Problem Statement
Customer churn is a major business problem in the telecom industry, where retaining an existing customer is usually cheaper than acquiring a new one.  
The objective of this project is to build multiple machine learning classification models that can predict whether a telecom customer will **churn (leave the service)** or not, based on customer demographic information, subscription services, contract type, and billing details.

This is formulated as a **binary classification** problem:
- `Churn = Yes`  → 1  
- `Churn = No`   → 0

The trained models are evaluated using multiple classification metrics and deployed as an **interactive Streamlit web application** where users can upload a CSV test dataset and select a model to view results.

---

## b) Dataset Description (Kaggle Dataset)
Dataset: **Telco Customer Churn Dataset (Extended Version)**  
Source: Kaggle (Telco Customer Churn dataset)

- **Type**: Binary Classification Dataset
- **Target Column**: `Churn`
- **Instances (Rows)**: ~ 7043
- **Input Features**: 19+ (mix of numerical and categorical)

Key feature examples:
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- `tenure`, `MonthlyCharges`, `TotalCharges`
- `Contract`, `PaymentMethod`, `InternetService`, `OnlineSecurity`, etc.

---

## c) Models Used and Evaluation Metrics Comparison

The following classification models were implemented and evaluated on the same dataset:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest Classifier (Ensemble)
6. XGBoost Classifier (Ensemble)

### Evaluation Metrics Used
For each model, the following metrics were computed:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Metrics Comparison Table
Model				Accuracy	AUC			Precision	Recall		F1			MCC
Logistic Regression	0.738112136	0.841297889	0.504302926	0.78342246	0.613612565	0.453140692
Decision Tree		0.728176011	0.649569867	0.487804878	0.481283422	0.484522207	0.299967773
KNN					0.765081618	0.804940711	0.555555556	0.57486631	0.565045992	0.404307071
Naive Bayes			0.694819021	0.807419463	0.458944282	0.836898396	0.59280303	0.424451896
Random Forest		0.782114975	0.819164794	0.618374558	0.467914439	0.532724505	0.400707579
XGBoost				0.791341377	0.832712031	0.626582278	0.529411765	0.573913043	0.439765387


After executing `utils/training_pipeline.py`, the metrics will be generated and saved in:
- `artifacts/model_comparison.csv`

---

## d) Observations

1. **Ensemble models (Random Forest and XGBoost) generally performed best** because they combine multiple weak learners/trees and reduce overfitting while capturing complex feature interactions.

2. **Logistic Regression performed as a strong baseline** but may underperform when churn patterns are not linearly separable.

3. **Decision Tree showed higher variance** and can overfit without pruning/tuning, especially with a large number of categorical features after one-hot encoding.

4. **KNN performance depended strongly on feature scaling**, since it is distance-based. Standardization of numerical features is essential for stable performance.

5. **Naive Bayes is fast and works as a baseline**, but it may underperform because it assumes conditional independence between features, which is often not true in real-world business datasets.

6. **MCC helped validate model performance** beyond accuracy, ensuring that results remain reliable even if churn/non-churn classes are not perfectly balanced.

---

## Project Structure

ML_Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── telco.csv
│
├── models/
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   └── xgboost_model.py
│
├── utils/
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── training_pipeline.py
│
└── artifacts/
    ├── model_comparison.csv
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    └── xgboost.joblib
