# models/logistic_regression.py

from sklearn.linear_model import LogisticRegression


def get_model():
    return LogisticRegression(max_iter=2000, class_weight="balanced")
