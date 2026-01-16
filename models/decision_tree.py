# models/decision_tree.py

from sklearn.tree import DecisionTreeClassifier


def get_model():
    return DecisionTreeClassifier(random_state=42, class_weight="balanced")
