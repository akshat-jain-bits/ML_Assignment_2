# utils/evaluation.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)


def evaluate_binary_classifier(model, X_test, y_test):
    """
    Returns:
      metrics_dict: dict
      y_pred: np.array
      y_proba: np.array (probability for class 1)
      cm: confusion matrix
    """

    y_pred = model.predict(X_test)

    # For AUC we need probabilities (or decision score)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        # Convert to pseudo probabilities using sigmoid (optional)
        y_proba = 1 / (1 + np.exp(-scores))
    else:
        y_proba = None

    metrics_dict = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    # AUC Score
    if y_proba is not None:
        metrics_dict["AUC"] = roc_auc_score(y_test, y_proba)
    else:
        metrics_dict["AUC"] = None

    cm = confusion_matrix(y_test, y_pred)
    return metrics_dict, y_pred, y_proba, cm
