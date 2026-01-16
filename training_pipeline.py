# utils/training_pipeline.py

import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from utils.preprocessing import load_telco_data, build_preprocessing_pipeline, split_data
from utils.evaluation import evaluate_binary_classifier

from models.logistic_regression import get_model as lr_model
from models.decision_tree import get_model as dt_model
from models.knn import get_model as knn_model
from models.naive_bayes import get_model as nb_model
from models.random_forest import get_model as rf_model
from models.xgboost_model import get_model as xgb_model


def train_and_evaluate_all(csv_path="data/telco.csv", output_dir="artifacts"):
    os.makedirs(output_dir, exist_ok=True)

    df = load_telco_data(csv_path)
    X, y, preprocessor = build_preprocessing_pipeline(df, target_col="Churn")

    X_train, X_test, y_train, y_test = split_data(X, y)

    models = {
        "Logistic Regression": lr_model(),
        "Decision Tree": dt_model(),
        "KNN": knn_model(),
        "Naive Bayes": nb_model(),
        "Random Forest": rf_model(),
        "XGBoost": xgb_model()
    }

    results = []

    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")

        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        clf.fit(X_train, y_train)

        metrics, y_pred, y_proba, cm = evaluate_binary_classifier(clf, X_test, y_test)

        row = {"Model": model_name}
        row.update(metrics)
        results.append(row)

        # Save model
        joblib.dump(clf, os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}.joblib"))

    results_df = pd.DataFrame(results)
    results_df = results_df[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]

    # Save metrics table
    results_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    print("\n✅ Model comparison saved:", os.path.join(output_dir, "model_comparison.csv"))
    print(results_df)

    return results_df


if __name__ == "__main__":
    train_and_evaluate_all()
