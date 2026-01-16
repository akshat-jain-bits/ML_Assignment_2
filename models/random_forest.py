# models/random_forest.py

from sklearn.ensemble import RandomForestClassifier


def get_model():
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
