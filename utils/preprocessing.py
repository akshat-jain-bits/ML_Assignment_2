# utils/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_telco_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop ID column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Fix TotalCharges: convert to numeric, handle blanks
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def build_preprocessing_pipeline(df: pd.DataFrame, target_col: str = "Churn"):
    X = df.drop(columns=[target_col])
    y = df[target_col].map({"Yes": 1, "No": 0})

    # Identify column types
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return X, y, preprocessor


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
