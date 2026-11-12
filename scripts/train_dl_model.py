"""Train the versioned deep MLP artifact used by DLPredictionAgent.

This is a reproducible educational model, not a clinically validated model.
The script keeps the test split untouched until final evaluation and chooses
the operating threshold on a separate validation split.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agents.metrics import best_threshold_f2, classification_metrics
from config import ROOT


FEATURES = [
    "gender", "age", "hypertension", "heart_disease", "smoking_history",
    "bmi", "hbA1c_level", "blood_glucose_level",
]
NUMERIC = ["age", "hypertension", "heart_disease", "bmi", "hbA1c_level", "blood_glucose_level"]
CATEGORICAL = ["gender", "smoking_history"]
VERSION = "diabetes-mlp-v1.0"


def train(data_path: Path, output_path: Path, metadata_path: Path, seed: int = 42):
    data = pd.read_csv(data_path).drop_duplicates().reset_index(drop=True)
    X, y = data[FEATURES], data["diabetes"].astype(int)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.2, random_state=seed, stratify=y_dev,
    )

    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])
    network = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=0.0005, batch_size=256, learning_rate="adaptive",
        learning_rate_init=0.001, max_iter=100, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=10, random_state=seed,
    )
    pipeline = Pipeline([("preprocess", preprocess), ("network", network)])
    pipeline.fit(X_train, y_train)

    val_probability = pipeline.predict_proba(X_val)[:, 1]
    threshold = best_threshold_f2(y_val, val_probability)
    test_probability = pipeline.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, test_probability, threshold)
    metrics["validation_auc"] = float(roc_auc_score(y_val, val_probability))
    metrics["test_rows"] = int(len(y_test))
    metrics["train_rows"] = int(len(y_train))
    metrics["validation_rows"] = int(len(y_val))
    metrics["network_iterations"] = int(network.n_iter_)

    artifact = {
        "pipe": pipeline,
        "thr": float(threshold),
        "features": FEATURES,
        "num": NUMERIC,
        "cat": CATEGORICAL,
        "model_version": VERSION,
        "model_architecture": "Feed-forward MLP: 8 inputs -> one-hot/scale -> Dense(64, ReLU) -> Dense(32, ReLU) -> sigmoid probability",
        "training_dataset": "Diabetes Prediction 100k (deduplicated; year/location/race excluded)",
        "preprocessing": "Median imputation + StandardScaler for numeric fields; most-frequent imputation + one-hot encoding for categorical fields",
        "output_format": {"score": "P(class=1)", "prediction": "score >= validation-selected F2 threshold"},
        "metrics": metrics,
        "trained_on": str(date.today()),
        "random_seed": seed,
        "clinical_validation": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path, compress=3)
    metadata_path.write_text(json.dumps({k: v for k, v in artifact.items() if k != "pipe"},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact


def main():
    parser = argparse.ArgumentParser(description="Train the diabetes MLP artifact")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "diabetes_dataset.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "diabetes_dl_v1.joblib")
    parser.add_argument("--metadata", type=Path, default=ROOT / "models" / "diabetes_dl_v1.metadata.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    artifact = train(args.data, args.output, args.metadata, args.seed)
    print(json.dumps({"model_version": artifact["model_version"],
                      "threshold": artifact["thr"], "metrics": artifact["metrics"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
