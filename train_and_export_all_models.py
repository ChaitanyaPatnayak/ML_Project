"""Script to train and export all 10 ML Capstone Rubric classifiers.
Saves individual pipeline artifacts and a master model registry with metrics and metadata.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.features import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    extract_text_features,
    extract_time_of_day,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train_sample_50k.csv"
MODELS_DIR = ROOT / "models" / "capstone_classifiers"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Define exact 10 Capstone Rubric algorithms
RUBRIC_MODELS = {
    # Classification Part A
    "Logistic Regression": {
        "filename": "logistic_regression.joblib",
        "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "track": "Part A",
        "description": "Linear classifier optimizing log-loss with L2 regularization.",
        "supports_proba": True,
    },
    "K-Nearest Neighbors (KNN)": {
        "filename": "k_nearest_neighbors.joblib",
        "estimator": KNeighborsClassifier(n_neighbors=5),
        "track": "Part A",
        "description": "Instance-based non-parametric classifier using k=5 nearest neighbors.",
        "supports_proba": True,
    },
    "Gaussian Naive Bayes": {
        "filename": "gaussian_naive_bayes.joblib",
        "estimator": GaussianNB(),
        "track": "Part A",
        "description": "Probabilistic classifier applying Bayes' theorem with Gaussian feature assumption.",
        "supports_proba": True,
    },
    "Decision Tree Classifier": {
        "filename": "decision_tree.joblib",
        "estimator": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "track": "Part A",
        "description": "Recursive binary partition tree minimizing Gini impurity (max depth 6).",
        "supports_proba": True,
    },
    "Support Vector Classifier (SVC)": {
        "filename": "support_vector_classifier.joblib",
        "estimator": SVC(kernel="linear", C=0.01, random_state=RANDOM_STATE, max_iter=10000),
        "track": "Part A",
        "description": "Linear maximal-margin separating hyperplane. Outputs decision distance (margin).",
        "supports_proba": False,
    },
    # Classification Part B
    "Random Forest Classifier": {
        "filename": "random_forest.joblib",
        "estimator": RandomForestClassifier(
            n_estimators=120,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "track": "Part B",
        "description": "Ensemble of 120 decorrelated decision trees with balanced bootstrap sampling.",
        "supports_proba": True,
    },
    "AdaBoost Classifier": {
        "filename": "adaboost.joblib",
        "estimator": AdaBoostClassifier(n_estimators=80, learning_rate=0.5, random_state=RANDOM_STATE),
        "track": "Part B",
        "description": "Adaptive boosting ensemble reweighting misclassified samples sequentially.",
        "supports_proba": True,
    },
    "Gradient Boosting Classifier": {
        "filename": "gradient_boosting.joblib",
        "estimator": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE
        ),
        "track": "Part B",
        "description": "Stage-wise additive tree ensemble minimizing deviance loss with gradient descent.",
        "supports_proba": True,
    },
    "Bagging (Decision Tree)": {
        "filename": "bagging_decision_tree.joblib",
        "estimator": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
            n_estimators=60,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "track": "Part B",
        "description": "Bootstrap aggregating ensemble of 60 balanced depth-6 decision trees.",
        "supports_proba": True,
    },
    "MLP Classifier (Neural Network)": {
        "filename": "mlp_classifier.joblib",
        "estimator": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.001,
            max_iter=120,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=RANDOM_STATE,
        ),
        "track": "Part B",
        "description": "Feedforward neural network with 2 hidden layers (64, 32 units) and ReLU activations.",
        "supports_proba": True,
    },
}


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df["comment_text"] = df["comment_text"].fillna("").astype(str)
    y = (df["target"] >= 0.5).astype(int)

    df_feat = extract_text_features(df, text_column="comment_text")
    df_feat["time_of_day"] = extract_time_of_day(df["created_date"])

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df_feat[feature_cols].copy()
    return X, y, feature_cols


def build_surface_preprocessor() -> ColumnTransformer:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def train_and_export_all():
    X, y, feature_cols = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Dataset: {len(X)} samples | Train: {len(X_train)} | Test: {len(X_test)}")
    preprocessor = build_surface_preprocessor()

    registry = {}
    comparison_records = []

    for name, config in RUBRIC_MODELS.items():
        print(f"\n--- Training {name} ({config['track']}) ---")
        pipe = Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", clone(config["estimator"])),
        ])

        start_time = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_duration = time.perf_counter() - start_time

        # Evaluation
        preds = pipe.predict(X_test)
        step_model = pipe.named_steps["model"]
        if hasattr(step_model, "predict_proba"):
            scores = pipe.predict_proba(X_test)[:, 1]
            supports_proba = True
        elif hasattr(step_model, "decision_function"):
            scores = pipe.decision_function(X_test)
            supports_proba = False
        else:
            scores = None
            supports_proba = False

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1_toxic = float(f1_score(y_test, preds, zero_division=0))
        weighted_f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))
        roc_auc = float(roc_auc_score(y_test, scores)) if scores is not None else np.nan
        cm = confusion_matrix(y_test, preds).tolist()

        print(f"Fit Time: {fit_duration:.2f}s | Acc: {acc:.4f} | Recall: {rec:.4f} | F1: {f1_toxic:.4f} | AUC: {roc_auc:.4f}")

        # Save model artifact
        artifact_path = MODELS_DIR / config["filename"]
        model_payload = {
            "pipeline": pipe,
            "model_name": name,
            "track": config["track"],
            "description": config["description"],
            "feature_columns": feature_cols,
            "supports_proba": supports_proba,
            "classes_": np.array([0, 1]),
            "metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_toxic": f1_toxic,
                "weighted_f1": weighted_f1,
                "roc_auc": roc_auc,
                "fit_time_seconds": fit_duration,
                "confusion_matrix": cm,
            },
        }
        joblib.dump(model_payload, artifact_path)
        print(f"Saved artifact to {artifact_path} ({artifact_path.stat().st_size / 1024:.1f} KB)")

        # Record in registry
        registry[name] = {
            "artifact_path": str(artifact_path.relative_to(ROOT)),
            "track": config["track"],
            "description": config["description"],
            "supports_proba": supports_proba,
            "metrics": model_payload["metrics"],
        }

        comparison_records.append({
            "Model": name,
            "Track": config["track"],
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 (Toxic)": f1_toxic,
            "Weighted F1": weighted_f1,
            "ROC-AUC": roc_auc,
            "Fit Time (s)": fit_duration,
            "Supports Probability": supports_proba,
        })

    # Save registry metadata
    registry_file = ROOT / "models" / "classifiers_registry.joblib"
    joblib.dump(registry, registry_file)
    print(f"\nSaved master registry to {registry_file}")

    # Save comparison table
    comp_df = pd.DataFrame(comparison_records).sort_values(by="F1 (Toxic)", ascending=False).reset_index(drop=True)
    comp_df.to_csv(REPORTS_DIR / "rubric_10_models_comparison.csv", index=False)
    print(f"Saved benchmark to {REPORTS_DIR / 'rubric_10_models_comparison.csv'}")
    print("\n--- Summary Comparison Table ---")
    print(comp_df.to_string())


if __name__ == "__main__":
    train_and_export_all()
