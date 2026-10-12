"""Train, evaluate, and serialize the production Hybrid Content Moderator model.
Combines TF-IDF n-grams with syntactic and temporal surface features under class-balanced loss.
"""
from __future__ import annotations

import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
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
from sklearn.pipeline import Pipeline

from src.features import (
    CATEGORICAL_FEATURES,
    HYBRID_FEATURES,
    NUMERICAL_FEATURES,
    TEXT_FEATURE,
    build_hybrid_preprocessor,
    extract_text_features,
    extract_time_of_day,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train_sample_50k.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df["comment_text"] = df["comment_text"].fillna("").astype(str)
    y = (df["target"] >= 0.5).astype(int)

    print("Extracting surface text features...")
    df_feat = extract_text_features(df, text_column=TEXT_FEATURE)
    df_feat["time_of_day"] = extract_time_of_day(df["created_date"])

    X = df_feat[HYBRID_FEATURES].copy()
    return X, y


def train_and_evaluate():
    X, y = load_and_prepare_data()

    print(f"Dataset Shape: X={X.shape}, y={y.shape} (Toxic rate: {y.mean():.2%})")

    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    preprocessor = build_hybrid_preprocessor(
        text_col=TEXT_FEATURE,
        num_cols=NUMERICAL_FEATURES,
        cat_cols=CATEGORICAL_FEATURES,
        max_text_features=35000,
    )

    clf = LogisticRegression(
        class_weight="balanced",
        C=1.5,
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", clf),
    ])

    print("Fitting Hybrid Production Pipeline...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    fit_duration = time.time() - t0
    print(f"Model fit completed in {fit_duration:.2f} seconds.")

    # Evaluate on held-out test set
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1_toxic = f1_score(y_test, preds, zero_division=0)
    weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    roc_auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print("\n================ TEST SET METRICS ================")
    print(f"Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
    print(f"Toxic Precision:  {prec:.4f} ({prec*100:.2f}%)")
    print(f"Toxic Recall:     {rec:.4f} ({rec*100:.2f}%)")
    print(f"Toxic F1-Score:   {f1_toxic:.4f}")
    print(f"Weighted F1:      {weighted_f1:.4f}")
    print(f"ROC-AUC:          {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print(f"True Negatives:  {cm[0,0]:>5} | False Positives: {cm[0,1]:>5}")
    print(f"False Negatives: {cm[1,0]:>5} | True Positives:  {cm[1,1]:>5}")

    # Load previous surface-only Random Forest metrics for comparison table
    prev_rf_metrics = {
        "Model": "Surface-Only Random Forest (Baseline)",
        "Accuracy": 0.5388,
        "Precision": 0.0946,
        "Recall": 0.5646,
        "F1 (Toxic)": 0.1621,
        "Weighted F1": 0.6408,
        "ROC-AUC": 0.5801,
        "Features": "9 Surface Numbers Only (No Words)",
    }

    hybrid_metrics = {
        "Model": "Hybrid Text + Syntactic Classifier (Production)",
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1 (Toxic)": float(f1_toxic),
        "Weighted F1": float(weighted_f1),
        "ROC-AUC": float(roc_auc),
        "Features": "35k TF-IDF N-grams + 9 Surface Numbers + Time",
    }

    df_comp = pd.DataFrame([prev_rf_metrics, hybrid_metrics])
    comp_path = REPORTS_DIR / "model_comparison_surface_vs_hybrid.csv"
    df_comp.to_csv(comp_path, index=False)
    print(f"\nSaved comparison to {comp_path}")

    # Serialize complete model package
    model_artifact_path = MODELS_DIR / "final_classifier.joblib"
    package = {
        "pipeline": pipeline,
        "model_name": "Hybrid Text & Syntactic Classifier (Production)",
        "feature_columns": HYBRID_FEATURES,
        "classes_": np.array([0, 1]),
        "thresholds": {
            "auto_approve": 0.30,
            "auto_quarantine": 0.70,
            "classification": 0.50,
        },
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_toxic": float(f1_toxic),
            "weighted_f1": float(weighted_f1),
            "roc_auc": float(roc_auc),
            "confusion_matrix": cm.tolist(),
        },
        "training_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "random_state": 42,
            "fit_time_seconds": float(fit_duration),
        },
    }

    joblib.dump(package, model_artifact_path)
    print(f"Serialized production model to {model_artifact_path} ({model_artifact_path.stat().st_size / (1024**2):.2f} MB)")


if __name__ == "__main__":
    train_and_evaluate()
