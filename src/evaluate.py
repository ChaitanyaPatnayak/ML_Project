"""Evaluation metrics, scoring utilities, and diagnostics for classification and clustering."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def evaluate_classifier(pipeline, X_eval, y_eval) -> dict:
    """Compute all mandatory Review 2 classification metrics on held-out evaluation set."""
    preds = pipeline.predict(X_eval)

    # Check for probability scores or decision function
    scores = None
    step_model = pipeline.named_steps.get('model', pipeline)
    if hasattr(step_model, 'predict_proba'):
        scores = pipeline.predict_proba(X_eval)[:, 1]
    elif hasattr(step_model, 'decision_function'):
        scores = pipeline.decision_function(X_eval)

    cm = confusion_matrix(y_eval, preds)
    acc = accuracy_score(y_eval, preds)
    prec = precision_score(y_eval, preds, zero_division=0)
    rec = recall_score(y_eval, preds, zero_division=0)
    f1_toxic = f1_score(y_eval, preds, zero_division=0)
    weighted_f1 = f1_score(y_eval, preds, average='weighted', zero_division=0)

    try:
        roc_auc = roc_auc_score(y_eval, scores) if scores is not None else np.nan
    except Exception:
        roc_auc = np.nan

    return {
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall': float(rec),
        'F1 (Toxic)': float(f1_toxic),
        'Weighted F1': float(weighted_f1),
        'ROC-AUC': float(roc_auc) if not np.isnan(roc_auc) else None,
        'Confusion Matrix': cm.tolist(),
        'Predictions': preds,
        'Scores': scores,
    }


def evaluate_clustering(X: np.ndarray, labels: np.ndarray, algorithm_name: str) -> dict:
    """Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2 or n_clusters >= len(X):
        return {
            'Algorithm': algorithm_name,
            'Clusters': n_clusters,
            'Silhouette Score': np.nan,
            'Davies-Bouldin Index': np.nan,
            'Calinski-Harabasz Index': np.nan,
        }

    # Silhouette on subset if sample is very large for computational efficiency
    if len(X) > 5000:
        sample_mask = np.random.RandomState(42).choice(len(X), size=5000, replace=False)
        sil = silhouette_score(X[sample_mask], labels[sample_mask])
    else:
        sil = silhouette_score(X, labels)

    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)

    return {
        'Algorithm': algorithm_name,
        'Clusters': int(n_clusters),
        'Silhouette Score': float(sil),
        'Davies-Bouldin Index': float(db),
        'Calinski-Harabasz Index': float(ch),
    }
