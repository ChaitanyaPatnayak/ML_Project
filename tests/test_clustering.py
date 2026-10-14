"""Tests for Unsupervised Clustering Pipeline and Validation Metrics (Review 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.app import (
    compute_all_clustering_comparison,
    compute_clustering_model,
    get_clustering_base_data,
)
from src.features import NUMERICAL_FEATURES, extract_single_comment_features


def test_clustering_base_data_integrity():
    """Verify base clustering feature space is strictly numerical, standardized, and label-free."""
    base = get_clustering_base_data()
    assert base is not None, "Base clustering data must load successfully"
    assert "X_scaled" in base
    assert "X_pca" in base
    assert "scaler" in base
    assert "pca" in base
    assert "feature_cols" in base

    # Check dimensions
    assert base["X_scaled"].shape[1] == len(NUMERICAL_FEATURES)
    assert base["X_pca"].shape[1] == 2
    assert base["feature_cols"] == NUMERICAL_FEATURES

    # Ensure no target column exists in feature set
    assert "target" not in base["feature_cols"]
    assert "toxic" not in base["feature_cols"]


@pytest.mark.parametrize(
    "algo_name",
    [
        "K-Means",
        "Agglomerative Clustering — Ward Linkage",
        "Agglomerative Clustering — Complete Linkage",
        "Agglomerative Clustering — Average Linkage",
    ],
)
def test_clustering_algorithms_k5(algo_name):
    """Verify all 4 clustering methods compute valid assignments and internal metrics at k=5."""
    res = compute_clustering_model(algo_name, k=5)
    assert res is not None
    assert res["k"] == 5
    assert len(res["labels"]) == len(res["X_pca"])

    # Labels must be in {0, 1, 2, 3, 4}
    unique_labels = set(res["labels"])
    assert len(unique_labels) <= 5

    # Check metrics validity
    assert -1.0 <= res["silhouette"] <= 1.0
    assert res["davies_bouldin"] >= 0.0
    assert res["calinski_harabasz"] > 0.0

    # Centroids shape
    assert res["centroids"].shape == (5, len(NUMERICAL_FEATURES))


def test_clustering_variable_k():
    """Verify algorithm updates correctly when k varies."""
    for k_val in [3, 6]:
        res = compute_clustering_model("K-Means", k=k_val)
        assert res["k"] == k_val
        assert res["centroids"].shape[0] == k_val
        assert len(set(res["labels"])) == k_val


def test_clustering_comparison_table():
    """Verify 4-method comparison table generates required schema and catches chaining."""
    df_comp = compute_all_clustering_comparison(k=5)
    assert not df_comp.empty
    assert len(df_comp) == 4

    required_cols = [
        "Clustering Method",
        "Linkage Type",
        "Silhouette Score (↑)",
        "Davies-Bouldin Index (↓)",
        "Calinski-Harabasz Index (↑)",
        "Largest Cluster Share",
        "Partition Health",
    ]
    for col in required_cols:
        assert col in df_comp.columns

    # Verify Complete and Average linkage show severe chaining
    complete_row = df_comp[df_comp["Clustering Method"].str.contains("Complete")].iloc[0]
    average_row = df_comp[df_comp["Clustering Method"].str.contains("Average")].iloc[0]
    assert "Severe Chaining" in complete_row["Partition Health"]
    assert "Severe Chaining" in average_row["Partition Health"]

    # Verify Ward and K-Means show balanced partition
    ward_row = df_comp[df_comp["Clustering Method"].str.contains("Ward")].iloc[0]
    kmeans_row = df_comp[df_comp["Clustering Method"].str.contains("K-Means")].iloc[0]
    assert "Balanced Partition" in ward_row["Partition Health"]
    assert "Balanced Partition" in kmeans_row["Partition Health"]


def test_single_comment_cluster_assignment():
    """Verify geometric projection of a single comment into the unsupervised space."""
    res = compute_clustering_model("K-Means", k=5)
    text = "This is a test comment to verify unsupervised centroid assignment."

    df_single = extract_single_comment_features(text, "Unknown", NUMERICAL_FEATURES)
    X_raw = df_single[NUMERICAL_FEATURES].copy()
    for col in ["comment_length", "word_count"]:
        if col in X_raw.columns:
            X_raw[col] = np.log1p(X_raw[col])

    X_scaled = res["scaler"].transform(X_raw)
    dists = np.linalg.norm(res["centroids"] - X_scaled, axis=1)
    assigned = int(np.argmin(dists))

    assert 0 <= assigned < 5
    pca_pt = res["pca"].transform(X_scaled)[0]
    assert len(pca_pt) == 2
