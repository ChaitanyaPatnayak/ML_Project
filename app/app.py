"""Real-Time Content Moderator - AI Analytics & Governance Dashboard
Capstone Review 2 (23CSE301 Machine Learning)
Authors: Chaitanya, Sai Krishna, Towfeeq Rameez
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path for imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features import (
    ALL_FEATURES,
    HYBRID_FEATURES,
    NUMERICAL_FEATURES,
    extract_single_comment_features,
    get_current_time_category,
)
from src.moderation_policy import ModerationDecision, ModerationPolicy

# Page configuration
st.set_page_config(
    page_title="AI Sentinel | Content Moderation Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Midnight Dark Theme Custom CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-main: #0B0F19;
        --bg-card: #131B2E;
        --bg-glass: rgba(19, 27, 46, 0.7);
        --accent-violet: #8B5CF6;
        --accent-cyan: #06B6D4;
        --accent-emerald: #10B981;
        --accent-amber: #F59E0B;
        --accent-rose: #F43F5E;
        --border-subtle: rgba(139, 92, 246, 0.2);
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
    }

    .stApp {
        background-color: var(--bg-main);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(6, 182, 212, 0.1) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #A78BFA 0%, #22D3EE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        margin: 0;
    }

    /* Metric Card */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: var(--accent-cyan);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #C084FC 0%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        font-weight: 600;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-violet { background: rgba(139, 92, 246, 0.2); color: #C084FC; border: 1px solid rgba(139, 92, 246, 0.4); }
    .badge-cyan { background: rgba(6, 182, 212, 0.2); color: #38BDF8; border: 1px solid rgba(6, 182, 212, 0.4); }
    .badge-emerald { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-amber { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-rose { background: rgba(244, 63, 94, 0.2); color: #FB7185; border: 1px solid rgba(244, 63, 94, 0.4); }

    /* Prediction Card */
    .pred-box {
        padding: 22px;
        border-radius: 12px;
        margin-top: 16px;
        border: 1px solid;
    }
    .pred-box-safe {
        background: rgba(16, 185, 129, 0.1);
        border-color: rgba(16, 185, 129, 0.4);
    }
    .pred-box-warning {
        background: rgba(245, 158, 11, 0.1);
        border-color: rgba(245, 158, 11, 0.4);
    }
    .pred-box-toxic {
        background: rgba(244, 63, 94, 0.1);
        border-color: rgba(244, 63, 94, 0.4);
    }

    /* Dataframe styling */
    .dataframe {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Data & Model Loader with Streamlit Caching
@st.cache_data
def load_dataset(sample_size: int = 5000) -> pd.DataFrame:
    path_candidates = [
        ROOT_DIR / "data" / "train_sample_50k.csv",
        ROOT_DIR / "train.csv",
    ]
    data_path = next((p for p in path_candidates if p.exists()), None)
    if not data_path:
        return pd.DataFrame()
    df = pd.read_csv(data_path)
    if len(df) > sample_size:
        return df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return df


@st.cache_data
def load_reports() -> dict[str, pd.DataFrame]:
    reports_dir = ROOT_DIR / "reports"
    files = {
        "classification": reports_dir / "classification_comparison.csv",
        "tuning": reports_dir / "tuning_comparison.csv",
        "delta": reports_dir / "tuning_delta.csv",
        "surface_vs_hybrid": reports_dir / "model_comparison_surface_vs_hybrid.csv",
        "rubric_10_models": reports_dir / "rubric_10_models_comparison.csv",
        "clustering": reports_dir / "clustering_metrics.csv",
        "linkage": reports_dir / "linkage_comparison.csv",
        "kmeans_profile": reports_dir / "kmeans_cluster_profile.csv",
        "ward_profile": reports_dir / "ward_cluster_profile.csv",
        "roc_curves": reports_dir / "roc_curves_data.csv",
    }
    loaded = {}
    for key, path in files.items():
        if path.exists():
            try:
                loaded[key] = pd.read_csv(path)
            except Exception:
                loaded[key] = pd.DataFrame()
        else:
            loaded[key] = pd.DataFrame()
    return loaded


@st.cache_resource
def load_trained_model():
    model_path = ROOT_DIR / "models" / "final_classifier.joblib"
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model artifact: {e}")
            return None
    return None


# Direct explicit artifact paths for each algorithm to prevent accidental substitution or fallback
MODEL_ARTIFACT_MAP: dict[str, Path] = {
    "Logistic Regression": ROOT_DIR / "models" / "capstone_classifiers" / "logistic_regression.joblib",
    "K-Nearest Neighbors (KNN)": ROOT_DIR / "models" / "capstone_classifiers" / "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": ROOT_DIR / "models" / "capstone_classifiers" / "gaussian_naive_bayes.joblib",
    "Decision Tree Classifier": ROOT_DIR / "models" / "capstone_classifiers" / "decision_tree.joblib",
    "Support Vector Classifier (SVC)": ROOT_DIR / "models" / "capstone_classifiers" / "support_vector_classifier.joblib",
    "Random Forest Classifier": ROOT_DIR / "models" / "capstone_classifiers" / "random_forest.joblib",
    "AdaBoost Classifier": ROOT_DIR / "models" / "capstone_classifiers" / "adaboost.joblib",
    "Gradient Boosting Classifier": ROOT_DIR / "models" / "capstone_classifiers" / "gradient_boosting.joblib",
    "Bagging (Decision Tree)": ROOT_DIR / "models" / "capstone_classifiers" / "bagging_decision_tree.joblib",
    "MLP Classifier (Neural Network)": ROOT_DIR / "models" / "capstone_classifiers" / "mlp_classifier.joblib",
    "🏆 Production Hybrid Model (TF-IDF + Syntactic)": ROOT_DIR / "models" / "final_classifier.joblib",
}


@st.cache_resource
def load_all_models_registry() -> dict:
    reg_path = ROOT_DIR / "models" / "classifiers_registry.joblib"
    if reg_path.exists():
        try:
            return joblib.load(reg_path)
        except Exception as e:
            st.warning(f"Warning: Could not load registry metadata: {e}")
    return {}


@st.cache_resource
def load_model_payload(model_key: str) -> dict:
    """Load the dedicated fitted model artifact for the selected algorithm.
    Enforces strict model isolation: NEVER silently substitutes another model or fallback.
    """
    if model_key in MODEL_ARTIFACT_MAP:
        p = MODEL_ARTIFACT_MAP[model_key]
    else:
        reg = load_all_models_registry()
        if model_key in reg:
            p = ROOT_DIR / reg[model_key]["artifact_path"]
        else:
            raise KeyError(f"Unknown algorithm selection: '{model_key}'. Allowed: {list(MODEL_ARTIFACT_MAP.keys())}")

    if not p.exists():
        raise FileNotFoundError(f"Model artifact file for '{model_key}' does not exist at '{p}'.")

    payload = joblib.load(p)
    if not isinstance(payload, dict) or "pipeline" not in payload:
        raise ValueError(f"Corrupt or invalid model payload for '{model_key}' at '{p}'.")

    # Strict identity check to guarantee no accidental cross-model reuse
    expected_name = "Hybrid Text & Syntactic Classifier (Production)" if "Hybrid" in model_key else model_key
    loaded_name = payload.get("model_name", "")
    if loaded_name != expected_name:
        raise ValueError(
            f"Model identity mismatch! Requested '{model_key}', but artifact at '{p}' contains '{loaded_name}'."
        )

    return payload


reports = load_reports()
model_artifact = load_trained_model()
models_registry = load_all_models_registry()
policy = ModerationPolicy(threshold_approve=0.30, threshold_quarantine=0.70, classification_threshold=0.50)

# Sidebar Navigation & Meta
with st.sidebar:
    st.markdown("### 🛡️ AI Sentinel")
    st.markdown(
        "<span class='badge badge-violet'>23CSE301</span><span class='badge badge-cyan'>Review 2</span>",
        unsafe_allow_html=True,
    )
    st.caption("Real-Time Content Moderation System")
    st.divider()

    navigation = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "🧪 Classification Lab",
            "🌌 Clustering Studio",
            "🔍 Dataset Explorer",
            "⚡ Prediction Playground",
            "📑 Reports & Governance",
        ],
        index=0,
    )

    st.divider()
    st.markdown("**Authors & Contributors**")
    st.markdown("• **Chaitanya**")
    st.markdown("• **Sai Krishna**")
    st.markdown("• **Towfeeq Rameez**")
    st.divider()
    st.caption("Scikit-Learn 1.8 | Streamlit | Plotly")


# Helper for Plotly dark layout
def dark_plotly_layout(fig, title: str = "", height: int = 420):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter", size=16, color="#F8FAFC")),
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#131B2E",
        font=dict(family="Inter", color="#94A3B8"),
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.1)", zerolinecolor="rgba(148, 163, 184, 0.2)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.1)", zerolinecolor="rgba(148, 163, 184, 0.2)")
    return fig


# ==========================================
# 1. OVERVIEW SECTION
# ==========================================
if navigation == "📊 Overview":
    st.markdown(
        """
    <div class="hero-container">
        <h1 class="hero-title">Real-Time Content Moderator</h1>
        <p class="hero-subtitle">
            An end-to-end Machine Learning intelligence engine built for toxicity triage,
            syntactic clustering, and explainable safety-critical content moderation.
        </p>
        <div style="margin-top: 14px;">
            <span class="badge badge-violet">Jigsaw Civil Comments (50k)</span>
            <span class="badge badge-cyan">10 Classifiers Evaluated</span>
            <span class="badge badge-emerald">Production Hybrid Pipeline</span>
            <span class="badge badge-amber">Synchronized Tri-Tier Policy</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Metric KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">50,000</div>
            <div class="metric-label">Dataset Samples</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">10</div>
            <div class="metric-label">Evaluated Classifiers</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">0.8870</div>
            <div class="metric-label">Production ROC-AUC</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">61.1%</div>
            <div class="metric-label">Toxic-Class Recall</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("🎯 Operational Problem Statement & Decision Policy")
        st.markdown(
            """
        Digital social platforms process millions of user comments every minute. Traditional deep transformer models (RoBERTa, BERT) 
        impose heavy GPU infrastructure costs and latency bottlenecks for high-throughput stream ingestion.
        
        This project designs a **rigorous two-tier scalable moderation system**:
        1. **Hybrid Text + Syntactic Representation**: Combines 35,000 TF-IDF word/char n-grams with 9 surface features (casing, punctuation, lexical ratios) 
           under class-balanced cross-entropy loss.
        2. **Safety-Critical Metric Formulation**: In online community safety, **False Negatives are catastrophic** (allowing severe hate speech or abuse to pass undetected). 
           We deliberately optimize **Toxic-Class Recall** and **Toxic-Class F1** rather than deceptive raw accuracy, which is artificially inflated by 92% non-toxic majority dominance.
        3. **Synchronized Moderation Policy**: Enforces single-source-of-truth thresholds where uncertain/borderline comments (e.g. 30%–70% score) 
           are strictly held for review and **never falsely approved**.
        """
        )

        st.subheader("🏆 Model Leaderboard Summary (Top 5)")
        if "classification" in reports and not reports["classification"].empty:
            df_lead = reports["classification"].copy()
            df_lead = df_lead.sort_values(by="F1 (Toxic)", ascending=False).head(5)
            st.dataframe(
                df_lead[
                    [
                        "Model",
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1 (Toxic)",
                        "Weighted F1",
                        "ROC-AUC",
                    ]
                ].style.format({
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1 (Toxic)": "{:.4f}",
                    "Weighted F1": "{:.4f}",
                    "ROC-AUC": "{:.4f}",
                }),
                use_container_width=True,
            )

    with col_right:
        st.subheader("⚙️ Synchronized Multi-Tier Policy")
        st.markdown(
            """
        ```
        Incoming Comment Stream
                 │
                 ▼
        ┌─────────────────────────┐
        │ Production Hybrid Model │ ◄ TF-IDF + Surface
        └────────────┬────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
      P < 0.30   0.30 - 0.70   P >= 0.70
     Tier 1:      Tier 2:       Tier 3:
     APPROVED   HELD FOR      QUARANTINED
     Auto-Pass   REVIEW        Auto-Block
    (Safe Only) (Escalated)   (High Risk)
        ```
        """
        )
        st.info(
            "💡 **Policy Invariant**: The decision rule strictly guarantees that borderline comments in Tier 2 ($30\\% \\le P < 70\\%$) "
            "are **NEVER APPROVED**, eliminating contradictions between decision badges and operational routing."
        )


# ==========================================
# 2. CLASSIFICATION LAB
# ==========================================
elif navigation == "🧪 Classification Lab":
    st.title("🧪 Classification Lab — All 10 Models & Production Pipeline")
    st.markdown(
        "Consolidated benchmarking of **5 Review 1 Part A** and **5 Review 2 Part B** algorithms on the identical 80/20 stratified test set."
    )

    # Production Comparison Banner
    if "surface_vs_hybrid" in reports and not reports["surface_vs_hybrid"].empty:
        st.subheader("🚀 Baseline (Surface-Only) vs Production (Hybrid Text+Surface)")
        st.dataframe(
            reports["surface_vs_hybrid"].style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 (Toxic)": "{:.4f}",
                "Weighted F1": "{:.4f}",
                "ROC-AUC": "{:.4f}",
            }).background_gradient(subset=["F1 (Toxic)", "ROC-AUC"], cmap="Blues"),
            use_container_width=True,
        )
        st.caption("Adding text n-grams to surface features increases ROC-AUC from 0.5801 to 0.8870 and Toxic F1 from 0.1621 to 0.5227 (over 3.2x relative gain).")

    st.markdown("---")

    if "classification" in reports and not reports["classification"].empty:
        df_class = reports["classification"].copy()

        # Filtering / Sorting Controls
        c_filter1, c_filter2 = st.columns([2, 2])
        with c_filter1:
            sort_by = st.selectbox(
                "Sort 10-Model Table By:",
                ["F1 (Toxic)", "Recall", "ROC-AUC", "Accuracy", "Weighted F1", "Precision"],
                index=0,
            )
        with c_filter2:
            model_type_filter = st.multiselect(
                "Filter Models:",
                options=df_class["Model"].unique(),
                default=df_class["Model"].unique(),
            )

        df_display = df_class[df_class["Model"].isin(model_type_filter)].sort_values(
            by=sort_by, ascending=False
        )

        st.subheader("📋 Consolidated 10-Algorithm Benchmark Table")
        st.dataframe(
            df_display.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 (Toxic)": "{:.4f}",
                "Weighted F1": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "Fit Time (s)": "{:.2f}s",
            }).background_gradient(subset=["F1 (Toxic)", "Recall", "ROC-AUC"], cmap="Purples"),
            use_container_width=True,
        )

        # Plotly Multi-Metric Bar Chart
        st.subheader("📊 Comparative Performance Bar Chart")
        fig_bar = px.bar(
            df_display,
            x="Model",
            y=["Accuracy", "Recall", "F1 (Toxic)", "ROC-AUC"],
            barmode="group",
            color_discrete_sequence=["#8B5CF6", "#06B6D4", "#10B981", "#F59E0B"],
        )
        dark_plotly_layout(fig_bar, "Model Performance Across Key Evaluated Metrics", height=450)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # Interactive ROC-AUC Curves Section
        st.subheader("📈 Receiver Operating Characteristic (ROC) & Area Under Curve (AUC)")
        st.caption("Interactive trade-off curves: True Positive Rate (Sensitivity / Recall) vs False Positive Rate across all continuous classification thresholds.")

        if "roc_curves" in reports and not reports["roc_curves"].empty:
            df_roc = reports["roc_curves"].copy()
            all_roc_models = list(df_roc["Model"].unique())
            key_benchmarks = [
                "Production Hybrid Model",
                "Bagging (Decision Tree)",
                "Random Forest Classifier",
                "Support Vector Classifier (SVC)",
            ]

            col_mode, _ = st.columns([3, 1])
            with col_mode:
                view_mode = st.radio(
                    "🎯 Model Selection View:",
                    options=["⭐ Key Benchmarks (Top 4)", "🌐 All 11 Classifiers", "🎛️ Custom Selection"],
                    horizontal=True,
                    index=0,
                    help="Switch between clean benchmark comparison, all models, or custom selection without UI crowding.",
                )

            if view_mode == "⭐ Key Benchmarks (Top 4)":
                selected_roc_models = [m for m in key_benchmarks if m in all_roc_models]
                st.caption(f"Displaying **{len(selected_roc_models)} core benchmark models** for high clarity. Switch to 'All 11 Classifiers' to overlay all models.")
            elif view_mode == "🌐 All 11 Classifiers":
                selected_roc_models = all_roc_models
                st.caption(f"Displaying **all {len(selected_roc_models)} classifiers** (10 rubric models + Flagship Hybrid Model).")
            else:
                selected_roc_models = st.multiselect(
                    "Select Models to Overlay:",
                    options=all_roc_models,
                    default=[m for m in key_benchmarks if m in all_roc_models],
                    help="Select classifiers to display on the ROC curve.",
                )

            if selected_roc_models:
                df_roc_filtered = df_roc[df_roc["Model"].isin(selected_roc_models)].copy()
                model_auc_map = df_roc_filtered.groupby("Model")["ROC_AUC"].first().to_dict()
                df_roc_filtered["Model_Legend"] = df_roc_filtered["Model"].apply(
                    lambda m: f"{m} (AUC={model_auc_map.get(m, 0.0):.3f})"
                )

                color_palette = [
                    "#10B981", "#8B5CF6", "#06B6D4", "#F59E0B", "#F43F5E",
                    "#38BDF8", "#EC4899", "#A855F7", "#14B8A6", "#EAB308",
                    "#6366F1", "#94A3B8"
                ]

                fig_roc = px.line(
                    df_roc_filtered,
                    x="FPR",
                    y="TPR",
                    color="Model_Legend",
                    labels={
                        "FPR": "False Positive Rate (1 - Specificity)",
                        "TPR": "True Positive Rate (Sensitivity / Recall)",
                        "Model_Legend": "Model (AUC Score)",
                    },
                    color_discrete_sequence=color_palette,
                )
                fig_roc.add_shape(
                    type="line", line=dict(dash="dash", color="#64748B", width=1.5),
                    x0=0, y0=0, x1=1, y1=1
                )
                dark_plotly_layout(fig_roc, "", height=520)
                fig_roc.update_layout(
                    title=dict(
                        text="ROC-AUC Curves Across Evaluated Classifiers",
                        font=dict(family="Inter", size=15, color="#F8FAFC"),
                        x=0.02,
                        y=0.96,
                        xanchor="left",
                        yanchor="top",
                    ),
                    legend=dict(
                        title=dict(text="<b>Model (AUC)</b>", font=dict(size=11, color="#CBD5E1")),
                        orientation="v",
                        yanchor="top",
                        y=0.95,
                        xanchor="left",
                        x=1.02,
                        font=dict(size=10, family="Inter", color="#94A3B8"),
                        bgcolor="rgba(19, 27, 46, 0.8)",
                        bordercolor="rgba(139, 92, 246, 0.25)",
                        borderwidth=1,
                    ),
                    margin=dict(l=55, r=260, t=65, b=50),
                )
                fig_roc.update_xaxes(
                    title_text="False Positive Rate (1 - Specificity)",
                    range=[-0.01, 1.01],
                    gridcolor="rgba(148, 163, 184, 0.1)",
                )
                fig_roc.update_yaxes(
                    title_text="True Positive Rate (Sensitivity / Recall)",
                    range=[-0.01, 1.01],
                    gridcolor="rgba(148, 163, 184, 0.1)",
                )
                st.plotly_chart(fig_roc, use_container_width=True)

                best_auc_m = max(model_auc_map, key=model_auc_map.get)
                st.info(
                    f"🏆 **Highest Discriminative Ability**: `{best_auc_m}` achieves an **AUC of {model_auc_map[best_auc_m]:.4f}**, "
                    "demonstrating strong capability to rank toxic content above civil comments across all threshold operating points."
                )
            else:
                st.warning("⚠️ Please select at least one model to display the ROC curve.")

        st.markdown("---")

        # Hyperparameter Tuning: Baseline vs Tuned Random Forest
        st.subheader("⚡ Hyperparameter Tuning — Baseline vs Tuned Random Forest")
        col_t1, col_t2 = st.columns([3, 2])
        with col_t1:
            if "tuning" in reports and not reports["tuning"].empty:
                df_tune = reports["tuning"].copy()
                st.dataframe(
                    df_tune.style.format({
                        "Accuracy": "{:.2%}",
                        "Precision": "{:.2%}",
                        "Recall": "{:.2%}",
                        "F1 (Toxic)": "{:.4f}",
                        "Weighted F1": "{:.4f}",
                        "ROC-AUC": "{:.4f}",
                    }),
                    use_container_width=True,
                )

            if "delta" in reports and not reports["delta"].empty:
                df_delta = reports["delta"].copy()
                fig_delta = px.bar(
                    df_delta,
                    x="Metric",
                    y="Change (Tuned - Baseline)",
                    color="Change (Tuned - Baseline)",
                    color_continuous_scale="RdBu",
                )
                dark_plotly_layout(fig_delta, "Metric Delta (Tuned vs Baseline)", height=320)
                st.plotly_chart(fig_delta, use_container_width=True)

        with col_t2:
            st.markdown(
                """
            <div style="background: #131B2E; border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 12px; padding: 18px;">
                <h4 style="color: #A78BFA; margin-top: 0;">🔧 Selected Hyperparameters</h4>
                <ul style="color: #94A3B8; font-size: 0.9rem; line-height: 1.8;">
                    <li><b>Estimator:</b> <code>RandomForestClassifier</code></li>
                    <li><b>n_estimators:</b> <code>60</code></li>
                    <li><b>max_depth:</b> <code>8</code></li>
                    <li><b>min_samples_leaf:</b> <code>2</code></li>
                    <li><b>class_weight:</b> <code>balanced_subsample</code></li>
                    <li><b>Optimization Objective:</b> <code>f1_score (toxic class)</code></li>
                </ul>
                <div style="margin-top: 12px; padding: 10px; background: rgba(16, 185, 129, 0.1); border-left: 3px solid #10B981; border-radius: 4px;">
                    <span style="color: #34D399; font-weight: 600; font-size: 0.85rem;">
                        Verified Improvement: Toxic Recall increased by <b>+56.08%</b>, F1 improved by <b>+0.1548</b>, and ROC-AUC improved by <b>+0.0286</b>.
                    </span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.subheader("🖼️ Evaluated Classification Visual Gallery (5 Mandatory Graphs)")
        st.caption("Publication-quality evaluation diagnostics computed directly from held-out test predictions (N = 10,000 comments).")

        c_vis_col1, c_vis_col2 = st.columns(2)
        with c_vis_col1:
            st.markdown("#### 1. 10-Classifier Performance Benchmark")
            p1 = ROOT_DIR / "assets" / "classification" / "01_model_performance_comparison.png"
            if p1.exists():
                st.image(str(p1), use_container_width=True)

            st.markdown("#### 3. Confusion Matrix Heatmaps (Selected Models)")
            p3 = ROOT_DIR / "assets" / "classification" / "03_confusion_matrix_selected_model.png"
            if p3.exists():
                st.image(str(p3), use_container_width=True)

            st.markdown("#### 5. Precision–Recall Curves (Imbalanced Data)")
            p5 = ROOT_DIR / "assets" / "classification" / "05_precision_recall_curves.png"
            if p5.exists():
                st.image(str(p5), use_container_width=True)

        with c_vis_col2:
            st.markdown("#### 2. Toxic-Class Precision vs Recall Trade-Off")
            p2 = ROOT_DIR / "assets" / "classification" / "02_precision_recall_tradeoff.png"
            if p2.exists():
                st.image(str(p2), use_container_width=True)

            st.markdown("#### 4. Multi-Model ROC Curves")
            p4 = ROOT_DIR / "assets" / "classification" / "04_roc_curves.png"
            if p4.exists():
                st.image(str(p4), use_container_width=True)


# ==========================================
# 3. CLUSTERING STUDIO
# ==========================================
elif navigation == "🌌 Clustering Studio":
    st.title("🌌 Clustering Studio — K-Means & Agglomerative Hierarchical")
    st.markdown(
        "Unsupervised exploration of structural text patterns **strictly without ground-truth labels during training**."
    )

    tab_eval, tab_vis, tab_profiles = st.tabs(
        ["📊 Metrics & Linkage Comparison", "🖼️ Visualizations & Projections", "👥 Cluster Profiles"]
    )

    with tab_eval:
        st.subheader("📏 Mandatory Internal Clustering Metrics")
        st.markdown(
            "Evaluation across all 3 rubric-mandated criteria: **Silhouette Score**, **Davies-Bouldin Index**, and **Calinski-Harabasz Index**."
        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### Primary Algorithms ($k=5$)")
            if "clustering" in reports and not reports["clustering"].empty:
                st.dataframe(
                    reports["clustering"].style.format({
                        "Silhouette Score (sampled)": "{:.4f}",
                        "Davies-Bouldin Index": "{:.4f}",
                        "Calinski-Harabasz Index": "{:.2f}",
                    }),
                    use_container_width=True,
                )

        with col_m2:
            st.markdown("#### Hierarchical Linkage Comparison ($k=5$)")
            if "linkage" in reports and not reports["linkage"].empty:
                st.dataframe(
                    reports["linkage"].style.format({
                        "Silhouette Score (sampled)": "{:.4f}",
                        "Davies-Bouldin Index": "{:.4f}",
                        "Calinski-Harabasz Index": "{:.2f}",
                    }),
                    use_container_width=True,
                )

        st.info(
            "🔍 **Linkage Analysis**: Complete and Average linkages exhibit artificially high Silhouette scores (>0.73) due to severe "
            "**chaining behavior** (isolating single outlier comments while bundling >98% of points into one giant cluster). "
            "**Ward's linkage** minimizes within-cluster variance, achieving balanced spherical clusters and a superior Calinski-Harabasz score (582.55 vs 121.17)."
        )

    with tab_vis:
        st.subheader("🖼️ Generated Clustering Plots (5 Rubric Visualizations)")
        vis_col1, vis_col2 = st.columns(2)

        with vis_col1:
            st.markdown("#### 1. K-Means Elbow Curve (Inertia vs $k$)")
            elbow_img = ROOT_DIR / "assets" / "clustering" / "01_kmeans_elbow_curve.png"
            if not elbow_img.exists():
                elbow_img = ROOT_DIR / "assets" / "elbow_curve.png"
            if elbow_img.exists():
                st.image(str(elbow_img), use_container_width=True)

            st.markdown("#### 3. PCA 2D Cluster Visualizations (K-Means vs Ward)")
            pca_img = ROOT_DIR / "assets" / "clustering" / "03_pca_clusters_2d.png"
            if not pca_img.exists():
                pca_img = ROOT_DIR / "assets" / "pca_clusters.png"
            if pca_img.exists():
                st.image(str(pca_img), use_container_width=True)

            st.markdown("#### 5. t-SNE 2D Manifold Visualization (K-Means)")
            tsne_img = ROOT_DIR / "assets" / "clustering" / "05_tsne_clusters_2d.png"
            if not tsne_img.exists():
                tsne_img = ROOT_DIR / "assets" / "tsne_kmeans.png"
            if tsne_img.exists():
                st.image(str(tsne_img), use_container_width=True)

        with vis_col2:
            st.markdown("#### 2. Clustering Quality Metrics Comparison")
            metrics_img = ROOT_DIR / "assets" / "clustering" / "02_clustering_metrics_comparison.png"
            if metrics_img.exists():
                st.image(str(metrics_img), use_container_width=True)

            st.markdown("#### 4. Hierarchical Clustering Dendrograms (3 Linkages)")
            dendro_img = ROOT_DIR / "assets" / "clustering" / "04_hierarchical_dendrograms.png"
            if not dendro_img.exists():
                dendro_img = ROOT_DIR / "assets" / "dendrograms.png"
            if dendro_img.exists():
                st.image(str(dendro_img), use_container_width=True)

    with tab_profiles:
        st.subheader("👥 Cluster Profiles & Post-Fit Toxicity Validation")
        st.caption("Ground-truth labels were strictly withheld during clustering and analyzed purely post-fit.")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("#### K-Means Cluster Profiles")
            if "kmeans_profile" in reports and not reports["kmeans_profile"].empty:
                st.dataframe(
                    reports["kmeans_profile"].style.format({
                        "Mean_Toxicity_Score": "{:.4f}",
                        "Toxic_Label_Rate": "{:.2%}",
                    }),
                    use_container_width=True,
                )

        with col_p2:
            st.markdown("#### Agglomerative (Ward) Cluster Profiles")
            if "ward_profile" in reports and not reports["ward_profile"].empty:
                st.dataframe(
                    reports["ward_profile"].style.format({
                        "Mean_Toxicity_Score": "{:.4f}",
                        "Toxic_Label_Rate": "{:.2%}",
                    }),
                    use_container_width=True,
                )

        st.warning(
            "⚠️ **Analytical Insight**: Adjusted Rand Index between clusters and actual toxicity is near zero (~0.001). "
            "This proves that syntax and text formatting alone cannot replace semantic NLP understanding; a well-formatted hateful comment "
            "and an informative technical paragraph share identical surface dimensions."
        )


# ==========================================
# 4. DATASET EXPLORER
# ==========================================
elif navigation == "🔍 Dataset Explorer":
    st.title("🔍 Dataset Explorer — Jigsaw Civil Comments")
    st.markdown("Audit the 50,000-sample dataset, explore feature distributions, and review data preprocessing integrity.")

    df_sample = load_dataset(sample_size=3000)

    if not df_sample.empty:
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Total Rows in Memory", f"{len(df_sample):,}")
        with col_d2:
            toxic_ratio = (df_sample["target"] >= 0.5).mean()
            st.metric("Toxic Prevalence Rate", f"{toxic_ratio:.2%}")
        with col_d3:
            st.metric("Missing Feature Values", "0 (Imputed)")

        tab_data, tab_dist, tab_corr = st.tabs(["📋 Data Preview", "📊 Distributions", "🔥 Correlation Heatmap"])

        with tab_data:
            st.dataframe(df_sample.head(20), use_container_width=True)

        with tab_dist:
            col_dist1, col_dist2 = st.columns(2)
            with col_dist1:
                # Target distribution
                df_pie = pd.DataFrame({
                    "Label": ["Civil (0)", "Toxic (1)"],
                    "Count": [(df_sample["target"] < 0.5).sum(), (df_sample["target"] >= 0.5).sum()],
                })
                fig_pie = px.pie(
                    df_pie,
                    names="Label",
                    values="Count",
                    color="Label",
                    color_discrete_map={"Civil (0)": "#8B5CF6", "Toxic (1)": "#F43F5E"},
                    hole=0.45,
                )
                dark_plotly_layout(fig_pie, "Binary Target Class Distribution", height=350)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_dist2:
                # Continuous toxicity score histogram
                fig_hist = px.histogram(
                    df_sample,
                    x="target",
                    nbins=30,
                    color_discrete_sequence=["#06B6D4"],
                )
                dark_plotly_layout(fig_hist, "Continuous Human Annotator Toxicity Score Distribution", height=350)
                st.plotly_chart(fig_hist, use_container_width=True)

        with tab_corr:
            # Correlation matrix of numerical features with toxicity
            text = df_sample["comment_text"].fillna("")
            df_feat = pd.DataFrame({
                "comment_length": text.str.len(),
                "word_count": text.str.split().str.len().fillna(0),
                "uppercase_ratio": text.apply(lambda s: sum(1 for c in s if c.isupper()) / max(len(s) + 1, 1)),
                "exclamation_count": text.str.count("!"),
                "question_count": text.str.count(r"\?"),
                "special_char_count": text.apply(lambda s: sum(1 for c in s if not c.isalnum() and not c.isspace())),
                "target": df_sample["target"],
            })
            corr = df_feat.corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Viridis",
            )
            dark_plotly_layout(fig_corr, "Feature Correlation Matrix with Toxicity Target", height=420)
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Dataset file data/train_sample_50k.csv not found.")


# ==========================================
# 5. PREDICTION PLAYGROUND — MULTI-ALGORITHM
# ==========================================
elif navigation == "⚡ Prediction Playground":
    st.title("⚡ Prediction Playground — Multi-Algorithm Lab")
    st.markdown(
        "Interactive toxicity evaluation laboratory featuring all **10 Capstone Rubric Classification Algorithms** "
        "plus the **Production Hybrid Pipeline**."
    )

    # 1. Model Selection Dropdown
    algo_catalog = {
        # Classification Part A
        "Logistic Regression": {
            "track": "Part A (Review 1)",
            "supports_proba": True,
            "desc": "Linear log-odds classifier with L2 regularization.",
        },
        "K-Nearest Neighbors (KNN)": {
            "track": "Part A (Review 1)",
            "supports_proba": True,
            "desc": "Non-parametric instance classifier (k=5 nearest Euclidean neighbors).",
        },
        "Gaussian Naive Bayes": {
            "track": "Part A (Review 1)",
            "supports_proba": True,
            "desc": "Probabilistic Bayesian classifier with Gaussian feature likelihoods.",
        },
        "Decision Tree Classifier": {
            "track": "Part A (Review 1)",
            "supports_proba": True,
            "desc": "Recursive binary partition tree minimizing Gini impurity (max depth 6).",
        },
        "Support Vector Classifier (SVC)": {
            "track": "Part A (Review 1)",
            "supports_proba": False,
            "desc": "Linear maximal-margin separating hyperplane. Outputs geometric margin distance.",
        },
        # Classification Part B
        "Random Forest Classifier": {
            "track": "Part B (Review 2)",
            "supports_proba": True,
            "desc": "Ensemble of 120 decorrelated decision trees with balanced bootstrap sampling.",
        },
        "AdaBoost Classifier": {
            "track": "Part B (Review 2)",
            "supports_proba": True,
            "desc": "Adaptive boosting ensemble adjusting sample weights sequentially.",
        },
        "Gradient Boosting Classifier": {
            "track": "Part B (Review 2)",
            "supports_proba": True,
            "desc": "Stage-wise additive decision trees minimizing deviance via gradient descent.",
        },
        "Bagging (Decision Tree)": {
            "track": "Part B (Review 2)",
            "supports_proba": True,
            "desc": "Bootstrap aggregating ensemble of 60 balanced depth-6 decision trees.",
        },
        "MLP Classifier (Neural Network)": {
            "track": "Part B (Review 2)",
            "supports_proba": True,
            "desc": "Multi-layer perceptron neural network (64, 32 units) with ReLU activations.",
        },
        # Production
        "🏆 Production Hybrid Model (TF-IDF + Syntactic)": {
            "track": "Production Engine",
            "supports_proba": True,
            "desc": "35k TF-IDF word/char n-grams + 9 surface features under class-balanced loss.",
        },
    }

    col_algo_sel, col_mode_sel = st.columns([3, 1])
    with col_algo_sel:
        selected_algo = st.selectbox(
            "🤖 Select Classification Algorithm:",
            options=list(algo_catalog.keys()),
            index=0,
            help="Select any of the 10 rubric-mandated algorithms to execute live inference with its actual trained model.",
        )
    with col_mode_sel:
        compare_all_mode = st.checkbox(
            "⚡ Compare All 10 Models",
            value=False,
            help="Run this comment through all 10 algorithms simultaneously and display a side-by-side comparison.",
        )

    # Active Algorithm Information Card
    algo_meta = algo_catalog[selected_algo]
    track_badge_class = (
        "badge-violet" if "Part A" in algo_meta["track"]
        else "badge-cyan" if "Part B" in algo_meta["track"]
        else "badge-emerald"
    )

    st.markdown(
        f"""
    <div style="background: #131B2E; border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 12px; padding: 14px 18px; margin-bottom: 16px;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
            <div>
                <span class="badge {track_badge_class}">{algo_meta["track"]}</span>
                <b style="color: #F8FAFC; font-size: 1.05rem;">{selected_algo}</b>
                <span style="color: #94A3B8; font-size: 0.9rem; margin-left: 10px;">— {algo_meta["desc"]}</span>
            </div>
            <div style="font-size: 0.82rem; color: #38BDF8; font-family: 'JetBrains Mono', monospace;">
                Output: {"Calibrated Probability P(Toxic)" if algo_meta["supports_proba"] else "Decision Margin Score (Hyperplane Distance)"}
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Policy threshold expander
    with st.expander("⚙️ Decision Threshold Policy & Policy Invariants", expanded=False):
        st.markdown(
            r"""
        - **Auto-Approve Threshold ($\tau_{\text{approve}}$):** `30.0%` (Only comments with $P(\text{Toxic}) < 30\%$ are auto-approved)
        - **Classification Cutoff:** `50.0%` ($P \ge 50\%$ maps to binary class 1)
        - **Auto-Quarantine Threshold ($\tau_{\text{quarantine}}$):** `70.0%` (Comments with $P(\text{Toxic}) \ge 70\%$ are quarantined)
        - **Borderline Ambiguity Window:** `[30.0%, 70.0%)` (Strictly held for secondary review; **NEVER APPROVED**)
        - **Non-Probability Classifiers (e.g. Linear SVC):** Operates on raw functional margins ($w^T x + b$). Positive margins ($\ge 0.000$) map to quarantine. Uncalibrated negative margins ($< 0.000$) are conservatively held for **Secondary Review** rather than auto-approved, preventing false safety assurances from uncalibrated baseline models lacking text representation.
        """
        )

    # Benchmark presets
    presets = {
        "Select an example...": "",
        "Borderline Admonition (Bug Case)": "Mind your fucking Language",
        "Severe Hostile Attack": "Shut your ugly mouth you disgusting pig and get lost.",
        "Civil Disagreement": "I respectfully disagree with your conclusion regarding tax policy, though your point on tariffs is well argued.",
        "Benign Shouting (High Caps)": "CONGRATULATIONS ON PASSING YOUR EXAMS BROTHER I AM SO PROUD OF YOU!!!!",
        "Colloquial / Enthusiastic Profanity": "Holy shit I cannot believe that actually worked",
        "Neutral Observation": "The weather today has been unusually calm for mid-October.",
    }

    col_preset, col_tod = st.columns([3, 1])
    with col_preset:
        selected_preset = st.selectbox("Quick Benchmark Presets:", list(presets.keys()))
    with col_tod:
        selected_tod = st.selectbox(
            "Time of Day:",
            ["Morning", "Afternoon", "Evening", "Night", "Unknown"],
            index=["Morning", "Afternoon", "Evening", "Night", "Unknown"].index(get_current_time_category()),
        )

    default_text = presets[selected_preset] if selected_preset != "Select an example..." else ""
    comment_input = st.text_area(
        "Enter Comment Text for Moderation:",
        value=default_text,
        height=125,
        placeholder="Type or paste any online comment here...",
    )

    btn_label = "⚡ Run Multi-Model Comparison" if compare_all_mode else f"🛡️ Run Moderation with {selected_algo}"
    if st.button(btn_label, type="primary", use_container_width=True):
        if not comment_input.strip():
            st.warning("Please enter valid comment text before running moderation.")
        else:
            # Load selected model payload with strict isolation
            try:
                selected_payload = load_model_payload(selected_algo)
            except Exception as err:
                st.error(f"❌ Error loading model artifact for **{selected_algo}**: {err}")
                selected_payload = None

            if selected_payload:
                active_pipeline = selected_payload["pipeline"]
                active_cols = selected_payload.get("feature_columns", ALL_FEATURES)
                df_single = extract_single_comment_features(comment_input, selected_tod, active_cols)

                # Execute timed inference with selected model
                t_start = time.perf_counter()
                decision = policy.evaluate(
                    pipeline=active_pipeline,
                    features_df=df_single,
                    raw_text=comment_input,
                    algorithm_name=selected_algo,
                    algorithm_track=algo_meta["track"],
                )
                latency_ms = (time.perf_counter() - t_start) * 1000.0

                st.subheader(f"🔍 Moderation Decision — {selected_algo}")

                # Metric KPI Row
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    if decision.supports_probability and decision.toxic_probability is not None:
                        st.metric("Toxic Probability P(Toxic)", f"{decision.toxic_probability:.1%}")
                    else:
                        st.metric("Decision Margin Score", f"{decision.decision_score:+.3f}")
                with m_col2:
                    if decision.supports_probability and decision.civil_probability is not None:
                        st.metric("Civil Probability P(Civil)", f"{decision.civil_probability:.1%}")
                    else:
                        st.metric("Boundary Cutoff", "0.000 (Hyperplane)")
                with m_col3:
                    st.metric("Binary Class", f"{decision.predicted_class} ({decision.predicted_label})")
                with m_col4:
                    st.metric("Inference Latency", f"{latency_ms:.2f} ms")

                res_col1, res_col2 = st.columns([2, 3])

                with res_col1:
                    score_label = (
                        f"{decision.toxic_probability:.1%} P(Toxic)"
                        if (decision.supports_probability and decision.toxic_probability is not None)
                        else f"Margin: {decision.decision_score:+.3f}"
                    )

                    if decision.decision_action == "APPROVED":
                        st.markdown(
                            f"""
                        <div class="pred-box pred-box-safe">
                            <h3 style="color: #34D399; margin: 0;">✅ APPROVED: SAFE</h3>
                            <p style="color: #94A3B8; margin: 6px 0 0 0;">Comment cleared safety cutoff for this model.</p>
                            <h2 style="color: #10B981; margin: 12px 0 0 0;">{score_label}</h2>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    elif decision.decision_action == "HELD FOR REVIEW":
                        st.markdown(
                            f"""
                        <div class="pred-box pred-box-warning">
                            <h3 style="color: #FBBF24; margin: 0;">🟡 HELD FOR REVIEW</h3>
                            <p style="color: #94A3B8; margin: 6px 0 0 0;">Ambiguous / Borderline. Escalated to human review.</p>
                            <h2 style="color: #F59E0B; margin: 12px 0 0 0;">{score_label}</h2>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                        <div class="pred-box pred-box-toxic">
                            <h3 style="color: #FB7185; margin: 0;">🔴 QUARANTINED</h3>
                            <p style="color: #94A3B8; margin: 6px 0 0 0;">Flagged for safety violation by {selected_algo}.</p>
                            <h2 style="color: #F43F5E; margin: 12px 0 0 0;">{score_label}</h2>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                with res_col2:
                    st.markdown("#### Operational Routing Action")
                    if decision.decision_action == "APPROVED":
                        st.success(f"🟢 **{decision.routing_tier}**: {decision.explanation}")
                    elif decision.decision_action == "HELD FOR REVIEW":
                        st.warning(f"🟡 **{decision.routing_tier}**: {decision.explanation}")
                    else:
                        st.error(f"🔴 **{decision.routing_tier}**: {decision.explanation}")

                    st.markdown(
                        f"**Epistemic Ambiguity Score:** `{decision.uncertainty_score:.2f}` "
                        "(1.0 = maximum ambiguity near decision boundary, 0.0 = decisive)"
                    )
                    if not decision.supports_probability:
                        st.caption(
                            "ℹ️ *Note on Linear SVC*: Computes geometric margin distance $w^T x + b$. "
                            "Platt probability scaling was disabled during training to preserve strict maximal-margin optimization."
                        )

                # Feature Vector Display
                st.markdown("#### Extracted Surface Feature Vector")
                feat_display = df_single.drop(columns=["comment_text"], errors="ignore").T
                st.dataframe(feat_display.rename(columns={0: "Feature Value"}), use_container_width=True)

                # ==========================================
                # SIDE-BY-SIDE 10-MODEL COMPARISON MODE
                # ==========================================
                if compare_all_mode:
                    st.markdown("---")
                    st.subheader("⚡ 10-Model Live Comparative Benchmark")
                    st.caption("Direct side-by-side comparison of every algorithm on this exact comment input.")

                    comparison_rows = []
                    for name in algo_catalog:
                        try:
                            m_payload = load_model_payload(name)
                        except Exception as err:
                            st.warning(f"Could not load {name}: {err}")
                            m_payload = None

                        if m_payload:
                            m_cols = m_payload.get("feature_columns", ALL_FEATURES)
                            m_single = extract_single_comment_features(comment_input, selected_tod, m_cols)
                            m_t0 = time.perf_counter()
                            m_dec = policy.evaluate(
                                pipeline=m_payload["pipeline"],
                                features_df=m_single,
                                raw_text=comment_input,
                                algorithm_name=name,
                                algorithm_track=algo_catalog[name]["track"],
                            )
                            m_lat = (time.perf_counter() - m_t0) * 1000.0

                            score_str = (
                                f"{m_dec.toxic_probability:.1%}"
                                if (m_dec.supports_probability and m_dec.toxic_probability is not None)
                                else f"Margin: {m_dec.decision_score:+.2f}"
                            )

                            action_icon = (
                                "✅ Approved" if m_dec.decision_action == "APPROVED"
                                else "🟡 Review" if m_dec.decision_action == "HELD FOR REVIEW"
                                else "🔴 Quarantined"
                            )

                            comparison_rows.append({
                                "Algorithm": name,
                                "Track": algo_catalog[name]["track"],
                                "Decision Action": action_icon,
                                "Predicted Class": f"Class {m_dec.predicted_class} ({m_dec.predicted_label})",
                                "Toxic Score / Margin": score_str,
                                "Latency (ms)": f"{m_lat:.2f} ms",
                                "Ambiguity": f"{m_dec.uncertainty_score:.2f}",
                            })

                    df_comp_table = pd.DataFrame(comparison_rows)
                    st.dataframe(df_comp_table, use_container_width=True)
                    st.caption("Notice how different algorithms express varying sensitivity and trade-offs on the same input.")


# ==========================================
# 6. REPORTS & GOVERNANCE
# ==========================================
elif navigation == "📑 Reports & Governance":
    st.title("📑 Reports & Data Exports")
    st.markdown("Download verified CSV evaluation artifacts, metric comparisons, and model outputs.")

    st.subheader("📥 Export Evaluated Datatables")
    r_col1, r_col2 = st.columns(2)

    with r_col1:
        if "surface_vs_hybrid" in reports and not reports["surface_vs_hybrid"].empty:
            st.download_button(
                label="📄 Download Baseline vs Production Comparison (CSV)",
                data=reports["surface_vs_hybrid"].to_csv(index=False),
                file_name="model_comparison_surface_vs_hybrid.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if "classification" in reports and not reports["classification"].empty:
            st.download_button(
                label="📄 Download 10-Model Classification Comparison (CSV)",
                data=reports["classification"].to_csv(index=False),
                file_name="classification_10_models_comparison.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if "tuning" in reports and not reports["tuning"].empty:
            st.download_button(
                label="📄 Download Hyperparameter Tuning Comparison (CSV)",
                data=reports["tuning"].to_csv(index=False),
                file_name="hyperparameter_tuning_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with r_col2:
        if "clustering" in reports and not reports["clustering"].empty:
            st.download_button(
                label="📄 Download Clustering Metrics Table (CSV)",
                data=reports["clustering"].to_csv(index=False),
                file_name="clustering_metrics_comparison.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if "linkage" in reports and not reports["linkage"].empty:
            st.download_button(
                label="📄 Download Hierarchical Linkage Comparison (CSV)",
                data=reports["linkage"].to_csv(index=False),
                file_name="hierarchical_linkage_comparison.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if "roc_curves" in reports and not reports["roc_curves"].empty:
            st.download_button(
                label="📄 Download ROC-AUC Coordinates Data (CSV)",
                data=reports["roc_curves"].to_csv(index=False),
                file_name="roc_curves_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
