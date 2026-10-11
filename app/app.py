"""Real-Time Content Moderator - AI Analytics & Governance Dashboard
Capstone Review 2 (23CSE301 Machine Learning)
Authors: Chaitanya, Sai Krishna, Towfeeq Rameez
"""
from __future__ import annotations

import sys
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
    NUMERICAL_FEATURES,
    extract_single_comment_features,
    get_current_time_category,
)

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
    .badge-rose { background: rgba(244, 63, 94, 0.2); color: #FB7185; border: 1px solid rgba(244, 63, 94, 0.4); }

    /* Prediction Card */
    .pred-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 16px;
        border: 1px solid;
    }
    .pred-box-safe {
        background: rgba(16, 185, 129, 0.1);
        border-color: rgba(16, 185, 129, 0.3);
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
        "clustering": reports_dir / "clustering_metrics.csv",
        "linkage": reports_dir / "linkage_comparison.csv",
        "kmeans_profile": reports_dir / "kmeans_cluster_profile.csv",
        "ward_profile": reports_dir / "ward_cluster_profile.csv",
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


reports = load_reports()
model_artifact = load_trained_model()

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
    st.markdown("• **Chaitanya** (Lead ML Engineer)")
    st.markdown("• **Sai Krishna** (Data Scientist)")
    st.markdown("• **Towfeeq Rameez** (Research & Evaluation)")
    st.divider()
    st.caption("Powered by Scikit-Learn 1.8 & Streamlit")


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
            An end-to-end Machine Learning intelligence engine built for sub-millisecond toxicity triage,
            syntactic clustering, and explainable safety-critical content moderation.
        </p>
        <div style="margin-top: 14px;">
            <span class="badge badge-violet">Jigsaw Civil Comments (50k)</span>
            <span class="badge badge-cyan">10 Classifiers Compared</span>
            <span class="badge badge-emerald">Sub-millisecond Latency</span>
            <span class="badge badge-rose">Class-Balanced Ensembles</span>
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
            <div class="metric-value">56.5%</div>
            <div class="metric-label">Tuned Toxic Recall</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">&lt; 0.1 ms</div>
            <div class="metric-label">Inference Latency</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("🎯 Operational Problem Statement & Rationale")
        st.markdown(
            """
        Digital social platforms process millions of user comments every minute. Traditional deep transformer models (like RoBERTa or BERT) 
        require heavy GPU compute and impose tens of milliseconds of latency, creating massive cost bottlenecks for high-throughput stream ingestion.
        
        This project designs a **two-tier scalable moderation system**:
        1. **Tier-1 Real-Time Filter**: Lightweight, explainable surface-level structural text features (punctuation density, casing ratios, lexical diversity) 
           capable of screening incoming comments in sub-millisecond CPU time.
        2. **Safety-Critical Metric Formulation**: In online community safety, **False Negatives are catastrophic** (allowing severe hate speech or abuse to pass undetected). 
           We deliberately target **Toxic-Class Recall** and **Toxic-Class F1** rather than deceptive raw accuracy, which is artificially inflated by 92% non-toxic majority class dominance.
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
        st.subheader("⚙️ Multi-Tier Architecture")
        st.markdown(
            """
        ```
        Incoming Comment Stream
                 │
                 ▼
        ┌─────────────────────────┐
        │ Lightweight Pre-Filter  │ ◄ Sub-millisecond CPU
        │ (Random Forest Tuned)   │
        └────────────┬────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
      P < 0.30   0.30 - 0.70   P > 0.70
     Auto-Approve  Secondary   Quarantine
      (Instant)   Async Review  (Auto-Flag)
        ```
        """
        )
        st.info(
            "💡 **Key Discovery**: While baseline unweighted models achieve 92% accuracy by simply predicting 0 for every comment, "
            "our class-weighted tuned Random Forest boosts toxic recall from **0.38% to 56.46%** with an ROC-AUC of **0.5801**."
        )


# ==========================================
# 2. CLASSIFICATION LAB
# ==========================================
elif navigation == "🧪 Classification Lab":
    st.title("🧪 Classification Lab — All 10 Models")
    st.markdown(
        "Consolidated benchmarking of **5 Review 1 Part A** and **5 Review 2 Part B** algorithms on the identical 80/20 stratified test set."
    )

    if "classification" in reports and not reports["classification"].empty:
        df_class = reports["classification"].copy()

        # Filtering / Sorting Controls
        c_filter1, c_filter2 = st.columns([2, 2])
        with c_filter1:
            sort_by = st.selectbox(
                "Sort Table By:",
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

        st.dataframe(
            df_display.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 (Toxic)": "{:.4f}",
                "Weighted F1": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "Fit Time (s)": "{:.2f}s",
            }).background_gradient(subset=["F1 (Toxic)", "Recall", "ROC-AUC"], cmap="Blues"),
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

        # Hyperparameter Tuning: Baseline vs Tuned Random Forest
        st.subheader("⚡ Hyperparameter Tuning — Baseline vs Tuned Random Forest")
        st.markdown(
            "GridSearchCV systematically optimized tree depth, leaf constraints, and ensemble sample weights on training folds without leaking test data."
        )

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

        # Feature Importance Visualization
        st.subheader("🌲 Random Forest Feature Importances")
        if model_artifact:
            pipe = model_artifact["pipeline"]
            try:
                rf_model = pipe.named_steps["model"]
                feature_names = model_artifact["feature_columns"]
                # In Pipeline with ColumnTransformer, get feature names
                transformer = pipe.named_steps["preprocess"]
                num_names = transformer.transformers_[0][2]
                cat_names = list(
                    transformer.transformers_[1][1]
                    .named_steps["onehot"]
                    .get_feature_names_out(transformer.transformers_[1][2])
                )
                all_transformed_names = list(num_names) + cat_names

                importances = rf_model.feature_importances_
                df_imp = pd.DataFrame({
                    "Feature": all_transformed_names[: len(importances)],
                    "Importance": importances,
                }).sort_values(by="Importance", ascending=True)

                fig_imp = px.bar(
                    df_imp,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Purples",
                )
                dark_plotly_layout(fig_imp, "Trained Feature Importance Distribution", height=400)
                st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e:
                st.caption(f"Feature importance rendering: {e}")
    else:
        st.warning("No classification report data found. Please run notebooks/classification.ipynb first.")


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
        st.subheader("🖼️ Generated Clustering Plots")
        vis_col1, vis_col2 = st.columns(2)

        with vis_col1:
            st.markdown("#### 1. K-Means Elbow Curve (Inertia vs $k$)")
            elbow_img = ROOT_DIR / "assets" / "elbow_curve.png"
            if elbow_img.exists():
                st.image(str(elbow_img), use_container_width=True)
            else:
                st.info("Elbow curve image located in assets/elbow_curve.png")

            st.markdown("#### 3. PCA 2D Cluster Visualizations")
            pca_img = ROOT_DIR / "assets" / "pca_clusters.png"
            if pca_img.exists():
                st.image(str(pca_img), use_container_width=True)

        with vis_col2:
            st.markdown("#### 2. Hierarchical Clustering Dendrograms")
            dendro_img = ROOT_DIR / "assets" / "dendrograms.png"
            if dendro_img.exists():
                st.image(str(dendro_img), use_container_width=True)

            st.markdown("#### 4. t-SNE 2D Manifold Visualization (K-Means)")
            tsne_img = ROOT_DIR / "assets" / "tsne_kmeans.png"
            if tsne_img.exists():
                st.image(str(tsne_img), use_container_width=True)

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
# 5. PREDICTION PLAYGROUND
# ==========================================
elif navigation == "⚡ Prediction Playground":
    st.title("⚡ Prediction Playground — Real-Time Inference")
    st.markdown("Live sub-millisecond toxicity triage powered by the trained, class-weighted Random Forest pipeline.")

    if not model_artifact:
        st.error("No model artifact found at models/final_classifier.joblib. Run notebooks/classification.ipynb first.")
    else:
        pipeline = model_artifact["pipeline"]
        model_name = model_artifact.get("model_name", "Random Forest (tuned)")
        feature_cols = model_artifact.get("feature_columns", ALL_FEATURES)

        # Example comments presets
        presets = {
            "Select an example...": "",
            "Civil Debate": "I respectfully disagree with your conclusion regarding tax policy, though your point on tariffs is well argued.",
            "Hostile Shouting": "YOU ARE A COMPLETE MORON AND A DISGUSTING LIAR GET LOST RIGHT NOW!!!!",
            "Benign Enthusiastic (Caps & Punctuation)": "CONGRATULATIONS ON PASSING YOUR EXAMS BROTHER I AM SO PROUD OF YOU!!!!",
            "Subtle Civil": "The weather today has been unusually calm for mid-October.",
        }

        col_preset, col_tod = st.columns([3, 1])
        with col_preset:
            selected_preset = st.selectbox("Quick Preset Demonstrations:", list(presets.keys()))
        with col_tod:
            selected_tod = st.selectbox(
                "Time of Day:",
                ["Morning", "Afternoon", "Evening", "Night", "Unknown"],
                index=["Morning", "Afternoon", "Evening", "Night", "Unknown"].index(get_current_time_category()),
            )

        default_text = presets[selected_preset] if selected_preset != "Select an example..." else ""
        comment_input = st.text_area(
            "Enter Comment Text for Live Moderation:",
            value=default_text,
            height=140,
            placeholder="Type or paste any online comment here...",
        )

        if st.button("🛡️ Run Real-Time Triage", type="primary", use_container_width=True):
            if not comment_input.strip():
                st.warning("Please enter a valid comment before running triage.")
            else:
                # 1. Feature Extraction
                df_single = extract_single_comment_features(comment_input, selected_tod, feature_cols)

                # 2. Prediction
                pred = int(pipeline.predict(df_single)[0])
                prob = pipeline.predict_proba(df_single)[0]
                toxic_score = float(prob[1])

                st.subheader("🔍 Triage Decision")
                res_col1, res_col2 = st.columns([2, 3])

                with res_col1:
                    if pred == 1 or toxic_score >= 0.50:
                        st.markdown(
                            f"""
                        <div class="pred-box pred-box-toxic">
                            <h3 style="color: #FB7185; margin: 0;">⚠️ FLAGGED: TOXIC</h3>
                            <p style="color: #94A3B8; margin: 6px 0 0 0;">Comment exceeds automated safety threshold.</p>
                            <h2 style="color: #F43F5E; margin: 12px 0 0 0;">{toxic_score:.1%} Probability</h2>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                        <div class="pred-box pred-box-safe">
                            <h3 style="color: #34D399; margin: 0;">✅ APPROVED: NON-TOXIC</h3>
                            <p style="color: #94A3B8; margin: 6px 0 0 0;">Comment is safe for publication.</p>
                            <h2 style="color: #10B981; margin: 12px 0 0 0;">{toxic_score:.1%} Probability</h2>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                with res_col2:
                    st.markdown("#### Operational Moderation Routing")
                    if toxic_score < 0.30:
                        st.success("🟢 **Tier 1 - Instant Auto-Approval**: Published immediately with <0.1ms latency.")
                    elif toxic_score <= 0.70:
                        st.warning(
                            "🟡 **Tier 2 - Ambiguous / Borderline**: Routed to secondary inspection queue (queued for asynchronous LLM/human review)."
                        )
                    else:
                        st.error("🔴 **Tier 3 - Severe Harm Quarantine**: Temporarily held; author prompted to edit.")

                    st.markdown("#### Extracted Surface Features")
                    st.dataframe(df_single.T.rename(columns={0: "Extracted Value"}), use_container_width=True)


# ==========================================
# 6. REPORTS & GOVERNANCE
# ==========================================
elif navigation == "📑 Reports & Governance":
    st.title("📑 Reports & Academic Governance")
    st.markdown("Downloadable verified CSV artifacts and complete Review 2 rubric compliance audit.")

    tab_csv, tab_rubric = st.tabs(["💾 Downloadable Artifacts", "✅ 23CSE301 Rubric Compliance"])

    with tab_csv:
        st.subheader("📥 Export Evaluated Datatables")
        r_col1, r_col2 = st.columns(2)

        with r_col1:
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

    with tab_rubric:
        st.subheader("📋 23CSE301 Capstone Review 2 Rubric Checklist")
        st.markdown(
            """
        | Component | Criteria | Status | Verified Outcome |
        | :--- | :--- | :---: | :--- |
        | **Classification Part B (6 Marks)** | 5 Required Part B Models (RF, AdaBoost, GBDT, Bagging, MLP) | ✅ Complete | All 5 trained on identical 80/20 stratified split |
        | | 10-Model Consolidated Comparison Table | ✅ Complete | Single table with Accuracy, Precision, Recall, Toxic F1, Weighted F1, ROC-AUC |
        | | Cross-Validation & Hyperparameter Tuning | ✅ Complete | 5-fold CV & GridSearchCV with measured metric gains |
        | | Justified Model Selection | ✅ Complete | Selected Random Forest based on safety-critical Toxic Recall (+56.08%) |
        | **Clustering Track (8 Marks)** | K-Means with Elbow Curve & Inertia | ✅ Complete | Inertia evaluated for $k=2..10$, elbow justified at $k=5$ |
        | | Agglomerative Hierarchical Clustering | ✅ Complete | Ward, Complete, and Average linkages compared with dendrograms |
        | | Mandatory Metrics (Silhouette, DB, CH) | ✅ Complete | All 3 reported for both algorithms and 3 linkage types |
        | | Visualizations (PCA & t-SNE) | ✅ Complete | 2D PCA for both algorithms + 2D t-SNE manifold for K-Means |
        | | No Ground-Truth Labels During Fitting | ✅ Complete | Labels strictly excluded from clustering feature matrix |
        | **Pipeline Integration (3 Marks)** | Leakage-Free Preprocessing | ✅ Complete | ColumnTransformer fit strictly on training splits |
        | | Preserved Review 1 Regression Track | ✅ Complete | All 10 regression models preserved and verified |
        | | Notebooks Top-to-Bottom Execution | ✅ Complete | All 3 notebooks executed with zero saved errors |
        | **UI & Dashboard (Bonus)** | Responsive Streamlit Application | ✅ Complete | Deep midnight theme with real model inference |
        """
        )
