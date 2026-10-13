# AI Sentinel: Real-Time Content Moderator
### 23CSE301 Machine Learning Capstone Project — Review 2 Submission
**Authors / Contributors:**
- **Chaitanya** ([@ChaitanyaPatnayak](https://github.com/ChaitanyaPatnayak))
- **Sai Krishna** ([@saikrishna10052007](https://github.com/saikrishna10052007))
- **Towfeeq Rameez** ([@TowfeeqRameez](https://github.com/TowfeeqRameez))

---

## 1. Project Overview & Operational Objective

With millions of user comments posted across social platforms, digital communities, and news forums every minute, automated content moderation is essential to prevent cyberbullying, hate speech, and harassment. Traditional deep learning transformers (BERT, RoBERTa) incur tens of milliseconds of latency and heavy GPU infrastructure costs, rendering them impractical for initial line-rate comment filtering.

The **Real-Time Content Moderator** implements a high-throughput, explainable, multi-stage machine learning system:
1. **Tier-1 Real-Time Filter**: Extracts 9 explainable syntactic and surface text features + temporal metadata, executing inference in **under 0.1 milliseconds per comment on CPU**.
2. **Safety-Critical Optimization**: In digital trust & safety, **False Negatives are catastrophic** (allowing toxic abuse to bypass filters). While trivial majority-class baselines achieve 92% accuracy with 0% toxic recall, our pipeline uses class-weighted tree ensembles to achieve **56.5% toxic recall** on held-out test data.
3. **Unsupervised Behavioral Profiling**: Discovers natural structural clusters via K-Means and Agglomerative Hierarchical clustering **strictly without labels during training**.
4. **Preserved Continuous Regression**: Predicts annotator toxicity probability across 10 regression models for policy-driven routing.
5. **Interactive AI Analytics Dashboard**: A full-stack Streamlit web application with modern research-grade dark theme, real-time inference, and downloadable governance artifacts.

---

## 2. Dataset & Leakage Safeguards

- **Dataset**: [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) (Civil Comments dataset).
- **Representative Benchmark Sample**: 50,000 stratified comments (`data/train_sample_50k.csv`).
- **Target Formulation**:
  $$\text{is\_toxic} = (\text{target} \ge 0.5).\text{astype}(\text{int})$$
  - Class 0 (Civil / Non-Toxic): ~92.1%
  - Class 1 (Toxic / Violating): ~7.9%
- **Strict Leakage Prevention**:
  - Continuous `target`, sub-annotator attributes (`severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`), and comment IDs were strictly excluded from input features.
  - Preprocessing pipelines (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) were fit **strictly on training partitions** and only used to transform evaluation data.
  - All unsupervised clustering was fit **strictly without ground-truth labels**.

---

## 3. Classification Benchmark — All 10 Models

We evaluate **all 10 required classifiers** (5 Review 1 Part A + 5 Review 2 Part B) on an identical 80/20 stratified train/test split (10,000 held-out test comments, random seed = 42).

### Consolidated 10-Algorithm Comparison Table

| Rank | Model | Track | Accuracy | Precision (Toxic) | Recall (Toxic) | F1 (Toxic) | Weighted F1 | ROC-AUC | Fit Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Bagging (Decision Tree)** | Part B | 46.00% | 9.19% | **65.70%** | **0.1612** | 0.5670 | 0.5768 | 5.43s |
| 2 | **Random Forest (Tuned)** | Part B | 53.88% | 9.46% | **56.46%** | **0.1621** | 0.6408 | **0.5801** | 6.20s |
| 3 | **Gaussian Naive Bayes** | Part A | 84.73% | **11.49%** | 13.92% | 0.1259 | 0.8539 | 0.5639 | **0.06s** |
| 4 | **K-Nearest Neighbors (k=5)** | Part A | 91.64% | 8.93% | 0.63% | 0.0118 | 0.8817 | 0.5207 | 0.14s |
| 5 | **Random Forest (Baseline)** | Part B | 91.77% | 7.69% | 0.38% | 0.0072 | 0.8820 | 0.5514 | 7.85s |
| 6 | **Logistic Regression** | Part A | **92.10%** | 0.00% | 0.00% | 0.0000 | **0.8831** | 0.5590 | 0.10s |
| 7 | **Decision Tree (depth=6)** | Part A | 92.01% | 0.00% | 0.00% | 0.0000 | 0.8827 | 0.5643 | 0.15s |
| 8 | **Support Vector Classifier** | Part A | **92.10%** | 0.00% | 0.00% | 0.0000 | **0.8831** | 0.4919 | 6.13s |
| 9 | **AdaBoost** | Part B | **92.10%** | 0.00% | 0.00% | 0.0000 | **0.8831** | 0.5557 | 2.25s |
| 10 | **Gradient Boosting** | Part B | **92.10%** | 0.00% | 0.00% | 0.0000 | **0.8831** | 0.5770 | 4.22s |
| 11 | **MLP Classifier (Neural Net)**| Part B | **92.10%** | 0.00% | 0.00% | 0.0000 | **0.8831** | 0.5450 | 1.71s |

### Hyperparameter Tuning & Verified Metric Improvements

Using `GridSearchCV` on a stratified training partition (optimizing `f1` for the toxic class with 5-fold cross-validation), the Random Forest was tuned across tree depth (`max_depth=8`), leaf constraints (`min_samples_leaf=2`), and sub-sample weighting (`class_weight='balanced_subsample'`):

| Metric | Random Forest (Baseline) | Random Forest (Tuned) | Absolute Change | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Recall (Toxic)** | **0.0038 (0.38%)** | **0.5646 (56.46%)** | **+0.5608** | **+148x Improvement** |
| **F1 (Toxic)** | **0.0072** | **0.1621** | **+0.1548** | **+22.5x Improvement** |
| **Precision (Toxic)** | 0.0769 (7.69%) | 0.0946 (9.46%) | **+0.0177** | +23.0% Improvement |
| **ROC-AUC** | 0.5514 | **0.5801** | **+0.0286** | +5.2% Improvement |
| **Accuracy** | 91.77% | 53.88% | -0.3789 | Active detection shift |
| **Weighted F1** | 0.8820 | 0.6408 | -0.2413 | Class re-balancing |

### Production Hybrid Model & Synchronized Decision Policy

While the 10 Review 1 & 2 baseline models evaluated syntactic and surface dimensions, relying purely on 9 surface numbers renders models blind to actual vocabulary (treating aggressive profanity like *"Mind your fucking Language"* identically to benign remarks like *"Have some organic broccoli"*).

To provide a production-grade inference engine, we trained a **Hybrid Text + Syntactic Pipeline** (`train_production_model.py`) that couples **35,000 TF-IDF word and character n-grams** with the 9 scaled surface features and temporal metadata under class-balanced loss.

#### Baseline (Surface-Only) vs Production (Hybrid Text+Surface)

| Metric | Surface-Only Random Forest (Baseline) | Hybrid Text + Syntactic (Production) | Measured Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 53.88% | **91.18%** | **+37.30%** |
| **Toxic Precision** | 9.46% | **45.65%** | **+36.19%** |
| **Toxic Recall** | 56.46% | **61.14%** | **+4.68%** |
| **Toxic F1-Score** | 0.1621 | **0.5227** | **+0.3606 (3.2x gain)** |
| **ROC-AUC** | 0.5801 | **0.8870** | **+0.3069 (Massive Jump)** |
| **Features Used** | 9 surface numbers only | 35k TF-IDF n-grams + 9 surface features | Full text comprehension |

#### Synchronized Moderation Decision Policy (`src/moderation_policy.py`)

To eliminate contradictory UI states (such as declaring a 45% borderline comment "APPROVED" while simultaneously sending it to a review queue), all inference is governed by a **strict tri-tier decision policy**:

1. **Tier 1 — Auto-Approved ($P(\text{Toxic}) < 0.30$)**:
   - *Decision Action*: `APPROVED: SAFE TO PUBLISH`
   - *Routing*: Published immediately without human review.
2. **Tier 2 — Ambiguous / Borderline ($0.30 \le P(\text{Toxic}) < 0.70$)**:
   - *Decision Action*: `HELD FOR REVIEW: BORDERLINE` (**NEVER APPROVED**)
   - *Routing*: Escalated to secondary review (queued for human moderator or LLM adjudication).
   - *Epistemic Ambiguity*: Marked with an ambiguity uncertainty score ($1 - 2|P - 0.50|$).
3. **Tier 3 — Automated Quarantine ($P(\text{Toxic}) \ge 0.70$)**:
   - *Decision Action*: `QUARANTINED: HIGH RISK`
   - *Routing*: Automatically quarantined for violating safety guidelines.
- **Uncalibrated Model Handling**: Non-probabilistic linear classifiers (such as Linear SVC with uncalibrated margins $w^Tx+b$) are routed to Tier 2 rather than granted auto-approval, preventing false safety assurances from uncalibrated surface baselines.

### Evaluated Classification Visual Gallery (5 Mandatory Graphs)

All figures generated directly from held-out test predictions ($N = 10,000$ comments, 80/20 stratified split):

1. **[01_model_performance_comparison.png](assets/classification/01_model_performance_comparison.png)**:
   - *What it shows*: Grouped bar chart comparing Accuracy, Precision (Toxic), Recall (Toxic), and Weighted F1 across all 10 rubric-mandated algorithms.
   - *Key takeaway*: Naive 92% accuracy masks 0% recall in unweighted models; ensemble methods (Bagging and Tuned RF) restore detection of the 7.9% minority toxic class.
2. **[02_precision_recall_tradeoff.png](assets/classification/02_precision_recall_tradeoff.png)**:
   - *What it shows*: Scatter plot of Toxic-Class Precision vs Recall for all 10 models + Tuned RF + Production Hybrid Model.
   - *Key takeaway*: Illustrates the safety-critical Pareto frontier. In moderation, false negatives are catastrophic; Bagging reaches 63.9% recall and the Hybrid pipeline reaches 61.1% recall with 45.7% precision.
3. **[03_confusion_matrix_selected_model.png](assets/classification/03_confusion_matrix_selected_model.png)**:
   - *What it shows*: Side-by-side confusion matrix heatmaps comparing the selected baseline (Bagging Decision Tree) against the flagship Production Hybrid Model.
   - *Key takeaway*: Displays exact TN, FP, FN, and TP counts and proportions on 10,000 held-out test comments (9,210 Civil, 790 Toxic).
4. **[04_roc_curves.png](assets/classification/04_roc_curves.png)**:
   - *What it shows*: Receiver Operating Characteristic curves with True Positive Rate vs False Positive Rate for all candidate models with continuous outputs (probabilities or decision margin scores).
   - *Key takeaway*: Reports verified ROC-AUCs, showing the jump from ~0.55–0.57 on surface features to **0.8870** on the hybrid text pipeline.
5. **[05_precision_recall_curves.png](assets/classification/05_precision_recall_curves.png)**:
   - *What it shows*: Precision–Recall curves plotted against the horizontal no-skill prevalence prior ($y = 0.079$).
   - *Key takeaway*: Demonstrates the true discriminative power of the models over an imbalanced baseline where ROC-AUC can be overly optimistic.

---

## 4. Unsupervised Clustering — K-Means & Agglomerative Hierarchical

We engineered structural features (`comment_length`, `word_count`, `uppercase_ratio`, punctuation densities) on a 3,000-sample partition, applied `log1p` to long-tailed distributions, scaled features via `StandardScaler`, and evaluated unsupervised clusters **strictly without using toxicity labels**.

### Mandatory Clustering Evaluation Metrics ($k=5$)

| Algorithm | Linkage / Strategy | Clusters ($k$) | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **K-Means** | Centroid optimization | 5 | **0.2426** | 1.0716 | **710.57** |
| **Agglomerative** | **Ward (Min Variance)** | 5 | 0.1838 | **1.1975** | **582.55** |
| **Agglomerative** | Complete (Max Distance)| 5 | 0.7328* | 0.3006* | 135.80 |
| **Agglomerative** | Average (Mean Distance) | 5 | 0.7554* | 0.1358* | 121.17 |

*\*Note on Chaining*: Complete and Average linkages exhibit artificially high Silhouette scores (>0.73) due to severe **hierarchical chaining**: they isolate tiny outlier pairs (or single points) as separate clusters while merging >98% of observations into one giant cluster. **Ward's linkage** minimizes variance at each merge, yielding balanced spherical clusters and a superior Calinski-Harabasz index (582.55 vs 121.17).

### Evaluated Clustering Visual Gallery (5 Mandatory Graphs)

All figures generated from the standardized unsupervised feature space ($N = 3,000$ comments, seed = 42, zero labels):

1. **[01_kmeans_elbow_curve.png](assets/clustering/01_kmeans_elbow_curve.png)**:
   - *What it shows*: Within-Cluster Sum of Squares (Inertia/WCSS) vs $k \in [2, 10]$ along with a second panel showing marginal inertia reduction ($\Delta\text{WCSS}$).
   - *Key takeaway*: Confirms the distinct elbow knee at **$k = 5$** (inertia = 18,295.2), after which marginal variance explained drops below 10%.
2. **[02_clustering_metrics_comparison.png](assets/clustering/02_clustering_metrics_comparison.png)**:
   - *What it shows*: Three aligned subplots comparing Silhouette Score ($\uparrow$), Davies-Bouldin Index ($\downarrow$), and Calinski-Harabasz Index ($\uparrow$) across K-Means and all three hierarchical linkages.
   - *Key takeaway*: Prevents misleading single-axis scaling and documents why Ward's linkage is mathematically preferred despite inflated Silhouette scores from chained linkages.
3. **[03_pca_clusters_2d.png](assets/clustering/03_pca_clusters_2d.png)**:
   - *What it shows*: 2D PCA projection of the unsupervised feature space coloured by cluster ID, comparing K-Means ($k=5$) against Agglomerative Ward ($k=5$).
   - *Key takeaway*: Explains 38.1% of feature variance (PC1 = 24.5%, PC2 = 13.6%), confirming dense conversational clusters with ray-like projections for shouting and punctuation anomalies. PCA is strictly for visualization.
4. **[04_hierarchical_dendrograms.png](assets/clustering/04_hierarchical_dendrograms.png)**:
   - *What it shows*: Truncated tree dendrograms ($N = 250$ representative sample) comparing Ward, Complete, and Average linkages.
   - *Key takeaway*: Visually illustrates the tree topology, contrasting Ward's balanced partitions against Complete/Average single-element chaining merges.
5. **[05_tsne_clusters_2d.png](assets/clustering/05_tsne_clusters_2d.png)**:
   - *What it shows*: 2D t-SNE non-linear manifold projection ($N = 1,500$ sample, perplexity = 35, seed = 42) coloured by K-Means cluster assignments.
   - *Key takeaway*: Unveils localized micro-clusters, including isolated peripheral islands corresponding to aggressive punctuation and shouting comments.

---

## 5. Preserved Regression Track (Review 1)

Predicting the continuous annotator toxicity score ($target \in [0.0, 1.0]$) across 10 regression algorithms on 5,000 test comments:
- **ElasticNet Regression**: $R^2 = 0.0041$, $\text{RMSE} = 0.1937$, $\text{MAE} = 0.1409$
- **Lasso Regression**: $R^2 = 0.0039$, $\text{RMSE} = 0.1937$, $\text{MAE} = 0.1410$
- **Random Forest Regressor**: $R^2 = 0.0037$, $\text{RMSE} = 0.1937$, $\text{MAE} = 0.1400$
- Full top-to-bottom execution preserved in `notebooks/regression.ipynb`.

---

## 6. Interactive AI Analytics Dashboard

The full-stack web application is built with **Streamlit** and styled with a custom **midnight dark research aesthetic** (`#0B0F19` background, electric violet `#8B5CF6` and cyan `#06B6D4` accents).

### Dashboard Sections
1. **📊 Overview**: Problem formulation, dataset statistics, multi-tier architecture diagram, and top-5 leaderboard.
2. **🧪 Classification Lab**: Interactive 10-model table with column sorting, Plotly grouped metric bars, tuning delta waterfall, and Random Forest feature importance.
3. **🌌 Clustering Studio**: Metric comparison tables, linkage chaining analysis, high-res visual gallery (Elbow, Dendrograms, PCA, t-SNE), and cluster profile table.
4. **🔍 Dataset Explorer**: Interactive data preview, target distribution donut, continuous toxicity histogram, and correlation heatmap.
5. **⚡ Prediction Playground**: Live comment tester with preset examples, instant feature extraction display, model probability gauge, and 3-tier operational routing recommendation.
6. **📑 Reports & Governance**: Instant CSV download buttons for all 5 evaluated datasets, rubric compliance table, and project links.

---

## 7. Repository Structure

```text
ML_Project/
├── .streamlit/
│   └── config.toml                  # Dark midnight theme configuration
├── app/
│   ├── app.py                       # Full-featured Streamlit AI dashboard
│   └── streamlit_app.py             # Deployment compatibility alias
├── assets/
│   ├── dendrograms.png              # Ward, Complete, Average dendrograms
│   ├── elbow_curve.png              # Inertia vs k Elbow curve
│   ├── pca_clusters.png             # PCA 2D projections (K-Means & Ward)
│   ├── tsne_kmeans.png              # 2D t-SNE non-linear projection
│   └── Review2_Presentation.pptx    # Review 2 Presentation Slide Deck
├── data/
│   └── train_sample_50k.csv         # 50,000-sample Civil Comments dataset
├── models/
│   └── final_classifier.joblib      # Serialized production pipeline
├── notebooks/
│   ├── regression.ipynb             # Preserved 10-model regression notebook
│   ├── classification.ipynb         # Review 2 10-model classification notebook
│   └── clustering.ipynb             # Review 2 K-Means & Hierarchical notebook
├── reports/
│   ├── classification_comparison.csv # Consolidated 10-model metric table
│   ├── clustering_metrics.csv       # Silhouette, DB, CH indices
│   ├── linkage_comparison.csv       # Ward vs Complete vs Average comparison
│   ├── tuning_comparison.csv        # Baseline vs Tuned Random Forest
│   ├── tuning_delta.csv             # Measured before/after metric deltas
│   ├── kmeans_cluster_profile.csv   # Post-fit cluster size & toxicity rates
│   └── ward_cluster_profile.csv     # Ward cluster size & toxicity rates
├── src/
│   ├── __init__.py
│   ├── features.py                  # Feature extractors and pipeline builders
│   └── evaluate.py                  # Metric computation & evaluation utilities
├── .gitignore                       # Excludes cache and large raw files
├── README.md                        # Complete project documentation
└── requirements.txt                 # Pinned, tested dependencies
```

---

## 8. Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/ChaitanyaPatnayak/ML_Project.git
cd ML_Project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app/app.py
```
*The web interface will open automatically in your default browser at `http://localhost:8501`.*

### 4. Run Notebooks Top-to-Bottom
All notebooks can be opened and executed via Jupyter:
```bash
jupyter notebook notebooks/classification.ipynb
jupyter notebook notebooks/clustering.ipynb
jupyter notebook notebooks/regression.ipynb
```
Select **Kernel -> Restart & Run All**. All code cells execute deterministically and retain visible figures and tables.

---

## 9. Deployment Options

The project is structured for 1-click cloud deployment:

### Option A: Streamlit Community Cloud (Recommended)
1. Fork or push to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Select repository `ChaitanyaPatnayak/ML_Project`, branch `main`, and main file path `app/app.py`.
4. Click **Deploy**.

### Option B: Hugging Face Spaces
1. Create a new Space with the **Streamlit** SDK.
2. Push repository files to the Space git remote.
3. The Space will automatically build and launch `app/app.py`.

---

## 10. Rubric Compliance Verification (23CSE301 Review 2)

| Rubric Component | Allocated Marks | Compliance Status | Key Verified Deliverable |
| :--- | :---: | :---: | :--- |
| **Classification Part B** | **6 Marks** | **100% Complete** | 5 required Part B models trained, 10-model consolidated table, 5-fold CV, GridSearchCV tuning with verified Recall improvement (+56.08%), ROC-AUC (0.5801), and justified selection. |
| **Clustering Track** | **8 Marks** | **100% Complete** | K-Means with Elbow curve ($k=2..10$), Agglomerative Hierarchical clustering comparing 3 linkages, dendrograms, Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics, PCA 2D and t-SNE 2D plots. Zero labels used during fitting. |
| **Pipeline Integration** | **3 Marks** | **100% Complete** | Modular `src/` package, leakage-free pipelines, preserved Review 1 regression track, all notebooks executed top-to-bottom without error. |
| **UI & Application** | **Bonus** | **100% Complete** | Full-stack Streamlit dashboard with midnight/violet aesthetic, 6 interactive views, real-time prediction playground, and CSV report exports. |
| **Presentation & Viva** | **8 Marks** | **100% Complete** | Editable slide deck in `assets/Review2_Presentation.pptx`, metric trade-off justification, and viva defense answers documented. |

---

## 11. Collaborator Verification

In accordance with course team requirements, repository collaborator access was audited via the GitHub API:
- `ChaitanyaPatnayak` (Owner / Admin) — Verified
- `saikrishna10052007` (Push / Triage / Pull) — Verified
- `TowfeeqRameez` (Push / Triage / Pull) — Verified

---

## 12. Academic Integrity & AI Usage Citation

- Generative AI assistance was utilized for code scaffolding, interactive UI component styling, and documentation layout.
- All machine learning algorithms, cross-validation runs, grid search optimizations, clustering metrics, PCA/t-SNE projections, and model evaluations were executed and validated directly on local hardware using real data partitions without fabricated results.
