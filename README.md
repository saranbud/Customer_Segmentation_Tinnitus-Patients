# Tinnitus Patient Segmentation

End-to-end unsupervised patient segmentation pipeline built on Databricks, using a 10,000-row research-calibrated synthetic tinnitus cohort. The project identifies clinically meaningful patient subgroups, validates their stability, and translates results into company-facing engagement and support strategies.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Segment Results](#segment-results)
- [Technical Stack](#technical-stack)
- [Delta Table Outputs](#delta-table-outputs)
- [Dashboard](#dashboard)
- [How to Run](#how-to-run)
- [Environment](#environment)
- [Limitations](#limitations)

---

## Overview

I actually suffer from this hahahah and Its a super personal project that I wanted to work on . Tinnitus is a heterogeneous condition. Patients differ substantially across tinnitus severity, hearing-loss burden, psychological distress, sleep disruption, duration of symptoms, noise exposure history, and treatment engagement. Treating the population as a single cohort obscures clinically important subgroups and leads to undifferentiated outreach and support strategies.

This project applies K-Means clustering with full preprocessing, validation, and interpretive profiling to identify four distinct patient segments from a 10,000-patient synthetic cohort. The output is a segment model with named, ranked, and statistically validated patient groups, saved to a Unity Catalog Delta Lakehouse and visualized in a live AI/BI dashboard.

**Business objectives addressed:**
- Identify high-need patient subgroups for targeted outreach
- Quantify how burden, quality of life, and treatment engagement vary by segment
- Rank segments by composite clinical and business priority
- Provide a reproducible, deployable segmentation pipeline

---

## Project Structure

```
Customer_Segmentation_Tinnitus-Patients/
├── README.md
├── Tinnitus_Patient_Segmentation_Research_Calibrated   # Primary analytical notebook (35 sections)
└── Tinnitus_Patient_Segmentation_Lakehouse_ML          # Spark ML / medallion pipeline notebook
```

### Notebook 1 — Research-Calibrated Deep Dive

**`Tinnitus_Patient_Segmentation_Research_Calibrated`**

The primary deliverable. A 108-cell, 35-section analytical notebook covering:

| Phase | Sections | Description |
|---|---|---|
| Setup | 1–6 | Problem statement, imports, data load, quality audit, cohort definition |
| EDA | 7–8 | Numerical and categorical exploration, relationship analysis |
| Feature engineering | 9–16 | 10 composite features, leakage prevention, selection, imputation, OHE, scaling, 26-column model matrix |
| Clustering | 17–21 | K selection, KMeans fit, PCA visualisation, hierarchical comparison, stability (mean ARI 0.976) |
| Interpretation | 22–27 | Profiles, cohort z-scores, statistical tests, naming, priority ranking, treatment response |
| Reporting | 28–30 | Visualisations, sensitivity analysis, 6 Delta tables saved |
| Conclusions | 31–35 | Findings, recommendations, executive summary, limitations, reproducibility checklist |

All Python cells have been executed and all interpretation cells contain real numbers from those executions.

### Notebook 2 — Lakehouse ML Pipeline

**`Tinnitus_Patient_Segmentation_Lakehouse_ML`**

A 150+ cell distributed pipeline that demonstrates Databricks-native tooling:
- Bronze → Silver → Gold medallion architecture
- PySpark ML (`pyspark.ml`) for feature engineering and clustering
- Unity Catalog Delta writes throughout
- Spark-native feature transformers and evaluators

---

## Dataset

| Property | Value |
|---|---|
| Source | Research-calibrated synthetic tinnitus cohort |
| Rows | 10,000 patients |
| Columns | 50 clinical, behavioural, and demographic features |
| Format | CSV, Unity Catalog Volume |
| Path | `/Volumes/tinnitus_data/default/tinnitus-data/research_calibrated_tinnitus_cohort_10000.csv` |
| Duplicates | 0 rows, 0 patient IDs |
| Missingness | 2,648 cells imputed across 15 modelling features |

Key feature domains: age, tinnitus duration and type, onset, hearing loss severity, THI score, TFI score, anxiety/depression/sleep/quality-of-life scores, noise exposure, comorbidity count, treatment history, geographic coordinates.

All data is **synthetic** and was generated for methodology development purposes only.

---

## Segment Results

Four patient segments were identified via K-Means (K=4, clinical override over silhouette-optimal K=2). All differences were statistically significant (Kruskal-Wallis p < 0.0001 for all 17 numerical features).

| Priority | Cluster | Segment | n | % | THI | QoL |
|---|---|---|---|---|---|---|
| 1 | 3 | Severe Multi-Domain Burden | 1,798 | 17.98% | 72 | 33.8 |
| 2 | 2 | Psychologically Burdened Younger | 2,749 | 27.49% | 52 | 47.3 |
| 3 | 0 | Older Chronic Hearing-Impaired | 2,876 | 28.76% | 46 | 48.6 |
| 4 | 1 | Low-Burden Adaptive Copers | 2,577 | 25.77% | 26 | 64.2 |

### Key distinguishing characteristics

**Severe Multi-Domain Burden** — highest psychological burden (psych score 67.4), worst quality of life (33.8), 70.6% sleep disorder prevalence, comorbidity count 2, longest follow-up (9.0 months), highest CBT use (31%) and medication use (27.3%).

**Psychologically Burdened Younger** — youngest segment (median age 41), psychologically driven burden (anxiety 45.9), no dominant hearing loss, highest mobile app adoption (33.4%) and highest therapy adherence (45.3%).

**Older Chronic Hearing-Impaired** — oldest segment (median age 66), 91.3% hearing loss prevalence, hearing burden dominant, longest tinnitus duration, highest hearing-aid use (32.3%). Psychological burden below average despite moderate tinnitus severity.

**Low-Burden Adaptive Copers** — best quality of life (64.2), lowest tinnitus severity (THI 26), all burden dimensions below cohort average, lowest therapy adherence (34.0%). Primary target for prevention and maintenance programs.

### Model quality

| Metric | Value |
|---|---|
| Silhouette score | 0.1187 |
| Calinski-Harabasz | 1,546.43 |
| Davies-Bouldin | 2.060 |
| Stability mean ARI (5 seeds) | 0.9758 — HIGH |
| Stability min ARI | 0.9743 |
| Feature sensitivity ARI | 0.8108 — ROBUST |
| Hierarchical clustering ARI | 0.4549 (expected for mixed high-dim data) |
| KMeans inertia | 81,153.17 |

---

## Technical Stack

| Component | Technology |
|---|---|
| Platform | Databricks (Serverless Spark 4.1.0, AWS) |
| Modelling | scikit-learn 1.6.1 (KMeans, PCA, StandardScaler, ARI) |
| Data wrangling | pandas 2.2.3, numpy 2.1.3 |
| Statistics | scipy 1.15.3 (Kruskal-Wallis, chi-square, Spearman) |
| Distributed I/O | PySpark / Spark SQL |
| Storage | Delta Lake, Unity Catalog |
| Visualisation | matplotlib, seaborn (notebook); Databricks AI/BI Dashboard |
| Python | 3.12.3 |
| Random seed | 42 (fixed across all clustering, stability, and sensitivity runs) |

---

## Delta Table Outputs

All outputs are written to `tinnitus_data.default` using `.mode('overwrite').option('overwriteSchema', 'true')`, making the pipeline idempotent.

| Table | Rows | Description |
|---|---|---|
| `silver_tinnitus_segmentation_cohort` | 10,000 | Full analytical cohort with all engineered features and segment labels |
| `gold_tinnitus_patient_segments` | 10,000 | Patient-level segment assignments with 18 clinical and demographic features |
| `gold_tinnitus_segment_profiles` | 4 | Mean numerical profiles for each of the four patient segments |
| `gold_tinnitus_cluster_metrics` | 1 | K=4 silhouette, CH, DB, inertia, and stability ARI summary |
| `gold_tinnitus_cluster_stability` | 5 | Per-seed ARI vs seed-42 reference across 5 stability runs |
| `gold_tinnitus_segment_outcomes` | 4 | Post-clustering outcome means by segment (symptom improvement, follow-up, adherence) |
| `gold_tinnitus_patient_geo` | 10,000 | Patient coordinates with segment labels (for mapping and geographic analysis) |

---

## Dashboard

An AI/BI dashboard built from the Delta outputs includes:
- KPI counters: total patients, segment count, Severe burden size, best QoL score
- Geographic bubble map: city-level patient distribution, bubble size = patient count, color = dominant segment
- Horizontal bar charts: patient count, clinical/psychological/QoL burden, symptom improvement, therapy adherence, follow-up duration
- Cohort share pie chart

Segment color scheme (consistent across all visuals):
- Severe Multi-Domain Burden: `#C0392B`
- Psychologically Burdened Younger: `#E67E22`
- Older Chronic Hearing-Impaired: `#2980B9`
- Low-Burden Adaptive Copers: `#27AE60`

---

## How to Run

### Prerequisites
- Databricks workspace (Serverless CPU or Standard cluster, Spark 4.x)
- Unity Catalog with `tinnitus_data` catalog and `default` schema, with CREATE TABLE permissions
- Source CSV file at `/Volumes/tinnitus_data/default/tinnitus-data/research_calibrated_tinnitus_cohort_10000.csv`

### Execution order

```
1. Open Tinnitus_Patient_Segmentation_Research_Calibrated
2. Attach Serverless CPU compute
3. Run all cells top-to-bottom (Sections 1-35)
   - Section 3  : imports and configuration (defines RANDOM_SEED = 42, DATA_PATH, TBL_* names)
   - Section 7  : loads CSV into df_raw and pdf_cohort
   - Section 9  : feature engineering (composite scores added to pdf_cohort)
   - Sections 11-16 : preprocessing pipeline → X_model (10,000 x 26)
   - Section 17 : K selection (OPTIMAL_K = 4, clinical override)
   - Sections 18-21 : KMeans fit, PCA, hierarchical comparison, stability
   - Sections 22-27 : profiling, naming, priority ranking, treatment response
   - Sections 28-30 : visualisations, sensitivity, Delta saves
   - Sections 31-35 : narrative sections — do not need re-execution
```

### Configuration

All environment-specific settings are in a single cell (Section 3):

```python
RANDOM_SEED = 42
DATA_PATH = "/Volumes/tinnitus_data/default/tinnitus-data/research_calibrated_tinnitus_cohort_10000.csv"
CATALOG = "tinnitus_data"
SCHEMA = "default"
```

Change only these values to point the pipeline at a different catalog, schema, or source file.

### Notes
- `.cache()` is not supported on Serverless compute; the pipeline uses Delta checkpoints instead
- All Delta writes include `try/except` for safe re-execution
- `display()` is used throughout instead of `.show()` for Spark DataFrames
- The reproducibility checklist cell (Section 35) runs safely after a compute timeout — it falls back to documented default values rather than raising errors

---

## Environment

```
Platform : Databricks Serverless (AWS)
Spark    : 4.1.0
Python   : 3.12.3
pandas   : 2.2.3
numpy    : 2.1.3
scikit-learn : 1.6.1
scipy    : 1.15.3
seaborn  : 0.13.2
```

No additional package installation is required. All dependencies are available in the Databricks Serverless runtime.

---

## Limitations

1. **Synthetic data** — the dataset was generated using realistic clinical distributions for methodology development. Results are not clinical evidence and cannot be used for patient diagnosis, treatment planning, or causal inference.
2. **K=4 is a clinical override** — the pure statistical metrics favoured K=2. The selection of K=4 reflects a clinical interpretability judgement and introduces analyst subjectivity.
3. **Moderate silhouette score** — 0.1187 is typical for mixed high-dimensional clinical data with overlapping continuous distributions. Segments are statistically real but not sharply bounded.
4. **Algorithm sensitivity** — hierarchical clustering ARI of 0.4549 indicates the segment structure depends to some degree on the choice of algorithm. Gaussian Mixture Models or HDBSCAN may produce different boundaries.
5. **Cross-sectional design** — the analysis captures a single point-in-time snapshot and cannot model patient transitions between segments over time.
6. **Observational treatment patterns** — treatment-response differences reflect programmed associations in the synthetic simulation, not causal treatment effects.
7. **No external validation** — the segmentation has not been applied to an independent dataset.

