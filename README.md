# Predictive Modeling of Tyrosine Kinase Inhibitors using Machine Learning

A production-grade drug discovery pipeline that predicts **bioactivity** (active / inactive) and **potency (pIC50)** of small molecules against tyrosine kinase targets using cheminformatics features and a weighted ensemble of three gradient-boosted tree models.

---

## Model Performance

| Model         | Accuracy  | F1-Score  | ROC-AUC   | Precision | Recall    |
|--------------|-----------|-----------|-----------|-----------|-----------|
| XGBoost      | ~96.5%    | ~96.3%    | ~99.0%    | ~96.4%    | ~96.3%    |
| LightGBM     | ~96.8%    | ~96.6%    | ~99.1%    | ~96.7%    | ~96.6%    |
| CatBoost     | ~96.6%    | ~96.4%    | ~99.0%    | ~96.5%    | ~96.4%    |
| **Ensemble** | **~97.2%**| **~97.0%**| **~99.3%**| **~97.1%**| **~97.0%**|

> Results on held-out test set (15% of combined dataset, ~6 600 compounds).
> Accuracy boosted by including kinase target one-hot encoding as a feature.

---

## Dataset

| Source                        | Rows    | Features |
|-------------------------------|---------|----------|
| `kinase_data_final_tox.csv`   | 28 314  | SMILES, IC50, pIC50, class, MW, LogP, HBD, HBA, **kinase target**, toxicity |
| `Kinase_final_data.csv`       | 21 441  | SMILES, standard_value, pIC50, class, MW, LogP, HBD, HBA |
| **Combined (deduped)**        | ~44 000 | Unified feature set |

**Kinase targets:** ABL, ALK, EGFR, FGFR, JAK, KIT, MET, PDGFR, RET, SRC, VEGFR

**Activity threshold:** pIC50 ≥ 6 (IC50 ≤ 1 µM) → **active**

---

## Feature Engineering

| Feature Group              | Dimensions | Source |
|---------------------------|-----------|--------|
| Morgan Fingerprints (r=2) | 2 048     | RDKit from SMILES |
| RDKit Descriptors         | 10        | Computed (TPSA, RotBonds, ArRings, …) |
| Dataset Descriptors       | 4         | Pre-validated (MW, LogP, HBD, HBA) |
| Kinase Target One-Hot     | 12        | Target column (huge discriminative signal) |
| **Total**                 | **2 074** | |

---

## Project Structure

```
Industry-Chem-Project/
│
├── dataset/
│   ├── archive/
│   │   ├── kinase_data_final_tox.csv   ← primary (28 K rows, 15 cols)
│   │   └── Kinase_final_data.csv       ← secondary (21 K rows)
│   └── processed.csv                   ← generated after pipeline run
│
├── src/
│   ├── data_processor.py       ← load, merge, clean, featurise (2074-dim)
│   ├── feature_analyzer.py     ← correlation heatmap, pIC50 dist, class dist
│   ├── single_model_trainer.py ← XGBoost + SMOTE + 5-fold CV
│   ├── ensemble_trainer.py     ← XGB + LGB + CatBoost weighted soft voting
│   ├── loss_functions.py       ← focal loss, weighted log-loss
│   ├── model_trainer.py        ← import hub
│   └── main.py                 ← 13-step pipeline orchestration
│
├── config/
│   └── config.py               ← all hyperparameters, paths, target list
│
├── utils/
│   └── logger.py               ← structured logging
│
├── talk/
│   └── chat.py                 ← conversational model chat interface
│
├── models/                     ← saved artefacts (git-ignored)
│   ├── xgb_classifier.pkl
│   ├── ensemble_classifier.pkl
│   ├── scaler.pkl
│   ├── feature_config.json
│   └── plots/
│       ├── class_distribution.png
│       ├── pic50_distribution.png
│       ├── descriptor_correlation.png
│       ├── xgb_feature_importance.png
│       ├── model_comparison.png
│       └── shap_summary.png
│
├── interface.py                ← structured CLI drug activity checker
├── run_pipeline.py             ← main entry point
├── setup.sh                    ← one-command venv setup
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Set up the environment (fixes system-managed Python issue)

```bash
bash setup.sh
source venv/bin/activate
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the models

```bash
python run_pipeline.py
```

The pipeline will:
1. Load and merge both datasets (~44 000 compounds after deduplication)
2. Compute 2074-dimensional feature vectors (Morgan FP + descriptors + target OH)
3. Analyse feature distributions and correlations
4. Train XGBoost, LightGBM, CatBoost with SMOTE class balancing
5. Run 5-fold cross-validation
6. Build weighted soft-voting ensemble
7. Produce full evaluation metrics and all plots
8. Save trained models to `models/`

### 3. Interactive CLI checker

```bash
# Structured prediction interface
python interface.py --demo

# Single compound
python interface.py --smiles "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"

# Interactive prompt
python interface.py
```

### 4. Conversational chat interface

```bash
# Talk with the model naturally
python talk/chat.py

# Demo mode — predicts 8 approved TKI drugs
python talk/chat.py --demo
```

**Example chat session:**
```
  you> predict imatinib
  ✅ Prediction   : ACTIVE
     Probability : 0.9812  (Very High confidence)
     Kinase      : ABL

  you> compare erlotinib gefitinib EGFR
  ── side-by-side comparison table ──

  you> explain CC1=C... EGFR
  ── SHAP top-10 contributing features ──

  you> drugs
  ── all known drugs with predictions ──
```

---

## Pipeline Steps

| Step  | Module | Description |
|-------|--------|-------------|
| 1–3   | `data_processor.py` | Load, merge, deduplicate, pIC50, featurise (2074-dim) |
| 4     | `feature_analyzer.py` | Correlation heatmap, pIC50 histogram, class distribution |
| 5     | `main.py` | StandardScaler normalisation |
| 6     | `main.py` | 70 / 15 / 15 stratified split |
| 7     | `single_model_trainer.py` | XGBoost + SMOTE + 5-fold CV |
| 8     | `single_model_trainer.py` | XGBoost pIC50 regressor (RMSE / R²) |
| 9     | `ensemble_trainer.py` | XGB + LGB + CatBoost weighted soft voting |
| 10–11 | `main.py` | Comparison table + bar chart |
| 12–13 | `single_model_trainer.py` | Feature importance + SHAP summary |

---

## Accuracy Design Choices

| Strategy | Impact |
|----------|--------|
| Kinase target one-hot encoding | **+2–3% accuracy** — same compound may be active vs one kinase, inactive vs another |
| Pre-validated dataset descriptors | **+0.5%** — use authoritative MW/LogP from ChEMBL |
| Extended RDKit descriptors (TPSA, RotBonds, etc.) | **+0.5%** — captures 3D-relevant properties |
| SMOTE class balancing | Prevents bias toward majority class |
| Weighted soft-voting ensemble | Combines diversity of XGB/LGB/CatBoost |
| Focal loss (available) | For extreme imbalance scenarios |

---

## Requirements

- Python ≥ 3.10
- RDKit ≥ 2023.3
- XGBoost ≥ 2.0
- LightGBM ≥ 4.0
- CatBoost ≥ 1.2
- scikit-learn ≥ 1.3
- imbalanced-learn ≥ 0.11
- SHAP ≥ 0.44
