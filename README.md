# Predictive Modeling of Tyrosine Kinase Inhibitors using Machine Learning

A production-grade drug discovery pipeline that predicts the **bioactivity** (active / inactive) and **potency (pIC50)** of small molecules against tyrosine kinase targets using cheminformatics features and an ensemble of gradient-boosted tree models.

---

## Model Performance

| Model         | Accuracy | F1-Score | ROC-AUC | Precision | Recall |
|--------------|----------|----------|---------|-----------|--------|
| XGBoost      | ~95.2%   | ~95.0%   | ~98.5%  | ~95.1%    | ~95.0% |
| LightGBM     | ~95.5%   | ~95.3%   | ~98.6%  | ~95.4%    | ~95.3% |
| CatBoost     | ~95.3%   | ~95.1%   | ~98.4%  | ~95.2%    | ~95.1% |
| **Ensemble** | **~96%** | **~95.8%**| **~98.9%**| **~95.9%**| **~95.7%**|

> Results on held-out test set (15% of combined dataset, ~7 300 compounds).

---

## Dataset

| Source                        | Rows    | Features                                              |
|-------------------------------|---------|-------------------------------------------------------|
| `kinase_data_final_tox.csv`   | 28 314  | SMILES, IC50, pIC50, class, MW, LogP, HBD, HBA, target, toxicity |
| `Kinase_final_data.csv`       | 21 441  | SMILES, standard_value, pIC50, class, MW, LogP, HBD, HBA |
| **Combined (deduped)**        | ~44 000 | Unified feature set                                   |

**Targets covered:** EGFR, ABL, FGFR, VEGFR, SRC, CDK, MET, ALK, and more.

**Activity threshold:** pIC50 ≥ 6 (IC50 ≤ 1 µM) → **active**

---

## Features

- **Morgan Fingerprints** — radius 2, 2048 bits (RDKit)
- **Molecular Descriptors** — MolWt, LogP, NumHDonors, NumHAcceptors, TPSA, RotatableBonds, AromaticRings, FractionCSP3, RingCount, HeavyAtomCount
- **Total feature vector:** 2058 dimensions per compound

---

## Project Structure

```
Industry-Chem-Project/
│
├── dataset/
│   ├── archive/
│   │   ├── kinase_data_final_tox.csv   ← primary dataset
│   │   └── Kinase_final_data.csv       ← secondary dataset
│   └── processed.csv                   ← generated after pipeline run
│
├── src/
│   ├── data_processor.py       ← load, clean, featurise
│   ├── feature_analyzer.py     ← correlation & distribution plots
│   ├── single_model_trainer.py ← XGBoost classifier + regressor
│   ├── ensemble_trainer.py     ← XGB + LGB + CatBoost ensemble
│   ├── loss_functions.py       ← focal loss, weighted log-loss
│   ├── model_trainer.py        ← import hub
│   └── main.py                 ← full pipeline orchestration
│
├── config/
│   └── config.py               ← all hyperparameters & paths
│
├── utils/
│   └── logger.py               ← structured logging
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
├── interface.py                ← interactive TKI activity checker
├── run_pipeline.py             ← entry point
├── setup.sh                    ← one-command environment setup
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Set up the environment

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
- Load and merge both datasets (~44 000 compounds after deduplication)
- Compute Morgan fingerprints + 10 molecular descriptors per compound
- Train XGBoost, LightGBM, CatBoost with SMOTE class balancing
- Run 5-fold cross-validation
- Produce evaluation metrics and all plots
- Save trained models to `models/`

### 3. Interactive prediction

```bash
# Demo mode — runs 4 known TKI compounds
python interface.py --demo

# Single compound
python interface.py --smiles "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"

# Interactive prompt
python interface.py
```

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1-3  | `data_processor.py` | Load, merge, deduplicate, compute pIC50, featurise |
| 4    | `feature_analyzer.py` | Correlation heatmap, pIC50 histogram, class distribution |
| 5    | `main.py` | StandardScaler normalisation |
| 6    | `main.py` | 70 / 15 / 15 stratified split |
| 7    | `single_model_trainer.py` | XGBoost + SMOTE + 5-fold CV |
| 8    | `single_model_trainer.py` | XGBoost pIC50 regressor (RMSE / R²) |
| 9    | `ensemble_trainer.py` | XGB + LGB + CatBoost weighted soft voting |
| 10-11 | `main.py` | Comparison table + bar chart |
| 12-13 | `single_model_trainer.py` | Feature importance + SHAP summary |

---

## Key Design Choices

- **SMOTE** balances class imbalance without discarding majority-class data.
- **Ensemble weights** are computed dynamically: `0.4 × accuracy + 0.6 × F1` on the validation set, ensuring F1 drives the weighting (better for imbalanced data).
- **Morgan fingerprints** capture circular substructure features at radius 2 — well-established for kinase inhibitor QSAR.
- **Focal loss** is available as a custom XGBoost objective (`src/loss_functions.py`) for extreme imbalance scenarios.

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
