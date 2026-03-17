"""Central configuration for the Tyrosine Kinase Inhibitor ML pipeline."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Dataset paths ──────────────────────────────────────────────────────────────
DATASET_PATHS = {
    "tox": os.path.join(BASE_DIR, "dataset", "archive", "kinase_data_final_tox.csv"),
    "kinase": os.path.join(BASE_DIR, "dataset", "archive", "Kinase_final_data.csv"),
}

# ── Column names ───────────────────────────────────────────────────────────────
SMILES_COL = "canonical_smiles"
IC50_COL_TOX = "IC50"
IC50_COL_KINASE = "standard_value"
LABEL_COL = "class"
PIC50_COL = "pIC50"

# ── Label mapping ──────────────────────────────────────────────────────────────
LABEL_MAP = {"active": 1, "inactive": 0, "intermediate": 0}
PIC50_THRESHOLD = 6.0      # pIC50 >= 6 → active

# ── Feature Engineering ────────────────────────────────────────────────────────
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
DESCRIPTORS = [
    "MolWt", "LogP", "NumHDonors", "NumHAcceptors",
    "TPSA", "NumRotatableBonds", "NumAromaticRings",
    "FractionCSP3", "RingCount", "HeavyAtomCount",
]

# All kinase targets present in the tox dataset (uppercase)
KINASE_TARGETS = ["ABL", "ALK", "EGFR", "FGFR", "JAK", "KIT", "MET",
                  "PDGFR", "RET", "SRC", "VEGFR", "UNKNOWN"]

# ── Data processing ────────────────────────────────────────────────────────────
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42

# SMOTE
SMOTE_SAMPLING_STRATEGY = "auto"
SMOTE_K_NEIGHBORS = 5

# ── XGBoost hyperparameters ────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 600,
    "max_depth": 8,
    "learning_rate": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "colsample_bylevel": 0.75,
    "min_child_weight": 3,
    "gamma": 0.05,
    "reg_alpha": 0.05,
    "reg_lambda": 1.5,
    "scale_pos_weight": 1,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
    "eval_metric": "logloss",
    "n_jobs": -1,
}

# ── LightGBM hyperparameters ───────────────────────────────────────────────────
LIGHTGBM_PARAMS = {
    "n_estimators": 600,
    "max_depth": 8,
    "learning_rate": 0.04,
    "num_leaves": 127,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "min_child_samples": 15,
    "reg_alpha": 0.05,
    "reg_lambda": 1.5,
    "random_state": RANDOM_STATE,
    "verbosity": -1,
    "n_jobs": -1,
}

# ── CatBoost hyperparameters ───────────────────────────────────────────────────
CATBOOST_PARAMS = {
    "iterations": 600,
    "depth": 8,
    "learning_rate": 0.04,
    "l2_leaf_reg": 3.0,
    "bagging_temperature": 0.5,
    "random_strength": 0.5,
    "random_seed": RANDOM_STATE,
    "verbose": 0,
    "thread_count": -1,
}

# ── Random Forest hyperparameters (baseline) ───────────────────────────────────
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ── Output paths ───────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(BASE_DIR, "models")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_classifier.pkl")
LGB_MODEL_PATH = os.path.join(MODELS_DIR, "lgb_classifier.pkl")
CAT_MODEL_PATH = os.path.join(MODELS_DIR, "cat_classifier.pkl")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_classifier.pkl")
ENSEMBLE_MODEL_PATH = os.path.join(MODELS_DIR, "ensemble_classifier.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_CONFIG_PATH = os.path.join(MODELS_DIR, "feature_config.json")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "dataset", "processed.csv")
PLOTS_DIR = os.path.join(MODELS_DIR, "plots")
