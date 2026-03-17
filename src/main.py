"""
Main pipeline — Predictive Modeling of Tyrosine Kinase Inhibitors.

Steps:
  1.  Load & merge datasets
  2.  Clean data
  3.  Featurise (Morgan FP + descriptors)
  4.  Correlation / distribution analysis
  5.  Normalise features
  6.  Train / validation / test split
  7.  Train single XGBoost model
  8.  Train XGBoost regressor (pIC50)
  9.  Train hybrid ensemble (XGB + LGB + CatBoost)
  10. Evaluate on validation and test sets
  11. Compare models
  12. Save artefacts
  13. Feature importance & SHAP
"""

import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.config import (
    RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE,
    SCALER_PATH, MODELS_DIR, PLOTS_DIR,
)
from src.data_processor import KinaseDataProcessor
from src.feature_analyzer import FeatureAnalyzer
from src.single_model_trainer import TKIPredictor
from src.ensemble_trainer import HybridEnsemble
from utils.logger import setup_logger

logger = setup_logger("MainPipeline")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Step 1-3: Load, clean, featurise ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1-3 | Data loading & featurisation")
    logger.info("=" * 60)
    processor = KinaseDataProcessor()
    X, y, y_reg = processor.run()
    feature_names = processor.feature_names

    # ── Step 4: Analysis ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 | Feature & distribution analysis")
    logger.info("=" * 60)
    analyzer = FeatureAnalyzer(X, y, feature_names)
    analyzer.plot_class_distribution()
    analyzer.plot_pic50_distribution(y_reg)
    analyzer.plot_correlation_heatmap()
    analyzer.report_multicollinearity()
    analyzer.target_correlation()

    # ── Step 5: Normalise ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5 | Feature normalisation")
    logger.info("=" * 60)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved → %s", SCALER_PATH)

    # ── Step 6: Split (70 / 15 / 15) ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6 | Train / validation / test split")
    logger.info("=" * 60)
    X_train, X_temp, y_train, y_temp, yr_train, yr_temp = train_test_split(
        X_scaled, y, y_reg,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    val_frac = VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE)
    X_val, X_test, y_val, y_test, yr_val, yr_test = train_test_split(
        X_temp, y_temp, yr_temp,
        test_size=1 - val_frac,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    logger.info("Train: %d  Val: %d  Test: %d", len(X_train), len(X_val), len(X_test))

    # ── Step 7: Single XGBoost classifier ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7 | Single XGBoost classifier")
    logger.info("=" * 60)
    predictor = TKIPredictor()
    predictor.train_classifier(X_train, y_train, X_val, y_val)
    val_metrics_xgb = predictor.evaluate_classifier(X_val, y_val, split="validation")
    test_metrics_xgb = predictor.evaluate_classifier(X_test, y_test, split="test")
    predictor.save()

    # ── Step 8: XGBoost regressor ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 8 | XGBoost pIC50 regressor")
    logger.info("=" * 60)
    predictor.train_regressor(X_train, yr_train, X_val, yr_val)
    predictor.evaluate_regressor(X_test, yr_test, split="test")

    # ── Step 9: Ensemble ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 9 | Hybrid ensemble (XGB + LGB + CatBoost)")
    logger.info("=" * 60)
    ensemble = HybridEnsemble()
    ensemble.train(X_train, y_train, X_val, y_val)
    val_metrics_ens = ensemble.evaluate(X_val, y_val, split="validation")
    test_metrics_ens = ensemble.evaluate(X_test, y_test, split="test")
    ensemble.save()

    # ── Step 10-11: Comparison ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 10-11 | Model comparison")
    logger.info("=" * 60)
    ensemble.plot_model_comparison()
    _print_comparison(test_metrics_xgb, test_metrics_ens)

    # ── Step 12-13: Plots ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 12-13 | Feature importance & SHAP")
    logger.info("=" * 60)
    predictor.plot_feature_importance(feature_names)
    predictor.run_shap(X_test, feature_names)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — all artefacts in ./models/")
    logger.info("=" * 60)


def _print_comparison(xgb_metrics: dict, ens_metrics: dict):
    logger.info("\n%s", "=" * 55)
    logger.info("%-30s %-12s %-12s", "Metric", "XGBoost", "Ensemble")
    logger.info("-" * 55)
    for k in ("accuracy", "f1", "roc_auc", "precision", "recall"):
        logger.info("%-30s %-12.4f %-12.4f", k.upper(), xgb_metrics[k], ens_metrics[k])
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
