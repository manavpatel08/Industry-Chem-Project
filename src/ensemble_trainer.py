"""Hybrid ensemble: XGBoost + LightGBM + CatBoost with weighted soft voting."""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

from config.config import (
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, CATBOOST_PARAMS,
    MODELS_DIR, PLOTS_DIR, ENSEMBLE_MODEL_PATH,
    LGB_MODEL_PATH, CAT_MODEL_PATH, RANDOM_STATE, SMOTE_K_NEIGHBORS,
)
from utils.logger import setup_logger

logger = setup_logger("EnsembleTrainer")


class HybridEnsemble:
    """
    Weighted soft-voting ensemble.

    Weights are derived from each model's validation performance:
      weight = 0.40 × accuracy + 0.60 × F1
    """

    def __init__(self):
        self.xgb: XGBClassifier | None = None
        self.lgb: lgb.LGBMClassifier | None = None
        self.cat: CatBoostClassifier | None = None
        self.weights: np.ndarray = np.array([1.0, 1.0, 1.0])
        self.history: dict = {}

    # ── Training ───────────────────────────────────────────────────────────────

    def _apply_smote(self, X, y):
        sm = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_STATE)
        return sm.fit_resample(X, y)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        apply_smote: bool = True,
    ):
        if apply_smote:
            logger.info("Applying SMOTE for ensemble training…")
            X_res, y_res = self._apply_smote(X_train, y_train)
        else:
            X_res, y_res = X_train, y_train

        model_metrics = {}

        # XGBoost — Conservative
        logger.info("Training XGBoost (Conservative)…")
        self.xgb = XGBClassifier(**XGBOOST_PARAMS)
        self.xgb.fit(X_res, y_res, eval_set=[(X_val, y_val)], verbose=False)
        model_metrics["xgb"] = self._val_metrics(self.xgb, X_val, y_val, "XGBoost")

        # LightGBM — Aggressive (fast, many leaves)
        logger.info("Training LightGBM (Aggressive)…")
        self.lgb = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
        self.lgb.fit(X_res, y_res, eval_set=[(X_val, y_val)])
        model_metrics["lgb"] = self._val_metrics(self.lgb, X_val, y_val, "LightGBM")

        # CatBoost — Balanced
        logger.info("Training CatBoost (Balanced)…")
        self.cat = CatBoostClassifier(**CATBOOST_PARAMS)
        self.cat.fit(X_res, y_res, eval_set=(X_val, y_val), use_best_model=True)
        model_metrics["cat"] = self._val_metrics(self.cat, X_val, y_val, "CatBoost")

        # Compute weights
        self.weights = np.array([
            0.40 * model_metrics[k]["accuracy"] + 0.60 * model_metrics[k]["f1"]
            for k in ("xgb", "lgb", "cat")
        ])
        self.weights /= self.weights.sum()
        logger.info("Ensemble weights — XGB: %.3f  LGB: %.3f  CAT: %.3f", *self.weights)

        self.history["individual_val"] = model_metrics

    def _val_metrics(self, model, X_val, y_val, label: str) -> dict:
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_prob)
        logger.info("  [%s] Acc=%.4f  F1=%.4f  AUC=%.4f", label, acc, f1, auc)
        return {"accuracy": acc, "f1": f1, "roc_auc": auc}

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        p_lgb = self.lgb.predict_proba(X)[:, 1]
        p_cat = self.cat.predict_proba(X)[:, 1]
        return (self.weights[0] * p_xgb +
                self.weights[1] * p_lgb +
                self.weights[2] * p_cat)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(self, X: np.ndarray, y: np.ndarray, split: str = "test") -> dict:
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        metrics = {
            "accuracy":  accuracy_score(y, y_pred),
            "f1":        f1_score(y, y_pred, zero_division=0),
            "roc_auc":   roc_auc_score(y, y_prob),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall":    recall_score(y, y_pred, zero_division=0),
        }
        logger.info(
            "[ENSEMBLE %s] Acc=%.4f  F1=%.4f  AUC=%.4f  Prec=%.4f  Rec=%.4f",
            split.upper(), metrics["accuracy"], metrics["f1"],
            metrics["roc_auc"], metrics["precision"], metrics["recall"],
        )
        self.history[split] = metrics
        self._plot_confusion_matrix(y, y_pred, f"Ensemble ({split})")
        return metrics

    # ── Plotting ───────────────────────────────────────────────────────────────

    def _plot_confusion_matrix(self, y_true, y_pred, title: str):
        os.makedirs(PLOTS_DIR, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["inactive", "active"])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, colorbar=False, cmap="Greens")
        ax.set_title(title)
        fname = title.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_cm.png"
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
        plt.close()

    def plot_model_comparison(self):
        if "individual_val" not in self.history:
            return
        models = ["XGBoost", "LightGBM", "CatBoost"]
        metrics = ["accuracy", "f1", "roc_auc"]
        vals = self.history["individual_val"]
        data = {m: [vals[k][m] for k in ("xgb", "lgb", "cat")] for m in metrics}

        x = np.arange(len(models))
        width = 0.25
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#3498db", "#2ecc71", "#e74c3c"]
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            bars = ax.bar(x + i * width, data[metric], width, label=metric.upper(), color=color, alpha=0.85)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Individual Model Comparison (Validation Set)")
        ax.legend()
        path = os.path.join(PLOTS_DIR, "model_comparison.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Model comparison → %s", path)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(ENSEMBLE_MODEL_PATH, "wb") as f:
            pickle.dump(self, f)
        logger.info("Ensemble saved → %s", ENSEMBLE_MODEL_PATH)

    @classmethod
    def load(cls) -> "HybridEnsemble":
        with open(ENSEMBLE_MODEL_PATH, "rb") as f:
            return pickle.load(f)
