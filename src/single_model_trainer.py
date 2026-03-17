"""Single-model trainer (XGBoost) with SMOTE, CV, and detailed metrics."""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    ConfusionMatrixDisplay,
    mean_squared_error, r2_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier, XGBRegressor

from config.config import (
    XGBOOST_PARAMS, MODELS_DIR, PLOTS_DIR, XGB_MODEL_PATH, RANDOM_STATE,
    SMOTE_K_NEIGHBORS,
)
from utils.logger import setup_logger

logger = setup_logger("SingleModelTrainer")


class TKIPredictor:
    """Trains an XGBoost classifier (+ optional regressor) for TKI bioactivity."""

    def __init__(self):
        self.classifier: XGBClassifier | None = None
        self.regressor: XGBRegressor | None = None
        self.history: dict = {}

    # ── Training ───────────────────────────────────────────────────────────────

    def train_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        apply_smote: bool = True,
    ) -> XGBClassifier:
        if apply_smote:
            logger.info("Applying SMOTE…")
            sm = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_STATE)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            logger.info("After SMOTE: %d samples", len(X_train))

        logger.info("Training XGBoost classifier…")
        model = XGBClassifier(**XGBOOST_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self.classifier = model

        # CV on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            XGBClassifier(**XGBOOST_PARAMS), X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        logger.info("5-fold CV ROC-AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())
        self.history["cv_auc"] = {"mean": cv_scores.mean(), "std": cv_scores.std()}
        return model

    def train_regressor(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> XGBRegressor:
        logger.info("Training XGBoost regressor (pIC50)…")
        params = {k: v for k, v in XGBOOST_PARAMS.items() if k != "eval_metric"}
        params["eval_metric"] = "rmse"
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.regressor = model
        return model

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate_classifier(self, X: np.ndarray, y: np.ndarray, split: str = "test") -> dict:
        assert self.classifier is not None, "Train classifier first"
        y_pred = self.classifier.predict(X)
        y_prob = self.classifier.predict_proba(X)[:, 1]

        metrics = {
            "accuracy":  accuracy_score(y, y_pred),
            "f1":        f1_score(y, y_pred, zero_division=0),
            "roc_auc":   roc_auc_score(y, y_prob),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall":    recall_score(y, y_pred, zero_division=0),
        }
        logger.info(
            "[XGB %s] Acc=%.4f  F1=%.4f  AUC=%.4f  Prec=%.4f  Rec=%.4f",
            split.upper(), metrics["accuracy"], metrics["f1"],
            metrics["roc_auc"], metrics["precision"], metrics["recall"],
        )
        self.history[split] = metrics
        self._plot_confusion_matrix(y, y_pred, title=f"XGBoost ({split})")
        return metrics

    def evaluate_regressor(self, X: np.ndarray, y: np.ndarray, split: str = "test") -> dict:
        assert self.regressor is not None
        y_pred = self.regressor.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2   = r2_score(y, y_pred)
        logger.info("[XGB-REG %s] RMSE=%.4f  R²=%.4f", split.upper(), rmse, r2)
        return {"rmse": rmse, "r2": r2}

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str = XGB_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.classifier, f)
        logger.info("XGBoost classifier → %s", path)

    def load(self, path: str = XGB_MODEL_PATH):
        with open(path, "rb") as f:
            self.classifier = pickle.load(f)

    # ── Plotting ───────────────────────────────────────────────────────────────

    def _plot_confusion_matrix(self, y_true, y_pred, title: str):
        os.makedirs(PLOTS_DIR, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["inactive", "active"])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title)
        fname = title.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_cm.png"
        path = os.path.join(PLOTS_DIR, fname)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def plot_feature_importance(self, feature_names: list[str], top_n: int = 25):
        assert self.classifier is not None
        importances = self.classifier.feature_importances_
        idx = np.argsort(importances)[-top_n:][::-1]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(range(top_n), importances[idx][::-1], color="#3498db", align="center")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in idx[::-1]], fontsize=8)
        ax.set_xlabel("Feature Importance (gain)")
        ax.set_title(f"XGBoost — Top {top_n} Features")
        path = os.path.join(PLOTS_DIR, "xgb_feature_importance.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Feature importance plot → %s", path)

    def run_shap(self, X_test: np.ndarray, feature_names: list[str]):
        try:
            import shap
            logger.info("Computing SHAP values…")
            explainer = shap.TreeExplainer(self.classifier)
            sample = X_test[:min(300, len(X_test))]
            shap_values = explainer.shap_values(sample)
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
            plt.figure()
            shap.summary_plot(sv, sample, feature_names=feature_names, show=False, max_display=20)
            path = os.path.join(PLOTS_DIR, "shap_summary.png")
            os.makedirs(PLOTS_DIR, exist_ok=True)
            plt.savefig(path, bbox_inches="tight", dpi=150)
            plt.close()
            logger.info("SHAP summary → %s", path)
        except Exception as e:
            logger.warning("SHAP skipped: %s", e)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.classifier.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(X)[:, 1]
