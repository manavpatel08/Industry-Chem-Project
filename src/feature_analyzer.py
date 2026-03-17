"""Correlation analysis and feature importance visualisation."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config.config import PLOTS_DIR, DESCRIPTORS
from utils.logger import setup_logger

logger = setup_logger("FeatureAnalyzer")


class FeatureAnalyzer:
    def __init__(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Descriptor-level correlation ───────────────────────────────────────────

    def descriptor_correlation(self) -> pd.DataFrame:
        """Correlation matrix of the molecular descriptors (last N columns)."""
        n_desc = len(DESCRIPTORS)
        X_desc = self.X[:, -n_desc:]
        df = pd.DataFrame(X_desc, columns=DESCRIPTORS)
        corr = df.corr()
        return corr

    def plot_correlation_heatmap(self):
        corr = self.descriptor_correlation()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                    linewidths=0.5, vmin=-1, vmax=1)
        ax.set_title("Molecular Descriptor Correlation Heatmap")
        path = os.path.join(PLOTS_DIR, "descriptor_correlation.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Correlation heatmap → %s", path)

    def report_multicollinearity(self, threshold: float = 0.80):
        corr = self.descriptor_correlation()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = [(c, r, upper.loc[r, c]) for c in upper.columns for r in upper.index
                 if abs(upper.loc[r, c]) > threshold]
        if pairs:
            logger.warning("High multicollinearity (|r| > %.2f):", threshold)
            for a, b, v in pairs:
                logger.warning("  %s ↔ %s: %.3f", a, b, v)
        else:
            logger.info("No high multicollinearity detected (threshold=%.2f)", threshold)

    # ── Target correlation ─────────────────────────────────────────────────────

    def target_correlation(self) -> pd.Series:
        n_desc = len(DESCRIPTORS)
        X_desc = self.X[:, -n_desc:]
        df = pd.DataFrame(X_desc, columns=DESCRIPTORS)
        df["label"] = self.y
        corr = df.corrwith(df["label"]).drop("label").abs().sort_values(ascending=False)
        logger.info("Top descriptor correlations with label:\n%s", corr.to_string())
        return corr

    # ── Class distribution ─────────────────────────────────────────────────────

    def plot_class_distribution(self):
        vals, counts = np.unique(self.y, return_counts=True)
        labels = ["inactive", "active"]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar([labels[v] for v in vals], counts, color=["#e74c3c", "#2ecc71"])
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    f"{count:,}", ha="center", va="bottom", fontsize=11)
        ax.set_title("Class Distribution — Active vs Inactive")
        ax.set_ylabel("Count")
        path = os.path.join(PLOTS_DIR, "class_distribution.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Class distribution → %s", path)

    # ── pIC50 distribution ─────────────────────────────────────────────────────

    def plot_pic50_distribution(self, y_reg: np.ndarray):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(y_reg, bins=60, color="#3498db", edgecolor="white", alpha=0.8)
        ax.axvline(6.0, color="red", linestyle="--", label="threshold (pIC50=6)")
        ax.set_xlabel("pIC50")
        ax.set_ylabel("Frequency")
        ax.set_title("pIC50 Distribution")
        ax.legend()
        path = os.path.join(PLOTS_DIR, "pic50_distribution.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("pIC50 distribution → %s", path)
