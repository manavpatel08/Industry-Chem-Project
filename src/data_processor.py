"""Data loading, merging, featurisation, and preprocessing.

Accuracy strategy:
  1. Morgan fingerprints (2048 bits, radius 2) — structural substructure features
  2. RDKit-computed descriptors (10 extended physicochemical properties)
  3. Pre-computed dataset descriptors (MW, LogP, HBD, HBA — already validated)
  4. Kinase target one-hot encoding (huge discriminative signal from tox dataset)
  Combined feature vector: 2048 + 10 + 4 + 12 = 2074 dimensions
"""

import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs

from config.config import (
    DATASET_PATHS, SMILES_COL, IC50_COL_TOX, IC50_COL_KINASE,
    LABEL_COL, PIC50_COL, LABEL_MAP, PIC50_THRESHOLD,
    MORGAN_RADIUS, MORGAN_NBITS, DESCRIPTORS, KINASE_TARGETS,
    PROCESSED_DATA_PATH, FEATURE_CONFIG_PATH, MODELS_DIR,
)
from utils.logger import setup_logger

logger = setup_logger("DataProcessor")

_DESCRIPTOR_FN = {
    "MolWt":             Descriptors.MolWt,
    "LogP":              Descriptors.MolLogP,
    "NumHDonors":        Descriptors.NumHDonors,
    "NumHAcceptors":     Descriptors.NumHAcceptors,
    "TPSA":              Descriptors.TPSA,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings":  Descriptors.NumAromaticRings,
    "FractionCSP3":      Descriptors.FractionCSP3,
    "RingCount":         Descriptors.RingCount,
    "HeavyAtomCount":    Descriptors.HeavyAtomCount,
}

# Pre-computed columns already validated in the dataset (use directly)
_PRECOMPUTED_COLS = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
_PRECOMPUTED_RENAMED = ["ds_MW", "ds_LogP", "ds_NumHDonors", "ds_NumHAcceptors"]


class KinaseDataProcessor:
    """Loads both kinase CSV datasets, harmonises columns, featurises SMILES."""

    def __init__(self):
        self.df_raw: pd.DataFrame | None = None
        self.df_processed: pd.DataFrame | None = None
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.y_reg: np.ndarray | None = None
        self.feature_names: list[str] = []

    # ── Loading & merging ──────────────────────────────────────────────────────

    def load_and_merge(self) -> pd.DataFrame:
        logger.info("Loading datasets…")
        dfs = []

        # Primary: tox dataset (has kinase target column)
        tox = pd.read_csv(DATASET_PATHS["tox"], encoding="utf-8-sig")
        tox = tox.rename(columns={IC50_COL_TOX: "ic50_nM"})
        tox["source"] = "tox"
        dfs.append(tox)
        logger.info("  tox dataset:    %d rows  (targets: %s)",
                    len(tox), sorted(tox["target"].dropna().unique().tolist()))

        # Secondary: kinase dataset (no target column → mark unknown)
        kin = pd.read_csv(DATASET_PATHS["kinase"])
        kin = kin.rename(columns={IC50_COL_KINASE: "ic50_nM"})
        kin["source"] = "kinase"
        if "target" not in kin.columns:
            kin["target"] = "unknown"
        dfs.append(kin)
        logger.info("  kinase dataset: %d rows", len(kin))

        df = pd.concat(dfs, ignore_index=True)
        logger.info("Combined: %d rows", len(df))
        self.df_raw = df
        return df

    # ── Cleaning ───────────────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=[SMILES_COL]).reset_index(drop=True)
        df = df.drop_duplicates(subset=[SMILES_COL]).reset_index(drop=True)
        logger.info("After dedup + drop-na SMILES: %d → %d rows", before, len(df))

        df = self._ensure_pic50(df)

        # Binary label: active=1, inactive=0, intermediate=0
        df["label"] = df[LABEL_COL].str.strip().str.lower().map(LABEL_MAP)
        missing_label = df["label"].isna()
        df.loc[missing_label, "label"] = (
            df.loc[missing_label, "pIC50_final"] >= PIC50_THRESHOLD
        ).astype(int)
        df = df.dropna(subset=["label", "pIC50_final"]).reset_index(drop=True)
        df["label"] = df["label"].astype(int)

        # Normalise target column
        df["target"] = df["target"].fillna("unknown").str.strip().str.upper()

        dist = df["label"].value_counts().to_dict()
        logger.info("Labels — active: %d  inactive: %d", dist.get(1, 0), dist.get(0, 0))
        return df

    def _ensure_pic50(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        existing = pd.to_numeric(df.get(PIC50_COL, pd.Series(dtype=float)), errors="coerce")
        ic50 = pd.to_numeric(df.get("ic50_nM", pd.Series(dtype=float)), errors="coerce")
        computed = np.where(ic50 > 0, -np.log10(ic50 * 1e-9), np.nan)
        df["pIC50_final"] = existing.fillna(pd.Series(computed, index=df.index))
        return df

    # ── Featurisation ──────────────────────────────────────────────────────────

    @staticmethod
    def _mol_from_smiles(smi: str):
        try:
            return Chem.MolFromSmiles(str(smi))
        except Exception:
            return None

    @staticmethod
    def _morgan_fp(mol, radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_NBITS) -> np.ndarray:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    @staticmethod
    def _rdkit_descriptors(mol, desc_names: list[str]) -> np.ndarray:
        return np.array([_DESCRIPTOR_FN[n](mol) for n in desc_names], dtype=np.float32)

    def _target_onehot(self, target_str: str) -> np.ndarray:
        """One-hot encode kinase target. Unknown → all-zeros vector."""
        vec = np.zeros(len(KINASE_TARGETS), dtype=np.float32)
        t = str(target_str).strip().upper()
        if t in KINASE_TARGETS:
            vec[KINASE_TARGETS.index(t)] = 1.0
        return vec

    def featurize(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_desc = len(DESCRIPTORS)
        n_pre  = len(_PRECOMPUTED_COLS)
        n_tgt  = len(KINASE_TARGETS)
        logger.info(
            "Featurising %d SMILES → Morgan(%d) + RDKit-desc(%d) + dataset-desc(%d) + target-OH(%d)",
            len(df), MORGAN_NBITS, n_desc, n_pre, n_tgt,
        )

        fps, rdkit_descs, pre_descs, target_ohs, valid_idx = [], [], [], [], []
        invalid = 0

        for i, row in df.iterrows():
            mol = self._mol_from_smiles(row[SMILES_COL])
            if mol is None:
                invalid += 1
                continue

            fps.append(self._morgan_fp(mol))
            rdkit_descs.append(self._rdkit_descriptors(mol, DESCRIPTORS))

            # Pre-computed descriptor columns (fall back to RDKit if NaN)
            pre = []
            for col in _PRECOMPUTED_COLS:
                val = pd.to_numeric(row.get(col, np.nan), errors="coerce")
                pre.append(float(val) if not np.isnan(val) else 0.0)
            pre_descs.append(np.array(pre, dtype=np.float32))

            target_ohs.append(self._target_onehot(row.get("target", "unknown")))
            valid_idx.append(i)

        if invalid:
            logger.warning("Skipped %d invalid SMILES (%.1f%%)", invalid, 100 * invalid / len(df))

        X = np.hstack([
            np.vstack(fps),
            np.vstack(rdkit_descs),
            np.vstack(pre_descs),
            np.vstack(target_ohs),
        ])

        valid_df = df.loc[valid_idx].reset_index(drop=True)
        y     = valid_df["label"].values.astype(int)
        y_reg = valid_df["pIC50_final"].values.astype(np.float32)

        self.feature_names = (
            [f"fp_{i}" for i in range(MORGAN_NBITS)]
            + DESCRIPTORS
            + _PRECOMPUTED_RENAMED
            + [f"target_{t}" for t in KINASE_TARGETS]
        )
        logger.info("Feature matrix: %s  (2048 FP + %d rdkit + %d pre + %d target-OH)",
                    X.shape, n_desc, n_pre, n_tgt)
        return X, y, y_reg

    # ── Public pipeline ────────────────────────────────────────────────────────

    def run(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        df = self.load_and_merge()
        df = self.clean(df)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        logger.info("Processed data → %s", PROCESSED_DATA_PATH)

        X, y, y_reg = self.featurize(df)
        self.X, self.y, self.y_reg = X, y, y_reg
        self.df_processed = df

        os.makedirs(MODELS_DIR, exist_ok=True)
        feat_cfg = {
            "morgan_radius":   MORGAN_RADIUS,
            "morgan_nbits":    MORGAN_NBITS,
            "descriptors":     DESCRIPTORS,
            "precomputed":     _PRECOMPUTED_COLS,
            "kinase_targets":  KINASE_TARGETS,
            "feature_names":   self.feature_names,
        }
        with open(FEATURE_CONFIG_PATH, "w") as f:
            json.dump(feat_cfg, f, indent=2)
        logger.info("Feature config → %s", FEATURE_CONFIG_PATH)

        return X, y, y_reg
