"""Data loading, merging, featurisation, and preprocessing."""

import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs

from config.config import (
    DATASET_PATHS, SMILES_COL, IC50_COL_TOX, IC50_COL_KINASE,
    LABEL_COL, PIC50_COL, LABEL_MAP, PIC50_THRESHOLD,
    MORGAN_RADIUS, MORGAN_NBITS, DESCRIPTORS,
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

        # Primary: tox dataset
        tox = pd.read_csv(DATASET_PATHS["tox"])
        tox = tox.rename(columns={IC50_COL_TOX: "ic50_nM"})
        tox["source"] = "tox"
        dfs.append(tox)
        logger.info("  tox dataset:    %d rows", len(tox))

        # Secondary: kinase dataset
        kin = pd.read_csv(DATASET_PATHS["kinase"])
        kin = kin.rename(columns={IC50_COL_KINASE: "ic50_nM"})
        kin["source"] = "kinase"
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

        # pIC50
        df = self._ensure_pic50(df)

        # binary label
        df["label"] = df[LABEL_COL].str.strip().str.lower().map(LABEL_MAP)
        # fallback: derive from pIC50
        missing_label = df["label"].isna()
        df.loc[missing_label, "label"] = (df.loc[missing_label, "pIC50_final"] >= PIC50_THRESHOLD).astype(int)
        df = df.dropna(subset=["label", "pIC50_final"]).reset_index(drop=True)
        df["label"] = df["label"].astype(int)

        dist = df["label"].value_counts().to_dict()
        logger.info("Labels — active: %d  inactive: %d", dist.get(1, 0), dist.get(0, 0))
        return df

    def _ensure_pic50(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use existing pIC50 where available, compute from ic50_nM otherwise."""
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
            mol = Chem.MolFromSmiles(str(smi))
            return mol
        except Exception:
            return None

    @staticmethod
    def _morgan_fp(mol, radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_NBITS) -> np.ndarray:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    @staticmethod
    def _compute_descriptors(mol, desc_names: list[str]) -> np.ndarray:
        return np.array([_DESCRIPTOR_FN[n](mol) for n in desc_names], dtype=np.float32)

    def featurize(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        logger.info("Featurising %d SMILES (Morgan r=%d, %d bits + %d descriptors)…",
                    len(df), MORGAN_RADIUS, MORGAN_NBITS, len(DESCRIPTORS))

        fps, descs, valid_idx = [], [], []
        invalid = 0
        for i, smi in enumerate(df[SMILES_COL]):
            mol = self._mol_from_smiles(smi)
            if mol is None:
                invalid += 1
                continue
            fps.append(self._morgan_fp(mol))
            descs.append(self._compute_descriptors(mol, DESCRIPTORS))
            valid_idx.append(i)

        if invalid:
            logger.warning("Skipped %d invalid SMILES (%.1f%%)", invalid, 100 * invalid / len(df))

        X_fp   = np.vstack(fps)
        X_desc = np.vstack(descs)
        X      = np.hstack([X_fp, X_desc])

        valid_df = df.iloc[valid_idx].reset_index(drop=True)
        y        = valid_df["label"].values.astype(int)
        y_reg    = valid_df["pIC50_final"].values.astype(np.float32)

        self.feature_names = [f"fp_{i}" for i in range(MORGAN_NBITS)] + DESCRIPTORS
        logger.info("Feature matrix: %s", X.shape)
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

        # save feature config
        os.makedirs(MODELS_DIR, exist_ok=True)
        feat_cfg = {"morgan_radius": MORGAN_RADIUS, "morgan_nbits": MORGAN_NBITS, "descriptors": DESCRIPTORS}
        with open(FEATURE_CONFIG_PATH, "w") as f:
            json.dump(feat_cfg, f, indent=2)
        logger.info("Feature config → %s", FEATURE_CONFIG_PATH)

        return X, y, y_reg
