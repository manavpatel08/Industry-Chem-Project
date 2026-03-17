"""
Interactive Drug Activity Checker for Tyrosine Kinase Inhibitors.

Usage:
    python interface.py
    python interface.py --smiles "CC1=C2C=C..."
    python interface.py --demo
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config.config import (
    ENSEMBLE_MODEL_PATH, SCALER_PATH, FEATURE_CONFIG_PATH,
    MORGAN_RADIUS, MORGAN_NBITS, DESCRIPTORS,
)
from utils.logger import setup_logger

logger = setup_logger("Interface")

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

DEMO_COMPOUNDS = {
    "Imatinib (active BCR-ABL)":
        "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "Erlotinib (active EGFR)":
        "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "Gefitinib (active EGFR)":
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "Inactive compound":
        "CC(C)(C)C(=O)N1Cc2c(NC(=O)c3cc(F)cc(F)c3)n[nH]c2C1(C)C",
}


class TKIChecker:
    """Loads the trained ensemble and runs predictions on SMILES."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feat_config: dict = {}
        self._load()

    def _load(self):
        for path, label in [(ENSEMBLE_MODEL_PATH, "ensemble"), (SCALER_PATH, "scaler")]:
            if not os.path.exists(path):
                logger.error("Model file not found: %s  — run `python run_pipeline.py` first", path)
                sys.exit(1)

        with open(ENSEMBLE_MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        if os.path.exists(FEATURE_CONFIG_PATH):
            with open(FEATURE_CONFIG_PATH) as f:
                self.feat_config = json.load(f)

    def _featurize(self, smiles: str) -> np.ndarray | None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, MORGAN_NBITS)
        fp_arr = np.zeros(MORGAN_NBITS, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, fp_arr)

        desc_names = self.feat_config.get("descriptors", DESCRIPTORS)
        desc_arr = np.array([_DESCRIPTOR_FN[n](mol) for n in desc_names], dtype=np.float32)
        return np.hstack([fp_arr, desc_arr]).reshape(1, -1)

    def predict(self, smiles: str) -> dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES string"}

        X = self._featurize(smiles)
        X_scaled = self.scaler.transform(X)

        prob = float(self.model.predict_proba(X_scaled)[0])
        label = "ACTIVE" if prob >= 0.5 else "INACTIVE"

        # Molecular properties
        props = {n: round(float(_DESCRIPTOR_FN[n](mol)), 3)
                 for n in ["MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA"]}

        return {
            "smiles":      smiles,
            "prediction":  label,
            "probability": round(prob, 4),
            "confidence":  _confidence(prob),
            "properties":  props,
            "lipinski":    _lipinski(props),
        }

    def print_result(self, result: dict):
        if "error" in result:
            print(f"\n  ERROR: {result['error']}")
            return

        icon = "✅" if result["prediction"] == "ACTIVE" else "❌"
        print("\n" + "─" * 55)
        print(f"  {icon}  Prediction  : {result['prediction']}")
        print(f"      Probability : {result['probability']:.4f}")
        print(f"      Confidence  : {result['confidence']}")
        print()
        print("  Molecular Properties:")
        for k, v in result["properties"].items():
            print(f"    {k:<22}: {v}")
        print()
        lipinski = result["lipinski"]
        status = "PASS" if lipinski["pass"] else "FAIL"
        print(f"  Lipinski Rule-of-5  : {status}")
        for violation in lipinski.get("violations", []):
            print(f"    ⚠  {violation}")
        print("─" * 55)


def _confidence(prob: float) -> str:
    p = max(prob, 1 - prob)
    if p >= 0.90:
        return "Very High"
    if p >= 0.75:
        return "High"
    if p >= 0.60:
        return "Moderate"
    return "Low"


def _lipinski(props: dict) -> dict:
    violations = []
    if props["MolWt"] > 500:
        violations.append(f"MolWt {props['MolWt']} > 500")
    if props["LogP"] > 5:
        violations.append(f"LogP {props['LogP']} > 5")
    if props["NumHDonors"] > 5:
        violations.append(f"HBD {props['NumHDonors']} > 5")
    if props["NumHAcceptors"] > 10:
        violations.append(f"HBA {props['NumHAcceptors']} > 10")
    return {"pass": len(violations) == 0, "violations": violations}


def interactive_mode(checker: TKIChecker):
    print("\n" + "=" * 55)
    print("  Tyrosine Kinase Inhibitor Activity Checker")
    print("  Type 'quit' to exit | 'demo' for examples")
    print("=" * 55)

    while True:
        smi = input("\n  Enter SMILES: ").strip()
        if smi.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break
        if smi.lower() == "demo":
            run_demo(checker)
            continue
        if not smi:
            continue
        result = checker.predict(smi)
        checker.print_result(result)


def run_demo(checker: TKIChecker):
    print("\n" + "=" * 55)
    print("  Demo — Known Tyrosine Kinase Inhibitors")
    print("=" * 55)
    for name, smi in DEMO_COMPOUNDS.items():
        print(f"\n  Compound: {name}")
        result = checker.predict(smi)
        checker.print_result(result)


def main():
    parser = argparse.ArgumentParser(description="TKI Activity Checker")
    parser.add_argument("--smiles", type=str, help="SMILES string to predict")
    parser.add_argument("--demo", action="store_true", help="Run demo compounds")
    args = parser.parse_args()

    checker = TKIChecker()

    if args.demo:
        run_demo(checker)
    elif args.smiles:
        result = checker.predict(args.smiles)
        checker.print_result(result)
    else:
        interactive_mode(checker)


if __name__ == "__main__":
    main()
