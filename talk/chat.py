"""
Talk with the TKI Model — Conversational Drug Activity Interface.

A natural-language-style chat that lets you ask questions about molecules,
predict activity, compare compounds, and explore model decisions.

Usage:
    python talk/chat.py
    python talk/chat.py --demo
"""

import json
import os
import pickle
import re
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config.config import (
    ENSEMBLE_MODEL_PATH, SCALER_PATH, FEATURE_CONFIG_PATH,
    MORGAN_RADIUS, MORGAN_NBITS, DESCRIPTORS, KINASE_TARGETS,
)
from utils.logger import setup_logger

logger = setup_logger("Chat")

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

KNOWN_DRUGS = {
    "imatinib":   ("Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", "ABL"),
    "erlotinib":  ("C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",                   "EGFR"),
    "gefitinib":  ("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",               "EGFR"),
    "sorafenib":  ("CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",   "VEGFR"),
    "sunitinib":  ("CCN(CC)CCNC(=O)c1c(C)[nH]c(-c2ccc(F)cc2)c1/C=C1\\C(=O)Nc2ccccc21", "VEGFR"),
    "dasatinib":  ("Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",     "ABL"),
    "crizotinib": ("Cc1cnn(-c2cc(Nc3ccc(F)cc3Cl)nc3cc(N4CCNCC4)cnc23)c1",          "ALK"),
    "lapatinib":  ("CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1", "EGFR"),
}

HELP_TEXT = """
╔══════════════════════════════════════════════════════╗
║       TKI Model Chat — Command Reference             ║
╠══════════════════════════════════════════════════════╣
║  predict <SMILES> [target]   Predict bioactivity     ║
║  predict <drug name>         E.g.: predict imatinib  ║
║  compare <SMILES1> <SMILES2> Side-by-side comparison ║
║  props <SMILES>              Molecular properties    ║
║  explain <SMILES> [target]   SHAP-based explanation  ║
║  drugs                       List known drugs        ║
║  targets                     List kinase targets     ║
║  help                        Show this menu          ║
║  quit / exit                 Exit                    ║
╚══════════════════════════════════════════════════════╝
"""


class ModelChat:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feat_config: dict = {}
        self._load_models()
        self.history: list[dict] = []

    def _load_models(self):
        for path, name in [(ENSEMBLE_MODEL_PATH, "ensemble"), (SCALER_PATH, "scaler")]:
            if not os.path.exists(path):
                print(f"\n  ⚠  {name} not found at: {path}")
                print("  Run `python run_pipeline.py` first to train the models.")
                sys.exit(1)
        with open(ENSEMBLE_MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        if os.path.exists(FEATURE_CONFIG_PATH):
            with open(FEATURE_CONFIG_PATH) as f:
                self.feat_config = json.load(f)

    def _featurize(self, smiles: str, target: str = "unknown") -> np.ndarray | None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, MORGAN_NBITS)
        fp_arr = np.zeros(MORGAN_NBITS, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, fp_arr)

        # RDKit descriptors
        desc_names = self.feat_config.get("descriptors", DESCRIPTORS)
        desc_arr = np.array([_DESCRIPTOR_FN[n](mol) for n in desc_names], dtype=np.float32)

        # Pre-computed (recompute here since we don't have the dataset row)
        pre_arr = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
        ], dtype=np.float32)

        # Target one-hot
        kinase_targets = self.feat_config.get("kinase_targets", KINASE_TARGETS)
        tgt_vec = np.zeros(len(kinase_targets), dtype=np.float32)
        t = target.strip().upper()
        if t in kinase_targets:
            tgt_vec[kinase_targets.index(t)] = 1.0

        return np.hstack([fp_arr, desc_arr, pre_arr, tgt_vec]).reshape(1, -1)

    def _predict(self, smiles: str, target: str = "unknown") -> dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES string. Please check the input."}

        X = self._featurize(smiles, target)
        X_scaled = self.scaler.transform(X)
        prob = float(self.model.predict_proba(X_scaled)[0])
        label = "ACTIVE" if prob >= 0.5 else "INACTIVE"

        props = {n: round(float(_DESCRIPTOR_FN[n](mol)), 3) for n in desc_names_short}
        return {
            "smiles":     smiles,
            "target":     target.upper(),
            "prediction": label,
            "probability": round(prob, 4),
            "confidence": _confidence(prob),
            "properties": props,
            "lipinski":   _lipinski(props),
            "drug_score": _drug_score(props),
        }

    def cmd_predict(self, args: str) -> str:
        args = args.strip()
        # Check if it's a known drug name
        lower = args.lower().split()[0] if args else ""
        if lower in KNOWN_DRUGS:
            smi, default_target = KNOWN_DRUGS[lower]
            parts = args.split()
            target = parts[1].upper() if len(parts) > 1 else default_target
            result = self._predict(smi, target)
            result["name"] = lower.capitalize()
        else:
            # Parse "SMILES [TARGET]"
            parts = args.split()
            if not parts:
                return "Usage: predict <SMILES> [kinase_target]"
            smi = parts[0]
            target = parts[1].upper() if len(parts) > 1 else "unknown"
            result = self._predict(smi, target)

        if "error" in result:
            return f"  Error: {result['error']}"

        name_line = f"  Compound     : {result.get('name', 'query')}\n" if "name" in result else ""
        icon = "✅" if result["prediction"] == "ACTIVE" else "❌"
        lines = [
            "",
            "─" * 52,
            f"  {icon} Prediction   : {result['prediction']}",
            f"     Probability : {result['probability']:.4f}  ({result['confidence']} confidence)",
            f"     Kinase      : {result['target']}",
        ]
        if name_line:
            lines.insert(2, f"  Compound     : {result.get('name', '')}")

        lines += [
            "",
            "  Physicochemical Properties:",
            *[f"    {k:<22}: {v}" for k, v in result["properties"].items()],
            "",
            f"  Lipinski Rule-of-5 : {'✅ PASS' if result['lipinski']['pass'] else '❌ FAIL'}",
            *[f"    ⚠  {v}" for v in result["lipinski"].get("violations", [])],
            f"  Drug-likeness Score: {result['drug_score']:.2f} / 5.00",
            "─" * 52,
        ]
        return "\n".join(lines)

    def cmd_compare(self, args: str) -> str:
        parts = args.strip().split()
        if len(parts) < 2:
            return "Usage: compare <SMILES1> <SMILES2> [target]"
        smi1, smi2 = parts[0], parts[1]
        target = parts[2].upper() if len(parts) > 2 else "unknown"
        r1 = self._predict(smi1, target)
        r2 = self._predict(smi2, target)
        if "error" in r1:
            return f"  Compound 1 error: {r1['error']}"
        if "error" in r2:
            return f"  Compound 2 error: {r2['error']}"

        lines = [
            "",
            "─" * 65,
            f"  {'Property':<25} {'Compound 1':<18} {'Compound 2':<18}",
            "─" * 65,
            f"  {'Prediction':<25} {r1['prediction']:<18} {r2['prediction']:<18}",
            f"  {'Probability':<25} {r1['probability']:<18.4f} {r2['probability']:<18.4f}",
            f"  {'Confidence':<25} {r1['confidence']:<18} {r2['confidence']:<18}",
        ]
        for k in r1["properties"]:
            lines.append(f"  {k:<25} {str(r1['properties'][k]):<18} {str(r2['properties'][k]):<18}")
        lines += [
            f"  {'Lipinski Pass':<25} {'Yes' if r1['lipinski']['pass'] else 'No':<18} {'Yes' if r2['lipinski']['pass'] else 'No':<18}",
            "─" * 65,
            f"  Winner (by probability): {'Compound 1' if r1['probability'] > r2['probability'] else 'Compound 2'}",
            "─" * 65,
        ]
        return "\n".join(lines)

    def cmd_props(self, args: str) -> str:
        smi = args.strip().split()[0] if args.strip() else ""
        if not smi:
            return "Usage: props <SMILES>"
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return "  Invalid SMILES."
        all_props = {n: round(float(_DESCRIPTOR_FN[n](mol)), 3) for n in _DESCRIPTOR_FN}
        lines = ["", "─" * 45, "  Full Molecular Profile:", ""]
        for k, v in all_props.items():
            lines.append(f"    {k:<25}: {v}")
        lip = _lipinski(all_props)
        lines += [
            "",
            f"  Lipinski RO5: {'✅ PASS' if lip['pass'] else '❌ FAIL'}",
            *[f"    ⚠  {v}" for v in lip.get("violations", [])],
            "─" * 45,
        ]
        return "\n".join(lines)

    def cmd_explain(self, args: str) -> str:
        """SHAP-based explanation for a prediction."""
        try:
            import shap
        except ImportError:
            return "  SHAP not installed. Run: pip install shap"

        parts = args.strip().split()
        if not parts:
            return "Usage: explain <SMILES> [target]"
        smi = parts[0]
        target = parts[1].upper() if len(parts) > 1 else "unknown"

        X = self._featurize(smi, target)
        if X is None:
            return "  Invalid SMILES."
        X_scaled = self.scaler.transform(X)

        try:
            explainer = shap.TreeExplainer(self.model.xgb)
            shap_vals = explainer.shap_values(X_scaled)
            sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

            feat_names = self.feat_config.get("feature_names", [])
            if feat_names:
                top_idx = np.argsort(np.abs(sv))[-10:][::-1]
                lines = ["", "─" * 52, "  Top 10 Contributing Features (SHAP):", ""]
                for i in top_idx:
                    direction = "▲ active" if sv[i] > 0 else "▼ inactive"
                    name = feat_names[i] if i < len(feat_names) else f"feat_{i}"
                    lines.append(f"    {name:<30}: {sv[i]:+.4f}  {direction}")
                lines.append("─" * 52)
                return "\n".join(lines)
        except Exception as e:
            return f"  SHAP explanation failed: {e}"

        return "  Explanation unavailable."

    def cmd_drugs(self, _: str) -> str:
        lines = ["", "─" * 52, "  Known Approved TKI Drugs in Database:", ""]
        for name, (smi, target) in KNOWN_DRUGS.items():
            r = self._predict(smi, target)
            icon = "✅" if r.get("prediction") == "ACTIVE" else "❌"
            lines.append(f"    {icon} {name.capitalize():<15} [{target}]  prob={r.get('probability', '?'):.4f}")
        lines.append("─" * 52)
        return "\n".join(lines)

    def cmd_targets(self, _: str) -> str:
        lines = ["", "  Available Kinase Targets:", ""]
        for t in KINASE_TARGETS:
            lines.append(f"    • {t}")
        lines.append("")
        return "\n".join(lines)

    def process(self, user_input: str) -> str:
        parts = user_input.strip().split(None, 1)
        if not parts:
            return ""
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        self.history.append({"role": "user", "content": user_input})

        COMMANDS = {
            "predict": self.cmd_predict,
            "pred":    self.cmd_predict,
            "p":       self.cmd_predict,
            "compare": self.cmd_compare,
            "cmp":     self.cmd_compare,
            "props":   self.cmd_props,
            "explain": self.cmd_explain,
            "shap":    self.cmd_explain,
            "drugs":   self.cmd_drugs,
            "targets": self.cmd_targets,
            "help":    lambda _: HELP_TEXT,
        }

        if cmd in COMMANDS:
            response = COMMANDS[cmd](args)
        else:
            # Fuzzy: if input looks like a SMILES, auto-predict
            if _looks_like_smiles(user_input.strip()):
                response = self.cmd_predict(user_input.strip())
            # If it's a known drug name, predict it
            elif user_input.strip().lower() in KNOWN_DRUGS:
                response = self.cmd_predict(user_input.strip().lower())
            else:
                response = (
                    f"  Unknown command: '{cmd}'\n"
                    "  Type 'help' to see all commands."
                )

        self.history.append({"role": "model", "content": response})
        return response


# ── Helpers ────────────────────────────────────────────────────────────────────

desc_names_short = ["MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds"]


def _confidence(prob: float) -> str:
    p = max(prob, 1 - prob)
    if p >= 0.92:  return "Very High"
    if p >= 0.80:  return "High"
    if p >= 0.65:  return "Moderate"
    return "Low"


def _lipinski(props: dict) -> dict:
    v = []
    if props.get("MolWt", 0) > 500:    v.append(f"MolWt {props['MolWt']} > 500")
    if props.get("LogP", 0) > 5:        v.append(f"LogP {props['LogP']} > 5")
    if props.get("NumHDonors", 0) > 5:  v.append(f"HBD {props['NumHDonors']} > 5")
    if props.get("NumHAcceptors", 0) > 10: v.append(f"HBA {props['NumHAcceptors']} > 10")
    return {"pass": len(v) == 0, "violations": v}


def _drug_score(props: dict) -> float:
    score = 5.0
    if props.get("MolWt", 0) > 500:    score -= 1
    if props.get("LogP", 0) > 5:        score -= 1
    if props.get("NumHDonors", 0) > 5:  score -= 1
    if props.get("NumHAcceptors", 0) > 10: score -= 1
    if props.get("TPSA", 0) > 140:      score -= 1
    return max(score, 0.0)


def _looks_like_smiles(s: str) -> bool:
    smiles_chars = set("CNOSPFClBrI()[]=#@+\\/-0123456789cnospfbri")
    return len(s) > 5 and sum(c in smiles_chars for c in s) / len(s) > 0.8


# ── Main ───────────────────────────────────────────────────────────────────────

def demo_mode(chat: ModelChat):
    print("\n" + "=" * 55)
    print("  Demo — Predicting Known TKI Drugs")
    print("=" * 55)
    print(chat.cmd_drugs(""))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TKI Model Chat Interface")
    parser.add_argument("--demo", action="store_true", help="Run demo predictions")
    args = parser.parse_args()

    chat = ModelChat()

    print("\n" + "=" * 55)
    print("  Tyrosine Kinase Inhibitor — Model Chat")
    print("  Predictive Modeling of TKIs using ML")
    print("=" * 55)
    print(HELP_TEXT)

    if args.demo:
        demo_mode(chat)
        return

    while True:
        try:
            user_input = input("  you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q", "bye"):
            print("  Goodbye!")
            break

        response = chat.process(user_input)
        print(response)


if __name__ == "__main__":
    main()
