"""Evaluate baseline vs fine-tuned MolSight on patent validation set.

Renders val SMILES to images, predicts with each model, and compares metrics.

Usage:
    python training/evaluate.py [--finetuned_checkpoint patent_grpo_final.pth]
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VAL_CSV = os.path.join(SCRIPT_DIR, "data", "real", "patent_val.csv")
VAL_IMG_DIR = os.path.join(SCRIPT_DIR, "data", "real", "patent_val")
MOLSIGHT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "Projects", "MolSight")
BASELINE_CKPT = "pubchem_uspto_smiles_edges_30.pth"

# The 11 known error compounds from WO2026024861 (compound IDs)
KNOWN_ERROR_IDS = [
    "1", "3", "4", "7", "11", "29", "33", "57", "64", "100", "128",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finetuned_checkpoint",
        default="patent_grpo_final.pth",
        help="Filename of fine-tuned checkpoint in MolSight dir",
    )
    parser.add_argument(
        "--val_csv",
        default=VAL_CSV,
        help="Path to validation CSV",
    )
    parser.add_argument(
        "--render_fresh",
        action="store_true",
        help="Re-render val images even if they exist",
    )
    return parser.parse_args()


def render_smiles_to_png(smiles: str, path: str, size=(512, 512)) -> bool:
    """Render SMILES to PNG using RDKit."""
    try:
        from rdkit.Chem.Draw import MolDraw2DCairo

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        drawer = MolDraw2DCairo(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        with open(path, "wb") as f:
            f.write(drawer.GetDrawingText())
        return True
    except Exception:
        return False


def canonicalize(smiles: str) -> str:
    """Canonicalize SMILES, return empty string on failure."""
    if not smiles or smiles == "NONE":
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


def tanimoto(s1: str, s2: str) -> float:
    """Compute Tanimoto similarity between two SMILES."""
    try:
        m1 = Chem.MolFromSmiles(s1)
        m2 = Chem.MolFromSmiles(s2)
        if m1 is None or m2 is None:
            return 0.0
        fp1 = Chem.RDKFingerprint(m1)
        fp2 = Chem.RDKFingerprint(m2)
        return DataStructs.FingerprintSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def predict_with_molsight(image_paths: list[str], checkpoint: str) -> list[str]:
    """Run MolSight prediction on a list of image paths."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(SCRIPT_DIR), "src"
    ))
    from pdf_to_smiles.core.molsight_predictor import MolSightPredictor

    predictor = MolSightPredictor(checkpoint_path=checkpoint)
    results = []
    from PIL import Image

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            # Use predict_single to avoid retry/fallback pipeline
            smiles = predictor.predict_single(img)
            results.append(smiles or "")
        except Exception as e:
            print(f"  Error on {path}: {e}")
            results.append("")

        if (i + 1) % 20 == 0:
            print(f"  Predicted {i + 1}/{len(image_paths)}...")

    predictor.close()
    return results


def compute_metrics(gold: list[str], pred: list[str]) -> dict:
    """Compute evaluation metrics."""
    n = len(gold)
    exact = 0
    valid = 0
    tani_scores = []

    for g, p in zip(gold, pred):
        gc = canonicalize(g)
        pc = canonicalize(p)
        if pc:
            valid += 1
        if gc and pc and gc == pc:
            exact += 1
        tani_scores.append(tanimoto(g, p))

    return {
        "n": n,
        "exact_match": exact / n if n > 0 else 0,
        "valid_smiles_rate": valid / n if n > 0 else 0,
        "avg_tanimoto": float(np.mean(tani_scores)) if tani_scores else 0,
        "n_exact": exact,
        "n_valid": valid,
    }


def main():
    args = parse_args()

    # Load validation data
    if not os.path.exists(args.val_csv):
        print(f"ERROR: Val CSV not found at {args.val_csv}")
        print("Run `python training/prepare_data.py` first.")
        sys.exit(1)

    val_df = pd.read_csv(args.val_csv)
    print(f"Loaded {len(val_df)} validation compounds")

    # Ensure images exist
    image_paths = []
    gold_smiles = []
    for _, row in val_df.iterrows():
        img_path = os.path.join(SCRIPT_DIR, "data", row["file_path"])
        if not os.path.exists(img_path) or args.render_fresh:
            render_smiles_to_png(row["SMILES"], img_path)
        if os.path.exists(img_path):
            image_paths.append(img_path)
            gold_smiles.append(row["SMILES"])

    print(f"  {len(image_paths)} images ready for evaluation\n")

    # ── Baseline evaluation ───────────────────────────────────
    baseline_ckpt = os.path.join(MOLSIGHT_DIR, BASELINE_CKPT)
    print(f"=== Baseline ({BASELINE_CKPT}) ===")
    if os.path.exists(baseline_ckpt):
        baseline_preds = predict_with_molsight(image_paths, BASELINE_CKPT)
        baseline_metrics = compute_metrics(gold_smiles, baseline_preds)
        print(json.dumps(baseline_metrics, indent=2))
    else:
        print(f"  Checkpoint not found: {baseline_ckpt}")
        baseline_preds = None
        baseline_metrics = None

    # ── Fine-tuned evaluation ─────────────────────────────────
    finetuned_ckpt = os.path.join(MOLSIGHT_DIR, args.finetuned_checkpoint)
    print(f"\n=== Fine-tuned ({args.finetuned_checkpoint}) ===")
    if os.path.exists(finetuned_ckpt):
        finetuned_preds = predict_with_molsight(image_paths, args.finetuned_checkpoint)
        finetuned_metrics = compute_metrics(gold_smiles, finetuned_preds)
        print(json.dumps(finetuned_metrics, indent=2))
    else:
        print(f"  Checkpoint not found: {finetuned_ckpt}")
        finetuned_preds = None
        finetuned_metrics = None

    # ── Comparison ────────────────────────────────────────────
    if baseline_metrics and finetuned_metrics:
        print("\n=== Comparison ===")
        for key in ("exact_match", "valid_smiles_rate", "avg_tanimoto"):
            b = baseline_metrics[key]
            f = finetuned_metrics[key]
            delta = f - b
            print(f"  {key}: {b:.4f} -> {f:.4f} ({delta:+.4f})")

    # ── Per-compound detail ───────────────────────────────────
    if baseline_preds and finetuned_preds:
        print("\n=== Per-compound results ===")
        results_df = pd.DataFrame({
            "image_id": val_df["image_id"].values[:len(gold_smiles)],
            "gold": gold_smiles,
            "baseline_pred": baseline_preds,
            "finetuned_pred": finetuned_preds,
            "baseline_match": [
                canonicalize(g) == canonicalize(p) and canonicalize(p) != ""
                for g, p in zip(gold_smiles, baseline_preds)
            ],
            "finetuned_match": [
                canonicalize(g) == canonicalize(p) and canonicalize(p) != ""
                for g, p in zip(gold_smiles, finetuned_preds)
            ],
        })

        # Show improvements and regressions
        improved = results_df[~results_df["baseline_match"] & results_df["finetuned_match"]]
        regressed = results_df[results_df["baseline_match"] & ~results_df["finetuned_match"]]

        print(f"\n  Improved (baseline wrong, finetuned correct): {len(improved)}")
        for _, row in improved.head(10).iterrows():
            print(f"    {row['image_id']}: {row['gold'][:60]}")

        print(f"\n  Regressed (baseline correct, finetuned wrong): {len(regressed)}")
        for _, row in regressed.head(10).iterrows():
            print(f"    {row['image_id']}: {row['gold'][:60]}")

        # Save detailed results
        out_path = os.path.join(SCRIPT_DIR, "evaluation_results.csv")
        results_df.to_csv(out_path, index=False)
        print(f"\n  Full results saved to {out_path}")


if __name__ == "__main__":
    main()
