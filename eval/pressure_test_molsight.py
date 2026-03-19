"""Pressure test MolSight against ground-truth SMILES from training set.

Runs MolSight prediction on all 92 structure images and compares
against manually curated SMILES. Reports accuracy metrics and
categorizes failure modes.
"""

import sys
import os
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdAll')


DATA_DIR = "/Users/hanmozhang/Downloads/MolSight_training_set_03142026"

PATENTS = [
    {
        "name": "WO_2026024861_A1",
        "xlsx": os.path.join(DATA_DIR, "WO_2026024861_A1_MolSight_training_03122026.xlsx"),
        "img_dir": os.path.join(DATA_DIR, "WO_2026024861_A1_Images"),
        "id_col": "Compound_ID",
        "smiles_col": "SMILES",
        "img_col": "Image_WO_2026024861_A1",
    },
    {
        "name": "WO2020132648A1",
        "xlsx": os.path.join(DATA_DIR, "WO2020132648A1_MolSight_training_03132026.xlsx"),
        "img_dir": os.path.join(DATA_DIR, "WO2020132648A1_Images"),
        "id_col": "Compound_ID",
        "smiles_col": "SMILES",
        "img_col": "Images_WO2020132648A1",
    },
    {
        "name": "WO2025235957",
        "xlsx": os.path.join(DATA_DIR, "WO2025235957 Y220C_MolSight_training_03122026.xlsx"),
        "img_dir": os.path.join(DATA_DIR, "WO2025235957_Images"),
        "id_col": "Compound Number",
        "smiles_col": "SMILES",
        "img_col": "Image file WO2025235957",
    },
]


def canonical(smiles):
    """Canonicalize SMILES, return None if invalid."""
    if not smiles or pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def tanimoto_similarity(smi1, smi2):
    """Compute Tanimoto similarity between two SMILES."""
    try:
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def load_test_cases():
    """Load all image-backed test cases from the three patents."""
    cases = []
    for patent in PATENTS:
        df = pd.read_excel(patent["xlsx"])
        for _, row in df.iterrows():
            img_ref = row.get(patent["img_col"])
            if pd.isna(img_ref):
                continue
            img_path = os.path.join(patent["img_dir"], str(img_ref))
            # Try case-insensitive match
            if not os.path.exists(img_path):
                for ext in ['.PNG', '.png', '.jpg', '.JPG']:
                    alt = os.path.splitext(img_path)[0] + ext
                    if os.path.exists(alt):
                        img_path = alt
                        break
            if not os.path.exists(img_path):
                continue
            gt_smiles = str(row[patent["smiles_col"]])
            gt_canonical = canonical(gt_smiles)
            if gt_canonical is None:
                print(f"  WARNING: invalid ground truth SMILES for {row[patent['id_col']]}: {gt_smiles[:50]}")
                continue
            cases.append({
                "patent": patent["name"],
                "compound_id": str(row[patent["id_col"]]),
                "img_path": img_path,
                "gt_smiles": gt_smiles,
                "gt_canonical": gt_canonical,
            })
    return cases


def run_molsight_test(cases, checkpoint_path=None):
    """Run MolSight on all test cases and compare to ground truth."""
    from pdf_to_smiles.core.molsight_predictor import MolSightPredictor
    from PIL import Image

    kwargs = {}
    if checkpoint_path:
        kwargs["checkpoint_path"] = checkpoint_path
    predictor = MolSightPredictor(**kwargs)

    results = []
    start_time = time.time()

    for i, case in enumerate(cases):
        img = Image.open(case["img_path"])
        t0 = time.time()
        predicted = predictor.predict(img, high_accuracy=False)
        elapsed = time.time() - t0

        pred_canonical = canonical(predicted) if predicted else None
        exact_match = (pred_canonical == case["gt_canonical"]) if pred_canonical else False
        tanimoto = tanimoto_similarity(pred_canonical, case["gt_canonical"]) if pred_canonical else 0.0

        result = {
            **case,
            "predicted_smiles": predicted,
            "pred_canonical": pred_canonical,
            "is_valid": pred_canonical is not None,
            "exact_match": exact_match,
            "tanimoto": tanimoto,
            "time_s": elapsed,
        }
        results.append(result)

        status = "EXACT" if exact_match else f"Tan={tanimoto:.3f}" if pred_canonical else "INVALID"
        print(f"  [{i+1}/{len(cases)}] {case['patent']}/{case['compound_id']}: {status} ({elapsed:.1f}s)")

    total_time = time.time() - start_time
    predictor.close()

    return results, total_time


def analyze_results(results):
    """Print detailed accuracy analysis."""
    total = len(results)
    valid = sum(1 for r in results if r["is_valid"])
    exact = sum(1 for r in results if r["exact_match"])
    high_sim = sum(1 for r in results if r["tanimoto"] >= 0.9)
    med_sim = sum(1 for r in results if 0.5 <= r["tanimoto"] < 0.9)
    low_sim = sum(1 for r in results if r["tanimoto"] < 0.5 and r["is_valid"])
    invalid = total - valid

    print(f"\n{'='*70}")
    print(f"MOLSIGHT PRESSURE TEST RESULTS")
    print(f"{'='*70}")
    print(f"Total test cases:     {total}")
    print(f"Valid SMILES output:  {valid}/{total} ({100*valid/total:.1f}%)")
    print(f"Invalid/failed:       {invalid}/{total} ({100*invalid/total:.1f}%)")
    print(f"")
    print(f"ACCURACY BREAKDOWN:")
    print(f"  Exact match:        {exact}/{total} ({100*exact/total:.1f}%)")
    print(f"  Tanimoto >= 0.9:    {high_sim}/{total} ({100*high_sim/total:.1f}%)")
    print(f"  Tanimoto 0.5-0.9:   {med_sim}/{total} ({100*med_sim/total:.1f}%)")
    print(f"  Tanimoto < 0.5:     {low_sim}/{total} ({100*low_sim/total:.1f}%)")
    print(f"  Invalid output:     {invalid}/{total} ({100*invalid/total:.1f}%)")

    # Per-patent breakdown
    patents = set(r["patent"] for r in results)
    print(f"\nPER-PATENT BREAKDOWN:")
    for pat in sorted(patents):
        pat_results = [r for r in results if r["patent"] == pat]
        n = len(pat_results)
        ex = sum(1 for r in pat_results if r["exact_match"])
        hi = sum(1 for r in pat_results if r["tanimoto"] >= 0.9)
        inv = sum(1 for r in pat_results if not r["is_valid"])
        avg_tan = sum(r["tanimoto"] for r in pat_results) / n if n > 0 else 0
        print(f"  {pat}: {n} images, {ex} exact ({100*ex/n:.0f}%), "
              f"{hi} Tan>=0.9 ({100*hi/n:.0f}%), {inv} invalid, avg Tan={avg_tan:.3f}")

    # Failure analysis
    failures = [r for r in results if not r["exact_match"]]
    if failures:
        print(f"\nFAILURE DETAILS ({len(failures)} cases):")
        print(f"{'Patent':<20} {'ID':<10} {'Valid':>5} {'Tanimoto':>8}  GT vs Predicted")
        print(f"{'-'*90}")
        for r in sorted(failures, key=lambda x: x["tanimoto"]):
            v = "Y" if r["is_valid"] else "N"
            gt_short = r["gt_canonical"][:40] if r["gt_canonical"] else "N/A"
            pred_short = (r["pred_canonical"] or "NONE")[:40]
            print(f"{r['patent']:<20} {r['compound_id']:<10} {v:>5} {r['tanimoto']:>8.3f}  "
                  f"{gt_short}  |  {pred_short}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned checkpoint (relative to MolSight dir)")
    args = parser.parse_args()

    if args.checkpoint:
        print(f"Using checkpoint: {args.checkpoint}")

    print("Loading test cases...")
    cases = load_test_cases()
    print(f"Loaded {len(cases)} test cases with images\n")

    print("Running MolSight predictions...")
    results, total_time = run_molsight_test(cases, checkpoint_path=args.checkpoint)

    print(f"\nTotal time: {total_time:.1f}s ({total_time/len(cases):.1f}s per image)")

    analyze_results(results)

    # Save results to CSV
    suffix = "_sft" if args.checkpoint and "sft" in args.checkpoint else ""
    out_path = os.path.join(
        os.path.dirname(__file__), "output", f"molsight_pressure_test{suffix}.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
