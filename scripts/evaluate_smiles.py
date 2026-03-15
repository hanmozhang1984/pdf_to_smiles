#!/usr/bin/env python3
"""Evaluate pipeline SMILES accuracy against ground-truth Excel data.

Compares pipeline output CSVs against ground-truth Excel files that contain
verified (Compound_ID, SMILES) pairs. Supports batch evaluation across
multiple patents.

Usage:
    # Single patent:
    python scripts/evaluate_smiles.py \\
        --pipeline-csv output.csv \\
        --ground-truth verified.xlsx

    # Batch (multiple patents):
    python scripts/evaluate_smiles.py \\
        --batch-dir ~/Downloads/MolSight_training_set_03142026 \\
        --pipeline-dir ~/Downloads/pipeline_outputs

    # Just evaluate a single CSV against a single Excel:
    python scripts/evaluate_smiles.py \\
        --pipeline-csv US20240366598A1_smiles_1926_XLM.csv \\
        --ground-truth GLP_ground_truth.xlsx

Metrics computed per-compound and aggregate:
  - Exact SMILES match (canonical)
  - Tanimoto similarity (fingerprint-based)
  - MW accuracy (within 1 Da)
  - Compound ID assignment accuracy
  - Detection recall (how many GT compounds were found at all)
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def normalize_id(raw_id: str) -> str:
    """Normalize a compound ID for matching."""
    s = str(raw_id).strip()
    # Remove trailing .0 from float-parsed integers
    if s.endswith(".0"):
        s = s[:-2]
    # Strip common prefixes
    s = re.sub(
        r"^(?:Example|Compound|Cpd\.?|Cmpd\.?|Ex\.?|No\.?|#)\s*",
        "", s, flags=re.IGNORECASE,
    ).strip()
    # Strip leading zeros from purely numeric IDs
    m = re.match(r"^0+(\d+)$", s)
    if m:
        s = m.group(1)
    return s


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string using RDKit."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def compute_tanimoto(smiles_a: str, smiles_b: str) -> float:
    """Compute Tanimoto similarity between two SMILES using Morgan fingerprints."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs

        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            return 0.0

        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    except Exception:
        return 0.0


def compute_mw(smiles: str) -> Optional[float]:
    """Compute molecular weight from SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Descriptors.MolWt(mol)
    except Exception:
        return None


def load_ground_truth_excel(excel_path: str) -> Dict[str, str]:
    """Load ground truth from Excel: {normalized_id: SMILES}.

    Handles varying column names across the training set Excel files.
    """
    try:
        import openpyxl
    except ImportError:
        print("Error: openpyxl required. pip install openpyxl")
        sys.exit(1)

    wb = openpyxl.load_workbook(excel_path, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not rows:
        return {}

    header = [str(h).lower().strip() if h else "" for h in rows[0]]

    # Find ID column
    id_idx = None
    for i, h in enumerate(header):
        if "compound" in h and ("id" in h or "number" in h):
            id_idx = i
            break
        if h in ("compound_id", "id", "example", "compound number"):
            id_idx = i
            break
    if id_idx is None:
        id_idx = 0  # First column fallback

    # Find SMILES column
    smiles_idx = None
    for i, h in enumerate(header):
        if "smiles" in h:
            smiles_idx = i
            break
    if smiles_idx is None:
        smiles_idx = 1  # Second column fallback

    gt = {}
    for row in rows[1:]:
        if len(row) <= max(id_idx, smiles_idx):
            continue
        raw_id = row[id_idx]
        smiles = row[smiles_idx]
        if raw_id is None or smiles is None:
            continue
        nid = normalize_id(str(raw_id))
        if nid and str(smiles).strip():
            gt[nid] = str(smiles).strip()

    return gt


def load_pipeline_csv(csv_path: str) -> List[Dict]:
    """Load pipeline output CSV."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def evaluate_single(
    pipeline_rows: List[Dict],
    ground_truth: Dict[str, str],
    verbose: bool = False,
) -> Dict:
    """Evaluate pipeline output against ground truth.

    Returns a dict with detailed metrics.
    """
    # Build pipeline lookup: normalized_id → best row
    # (prefer single-compound SMILES over multi-compound)
    pipeline_by_id: Dict[str, List[Dict]] = defaultdict(list)
    for row in pipeline_rows:
        cid = row.get("Compound_ID", "")
        if not cid:
            continue
        nid = normalize_id(cid)
        pipeline_by_id[nid].append(row)

    # For each ground-truth compound, find the best pipeline match
    results = []
    matched = 0
    exact_match = 0
    tanimoto_sum = 0.0
    mw_match = 0
    id_found = 0
    multi_structure_count = 0

    for gt_id, gt_smiles in sorted(ground_truth.items()):
        gt_canonical = canonicalize_smiles(gt_smiles)
        gt_mw = compute_mw(gt_smiles)

        entry = {
            "gt_id": gt_id,
            "gt_smiles": gt_smiles,
            "gt_canonical": gt_canonical,
            "gt_mw": gt_mw,
            "found": False,
            "pipeline_smiles": None,
            "pipeline_canonical": None,
            "exact_match": False,
            "tanimoto": 0.0,
            "mw_error": None,
            "is_multi_structure": False,
            "compound_type": None,
        }

        candidates = pipeline_by_id.get(gt_id, [])
        if not candidates:
            results.append(entry)
            continue

        id_found += 1
        entry["found"] = True

        # Pick the best candidate: prefer single-compound, valid SMILES
        best = None
        best_tanimoto = -1.0
        for cand in candidates:
            smi = cand.get("Canonical_SMILES") or cand.get("SMILES", "")
            if not smi:
                continue
            is_multi = "." in smi
            can = canonicalize_smiles(smi)
            if can and gt_canonical:
                t = compute_tanimoto(can, gt_canonical)
            else:
                t = 0.0

            # Prefer single-compound with higher Tanimoto
            score = t + (0.0 if is_multi else 0.5)
            if score > best_tanimoto:
                best_tanimoto = score
                best = cand

        if best is None:
            results.append(entry)
            continue

        matched += 1
        p_smi = best.get("Canonical_SMILES") or best.get("SMILES", "")
        p_canonical = canonicalize_smiles(p_smi)
        is_multi = "." in p_smi

        entry["pipeline_smiles"] = p_smi
        entry["pipeline_canonical"] = p_canonical
        entry["is_multi_structure"] = is_multi
        entry["compound_type"] = best.get("Compound_Type", "")

        if is_multi:
            multi_structure_count += 1

        # Exact match
        if gt_canonical and p_canonical and gt_canonical == p_canonical:
            entry["exact_match"] = True
            exact_match += 1

        # Tanimoto
        if gt_canonical and p_canonical:
            t = compute_tanimoto(gt_canonical, p_canonical)
            entry["tanimoto"] = t
            tanimoto_sum += t
        elif entry["exact_match"]:
            entry["tanimoto"] = 1.0
            tanimoto_sum += 1.0

        # MW accuracy
        p_mw = compute_mw(p_smi)
        if gt_mw and p_mw:
            entry["mw_error"] = abs(gt_mw - p_mw)
            if entry["mw_error"] < 1.0:
                mw_match += 1

        results.append(entry)

    total_gt = len(ground_truth)
    avg_tanimoto = tanimoto_sum / matched if matched > 0 else 0.0

    # Compounds in pipeline but NOT in ground truth
    pipeline_ids = set(pipeline_by_id.keys())
    gt_ids = set(ground_truth.keys())
    extra_ids = pipeline_ids - gt_ids

    summary = {
        "total_gt_compounds": total_gt,
        "total_pipeline_rows": len(pipeline_rows),
        "gt_found_in_pipeline": id_found,
        "gt_with_smiles_match": matched,
        "detection_recall": round(id_found / total_gt, 4) if total_gt > 0 else 0.0,
        "exact_smiles_match": exact_match,
        "exact_match_rate": round(exact_match / matched, 4) if matched > 0 else 0.0,
        "avg_tanimoto": round(avg_tanimoto, 4),
        "mw_match_count": mw_match,
        "mw_match_rate": round(mw_match / matched, 4) if matched > 0 else 0.0,
        "multi_structure_count": multi_structure_count,
        "multi_structure_rate": round(multi_structure_count / matched, 4) if matched > 0 else 0.0,
        "extra_pipeline_ids": len(extra_ids),
        "details": results,
    }

    if verbose:
        print("\nPer-compound results:")
        print(f"{'ID':>8s}  {'Found':>5s}  {'Exact':>5s}  {'Tanimoto':>8s}  {'MW_err':>8s}  {'Multi':>5s}")
        print("-" * 55)
        for r in results:
            found = "Y" if r["found"] else "N"
            exact = "Y" if r["exact_match"] else "N"
            tani = f"{r['tanimoto']:.3f}" if r["found"] else "-"
            mw_err = f"{r['mw_error']:.1f}" if r["mw_error"] is not None else "-"
            multi = "Y" if r["is_multi_structure"] else "N"
            print(f"{r['gt_id']:>8s}  {found:>5s}  {exact:>5s}  {tani:>8s}  {mw_err:>8s}  {multi:>5s}")

    return summary


def print_summary(summary: Dict, label: str = "") -> None:
    """Print evaluation summary."""
    print()
    print("=" * 65)
    if label:
        print(f"  {label}")
        print("=" * 65)
    print(f"  Ground truth compounds:       {summary['total_gt_compounds']}")
    print(f"  Pipeline output rows:         {summary['total_pipeline_rows']}")
    print(f"  GT compounds found by ID:     {summary['gt_found_in_pipeline']} "
          f"({summary['detection_recall']:.1%} recall)")
    print(f"  GT compounds with SMILES:     {summary['gt_with_smiles_match']}")
    print()
    print(f"  Exact SMILES match:           {summary['exact_smiles_match']} "
          f"({summary['exact_match_rate']:.1%})")
    print(f"  Average Tanimoto similarity:  {summary['avg_tanimoto']:.4f}")
    print(f"  MW within 1 Da:               {summary['mw_match_count']} "
          f"({summary['mw_match_rate']:.1%})")
    print()
    print(f"  Multi-structure crops:        {summary['multi_structure_count']} "
          f"({summary['multi_structure_rate']:.1%})")
    print(f"  Extra pipeline IDs (not in GT): {summary['extra_pipeline_ids']}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline SMILES accuracy against ground-truth Excel"
    )
    parser.add_argument(
        "--pipeline-csv",
        help="Path to pipeline output CSV",
    )
    parser.add_argument(
        "--ground-truth",
        help="Path to ground truth Excel file",
    )
    parser.add_argument(
        "--batch",
        nargs="*",
        metavar="CSV:XLSX",
        help="Batch mode: pairs of csv:xlsx paths (e.g., output.csv:verified.xlsx)",
    )
    parser.add_argument(
        "--output",
        help="Save detailed results JSON to this path",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-compound details",
    )

    args = parser.parse_args()

    # Suppress RDKit warnings
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        print("Error: RDKit required. pip install rdkit")
        sys.exit(1)

    pairs = []

    if args.pipeline_csv and args.ground_truth:
        pairs.append((args.pipeline_csv, args.ground_truth))

    if args.batch:
        for pair_str in args.batch:
            if ":" in pair_str:
                csv_path, xlsx_path = pair_str.split(":", 1)
                pairs.append((csv_path.strip(), xlsx_path.strip()))
            else:
                print(f"Warning: skipping invalid batch pair '{pair_str}' (use csv:xlsx format)")

    if not pairs:
        parser.print_help()
        print("\nError: Provide --pipeline-csv + --ground-truth, or --batch pairs")
        sys.exit(1)

    all_results = {}
    aggregate = {
        "total_gt_compounds": 0,
        "total_pipeline_rows": 0,
        "gt_found_in_pipeline": 0,
        "gt_with_smiles_match": 0,
        "exact_smiles_match": 0,
        "mw_match_count": 0,
        "multi_structure_count": 0,
        "tanimoto_sum": 0.0,
        "matched_count": 0,
    }

    for csv_path, xlsx_path in pairs:
        label = os.path.basename(csv_path)
        print(f"\nLoading: {label}")
        print(f"  Ground truth: {os.path.basename(xlsx_path)}")

        gt = load_ground_truth_excel(xlsx_path)
        print(f"  Loaded {len(gt)} ground-truth compounds")

        rows = load_pipeline_csv(csv_path)
        print(f"  Loaded {len(rows)} pipeline rows")

        summary = evaluate_single(rows, gt, verbose=args.verbose)
        print_summary(summary, label)

        all_results[label] = summary

        # Aggregate
        aggregate["total_gt_compounds"] += summary["total_gt_compounds"]
        aggregate["total_pipeline_rows"] += summary["total_pipeline_rows"]
        aggregate["gt_found_in_pipeline"] += summary["gt_found_in_pipeline"]
        aggregate["gt_with_smiles_match"] += summary["gt_with_smiles_match"]
        aggregate["exact_smiles_match"] += summary["exact_smiles_match"]
        aggregate["mw_match_count"] += summary["mw_match_count"]
        aggregate["multi_structure_count"] += summary["multi_structure_count"]
        m = summary["gt_with_smiles_match"]
        aggregate["tanimoto_sum"] += summary["avg_tanimoto"] * m
        aggregate["matched_count"] += m

    # Print aggregate if multiple patents
    if len(pairs) > 1:
        total = aggregate["total_gt_compounds"]
        found = aggregate["gt_found_in_pipeline"]
        matched = aggregate["matched_count"]
        exact = aggregate["exact_smiles_match"]
        mw = aggregate["mw_match_count"]
        multi = aggregate["multi_structure_count"]
        avg_t = aggregate["tanimoto_sum"] / matched if matched > 0 else 0.0

        print("\n" + "=" * 65)
        print("  AGGREGATE ACROSS ALL PATENTS")
        print("=" * 65)
        print(f"  Total GT compounds:           {total}")
        print(f"  Detection recall:             {found}/{total} ({found/total:.1%})" if total else "")
        print(f"  Exact SMILES match:           {exact}/{matched} ({exact/matched:.1%})" if matched else "")
        print(f"  Average Tanimoto:             {avg_t:.4f}")
        print(f"  MW within 1 Da:               {mw}/{matched} ({mw/matched:.1%})" if matched else "")
        print(f"  Multi-structure crops:         {multi}/{matched} ({multi/matched:.1%})" if matched else "")
        print("=" * 65)

    # Save results
    if args.output:
        # Strip non-serializable details for JSON
        save_data = {}
        for k, v in all_results.items():
            save_copy = {kk: vv for kk, vv in v.items() if kk != "details"}
            save_copy["details"] = [
                {dk: dv for dk, dv in d.items()}
                for d in v["details"]
            ]
            save_data[k] = save_copy

        with open(args.output, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
