"""Prepare patent SMILES data for MolSight fine-tuning.

Reads 4 Excel files containing ground-truth SMILES, validates with RDKit,
creates 90/10 train/val split stratified by patent, and renders val images.

Usage:
    python training/prepare_data.py
"""

import os
import sys
import glob

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# ──────────────────────────────────────────────────────────────
# Patent Excel files — (path, sheet_name or None for auto)
# ──────────────────────────────────────────────────────────────

HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, "Downloads", "SMILES_lists_inspected_03102026")

PATENT_FILES = [
    {
        "patent": "WO2026024861",
        "path": os.path.join(BASE, "WO_2026024861_A1_MolSight_training_03102026.xlsx"),
        "sheet": "WO_2026024861_A1_1st_pass",
    },
    {
        "patent": "WO2020132648",
        "path": os.path.join(BASE, "WO2020132648A1_MolSight_training_03102026.xlsx"),
        "sheet": None,  # single sheet
    },
    {
        "patent": "WO2025235957",
        "path": os.path.join(BASE, "WO2025235957 Y220C_MolSight_training_03102026.xlsx"),
        "sheet": "Combined_cleaned_wSMILES",
    },
    {
        "patent": "WO2025184668",
        "path": os.path.join(BASE, "WO_2025184668_A1_MolSight_training_03102026.xlsx"),
        "sheet": "ChemOffice1",
    },
]

# ──────────────────────────────────────────────────────────────
# Output paths (relative to project root)
# ──────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(SCRIPT_DIR, "data", "pubchem", "train_1m.csv")
VAL_CSV = os.path.join(SCRIPT_DIR, "data", "real", "patent_val.csv")
VAL_IMG_DIR = os.path.join(SCRIPT_DIR, "data", "real", "patent_val")

VAL_FRACTION = 0.10
SEED = 42


def resolve_path(entry: dict) -> str:
    """Resolve file path, using glob if needed."""
    if entry["path"] and os.path.exists(entry["path"]):
        return entry["path"]
    if "glob_dir" in entry:
        matches = glob.glob(os.path.join(entry["glob_dir"], entry["glob_pattern"]))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Cannot find Excel file for {entry['patent']}")


def read_patent(entry: dict) -> pd.DataFrame:
    """Read a single patent Excel file and return DataFrame with SMILES + patent columns."""
    path = resolve_path(entry)
    kwargs = {}
    if entry.get("sheet"):
        kwargs["sheet_name"] = entry["sheet"]
    df = pd.read_excel(path, **kwargs)

    if "SMILES" not in df.columns:
        raise ValueError(f"No 'SMILES' column in {path} (columns: {list(df.columns)})")

    out = pd.DataFrame()
    out["SMILES"] = df["SMILES"].astype(str).str.strip()
    out["patent"] = entry["patent"]
    return out


def validate_smiles(smiles: str) -> str | None:
    """Canonicalize SMILES, return None if invalid."""
    if not smiles or smiles in ("nan", "None", ""):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() < 3:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def render_smiles_to_png(smiles: str, path: str, size: tuple = (512, 512)) -> bool:
    """Render a SMILES string to a PNG file using RDKit."""
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
    except Exception as e:
        print(f"  Warning: could not render {smiles[:40]}... — {e}")
        return False


def main():
    np.random.seed(SEED)

    # ── Read all patents ──────────────────────────────────────
    all_dfs = []
    for entry in PATENT_FILES:
        try:
            df = read_patent(entry)
            print(f"  {entry['patent']}: {len(df)} rows from {resolve_path(entry)}")
            all_dfs.append(df)
        except Exception as e:
            print(f"  ERROR reading {entry['patent']}: {e}", file=sys.stderr)
            sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal raw rows: {len(combined)}")

    # ── Validate & canonicalize ───────────────────────────────
    combined["canonical"] = combined["SMILES"].apply(validate_smiles)
    n_invalid = combined["canonical"].isna().sum()
    combined = combined.dropna(subset=["canonical"]).reset_index(drop=True)
    print(f"Dropped {n_invalid} invalid SMILES, {len(combined)} remaining")

    # Deduplicate by canonical SMILES
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=["canonical"]).reset_index(drop=True)
    print(f"Dropped {n_before - len(combined)} duplicates, {len(combined)} remaining")

    # ── Stratified train/val split ────────────────────────────
    val_dfs = []
    train_dfs = []
    for patent, group in combined.groupby("patent"):
        n_val = max(1, int(len(group) * VAL_FRACTION))
        shuffled = group.sample(frac=1, random_state=SEED).reset_index(drop=True)
        val_dfs.append(shuffled.iloc[:n_val])
        train_dfs.append(shuffled.iloc[n_val:])

    val_df = pd.concat(val_dfs, ignore_index=True)
    train_df = pd.concat(train_dfs, ignore_index=True)

    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}")
    for patent in combined["patent"].unique():
        n_t = (train_df["patent"] == patent).sum()
        n_v = (val_df["patent"] == patent).sum()
        print(f"  {patent}: train={n_t}, val={n_v}")

    # ── Write train CSV (SMILES column only — matches pubchem format) ──
    os.makedirs(os.path.dirname(TRAIN_CSV), exist_ok=True)
    train_out = pd.DataFrame({"SMILES": train_df["canonical"]})
    train_out.to_csv(TRAIN_CSV, index=False)
    print(f"\nWrote train CSV: {TRAIN_CSV} ({len(train_out)} rows)")

    # ── Render val images & write val CSV ─────────────────────
    os.makedirs(VAL_IMG_DIR, exist_ok=True)

    val_records = []
    for i, row in val_df.iterrows():
        image_id = f"patent_val_{i:04d}"
        img_path = os.path.join(VAL_IMG_DIR, f"{image_id}.png")
        rel_path = os.path.join("real", "patent_val", f"{image_id}.png")

        if render_smiles_to_png(row["canonical"], img_path):
            val_records.append({
                "image_id": image_id,
                "file_path": rel_path,
                "SMILES": row["canonical"],
            })

    val_out = pd.DataFrame(val_records)
    os.makedirs(os.path.dirname(VAL_CSV), exist_ok=True)
    val_out.to_csv(VAL_CSV, index=False)
    print(f"Wrote val CSV: {VAL_CSV} ({len(val_out)} rows)")
    print(f"Rendered {len(val_records)} val images to {VAL_IMG_DIR}/")

    print("\nDone!")


if __name__ == "__main__":
    main()
