"""Prepare real patent image training dataset for MolSight fine-tuning.

Creates a CSV mapping real patent structure images to ground-truth SMILES,
compatible with MolSight's USPTODataset format.

This enables training on actual patent figures (ChemDraw-rendered, scanned,
compressed) rather than only Indigo-rendered clean images.

Usage:
    python training/prepare_real_image_dataset.py

Output:
    training/data/real/patent_train.csv  — CSV with file_path + SMILES columns
    training/data/real/patent_images/    — copied patent images
"""

import os
import sys
import shutil

import pandas as pd

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdAll")

# ──────────────────────────────────────────────────────────────
# Source data: the MolSight training set with real images
# ──────────────────────────────────────────────────────────────

HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME, "Downloads", "MolSight_training_set_03142026")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

# Output paths
OUT_IMG_DIR = os.path.join(SCRIPT_DIR, "data", "real", "patent_images")
OUT_TRAIN_CSV = os.path.join(SCRIPT_DIR, "data", "real", "patent_train.csv")
OUT_VAL_CSV = os.path.join(SCRIPT_DIR, "data", "real", "patent_realimg_val.csv")

SEED = 42
VAL_FRACTION = 0.10


def canonical(smiles):
    """Canonicalize SMILES, return None if invalid."""
    if not smiles or pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None or mol.GetNumAtoms() < 3:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def resolve_image_path(img_ref, img_dir):
    """Find the actual image file, handling case-insensitive extensions."""
    if pd.isna(img_ref):
        return None
    img_path = os.path.join(img_dir, str(img_ref))
    if os.path.exists(img_path):
        return img_path
    # Try alternate extensions
    base = os.path.splitext(img_path)[0]
    for ext in [".PNG", ".png", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
        alt = base + ext
        if os.path.exists(alt):
            return alt
    return None


def main():
    import numpy as np
    np.random.seed(SEED)

    print("=" * 60)
    print("Prepare Real Patent Image Dataset for MolSight Training")
    print("=" * 60)

    os.makedirs(OUT_IMG_DIR, exist_ok=True)

    records = []
    skipped_no_img = 0
    skipped_bad_smiles = 0

    for patent in PATENTS:
        print(f"\nProcessing {patent['name']}...")
        if not os.path.exists(patent["xlsx"]):
            print(f"  WARNING: Excel file not found: {patent['xlsx']}")
            continue

        df = pd.read_excel(patent["xlsx"])
        n_found = 0

        for _, row in df.iterrows():
            # Find image
            img_path = resolve_image_path(row.get(patent["img_col"]), patent["img_dir"])
            if img_path is None:
                skipped_no_img += 1
                continue

            # Validate SMILES
            gt_smiles = str(row[patent["smiles_col"]])
            can_smiles = canonical(gt_smiles)
            if can_smiles is None:
                skipped_bad_smiles += 1
                continue

            # Copy image to output directory with unique name
            compound_id = str(row[patent["id_col"]])
            ext = os.path.splitext(img_path)[1]
            out_name = f"{patent['name']}_{compound_id}{ext}"
            out_path = os.path.join(OUT_IMG_DIR, out_name)
            # Relative path from data/ directory (what MolSight expects)
            rel_path = os.path.join("real", "patent_images", out_name)

            if not os.path.exists(out_path):
                shutil.copy2(img_path, out_path)

            records.append({
                "file_path": rel_path,
                "SMILES": can_smiles,
                "patent": patent["name"],
                "compound_id": compound_id,
            })
            n_found += 1

        print(f"  Found {n_found} image-SMILES pairs")

    print(f"\nTotal: {len(records)} pairs")
    print(f"Skipped: {skipped_no_img} no image, {skipped_bad_smiles} bad SMILES")

    if not records:
        print("ERROR: No records to write!")
        sys.exit(1)

    # Stratified train/val split
    full_df = pd.DataFrame(records)
    train_dfs = []
    val_dfs = []

    for patent, group in full_df.groupby("patent"):
        n_val = max(1, int(len(group) * VAL_FRACTION))
        shuffled = group.sample(frac=1, random_state=SEED).reset_index(drop=True)
        val_dfs.append(shuffled.iloc[:n_val])
        train_dfs.append(shuffled.iloc[n_val:])

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)

    # Write CSVs (only file_path and SMILES — what MolSight expects)
    train_out = train_df[["file_path", "SMILES"]]
    train_out.to_csv(OUT_TRAIN_CSV, index=False)
    print(f"\nTrain CSV: {OUT_TRAIN_CSV} ({len(train_out)} rows)")

    val_out = val_df[["file_path", "SMILES"]]
    val_out.to_csv(OUT_VAL_CSV, index=False)
    print(f"Val CSV:   {OUT_VAL_CSV} ({len(val_out)} rows)")

    # Per-patent breakdown
    print("\nPer-patent breakdown:")
    for patent in full_df["patent"].unique():
        n_train = (train_df["patent"] == patent).sum()
        n_val = (val_df["patent"] == patent).sum()
        print(f"  {patent}: train={n_train}, val={n_val}")

    print("\n" + "=" * 60)
    print("Done! To use in MolSight training:")
    print("  1. Add 'patent' to data_path_map in MolSight/train.py:")
    print("     'patent': 'real/patent_train.csv'")
    print("  2. Add 'patent' handling in HybridDataset (same as 'uspto')")
    print("  3. Train with: --train_datasets pubchem,patent")
    print("=" * 60)


if __name__ == "__main__":
    main()
