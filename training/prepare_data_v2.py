"""Prepare expanded fine-tuning dataset for MolSight Phase 2.

Combines:
  1. Original patent SMILES (3 patents, ~1,100 compounds)
  2. Hard examples from pressure test failures (oversampled 5×)
  3. Macrocyclic SMILES from PubChem (to fix macrocycle blindness)
  4. Diverse heterocyclic scaffolds from ChEMBL/PubChem

Usage:
    python training/prepare_data_v2.py
"""

import os
import sys
import re
import json
import time
import urllib.request
import urllib.parse

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────

HOME = os.path.expanduser("~")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Source data directories
TRAINING_DIR = os.path.join(HOME, "Downloads", "MolSight_training_set_03142026")
INSPECTED_DIR = os.path.join(HOME, "Downloads", "SMILES_lists_inspected_03102026")

# Pressure test results (for hard-example mining)
PRESSURE_TEST_CSV = os.path.join(
    os.path.dirname(SCRIPT_DIR), "eval", "output", "molsight_pressure_test.csv"
)

# Output paths
TRAIN_CSV = os.path.join(SCRIPT_DIR, "data", "pubchem", "train_1m.csv")
VAL_CSV = os.path.join(SCRIPT_DIR, "data", "real", "patent_val.csv")
VAL_IMG_DIR = os.path.join(SCRIPT_DIR, "data", "real", "patent_val")
SUPPLEMENTARY_DIR = os.path.join(SCRIPT_DIR, "data", "supplementary")

VAL_FRACTION = 0.10
HARD_EXAMPLE_REPEATS = 5  # Oversample structural failures
SEED = 42

# ──────────────────────────────────────────────────────────────
# Patent Excel files
# ──────────────────────────────────────────────────────────────

PATENT_FILES = [
    {
        "patent": "WO2026024861",
        "path": os.path.join(TRAINING_DIR, "WO_2026024861_A1_MolSight_training_03122026.xlsx"),
        "fallback": os.path.join(INSPECTED_DIR, "WO_2026024861_A1_MolSight_training_03102026.xlsx"),
    },
    {
        "patent": "WO2020132648",
        "path": os.path.join(TRAINING_DIR, "WO2020132648A1_MolSight_training_03132026.xlsx"),
        "fallback": os.path.join(INSPECTED_DIR, "WO2020132648A1_MolSight_training_03102026.xlsx"),
    },
    {
        "patent": "WO2025235957",
        "path": os.path.join(TRAINING_DIR, "WO2025235957 Y220C_MolSight_training_03122026.xlsx"),
        "fallback": os.path.join(INSPECTED_DIR, "WO2025235957 Y220C_MolSight_training_03102026.xlsx"),
    },
]


# ──────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────


def validate_smiles(smiles: str) -> str | None:
    """Canonicalize SMILES, return None if invalid or too small."""
    if not smiles or smiles in ("nan", "None", ""):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None or mol.GetNumAtoms() < 3:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def is_macrocycle(smiles: str) -> bool:
    """Check if SMILES contains a macrocyclic ring (>= 12 atoms)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) >= 12:
            return True
    return False


def fetch_pubchem_smiles(query: str, max_results: int = 500) -> list[str]:
    """Fetch SMILES from PubChem using a structure search query."""
    try:
        encoded = urllib.parse.quote(query)
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
            f"fastsubstructure/smarts/{encoded}/property/CanonicalSMILES/JSON"
            f"?MaxRecords={max_results}"
        )
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        props = data.get("PropertyTable", {}).get("Properties", [])
        return [p["CanonicalSMILES"] for p in props if "CanonicalSMILES" in p]
    except Exception as e:
        print(f"  PubChem query failed: {e}")
        return []


def fetch_pubchem_by_cids(cids: list[int]) -> list[str]:
    """Fetch SMILES for specific PubChem CIDs."""
    results = []
    # Process in batches of 100
    for i in range(0, len(cids), 100):
        batch = cids[i : i + 100]
        cid_str = ",".join(str(c) for c in batch)
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
            f"cid/{cid_str}/property/IsomericSMILES/JSON"
        )
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            props = data.get("PropertyTable", {}).get("Properties", [])
            results.extend(p["SMILES"] for p in props if "SMILES" in p)
        except Exception as e:
            print(f"  PubChem CID batch failed: {e}")
        time.sleep(0.3)
    return results


# ──────────────────────────────────────────────────────────────
# Data loading functions
# ──────────────────────────────────────────────────────────────


def load_patent_smiles() -> pd.DataFrame:
    """Load SMILES from all patent Excel files."""
    all_dfs = []
    for entry in PATENT_FILES:
        path = entry["path"] if os.path.exists(entry["path"]) else entry.get("fallback")
        if not path or not os.path.exists(path):
            print(f"  WARNING: No file found for {entry['patent']}")
            continue
        df = pd.read_excel(path)
        smiles_col = [c for c in df.columns if "smiles" in c.lower()]
        if not smiles_col:
            print(f"  WARNING: No SMILES column in {path}")
            continue
        out = pd.DataFrame()
        out["SMILES"] = df[smiles_col[0]].astype(str).str.strip()
        out["patent"] = entry["patent"]
        out["source"] = "patent"
        print(f"  {entry['patent']}: {len(out)} rows from {os.path.basename(path)}")
        all_dfs.append(out)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def load_hard_examples() -> pd.DataFrame:
    """Load structural failure SMILES from pressure test, oversampled."""
    if not os.path.exists(PRESSURE_TEST_CSV):
        print("  WARNING: Pressure test results not found — skipping hard examples")
        return pd.DataFrame()

    df = pd.read_csv(PRESSURE_TEST_CSV)
    # Structural failures: wrong connectivity (Tanimoto < 0.95)
    failures = df[(~df["exact_match"]) & (df["tanimoto"] < 0.95)]
    if failures.empty:
        return pd.DataFrame()

    hard = pd.DataFrame()
    hard["SMILES"] = failures["gt_canonical"]
    hard["patent"] = failures["patent"]
    hard["source"] = "hard_example"

    # Oversample: repeat hard examples
    hard_repeated = pd.concat([hard] * HARD_EXAMPLE_REPEATS, ignore_index=True)
    print(f"  Hard examples: {len(failures)} unique × {HARD_EXAMPLE_REPEATS} = {len(hard_repeated)}")
    return hard_repeated


def generate_macrocycle_smiles() -> pd.DataFrame:
    """Generate macrocyclic SMILES — curated set + PubChem fetch."""
    print("  Generating macrocycle training data...")

    # Curated macrocyclic SMILES representative of common drug-like macrocycles
    curated_macrocycles = [
        # PROTACs and molecular glues (linker-based macrocycles)
        "O=C(NCCCCCCCNC(=O)c1ccc(N2CCN(c3ccccc3)CC2)cc1)c1ccc2[nH]ccc2c1",
        "O=C(NCCOCCOCCNC(=O)c1cccc(NC(=O)c2ccc(F)cc2)c1)c1ccc2c(c1)OCO2",
        # Macrolide-type structures
        "CC1CC(=O)OC(CC(OC)CC(CC=CC(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)C(=O)C(C)CC(C)C(=O)O1)C)C",
        "CCC(=O)OC1CC(C)C(=O)C(C)CC(OC2OC(C)CC(N(C)C)C2O)C(C)CC(CC=CC(=O)O1)OC",
        # Cyclic peptide-like
        "O=C1CCC(NC(=O)CNC(=O)c2ccccc2NC(=O)CNC(=O)c2ccccc2N1)=O",
        "CC(NC(=O)C(CC(=O)N1CCCC1C(=O)NC(C)C(=O)NC(Cc1ccccc1)C(=O)O)NC(=O)C(C)N)C(=O)O",
        # PROTAC-like with long alkyl linkers (similar to compounds 860/862)
        "C=C(c1nc2ccccc2c(NC3CCN(CCCCCCCCNC(=O)c4ccnn4C)CC3)c1)C(F)(F)F",
        "C=C(c1nc2ccccc2c(NC3CCN(CCCCCCCCCNC(=O)c4ccnn4C)CC3)c1)C(F)(F)F",
        "O=C(c1ccnn1CCCCCCCCN1CCC(Nc2cccc3c2cc(C=CCNC(=O)c2ccccc2)nc3)CC1)NC1CC1",
        # Ring sizes 12-20
        "C1CCCCCCCCCCCC1",  # cyclododecane
        "C1CCCCCCCCCCCCCCC1",  # cyclohexadecane
        "O=C1CCCCCCCCCCNC(=O)CCCCCCCCCCN1",
        "c1ccc(CCCCOCCCCc2ccccc2OCCCC2ccccc21)cc1",
        # Drug-like macrocycles
        "CC1CCC(NC(=O)c2cc(OC)cc(OC)c2)C(=O)NC(Cc2ccc(O)cc2)C(=O)NC(CCCCN)C(=O)N1",
        "O=C(NC1CCCCNC(=O)c2ccccc2NC(=O)C(Cc2c[nH]c3ccccc23)NC1=O)c1ccc(F)cc1",
        # Macrocyclic kinase inhibitors
        "C(=O)(NC1CCN(CCCCCC2=NC3=CC=CC=C3C(=O)N2)CC1)C1=CC=CC=C1",
        "O=C1NC2=CC=CC=C2C(=O)NCCCCCN1CC1=CC=CC=C1",
        # PEG-linked macrocycles (common in PROTACs)
        "O=C(NCCOCCOCCOCCNC(=O)c1ccc(NC(=O)c2ccccc2)cc1)c1ccc(F)cc1",
        "O=C(NCCOCCOCCOCCOCCCNC(=O)c1ccncc1)c1ccc(Cl)cc1",
        # Macrocyclic HCV protease inhibitor-type
        "CC1=CC=C(C=C1)S(=O)(=O)NC(CC2CCCCC2)C(=O)N3CCCC3C(=O)NC(CC4=CC=CC=C4)C(=O)O",
        # Macrocyclic with alkyl + amide linkers (closest to 860/862 pattern)
        "C=C(c1nc(/C=C/CN2C(=O)c3ccnn3CCCCCCCCN3CC[C@@H](Nc4cccc5c4cc2cc5)C[C@H]3F)cc3ccccc13)C(F)(F)F",
        "C=C(c1nc(/C=C/CN2C(=O)c3ccnn3CCCCCCCCCN3CC[C@@H](Nc4cccc5c4cc2cc5)C[C@H]3F)cc3ccccc13)C(F)(F)F",
        # Variations of the 860/862 scaffold with different linker lengths
        "C=C(c1nc2cc3c(cccc13)N[C@@H]1CCN(CCCCCCCN3N=CC=C3C(=O)NCC=C2)C[C@@H]1F)C(F)(F)F",
        "C=C(c1nc2cc3c(cccc13)N[C@@H]1CCN(CCCCCCCCCCN3N=CC=C3C(=O)NCC=C2)C[C@@H]1F)C(F)(F)F",
        "C=C(c1nc2cc3c(cccc13)N[C@@H]1CCN(CCCCCCCN3N=CC=C3C(=O)NCC=C2)C[C@@H]1F)C(F)(F)F",
    ]

    # Try fetching more macrocycles from PubChem
    pubchem_macrocycles = []
    print("  Fetching macrocycles from PubChem...")

    # Known macrocyclic drug CIDs
    known_macrocycle_cids = [
        5311497,    # Rapamycin/Sirolimus
        5284616,    # Erythromycin
        84691,      # Cyclosporine
        5280440,    # Vancomycin
        16051692,   # Lorlatinib (macrocyclic kinase inhibitor)
        71496458,   # Glecaprevir (macrocyclic HCV)
        25154714,   # Simeprevir
        44246704,   # Paritaprevir
        46930998,   # Voxilaprevir
        54671008,   # Rifampicin
        5281040,    # Azithromycin
        5280953,    # Fidaxomicin
        11979044,   # Ixazomib citrate
        11556711,   # Danoprevir
        49869906,   # Dalbavancin
        25195440,   # Faldaprevir
    ]
    pubchem_macrocycles = fetch_pubchem_by_cids(known_macrocycle_cids)
    print(f"  Fetched {len(pubchem_macrocycles)} macrocycles from PubChem CIDs")

    all_macrocycles = curated_macrocycles + pubchem_macrocycles

    # Validate and keep only macrocycles with >= 12-membered rings
    valid = []
    for smi in all_macrocycles:
        can = validate_smiles(smi)
        if can and is_macrocycle(can):
            valid.append(can)
    valid = list(set(valid))  # deduplicate

    if not valid:
        print("  WARNING: No valid macrocycles generated")
        return pd.DataFrame()

    # Oversample macrocycles since they're a critical gap
    MACROCYCLE_REPEATS = 10
    df = pd.DataFrame({
        "SMILES": valid * MACROCYCLE_REPEATS,
        "patent": "macrocycle_supplementary",
        "source": "macrocycle",
    })
    print(f"  Macrocycles: {len(valid)} unique × {MACROCYCLE_REPEATS} = {len(df)}")
    return df


def generate_diverse_heterocycles() -> pd.DataFrame:
    """Generate diverse heterocyclic scaffolds underrepresented in training data."""
    print("  Generating diverse heterocycle training data...")

    # Scaffolds that appear in failures but are underrepresented
    diverse_scaffolds = [
        # Quinazoline core with various substituents (WO2025235957 scaffold)
        "C=C(c1nc(N)cc2ccccc12)C(F)(F)F",
        "C=C(c1nc(NC2CCNCC2)cc2ccccc12)C(F)(F)F",
        "C=C(c1nc(/C=C/CN)cc2ccccc12)C(F)(F)F",
        "C=C(c1nc(CCCN)cc2ccccc12)C(F)(F)F",
        "C=C(c1nc(-c2cccc(CN)c2)cc2ccccc12)C(F)(F)F",

        # Pyrazole-amide warheads (common in failures)
        "O=C(NCC=Cc1ccccc1)c1ccn(C)n1",
        "O=C(NCC=Cc1ccccc1)c1ccn(CC)n1",
        "O=C(NCC=Cc1ccccc1)c1ccn(CCOC)n1",
        "O=C(NCC=Cc1ccccc1)c1ccn(CC(F)F)n1",
        "O=C(NCC=Cc1ccccc1)c1ccnn1C(F)(F)F",

        # Bridged bicyclic amines (failures 83, 99, 100, 353, 407)
        "C1CC2CCC1N2",
        "C1CC2CCC(N2)C1F",
        "C1CC2CCC(NC3=CC=CC4=CC=CC=C34)C1N2C",
        "FC1CN(C)CC1Nc1cccc2ccccc12",

        # Indazole/pyrrolo[2,3-b]pyridine (WO2026024861 scaffold)
        "FC(F)(F)Sc1c(C#CCN)nn2ccccc12",
        "FC(F)(F)Sc1c(C#CCN)cc2cccnc12",

        # Sulfonamide variations (WO2020132648 failures)
        "O=S(=O)(CCO)Nc1ccc(C(=O)N)cc1N1CCC2(CC1)CC2",
        "O=S(=O)(C(C)CO)c1ccc(C(=O)N)cc1N1CCC2(CC1)CC2",

        # Spiro compounds
        "O=C(Nc1ccnc(N2CCC23COC3)n1)c1ccccc1",
        "O=C(Nc1ccnc(N2CCC23CC3)n1)c1ccccc1",

        # Kinase inhibitor scaffolds (common patent structures)
        "Nc1ncnc2[nH]ccc12",  # 7H-pyrrolo[2,3-d]pyrimidine
        "Nc1ncnc2oc(C3CC3)cc12",  # fused oxazole-pyrimidine
        "O=C(Nc1ccc(-c2cnc3ccccc3n2)cc1)c1ccccc1",  # quinoxaline
        "Nc1cc(-c2ccc(F)c(Cl)c2)nc2ncccc12",  # aminopyridine

        # Complex fused ring systems
        "c1ccc2c(c1)ncc1[nH]c3ccccc3c12",
        "O=c1[nH]c2ccccc2c2c1CCC2",
        "c1ccc2c(c1)c1ccccc1[nH]2",

        # Deuterium-containing (to help with [2H] recognition)
        "[2H]C([2H])([2H])n1ccc(C(=O)NCC=Cc2ccccc2)n1",
        "[2H]C([2H])([2H])N1CCC(Nc2ccccc2)CC1",
        "[2H]C([2H])([2H])C([2H])([2H])n1nccc1C(=O)N",
        "[2H]C(C)n1nccc1C(=O)NCC=Cc1ccccc1",
        "[2H]Cn1ccc(C(=O)N)n1",
        "[2H]Cn1nc(C(=O)N)cc1",
    ]

    # Validate
    valid = []
    for smi in diverse_scaffolds:
        can = validate_smiles(smi)
        if can:
            valid.append(can)
    valid = list(set(valid))

    DIVERSE_REPEATS = 3
    df = pd.DataFrame({
        "SMILES": valid * DIVERSE_REPEATS,
        "patent": "diverse_supplementary",
        "source": "diverse_heterocycle",
    })
    print(f"  Diverse scaffolds: {len(valid)} unique × {DIVERSE_REPEATS} = {len(df)}")
    return df


def generate_bridged_bicyclics() -> pd.DataFrame:
    """Generate bridged bicyclic amine SMILES — the #1 remaining failure mode.

    Compounds 51, 83, 99, 100, 102 all contain 2-azabicyclo[2.2.1]heptane
    (2-azanorbornane) that the model consistently misreads.
    """
    print("  Generating bridged bicyclic amine training data...")

    bridged = [
        # ── 2-Azabicyclo[2.2.1]heptane (2-azanorbornane) core ──
        "C1CC2CCC1N2",                           # parent
        "CN1C2CCC1CC2",                           # N-methyl
        "CCN1C2CCC1CC2",                          # N-ethyl
        "C(CF)N1C2CCC1CC2",                       # N-CH2CH2F
        "CC(=O)CN1C2CCC1CC2",                     # N-CH2C(=O)CH3
        "CN(C)C(=O)CN1C2CCC1CC2",                 # N-CH2CONMe2
        "O=C(CN1C2CCC1CC2)N(C)C",                 # same, different SMILES order

        # Stereoisomers (exo/endo)
        "[C@@H]1(NC2CC1CC2)c1ccccc1",             # exo, aryl-substituted
        "[C@H]1(NC2CC1CC2)c1ccccc1",              # endo, aryl-substituted
        "C[C@@H]1CC2CC[C@H]1N2",                  # methyl on bridge
        "C[C@H]1CC2CC[C@@H]1N2",                  # methyl, other config
        "[C@@H]1(N2CC3CC2C1)Nc1ccccc1",           # aryl-NH on ring carbon
        "N[C@@H]1C2CCC(C2)N1C",                   # amino on bridge carbon

        # ── With the actual WO2026024861 scaffold context ──
        # Indazole-SCF3 + azanorbornane (compounds 51, 83, 99, 100, 102)
        "FC(F)(F)Sc1c(C#CCN)nn2c(N[C@@H]3CCC4CCC3N4)cccc12",
        "FC(F)(F)Sc1c(C#CCN)nn2c(N[C@H]3CCC4CCC3N4)cccc12",
        "FC(F)(F)Sc1c(C#CCN)nn2c(N[C@@H]3CCC4CCC3N4C)cccc12",
        "FC(F)(F)Sc1c(C#CCN)nn2c(N[C@H]3CCC4CCC3N4C)cccc12",
        "FC(F)(F)Sc1c(C#CCN)cc2c(N[C@@H]3CCC4CCC3N4)cccn12",
        "FC(F)(F)Sc1c(C#CCN)cc2c(N[C@H]3CCC4CCC3N4)cccn12",
        # N-substituted azanorbornane on scaffold
        "FC(F)(F)Sc1c(C#CCN)nn2c(N[C@@H]3CCC4CCC3N4CCF)cccc12",
        "FC(F)(F)Sc1c(C#CCN)nn2c(N[C@@H]3CCC4CCC3N4CC(=O)N(C)C)cccc12",
        "FC(F)(F)Sc1c(C#CCN)nn2c(N[C@@H]3CCC4CCC3N4C5(C)COC5)cccc12",

        # ── 2-Azabicyclo[2.1.1]hexane (smaller bridged) ──
        "C1CC2(C1)CN2",                           # parent
        "CN1CC2(CC1)C2",                           # N-methyl
        "c1ccc(N2CC3(CC2)C3)cc1",                 # aryl-substituted

        # ── 3-Azabicyclo[3.1.0]hexane (fused cyclopropane) ──
        "C1CC2CC1N2",                              # parent
        "CN1CC2CC1C2",                             # N-methyl
        "c1ccc(NC2CC3CC2N3C)cc1",                  # aryl-amino

        # ── 2,5-Diazabicyclo[2.2.1]heptane ──
        "C1NC2CC1NC2",                             # parent
        "CN1CC2CC1NC2",                            # N-methyl

        # ── Oxetane-fused azanorbornane (compound 83 substituent) ──
        "C1(C)COC1",                               # simple oxetane
        "CN1C2CCC1CC2C1(C)COC1",                   # azanorbornane + oxetane
    ]

    # Validate
    valid = []
    for smi in bridged:
        can = validate_smiles(smi)
        if can:
            valid.append(can)
    valid = list(set(valid))

    BRIDGED_REPEATS = 10
    df = pd.DataFrame({
        "SMILES": valid * BRIDGED_REPEATS,
        "patent": "bridged_bicyclic_supplementary",
        "source": "bridged_bicyclic",
    })
    print(f"  Bridged bicyclics: {len(valid)} unique × {BRIDGED_REPEATS} = {len(df)}")
    return df


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


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────


def main():
    np.random.seed(SEED)

    print("=" * 60)
    print("MolSight Fine-Tuning Dataset Preparation (Phase 2)")
    print("=" * 60)

    # ── Step 1: Load original patent SMILES ───────────────────
    print("\n1. Loading patent SMILES...")
    patent_df = load_patent_smiles()
    print(f"   Total patent SMILES: {len(patent_df)}")

    # ── Step 2: Load hard examples ────────────────────────────
    print("\n2. Loading hard examples from pressure test...")
    hard_df = load_hard_examples()

    # ── Step 3: Generate macrocycle SMILES ─────────────────────
    print("\n3. Generating macrocycle supplementary data...")
    macro_df = generate_macrocycle_smiles()

    # ── Step 4: Generate diverse heterocycle scaffolds ─────────
    print("\n4. Generating diverse heterocycle scaffolds...")
    diverse_df = generate_diverse_heterocycles()

    # ── Step 4b: Generate bridged bicyclic amines ─────────────
    print("\n4b. Generating bridged bicyclic amines...")
    bridged_df = generate_bridged_bicyclics()

    # ── Step 5: Combine all sources ───────────────────────────
    print("\n5. Combining all data sources...")
    all_dfs = [df for df in [patent_df, hard_df, macro_df, diverse_df, bridged_df] if not df.empty]
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"   Total raw rows: {len(combined)}")

    # ── Step 6: Validate & canonicalize ───────────────────────
    print("\n6. Validating SMILES...")
    combined["canonical"] = combined["SMILES"].apply(validate_smiles)
    n_invalid = combined["canonical"].isna().sum()
    combined = combined.dropna(subset=["canonical"]).reset_index(drop=True)
    print(f"   Dropped {n_invalid} invalid, {len(combined)} remaining")

    # ── Step 7: Stratified train/val split (patents only) ─────
    print("\n7. Creating train/val split...")

    # Only split patent data for validation; supplementary data is train-only
    patent_mask = combined["source"] == "patent"
    patent_rows = combined[patent_mask]
    supplementary_rows = combined[~patent_mask]

    val_dfs = []
    train_dfs = []
    for patent, group in patent_rows.groupby("patent"):
        n_val = max(1, int(len(group) * VAL_FRACTION))
        shuffled = group.sample(frac=1, random_state=SEED).reset_index(drop=True)
        val_dfs.append(shuffled.iloc[:n_val])
        train_dfs.append(shuffled.iloc[n_val:])

    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    train_patent = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()

    # Combine patent train + all supplementary for training
    train_parts = [df for df in [train_patent, supplementary_rows] if not df.empty]
    train_df = pd.concat(train_parts, ignore_index=True)

    print(f"   Train: {len(train_df)} (patent: {len(train_patent)}, supplementary: {len(supplementary_rows)})")
    print(f"   Val: {len(val_df)}")

    # Source breakdown
    print("\n   Training set composition:")
    for source, group in train_df.groupby("source"):
        n_unique = group["canonical"].nunique()
        print(f"     {source}: {len(group)} rows ({n_unique} unique)")

    # ── Step 8: Write train CSV ───────────────────────────────
    print("\n8. Writing output files...")
    os.makedirs(os.path.dirname(TRAIN_CSV), exist_ok=True)
    train_out = pd.DataFrame({"SMILES": train_df["canonical"]})
    train_out.to_csv(TRAIN_CSV, index=False)
    print(f"   Train CSV: {TRAIN_CSV} ({len(train_out)} rows)")

    # ── Step 9: Render val images & write val CSV ─────────────
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

    if val_records:
        val_out = pd.DataFrame(val_records)
        os.makedirs(os.path.dirname(VAL_CSV), exist_ok=True)
        val_out.to_csv(VAL_CSV, index=False)
        print(f"   Val CSV: {VAL_CSV} ({len(val_out)} rows)")

    # ── Step 10: Save supplementary data separately ───────────
    os.makedirs(SUPPLEMENTARY_DIR, exist_ok=True)
    supplementary_path = os.path.join(SUPPLEMENTARY_DIR, "dataset_manifest.json")
    manifest = {
        "total_train": len(train_out),
        "total_val": len(val_records),
        "sources": {},
    }
    for source, group in train_df.groupby("source"):
        manifest["sources"][source] = {
            "total_rows": len(group),
            "unique_smiles": int(group["canonical"].nunique()),
        }
    with open(supplementary_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"   Manifest: {supplementary_path}")

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print(f"  Total training SMILES: {len(train_out)}")
    print(f"  Total validation images: {len(val_records)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
