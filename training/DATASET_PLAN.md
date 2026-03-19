# Fine-Tuning Dataset Plan for MolSight

## Current State

**Training set**: 1,010 SMILES from 3 patents (rendered on-the-fly by Indigo during training)
- WO2026024861: ~170 (pyrimidine-sulfonamide series)
- WO2020132648: ~180 (similar sulfonamide series)
- WO2025235957: ~650 (quinazoline Y220C series)
- WO2025184668: 0 (empty Excel — data missing)

**Pressure test**: 92 real patent images → 42.4% exact match, 71.7% Tan≥0.9

## Failure Mode Analysis (53 failures)

| Category | Count | % of Total | Root Cause |
|----------|-------|-----------|------------|
| Stereo-only (Tan≥0.95) | 22 | 24% | Missing/wrong @/@@ |
| Minor structural (0.7-0.95) | 19 | 21% | Ring substitution, small group swaps |
| Medium structural (0.5-0.7) | 8 | 9% | Side chain loss, wrong heterocycles |
| Catastrophic (<0.5) | 4 | 4% | Macrocycles completely lost |

### Patent-specific failure rates
- **WO2025235957**: 32/44 failures (73%) — worst performer
- **WO_2026024861_A1**: 15/35 failures (43%)
- **WO2020132648A1**: 6/13 failures (46%)

## Identified Data Gaps

### 1. Macrocycles / PROTACs (CRITICAL)
- **Problem**: Compounds 860, 862 (Tan=0.22) are macrocyclic structures with 8-10 carbon linkers connecting a quinazoline to a pyrazole warhead. MolSight completely collapses these to simple bicyclic structures.
- **Current training data**: Zero SMILES > 150 chars. No macrocycles represented.
- **Need**: 200-500 macrocyclic SMILES (PROTACs, macrolides, macrocyclic peptides) to teach ring-closure over long spans.
- **Source**: PubChem CID search for macrocycles, ChEMBL PROTAC sets, or curated macrocycle databases.

### 2. Complex Multi-Ring Heterocycles (HIGH)
- **Problem**: WO2025235957 failures involve quinazoline cores with elaborate substituents (pyrazoles, cyclopropyl ureas, bridged amines) that get simplified or deleted.
- **Current training data**: Dominated by pyrimidine-sulfonamide scaffolds from WO2026024861/WO2020132648. The WO2025235957 quinazoline series is there but with only ~650 examples.
- **Need**: More structural diversity — particularly fused heterocycles with 3+ substitution points.
- **Source**: PubChem/ChEMBL patent compound sets spanning diverse scaffolds.

### 3. Real Patent Image Domain (HIGH)
- **Problem**: Training uses Indigo-rendered images (clean 2D with consistent style). Test images are real patent figures with:
  - Variable line thickness and bond angles
  - ChemDraw/ChemOffice rendering artifacts
  - Occasional hand-drawn appearance
  - Background noise, compression artifacts
  - Stereodescriptor labels ((R), (S)) embedded in the image
- **Need**: Train on actual patent images, not just Indigo renders. The MolSight training framework renders SMILES on-the-fly with Indigo — there's no built-in mechanism to train on real images.
- **Options**:
  a. Modify MolSight's `HybridDataset` to accept real image paths + SMILES pairs
  b. Mix Indigo renders with augmented ChemDraw-style renders
  c. Use the 92 pressure test images as hard-example training data

### 4. Stereochemistry Enhancement (MEDIUM)
- **Problem**: 22 compounds have correct connectivity but wrong stereochemistry.
- **Current training data**: 73% have stereo — good coverage, but the failures suggest the model sometimes drops @ annotations when the visual cues are ambiguous (wedge/dash bonds).
- **Need**: Training augmentation that varies wedge/dash bond rendering styles. Indigo already does some of this, but patent images have more variation.
- **Source**: Current dataset may be sufficient if augmentation is improved.

### 5. Deuterium (LOW — already partially fixed)
- **Problem**: 9 SMILES in training contain [2H], but patent images show "D" labels that MolSight reads as wildcards.
- **Status**: Post-processing fix already handles most cases.
- **Need**: Minimal — the post-processing workaround is effective.

### 6. WO2025184668 Patent Data (HIGH)
- **Problem**: The 4th patent (WO2025184668) has 0 rows in the training Excel — this data is completely missing.
- **Need**: Curate SMILES for WO2025184668 compounds. This patent likely contributes additional scaffold diversity.

## Recommended Dataset Composition

### Phase 2a: Expand SMILES diversity (~2,000 additional SMILES)
1. **Macrocycles**: 300 SMILES from PubChem/ChEMBL (PROTAC, macrolide, macrocyclic peptide classes)
2. **Diverse heterocycles**: 500 SMILES from ChEMBL patent compounds (kinase inhibitors, PPI modulators, etc.)
3. **WO2025184668 data**: Curate ground truth SMILES for this patent
4. **Hard examples**: Include the 31 structural failure SMILES weighted 3-5× in training

### Phase 2b: Real image training (major infrastructure change)
1. Modify `HybridDataset` to support `(image_path, SMILES)` pairs alongside Indigo renders
2. Use the 92 pressure test images + their ground truth SMILES as real-domain training data
3. Leave 10-20% as held-out validation
4. Mix ratio: ~70% Indigo renders, ~30% real patent images

### Phase 2c: Augmentation improvements
1. Vary bond thickness, atom label fonts, line angles during Indigo rendering
2. Add JPEG compression artifacts to rendered images
3. Random background noise / slight rotation
4. Explicit wedge/dash bond style variation for stereo training

## Expected Impact

| Intervention | Compounds Fixed | Tan≥0.9 Gain |
|-------------|----------------|--------------|
| Phase 2a (SMILES diversity) | 5-10 structural | +5-10% |
| Phase 2b (real images) | 10-15 structural + 5-10 stereo | +10-15% |
| Phase 2c (augmentation) | 5-10 stereo | +5-10% |
| **Combined** | **~25-30** | **+25-35%** |

Target: 71.7% → ~95% Tan≥0.9

## Next Steps

1. Prepare macrocycle + diverse heterocycle SMILES (pull from PubChem/ChEMBL)
2. Update `prepare_data.py` to include new sources + weight hard examples
3. Modify MolSight training to accept real image pairs (Phase 2b)
4. Run SFT with expanded dataset, evaluate on held-out test set
