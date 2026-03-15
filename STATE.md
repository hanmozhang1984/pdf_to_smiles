# Structure Detection Improvement — Handoff State

## Project Overview

Chemical structure detection from patent PDFs. The `DoclingDetector` class in
`src/pdf_to_smiles/core/docling_detector.py` uses Docling's RT-DETR layout
classifier to find chemical structures, then post-processes to fill gaps in
table-based pages.

## Current Eval Results (as of 2026-03-15)

```
OVERALL: 30/36 pages with exact count (83%)
Total structures: expected=127, detected=123, delta=-4
```

| Patent | Score | Notes |
|--------|-------|-------|
| GLP (`US20240366598A1.pdf`) | 16/21 (76%) | 5 remaining failures |
| Merck (`WO_2026024861_A1.pdf`) | 11/12 (92%) | 1 remaining failure |
| KAT6 (`US11492346_KAT6pages.pdf`) | 3/3 (100%) | Fully passing |

**Target: >95% accuracy (need 35/36 = 97%)**

## What Was Done (this session)

### 1. Whitespace Gap Detection (`_subdivide_tall_segments`)
Added to `_detect_table_row_boundaries`. When horizontal line detection finds
row segments taller than 600px, it analyses per-row ink density to find
whitespace gaps and splits there. Uses two-tier acceptance:
- Wide gaps (>=30px): always accepted
- Narrow gaps (>=15px): accepted only if min density < 0.5%

**Impact**: GLP went from 4/21 → 16/21. Fixed pages where the classifier found
0 structures and the table had no horizontal separator lines between data rows.

### 2. Min Row Height Raised (120 → 160px)
`_MIN_STRUCTURE_ROW_HEIGHT` raised from 120 to 160 to filter out table header
rows (typically 110-151px) that were being included as false structure detections.

## Remaining 6 Failures — Root Cause Analysis

### Diastereomer Pair Splitting Issues (GLP pages 130, 131, 147, 148)

These pages have boxes containing TWO chemical structures (diastereomers)
stacked vertically, separated by "or" text or "DIAST-1/DIAST-2" labels.

**Pages 130, 131** (expected 3, detected 2):
- No table region detected by Docling (`_get_table_boxes` returns [])
- Classifier finds 2 boxes, one is very tall (~960px) covering both diastereomers
- Neither `_fill_table_gaps` nor `_scan_example_tables` activates
- **Fix**: Split tall boxes (>600px) using whitespace gap detection

**Pages 147, 148** (expected 4, detected 3):
- Table IS detected, but the tall classifier box (~700-740px) covers 2 row
  segments, both marked "covered" in `_fill_table_gaps`
- **Fix**: Same — split tall boxes post-pipeline

**Verified data for tall boxes:**
```
Page 130 box[1]: 692x993 — main gap at y_rel=444-627 (w=183, density=0.0)
Page 131 box[0]: 692x945 — main gap at y_rel=445-584 (w=139, density=0.0)
Page 147 box[1]: 647x769 — main gap at y_rel=329-433 (w=104, density=0.0)
Page 148 box[0]: 659x770 — main gap at y_rel=362-401 (w=39, density=0.0)
```

### Over-Detection (GLP page 146)

**Expected 3, detected 4.** Example 221 has 2 diastereomers that the ground truth
counts as ONE box (they share the same NMR data cell). But line detection finds
a real horizontal rule between them, creating 2 synthetic boxes.

This is the hardest to fix. Would need to distinguish "main row separators"
from "within-cell formatting lines." Possibly fixable by checking if adjacent
synthetic boxes share a text/data column.

**Skip this for now** — fixing the other 5 gets us to 35/36 (97%).

### Side-by-Side Structures (Merck page 155)

**Expected 3, detected 2.** Examples 137/138 are drawn as two structures
side-by-side BELOW the table. Classifier treats them as one wide box.

```
Box: 981x243 (ratio=4.0) at (380,1444)-(1361,1687), OUTSIDE table
Vertical gap at x_rel=450-526 (w=76, density=0.0) — clear split point
```

**Fix**: Split wide boxes (aspect ratio >2.5) outside table regions using
vertical whitespace gap detection.

## Planned Implementation (NOT YET DONE)

### New method: `_split_tall_boxes(boxes, page_image)`

Add as the LAST step in `detect_structures_with_boxes`, after `_close_row_gaps`.

```python
# In detect_structures_with_boxes:
boxes = self._close_row_gaps(boxes, page_num, page_image.size)
boxes = self._split_compound_boxes(boxes, page_image, page_num)  # NEW
```

Algorithm:
1. For each box with height > `_MAX_ROW_SEGMENT_HEIGHT` (600px):
   - Compute smoothed horizontal ink-density profile (same as `_subdivide_tall_segments`)
   - Find the Y-position with MINIMUM density in the middle 60% of the box
   - If min density < 0.02, split at that point
   - Validate BOTH resulting sub-boxes with `_validate_candidate`
   - Both sub-boxes must have height >= `_MIN_STRUCTURE_ROW_HEIGHT` (160px)
   - Only keep split if BOTH sub-boxes pass all checks
   - Otherwise keep original box

2. For each box OUTSIDE all table regions, with aspect ratio > 2.5 and h > 150:
   - Same approach but for VERTICAL (column-wise) density profile
   - Find X-position with minimum density in middle 60%
   - Validate both sub-boxes pass `_has_structure_ink`
   - Both sub-boxes must have aspect ratio < 3.0 (safety check)
   - Only keep split if both pass

### Safety — Verified No Regressions

**All currently-passing GLP pages** have max box height < 600px (range: 278-457px).
No tall-box splitting would trigger.

**Merck page 120** (PASSING, expected=2): Has a wide box 1146x275 (ratio=4.2)
outside table. BUT after split, the right sub-box would have aspect ratio 3.3
which exceeds the 3.0 safety limit → split rejected → no regression.

**Merck page 152** (PASSING, expected=2): Has a tall box 418x791 (h>600) inside
table. After split at the minimum density point, one sub-box would have
h < 160px → fails height check → split rejected → no regression.

**KAT6 page 1** (PASSING, expected=4): Has a box 923x849. After split, only ONE
sub-box would pass `_has_structure_ink` (the other is IUPAC text) → split
rejected (need 2 valid) → no regression.

## Key Files

| File | Purpose |
|------|---------|
| `src/pdf_to_smiles/core/docling_detector.py` | **Main file to modify** — all detection post-processing |
| `src/pdf_to_smiles/core/docling_classifier.py` | Docling RT-DETR classifier — read-only, provides `_cached_layout` |
| `eval/run_eval.py` | Evaluation harness |
| `eval/ground_truth.json` | Ground truth (35 test pages across 3 patents) |
| `eval/output/results.json` | Latest eval results |

## How to Run Eval

```bash
cd /Users/hanmozhang/Documents/Projects/pdf_to_smiles
source venv/bin/activate

# Full eval (all 3 patents, ~60s)
python eval/run_eval.py

# Single patent
python eval/run_eval.py --patent GLP
python eval/run_eval.py --patent Merck
python eval/run_eval.py --patent KAT6

# Specific pages
python eval/run_eval.py --patent GLP --pages 130 131 147 148
```

PDFs are in `~/Downloads/Sample patents for testing/`.
Annotated output images go to `eval/output/{patent_id}/page_NNN.png`.

## Detection Pipeline Flow

```
detect_structures_with_boxes(page_image, page_num)
  │
  ├─ get_structure_boxes()      ← RT-DETR classifier boxes
  ├─ _merge_split_boxes()       ← merge horizontally-split boxes (union-find)
  ├─ _fill_table_gaps()         ← fill missed rows in tables with ≥1 detection
  │    └─ _detect_table_row_boundaries()
  │         └─ _subdivide_tall_segments()  ← whitespace gap fallback
  ├─ _scan_example_tables()     ← rescue tables with 0 detections (OCR caption)
  ├─ _close_row_gaps()          ← expand box tops to include example numbers
  └─ [NEW] _split_compound_boxes()  ← split tall/wide boxes containing 2 structures
```

## Key Thresholds (current values)

```python
_MIN_STRUCTURE_ROW_HEIGHT = 160   # min height for a valid structure cell
_MAX_ROW_SEGMENT_HEIGHT = 600     # trigger whitespace subdivision above this
_MIN_WHITESPACE_GAP = 30          # min gap width for normal acceptance
_NARROW_GAP_MIN = 15              # min gap width for near-zero density
_NARROW_GAP_DENSITY = 0.005       # max density for narrow gap acceptance
_INK_DARK_THRESHOLD = 180         # grayscale dark pixel threshold
_INK_RATIO_THRESHOLD = 0.01       # min dark pixel fraction
_STRUCTURE_INK_MIN_CELLS = 6      # min occupied cells in 4x4 grid
_TEMPLATE_SIMILARITY_THRESHOLD = -0.3  # very permissive template match
```

## Debug Scripts (in /tmp)

- `/tmp/debug_lines/` — line detection visualizations for GLP page 100
- `/tmp/debug_failures/` — annotated images for all 6 failing pages
- `/tmp/debug_crops2/` — Merck page crops

## Expected Results After Fix

If `_split_compound_boxes` is implemented correctly:

| Page | Before | After | Fix |
|------|--------|-------|-----|
| GLP 130 | 2 → 3 | PASS | tall box split |
| GLP 131 | 2 → 3 | PASS | tall box split |
| GLP 146 | 4 → 3 | still OVER(+1) | not addressed |
| GLP 147 | 3 → 4 | PASS | tall box split |
| GLP 148 | 3 → 4 | PASS | tall box split |
| Merck 155 | 2 → 3 | PASS | wide box split |

**Projected: 35/36 (97%)** — exceeds 95% target.

## Session Context

- Was in **plan mode** when handoff was created (plan file at
  `/Users/hanmozhang/.claude/plans/nested-humming-lecun.md` is stale — from
  the earlier line-detection work, not the current `_split_compound_boxes` plan)
- The plan mode should be exited before implementing
- No uncommitted changes exist beyond the `docling_detector.py` modifications
  already in the working tree (whitespace gap detection + min height raise)
