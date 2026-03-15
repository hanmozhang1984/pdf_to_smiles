# Handoff State: Adding Frontier Patent to Ground Truth

## Task
Add ground truth entries for the Frontier patent (`WO2025235957 Y220C vinyl pyridine Frontier (1).pdf`) to `eval/ground_truth.json` to strengthen the evaluation harness with a diverse new patent.

## PDF Location
```
/Users/hanmozhang/Downloads/Sample patents for testing/Frontier P53YC/WO2025235957 Y220C vinyl pyridine Frontier (1).pdf
```
- **Total pages**: 549
- **Total compounds**: 734 (from Excel sheet)
- **Patent number**: WO 2025/235957, PCT/US2025/028753

## What Was Done
- Rendered and visually inspected ~30 pages across the document
- Identified the table structure and page ranges
- Counted structures on inspected pages (see below)
- Did NOT yet update `ground_truth.json` — that is the remaining work

## Patent Layout

### Table Structure (Compound Structure Tables — "Table 1")
- **Format**: Two side-by-side sub-tables, each with columns: `Cpd. #` | `Structure`
- **Typical layout**: 7 rows per sub-table column = **14 structures per page** on full pages (BUT some pages have 6 rows per column = 12, and the early pages with CF3 groups have smaller structures allowing 7 rows)
- **Table lines**: Clear horizontal and vertical separator lines between every cell
- **Page range for Table 1**: Approximately pages **104–176** (compound numbers ~30 through ~981)
  - Compound numbering is NOT sequential with page numbers — there are gaps (e.g., page 106 has cpds 58-71, page 107 has 72-85)

### Key Observations
- The tables use a consistent 2-column layout throughout
- Structures vary in size/complexity but are uniformly placed in cells
- **No "examples" column** — just Cpd. # and Structure (simpler than GLP/Merck patents)

### Special Pages Identified
- **Page 105** (p104 printed): First full table page seen — compounds 30-43, **14 structures** (7 per column)
- **Page 106** (p105 printed): compounds 44-57, **14 structures** (7 per column)
- **Page 107** (p106 printed): compounds 58-71, **14 structures** (7 per column)
- **Page 108** (p107 printed): compounds 72-85, **14 structures** (7 per column)
- **Page 110** (p109 printed): compounds 100-110a, **12 structures** (6 per column, BUT right column has "110a" so technically 12 including 110a)
  - **IMPORTANT**: Page 110 has compound "110a" — a non-numeric compound ID. Count carefully.
- **Page 115** (p114 printed): compounds 204-215, **12 structures** (6 per column)
- **Page 117** (p116 printed): compounds 228-239, **12 structures** (6 per column)
- **Page 118** (p117 printed): compounds 240-251, **12 structures** (6 per column)
- **Page 120** (p119 printed): compounds 264-276, **12 structures** (6 left, 6 right)
- **Page 125**: NOT inspected — need to check
- **Page 130**: NOT inspected — need to check
- **Page 140** (p139 printed): compounds 514-527, **14 structures** (7 per column)
- **Page 160** (p159 printed): compounds 761-772, **12 structures** (6 per column)
- **Page 170** (p169 printed): compounds 888-899, **12 structures** (6 per column)
- **Page 175** (p174 printed): compounds 949-961, **14 structures** (7 left, 7 right) — wait, actually left has 949-955 (7) and right has 956-961 (6) = **13 structures**. NEED TO RECOUNT.
- **Page 176** (p175 printed): **MIXED PAGE** — top has last compounds of Table 1 (976-981, 979-981 = 6 structures in 2-col table with 3 rows each), THEN text paragraph [0178], THEN **Table 2** starts with intermediates (Int-1 through Int-12ish) in a **4-column grid** layout. This is a tricky boundary page.
  - Table 1 portion: 6 structures (compounds 976-981)
  - Table 2 portion: 12 intermediates (Int-1 through Int-12, in 4x3 grid)
  - **Total structures on page 176: 18** (6 + 12)? Or should we count Table 2 intermediates separately? Need to decide.
- **Page 177** (p176 printed): Table 2 continued — intermediates in **4-column grid**, Int-9 through Int-34 = **26 structures**. Very different layout from Table 1.

### Non-Table Pages (expected_count = 0)
- **Page 50** (p49 printed): text only (claims/description)
- **Page 80** (p79 printed): inline structures in text — these are NOT in tables, they are Markush fragments. **Should count as 0** for table detection since they're inline with text, not in structured tables.
- **Page 100** (p99 printed): text with 2 inline structures (Formula I-4 and one fragment). **0 for table detection**.
- **Page 180** (p179 printed): text only (description of p53)
- **Page 200** (p199 printed): text with inline Markush fragments — **0 for table detection**

## Recommended Diverse Page Set for Ground Truth

Pick ~12-15 pages covering these scenarios:

### Full Table 1 pages (typical, 12-14 structures)
1. **Page 105**: 14 structures (early table, CF3 compounds)
2. **Page 107**: 14 structures (compounds 72-85)
3. **Page 115**: 12 structures (compounds 204-215)
4. **Page 120**: 12 structures (compounds 264-276) — VERIFY exact count, saw "276" as last
5. **Page 140**: 14 structures (compounds 514-527)
6. **Page 160**: 12 structures (compounds 761-772)
7. **Page 170**: 12 structures (compounds 888-899)

### Edge cases
8. **Page 110**: 12 structures — has compound "110a" (non-numeric ID)
9. **Page 175**: ~13 structures — RECOUNT NEEDED (last full-ish table page)
10. **Page 176**: Mixed page — last Table 1 rows + Table 2 start. Complex.

### Table 2 pages (4-column intermediate grid)
11. **Page 177**: ~26 intermediates in 4-col grid — very different layout

### No-structure pages
12. **Page 50**: text only → expected_count = 0
13. **Page 100**: text with inline structures → expected_count = 0
14. **Page 180**: text only → expected_count = 0

## CRITICAL: Counts Need Verification

**Several pages need exact recounting before writing to ground_truth.json:**
- Pages 105-108: I counted 14 per page (7 per column) — verify
- Page 110: count carefully (has "110a")
- Page 175: recount — left col may have 7, right may have fewer
- Page 176: decide how to count (Table 1 + Table 2 mixed)
- Page 177: count all intermediates in the 4-col grid

**Verification method**: Render each page at high DPI, count structures row by row in each column.

## How to Resume

1. Read this file and `eval/ground_truth.json`
2. Re-render the recommended pages and do exact counts
3. Add a new patent entry to `ground_truth.json`:
```json
{
  "id": "Frontier",
  "filename": "WO2025235957 Y220C vinyl pyridine Frontier (1).pdf",
  "notes": "Y220C vinyl pyridine patent. Table 1 has 2-column layout (Cpd.#/Structure). Table 2 has 4-column intermediate grid.",
  "pages": [
    {"page": 105, "expected_count": 14, "examples": "30-43"},
    ...
  ]
}
```
4. The PDF must be copied/symlinked to `PDF_DIR` (`~/Downloads/Sample patents for testing/`) or the eval harness path updated. Currently it's in a subdirectory `Frontier P53YC/`.
5. Run eval: `python eval/run_eval.py --patent Frontier`

## File Locations
- Ground truth: `/Users/hanmozhang/Documents/Projects/pdf_to_smiles/eval/ground_truth.json`
- Eval harness: `/Users/hanmozhang/Documents/Projects/pdf_to_smiles/eval/run_eval.py`
- PDF_DIR in eval: `~/Downloads/Sample patents for testing/` (currently PDF is in subdirectory — needs path fix or file move)
- Excel with compound data: `/Users/hanmozhang/Downloads/Sample patents for testing/Frontier P53YC/Pages from WO2025235957 Y220C vinyl pyridine Frontier (1) (2).xlsx`
- Rendered page images (temporary): `/tmp/frontier_pages/page_NNN.png`
- Active plan: `/Users/hanmozhang/.claude/plans/nested-humming-lecun.md` (line-based table cell detection plan — separate from this ground truth task)

## Related Context
- The broader project is building a PDF-to-SMILES pipeline for pharmaceutical patents
- The eval harness tests structure *detection* (bounding box count), not SMILES conversion
- The active plan (`nested-humming-lecun.md`) is about improving the detector using line-based table cell detection — this ground truth work supports testing those improvements
- The Frontier patent is interesting because it has a different table layout (2-col vs GLP's 4-col vs Merck's 6-per-page) and includes a Table 2 with a 4-column intermediate grid
