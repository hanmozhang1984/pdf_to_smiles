"""Validate OCSR output by comparing RDKit-computed formulas against patent references.

Compares molecular formulas and exact masses from SMILES predictions against
analytical data (MS, HRMS, LCMS) extracted from patent text.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Proton mass for [M+H]+ calculation
_PROTON_MASS = 1.00728


@dataclass
class FormulaValidationResult:
    """Result of comparing a SMILES against reference formula data."""

    status: str  # "match", "mismatch", "mass_only_match", "no_reference"
    reference_formula: Optional[str] = None
    computed_formula: Optional[str] = None
    reference_mass: Optional[float] = None
    computed_mass: Optional[float] = None
    mass_error_ppm: Optional[float] = None


class FormulaValidator:
    """Compares RDKit-computed molecular properties against extracted reference data."""

    def validate(
        self,
        smiles: str,
        references: list,
        compound_id: Optional[str] = None,
    ) -> FormulaValidationResult:
        """Compare SMILES against reference formulas.

        Strategy:
        1. Compute molecular formula from SMILES via RDKit.
        2. Try to match by compound_id first.
        3. If no ID match, try all references.
        4. Compare formulas (normalize atom ordering).
        5. If formula doesn't match, check mass (within 10 ppm for [M+H]+).
        6. Return validation result.

        Args:
            smiles: The SMILES string to validate.
            references: List of FormulaReference objects from the same page.
            compound_id: Optional compound ID for targeted matching.

        Returns:
            FormulaValidationResult with match status and details.
        """
        if not references:
            return FormulaValidationResult(status="no_reference")

        # Compute formula and mass from SMILES
        computed_formula, computed_mass = _compute_from_smiles(smiles)
        if computed_formula is None:
            return FormulaValidationResult(status="no_reference")

        computed_atoms = _normalize_formula(computed_formula)

        # Try compound_id match first
        if compound_id:
            norm_id = _normalize_compound_id(compound_id)
            for ref in references:
                if ref.compound_id and _normalize_compound_id(ref.compound_id) == norm_id:
                    return _compare(
                        computed_formula, computed_atoms, computed_mass, ref
                    )

        # No ID match — try all references, return best match
        best_result = None
        for ref in references:
            result = _compare(computed_formula, computed_atoms, computed_mass, ref)
            if result.status == "match":
                return result
            if result.status == "mass_only_match" and (
                best_result is None or best_result.status != "mass_only_match"
            ):
                best_result = result

        # Return best result found, or mismatch against first reference
        if best_result is not None:
            return best_result

        # Default: compare against first reference
        ref = references[0]
        return _compare(computed_formula, computed_atoms, computed_mass, ref)


def _compute_from_smiles(smiles: str):
    """Compute molecular formula and monoisotopic mass from SMILES.

    Returns:
        Tuple of (formula_string, monoisotopic_mass) or (None, None) on failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        mass = Descriptors.ExactMolWt(mol)
        return formula, mass
    except Exception:
        return None, None


def _normalize_formula(formula: str) -> Dict[str, int]:
    """Parse molecular formula into {element: count} dict.

    Handles standard Hill notation: "C28H30F4N8O2S" -> {"C": 28, "H": 30, ...}
    """
    atoms = {}
    for element, count_str in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if not element:
            continue
        count = int(count_str) if count_str else 1
        atoms[element] = atoms.get(element, 0) + count
    return atoms


def _normalize_compound_id(cid: str) -> str:
    """Normalize compound ID for fuzzy matching.

    Strips common prefixes and whitespace:
    "Example 85" -> "85", "Compound 1a" -> "1a", "85" -> "85"
    """
    cid = cid.strip()
    # Remove common prefixes
    for prefix in ("Example", "Compound", "Cpd.", "Cpd", "Ex.", "Ex"):
        if cid.lower().startswith(prefix.lower()):
            cid = cid[len(prefix):].strip().lstrip(".")
            break
    return cid.strip()


def _compare(
    computed_formula: str,
    computed_atoms: Dict[str, int],
    computed_mass: float,
    ref,
) -> FormulaValidationResult:
    """Compare computed properties against a single reference."""
    ref_atoms = _normalize_formula(ref.molecular_formula) if ref.molecular_formula else {}

    # Formula match (order-independent) — skip if ref has no formula
    if ref_atoms and computed_atoms == ref_atoms:
        # Also compute mass error if reference mass available
        mass_error = None
        if ref.expected_mh_mass is not None and computed_mass is not None:
            computed_mh = computed_mass + _PROTON_MASS
            mass_error = _ppm_error(computed_mh, ref.expected_mh_mass)

        return FormulaValidationResult(
            status="match",
            reference_formula=ref.molecular_formula,
            computed_formula=computed_formula,
            reference_mass=ref.expected_mh_mass,
            computed_mass=computed_mass + _PROTON_MASS if computed_mass else None,
            mass_error_ppm=mass_error,
        )

    # Formula mismatch — check mass as fallback
    if ref.expected_mh_mass is not None and computed_mass is not None:
        computed_mh = computed_mass + _PROTON_MASS
        mass_error = _ppm_error(computed_mh, ref.expected_mh_mass)

        # Use Da tolerance for integer (low-res) masses, ppm for precise masses
        is_integer_mass = ref.expected_mh_mass == int(ref.expected_mh_mass)
        mass_matches = (
            abs(computed_mh - ref.expected_mh_mass) <= 0.5
            if is_integer_mass
            else abs(mass_error) <= 10.0
        )

        if mass_matches:
            return FormulaValidationResult(
                status="mass_only_match",
                reference_formula=ref.molecular_formula,
                computed_formula=computed_formula,
                reference_mass=ref.expected_mh_mass,
                computed_mass=computed_mh,
                mass_error_ppm=mass_error,
            )

    # Full mismatch
    return FormulaValidationResult(
        status="mismatch",
        reference_formula=ref.molecular_formula,
        computed_formula=computed_formula,
        reference_mass=ref.expected_mh_mass,
        computed_mass=computed_mass + _PROTON_MASS if computed_mass else None,
        mass_error_ppm=None,
    )


def correct_compound_ids(
    structures: List[dict],
    references: List,
) -> Dict[int, str]:
    """Return {structure_index: corrected_compound_id} based on mass matching.

    Computes [M+H]+ for each structure from its SMILES, then matches against
    reference masses using greedy best-first assignment (lowest mass error
    first). Only reassigns when a clear mass match exists within tolerance.

    Args:
        structures: List of dicts with at least "index" (int) and "smiles" (str|None).
        references: List of FormulaReference objects from the same page.

    Returns:
        Dict mapping structure index to corrected compound_id. Structures
        without a match are not included.
    """
    # Compute [M+H]+ for each structure
    struct_masses = {}  # index -> computed [M+H]+
    for s in structures:
        smiles = s.get("canonical_smiles") or s.get("smiles")
        if not smiles:
            continue
        _, mass = _compute_from_smiles(smiles)
        if mass is not None:
            struct_masses[s["index"]] = mass + _PROTON_MASS

    if not struct_masses:
        return {}

    # Filter references to those with both compound_id and expected_mh_mass
    valid_refs = [
        r for r in references
        if r.compound_id and r.expected_mh_mass is not None
    ]
    if not valid_refs:
        return {}

    # Build all (struct_index, ref_index, mass_error) pairs
    pairs = []
    for s_idx, computed_mh in struct_masses.items():
        for r_idx, ref in enumerate(valid_refs):
            error = abs(computed_mh - ref.expected_mh_mass)
            pairs.append((error, s_idx, r_idx))

    # Sort by mass error (best matches first)
    pairs.sort()

    # Greedy assignment: assign best matches first, skip if either already assigned
    assigned_structs = set()
    assigned_refs = set()
    corrections = {}

    for error, s_idx, r_idx in pairs:
        if s_idx in assigned_structs or r_idx in assigned_refs:
            continue

        ref = valid_refs[r_idx]
        computed_mh = struct_masses[s_idx]

        # Check tolerance: 0.5 Da for integer masses, 10 ppm for precise
        is_integer_mass = ref.expected_mh_mass == int(ref.expected_mh_mass)
        if is_integer_mass:
            if error > 0.5:
                continue
        else:
            ppm = abs((computed_mh - ref.expected_mh_mass) / ref.expected_mh_mass) * 1e6
            if ppm > 10.0:
                continue

        corrections[s_idx] = ref.compound_id
        assigned_structs.add(s_idx)
        assigned_refs.add(r_idx)

    if corrections:
        logger.info(
            "Mass-based ID correction: reassigned %d structure(s): %s",
            len(corrections),
            {idx: cid for idx, cid in corrections.items()},
        )

    return corrections


def _ppm_error(computed: float, reference: float) -> float:
    """Calculate mass error in parts per million."""
    if reference == 0:
        return 0.0
    return ((computed - reference) / reference) * 1e6
