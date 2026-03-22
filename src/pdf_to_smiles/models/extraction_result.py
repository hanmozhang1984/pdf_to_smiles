"""Data classes for extraction results."""

from dataclasses import dataclass, field
from typing import Dict, Optional
from PIL import Image


@dataclass
class ExtractionResult:
    """Holds the result of extracting and processing a single chemical structure.

    Attributes:
        page_number: The PDF page number (1-indexed) where the structure was found.
        structure_index: The index of the structure on the page (0-indexed).
        source_file: The filename of the source PDF.
        original_image: The cropped image of the detected chemical structure.
        smiles: The predicted SMILES string, or None if prediction failed.
        is_valid: Whether the SMILES was validated by RDKit.
        canonical_smiles: The canonicalized SMILES from RDKit, if valid.
        rdkit_image: The 2D structure rendered by RDKit, if SMILES is valid.
        error_message: Error message if processing failed.
        molecular_weight: Molecular weight in Da.
        clogp: Calculated LogP (lipophilicity).
        tpsa: Topological polar surface area in Å².
        num_rotatable_bonds: Number of rotatable bonds.
        num_stereocenters: Number of stereogenic centers.
        molecular_formula: Molecular formula string.
        compound_id: Detected compound/example number from nearby text (e.g., "Compound 1").
        compound_type: Classification label from LLM classifier ("example_compound" or "other").
        bounding_box: Structure position in page coordinates (x1, y1, x2, y2).
        ic50: IC50 value from biological data tables.
        ec50: EC50 value from biological data tables.
        ki: Ki value from biological data tables.
        kd: Kd value from biological data tables.
    """
    page_number: int
    structure_index: int
    source_file: Optional[str] = None
    original_image: Optional[Image.Image] = None
    smiles: Optional[str] = None
    is_valid: bool = False
    canonical_smiles: Optional[str] = None
    rdkit_image: Optional[Image.Image] = None
    error_message: Optional[str] = None
    # Physicochemical properties
    molecular_weight: Optional[float] = None
    clogp: Optional[float] = None
    tpsa: Optional[float] = None
    num_rotatable_bonds: Optional[int] = None
    num_stereocenters: Optional[int] = None
    molecular_formula: Optional[str] = None
    # Compound identification
    compound_id: Optional[str] = None  # Detected compound/example number (e.g., "Compound 1")
    compound_type: Optional[str] = None  # "example_compound" or "other" (from LLM classifier)
    bounding_box: Optional[tuple] = None  # (x1, y1, x2, y2) in page coordinates
    # Biological data - legacy fields (kept for backwards compatibility)
    ic50: Optional[str] = None
    ec50: Optional[str] = None
    ki: Optional[str] = None
    kd: Optional[str] = None
    # Dynamic biological data (assay_name -> value) for any assay type
    bio_data: Dict[str, str] = field(default_factory=dict)
    # Formula validation (from patent analytical data)
    reference_formula: Optional[str] = None
    formula_validation: Optional[str] = None  # "match", "mismatch", "mass_only_match", "no_reference"
    reference_mass: Optional[float] = None
    mass_error_ppm: Optional[float] = None

    @property
    def display_smiles(self) -> str:
        """Return the SMILES for display, or 'INVALID' if not available."""
        if self.smiles:
            return self.smiles
        if self.error_message:
            return f"ERROR: {self.error_message}"
        return "N/A"

    @property
    def validation_status(self) -> str:
        """Return a human-readable validation status."""
        if self.is_valid:
            return "Valid"
        if self.smiles:
            return "Invalid"
        return "N/A"


@dataclass
class ProcessingProgress:
    """Tracks the progress of PDF processing.

    Attributes:
        current_page: The page currently being processed.
        total_pages: Total number of pages in the PDF.
        current_structure: The structure currently being processed on the page.
        total_structures: Total structures detected on the current page.
        status_message: Human-readable status message.
    """
    current_page: int = 0
    total_pages: int = 0
    current_structure: int = 0
    total_structures: int = 0
    status_message: str = ""

    @property
    def overall_progress(self) -> float:
        """Calculate overall progress as a percentage (0-100)."""
        if self.total_pages == 0:
            return 0.0
        page_progress = (self.current_page - 1) / self.total_pages
        if self.total_structures > 0:
            structure_progress = self.current_structure / self.total_structures / self.total_pages
        else:
            structure_progress = 0
        return min(100.0, (page_progress + structure_progress) * 100)


@dataclass
class PDFInfo:
    """Information about a loaded PDF file.

    Attributes:
        file_path: Path to the PDF file.
        page_count: Number of pages in the PDF.
        file_name: Name of the PDF file.
    """
    file_path: str
    page_count: int

    @property
    def file_name(self) -> str:
        """Extract the file name from the path."""
        import os
        return os.path.basename(self.file_path)
