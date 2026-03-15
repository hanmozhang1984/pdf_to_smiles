"""Export handler for saving extraction results to various formats."""

import csv
from pathlib import Path
from typing import List, Optional
import os

from ..models.extraction_result import ExtractionResult


class ExportHandler:
    """Handles exporting extraction results to CSV and TXT formats."""

    @staticmethod
    def export_to_csv(
        results: List[ExtractionResult],
        output_path: str,
        include_invalid: bool = True,
        example_only: bool = False,
        source_file: Optional[str] = None,
        custom_columns: Optional[List[str]] = None,
        custom_data: Optional[dict] = None
    ) -> None:
        """Export results to a CSV file.

        Args:
            results: List of ExtractionResult objects to export.
            output_path: Path for the output CSV file.
            include_invalid: Whether to include invalid/failed predictions.
            example_only: If True, only export example compounds (exclude "other").
            source_file: Optional source PDF filename to include in output.
            custom_columns: Optional list of custom column names.
            custom_data: Optional dict of {(row, col_name): value} for custom data.

        Raises:
            IOError: If file cannot be written.
        """
        custom_columns = custom_columns or []
        custom_data = custom_data or {}

        # Filter results if needed
        if example_only:
            results = [r for r in results if r.compound_type != "other"]
        if not include_invalid:
            results = [r for r in results if r.is_valid]

        import re

        # Collect all unique bio_data keys from results for dynamic columns
        all_bio_keys = set()
        for result in results:
            all_bio_keys.update(result.bio_data.keys())

        # Priority patterns for common assay types
        priority_patterns = [
            re.compile(r'ic\s*[-_]?\s*50', re.IGNORECASE),
            re.compile(r'ec\s*[-_]?\s*50', re.IGNORECASE),
            re.compile(r'gi\s*[-_]?\s*50', re.IGNORECASE),
            re.compile(r'\bki\b', re.IGNORECASE),
            re.compile(r'\bkd\b', re.IGNORECASE),
            re.compile(r'emax', re.IGNORECASE),
        ]

        def get_priority(key):
            for idx, pattern in enumerate(priority_patterns):
                if pattern.search(key):
                    return idx
            return len(priority_patterns)

        # Sort: priority items first, then alphabetically
        sorted_bio_keys = sorted(all_bio_keys, key=lambda k: (get_priority(k), k.lower()))

        # Ensure parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header - base columns + dynamic bio columns + custom columns
            headers = [
                'Source_File', 'Compound_ID', 'Compound_Type', 'Page', 'Structure', 'SMILES', 'Canonical_SMILES', 'Valid',
                'MW_Da', 'cLogP', 'TPSA', 'Rotatable_Bonds', 'Stereocenters',
                'Molecular_Formula', 'Error'
            ]
            # Add dynamic bio data column headers
            headers.extend(sorted_bio_keys)
            # Add custom column headers
            headers.extend(custom_columns)
            writer.writerow(headers)

            # Write data rows
            for row_idx, result in enumerate(results):
                # Use result's source_file, fall back to parameter, then empty
                row_source = result.source_file or source_file or ''
                row = [
                    row_source,
                    result.compound_id or '',
                    result.compound_type or '',
                    result.page_number,
                    result.structure_index + 1,  # 1-indexed for users
                    result.smiles or '',
                    result.canonical_smiles or '',
                    'Yes' if result.is_valid else 'No',
                    f"{result.molecular_weight:.2f}" if result.molecular_weight else '',
                    f"{result.clogp:.2f}" if result.clogp is not None else '',
                    f"{result.tpsa:.1f}" if result.tpsa is not None else '',
                    result.num_rotatable_bonds if result.num_rotatable_bonds is not None else '',
                    result.num_stereocenters if result.num_stereocenters is not None else '',
                    result.molecular_formula or '',
                    result.error_message or ''
                ]
                # Add dynamic bio data values
                for bio_key in sorted_bio_keys:
                    row.append(result.bio_data.get(bio_key, ''))
                # Add custom column values
                for col_name in custom_columns:
                    row.append(custom_data.get((row_idx, col_name), ''))
                writer.writerow(row)

    @staticmethod
    def export_to_txt(
        results: List[ExtractionResult],
        output_path: str,
        include_invalid: bool = True,
        example_only: bool = False,
        source_file: Optional[str] = None,
        custom_columns: Optional[List[str]] = None,
        custom_data: Optional[dict] = None
    ) -> None:
        """Export results to a plain text file.

        Args:
            results: List of ExtractionResult objects to export.
            output_path: Path for the output TXT file.
            include_invalid: Whether to include invalid/failed predictions.
            example_only: If True, only export example compounds (exclude "other").
            source_file: Optional source PDF filename to include in output.
            custom_columns: Optional list of custom column names.
            custom_data: Optional dict of {(row, col_name): value} for custom data.

        Raises:
            IOError: If file cannot be written.
        """
        custom_columns = custom_columns or []
        custom_data = custom_data or {}

        # Filter results if needed
        if example_only:
            results = [r for r in results if r.compound_type != "other"]
        if not include_invalid:
            results = [r for r in results if r.is_valid]

        # Ensure parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("=" * 60 + "\n")
            f.write("PDF Chemical Structure Extraction Results\n")
            f.write("=" * 60 + "\n\n")

            # Summary statistics
            f.write(f"Total structures: {len(results)}\n")
            valid_count = sum(1 for r in results if r.is_valid)
            f.write(f"Valid SMILES: {valid_count}\n")
            f.write(f"Invalid/Failed: {len(results) - valid_count}\n")

            # List unique source files
            source_files = set(r.source_file for r in results if r.source_file)
            if source_files:
                f.write(f"Source files: {len(source_files)}\n")
                for sf in sorted(source_files):
                    f.write(f"  - {sf}\n")

            f.write("\n" + "-" * 60 + "\n\n")

            # Write results
            for row_idx, result in enumerate(results):
                row_source = result.source_file or source_file or 'Unknown'
                f.write(f"[{row_source}] Page {result.page_number}, Structure {result.structure_index + 1}\n")
                if result.compound_id:
                    f.write(f"  Compound ID: {result.compound_id}\n")
                if result.compound_type:
                    f.write(f"  Compound Type: {result.compound_type}\n")
                f.write(f"  SMILES: {result.smiles or 'N/A'}\n")
                if result.canonical_smiles and result.canonical_smiles != result.smiles:
                    f.write(f"  Canonical: {result.canonical_smiles}\n")
                f.write(f"  Valid: {'Yes' if result.is_valid else 'No'}\n")
                # Physicochemical properties
                if result.is_valid:
                    if result.molecular_formula:
                        f.write(f"  Formula: {result.molecular_formula}\n")
                    if result.molecular_weight:
                        f.write(f"  MW: {result.molecular_weight:.2f} Da\n")
                    if result.clogp is not None:
                        f.write(f"  cLogP: {result.clogp:.2f}\n")
                    if result.tpsa is not None:
                        f.write(f"  TPSA: {result.tpsa:.1f} A^2\n")
                    if result.num_rotatable_bonds is not None:
                        f.write(f"  Rotatable Bonds: {result.num_rotatable_bonds}\n")
                    if result.num_stereocenters is not None:
                        f.write(f"  Stereocenters: {result.num_stereocenters}\n")
                # Biological data (dynamic)
                for bio_key, bio_value in result.bio_data.items():
                    f.write(f"  {bio_key}: {bio_value}\n")
                # Custom columns
                for col_name in custom_columns:
                    value = custom_data.get((row_idx, col_name), '')
                    if value:
                        f.write(f"  {col_name}: {value}\n")
                if result.error_message:
                    f.write(f"  Error: {result.error_message}\n")
                f.write("\n")

    @staticmethod
    def export_smiles_only(
        results: List[ExtractionResult],
        output_path: str,
        canonical: bool = True,
        example_only: bool = False,
    ) -> None:
        """Export only valid SMILES strings, one per line.

        Args:
            results: List of ExtractionResult objects to export.
            output_path: Path for the output file.
            canonical: If True, export canonical SMILES; otherwise raw predictions.
            example_only: If True, only export example compounds (exclude "other").

        Raises:
            IOError: If file cannot be written.
        """
        if example_only:
            results = [r for r in results if r.compound_type != "other"]

        # Ensure parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                if result.is_valid:
                    smiles = result.canonical_smiles if canonical else result.smiles
                    if smiles:
                        f.write(smiles + '\n')

    @staticmethod
    def export_to_sdf(
        results: List[ExtractionResult],
        output_path: str,
        source_file: Optional[str] = None,
        example_only: bool = False,
    ) -> int:
        """Export valid results to an SDF (Structure Data File) format.

        Args:
            results: List of ExtractionResult objects to export.
            output_path: Path for the output SDF file.
            source_file: Optional source PDF filename to include as property.
            example_only: If True, only export example compounds (exclude "other").

        Returns:
            Number of molecules successfully written.

        Raises:
            IOError: If file cannot be written.
            ImportError: If RDKit is not available.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            raise ImportError("RDKit is required for SDF export")

        if example_only:
            results = [r for r in results if r.compound_type != "other"]

        # Ensure parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Filter to valid results only (SDF requires valid structures)
        valid_results = [r for r in results if r.is_valid and r.canonical_smiles]

        writer = Chem.SDWriter(output_path)
        molecules_written = 0

        for result in valid_results:
            try:
                mol = Chem.MolFromSmiles(result.canonical_smiles)
                if mol is None:
                    continue

                # Generate 2D coordinates for better visualization
                AllChem.Compute2DCoords(mol)

                # Set molecule name
                mol.SetProp("_Name", f"Page{result.page_number}_Struct{result.structure_index + 1}")

                # Add properties - use result's source_file, fall back to parameter
                row_source = result.source_file or source_file or ""
                if row_source:
                    mol.SetProp("Source_File", row_source)
                if result.compound_id:
                    mol.SetProp("Compound_ID", result.compound_id)
                if result.compound_type:
                    mol.SetProp("Compound_Type", result.compound_type)
                mol.SetProp("Page_Number", str(result.page_number))
                mol.SetProp("Structure_Index", str(result.structure_index + 1))
                mol.SetProp("Original_SMILES", result.smiles or "")
                mol.SetProp("Canonical_SMILES", result.canonical_smiles or "")

                # Add physicochemical properties
                if result.molecular_formula:
                    mol.SetProp("Molecular_Formula", result.molecular_formula)
                if result.molecular_weight is not None:
                    mol.SetProp("Molecular_Weight_Da", f"{result.molecular_weight:.2f}")
                if result.clogp is not None:
                    mol.SetProp("cLogP", f"{result.clogp:.2f}")
                if result.tpsa is not None:
                    mol.SetProp("TPSA", f"{result.tpsa:.1f}")
                if result.num_rotatable_bonds is not None:
                    mol.SetProp("Rotatable_Bonds", str(result.num_rotatable_bonds))
                if result.num_stereocenters is not None:
                    mol.SetProp("Stereocenters", str(result.num_stereocenters))

                # Add biological data (dynamic)
                for bio_key, bio_value in result.bio_data.items():
                    # Sanitize property name for SDF (replace special chars)
                    safe_key = bio_key.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
                    mol.SetProp(safe_key, bio_value)

                writer.write(mol)
                molecules_written += 1

            except Exception:
                # Skip molecules that fail to process
                continue

        writer.close()
        return molecules_written

    @staticmethod
    def get_summary(results: List[ExtractionResult]) -> dict:
        """Get a summary of extraction results.

        Args:
            results: List of ExtractionResult objects.

        Returns:
            Dictionary with summary statistics.
        """
        total = len(results)
        valid = sum(1 for r in results if r.is_valid)
        invalid = sum(1 for r in results if r.smiles and not r.is_valid)
        failed = sum(1 for r in results if not r.smiles)

        pages = set(r.page_number for r in results)

        return {
            'total_structures': total,
            'valid_smiles': valid,
            'invalid_smiles': invalid,
            'failed_predictions': failed,
            'pages_with_structures': len(pages),
            'success_rate': (valid / total * 100) if total > 0 else 0
        }
