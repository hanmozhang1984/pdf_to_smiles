"""SMILES validation and 2D structure rendering using RDKit."""

from typing import Optional, Tuple
from PIL import Image
import io


class SMILESValidator:
    """Validates SMILES strings and generates 2D structure renderings.

    Uses RDKit for SMILES parsing, canonicalization, and 2D depiction.
    """

    DEFAULT_IMAGE_SIZE = (200, 200)

    def __init__(self, image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE):
        """Initialize the SMILES validator.

        Args:
            image_size: Size (width, height) for rendered structure images.
        """
        self.image_size = image_size
        self._rdkit_available = None

    def _check_rdkit(self) -> bool:
        """Check if RDKit is available."""
        if self._rdkit_available is None:
            try:
                from rdkit import Chem
                self._rdkit_available = True
            except ImportError:
                self._rdkit_available = False
        return self._rdkit_available

    def validate(self, smiles: str) -> Tuple[bool, Optional[str]]:
        """Validate a SMILES string and return canonical form.

        Args:
            smiles: SMILES string to validate.

        Returns:
            Tuple of (is_valid, canonical_smiles).
            If invalid, canonical_smiles is None.
        """
        if not smiles or not self._check_rdkit():
            return False, None

        try:
            from rdkit import Chem

            # Attempt to parse the SMILES
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                return False, None

            # Get canonical SMILES
            canonical = Chem.MolToSmiles(mol, canonical=True)
            return True, canonical

        except Exception:
            return False, None

    def render_structure(self, smiles: str) -> Optional[Image.Image]:
        """Render a SMILES string as a 2D structure image.

        Args:
            smiles: SMILES string to render.

        Returns:
            PIL Image of the 2D structure, or None if rendering fails.
        """
        if not smiles or not self._check_rdkit():
            return None

        try:
            from rdkit import Chem
            from rdkit.Chem import Draw

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Generate 2D coordinates if needed
            from rdkit.Chem import AllChem
            AllChem.Compute2DCoords(mol)

            # Render to PNG
            img = Draw.MolToImage(mol, size=self.image_size)

            return img

        except Exception as e:
            print(f"Structure rendering failed: {e}")
            return None

    def validate_and_render(
        self, smiles: str
    ) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
        """Validate SMILES and render structure in one call.

        Args:
            smiles: SMILES string to process.

        Returns:
            Tuple of (is_valid, canonical_smiles, structure_image).
            Invalid SMILES returns (False, None, None).
        """
        is_valid, canonical = self.validate(smiles)

        if not is_valid:
            return False, None, None

        image = self.render_structure(canonical or smiles)
        return True, canonical, image

    def get_molecular_formula(self, smiles: str) -> Optional[str]:
        """Get the molecular formula for a SMILES string.

        Args:
            smiles: SMILES string.

        Returns:
            Molecular formula string, or None if invalid.
        """
        if not smiles or not self._check_rdkit():
            return None

        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            return rdMolDescriptors.CalcMolFormula(mol)

        except Exception:
            return None

    def get_molecular_weight(self, smiles: str) -> Optional[float]:
        """Get the molecular weight for a SMILES string.

        Args:
            smiles: SMILES string.

        Returns:
            Molecular weight in Da, or None if invalid.
        """
        if not smiles or not self._check_rdkit():
            return None

        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            return Descriptors.ExactMolWt(mol)

        except Exception:
            return None

    def get_clogp(self, smiles: str) -> Optional[float]:
        """Get the calculated LogP (lipophilicity) for a SMILES string.

        Args:
            smiles: SMILES string.

        Returns:
            cLogP value, or None if invalid.
        """
        if not smiles or not self._check_rdkit():
            return None

        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            return Descriptors.MolLogP(mol)

        except Exception:
            return None

    def get_tpsa(self, smiles: str) -> Optional[float]:
        """Get the topological polar surface area for a SMILES string.

        Args:
            smiles: SMILES string.

        Returns:
            TPSA in Å², or None if invalid.
        """
        if not smiles or not self._check_rdkit():
            return None

        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            return Descriptors.TPSA(mol)

        except Exception:
            return None

    def get_num_rotatable_bonds(self, smiles: str) -> Optional[int]:
        """Get the number of rotatable bonds for a SMILES string.

        Args:
            smiles: SMILES string.

        Returns:
            Number of rotatable bonds, or None if invalid.
        """
        if not smiles or not self._check_rdkit():
            return None

        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            return Descriptors.NumRotatableBonds(mol)

        except Exception:
            return None

    def get_num_stereocenters(self, smiles: str) -> Optional[int]:
        """Get the number of stereogenic centers for a SMILES string.

        Args:
            smiles: SMILES string.

        Returns:
            Number of stereocenters, or None if invalid.
        """
        if not smiles or not self._check_rdkit():
            return None

        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Find chiral centers (includeUnassigned=True to count all potential stereocenters)
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            return len(chiral_centers)

        except Exception:
            return None

    def get_all_properties(self, smiles: str) -> dict:
        """Get all physicochemical properties for a SMILES string.

        Args:
            smiles: SMILES string.

        Returns:
            Dictionary with all calculated properties.
        """
        return {
            'molecular_weight': self.get_molecular_weight(smiles),
            'molecular_formula': self.get_molecular_formula(smiles),
            'clogp': self.get_clogp(smiles),
            'tpsa': self.get_tpsa(smiles),
            'num_rotatable_bonds': self.get_num_rotatable_bonds(smiles),
            'num_stereocenters': self.get_num_stereocenters(smiles),
        }
