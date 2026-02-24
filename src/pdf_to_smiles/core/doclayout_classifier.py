"""Page classifier using DocLayout-YOLO for detecting pages with chemical structures or bio data.

Uses a pre-trained document layout analysis model to classify page content.
Dramatically outperforms pixel heuristics (~90%+ vs ~59% accuracy).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pypdfium2 as pdfium
from PIL import Image


# Categories that indicate the page should be processed
_STRUCTURE_CATEGORIES = {"figure", "isolate_formula", "formula_caption"}
_TABLE_CATEGORIES = {"table"}
# Categories that indicate text-only content (skip)
_TEXT_ONLY_CATEGORIES = {"plain text", "title", "abandon", "caption", "footnote"}


@dataclass
class PageClassification:
    """Classification result for a single page."""
    has_structures: bool = False
    has_tables: bool = False
    is_text_only: bool = True
    categories: Dict[str, int] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def should_process(self) -> bool:
        """Whether this page should be sent for structure extraction."""
        return self.has_structures or self.has_tables


class DocLayoutClassifier:
    """Classify PDF pages using DocLayout-YOLO document layout analysis."""

    DPI = 200  # YOLO needs decent resolution
    CONFIDENCE_THRESHOLD = 0.25
    IMGSZ = 1024

    def __init__(self):
        self._model = None

    def _load_model(self):
        """Lazy-load the YOLO model on first use."""
        if self._model is not None:
            return

        from doclayout_yolo import YOLOv10

        # Use HuggingFace cached weights
        model_path = self._find_model_weights()
        self._model = YOLOv10(model_path)

    def _find_model_weights(self) -> str:
        """Find DocLayout-YOLO weights, downloading if needed."""
        import os
        from pathlib import Path

        # Check HuggingFace cache first
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_dir = cache_dir / "models--juliozhao--DocLayout-YOLO-DocStructBench"

        if model_dir.exists():
            # Find the weights file in snapshots
            for weights_file in model_dir.rglob("doclayout_yolo_docstructbench_imgsz1024.pt"):
                return str(weights_file)

        # Try downloading via huggingface_hub
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                filename="doclayout_yolo_docstructbench_imgsz1024.pt",
            )
        except ImportError:
            raise FileNotFoundError(
                "DocLayout-YOLO weights not found. Install huggingface_hub and run:\n"
                "  pip install huggingface_hub\n"
                "  python -c \"from huggingface_hub import hf_hub_download; "
                "hf_hub_download('juliozhao/DocLayout-YOLO-DocStructBench', "
                "'doclayout_yolo_docstructbench_imgsz1024.pt')\""
            )

    def _get_device(self) -> str:
        """Get the best available device for inference."""
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except (ImportError, AttributeError):
            pass
        return "cpu"

    def classify_page(self, pil_image: Image.Image) -> PageClassification:
        """Classify a single page image.

        Args:
            pil_image: PIL Image of the page (should be ~200 DPI).

        Returns:
            PageClassification with detected categories.
        """
        import tempfile
        import os

        self._load_model()

        # YOLO needs a file path, save temp image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            pil_image.save(tmp_path)

        try:
            results = self._model.predict(
                tmp_path,
                imgsz=self.IMGSZ,
                conf=self.CONFIDENCE_THRESHOLD,
                device=self._get_device(),
            )
        finally:
            os.unlink(tmp_path)

        # Parse detections
        categories = {}
        confidence_scores = {}
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            conf = float(box.conf[0])
            if conf < self.CONFIDENCE_THRESHOLD:
                continue
            categories[cls_name] = categories.get(cls_name, 0) + 1
            # Track max confidence per category
            confidence_scores[cls_name] = max(
                confidence_scores.get(cls_name, 0.0), conf
            )

        has_structures = any(c in categories for c in _STRUCTURE_CATEGORIES)
        has_tables = any(c in categories for c in _TABLE_CATEGORIES)

        # Text-only if no structures and no tables detected
        is_text_only = not has_structures and not has_tables

        return PageClassification(
            has_structures=has_structures,
            has_tables=has_tables,
            is_text_only=is_text_only,
            categories=categories,
            confidence_scores=confidence_scores,
        )

    def detect_structure_pages(
        self,
        pdf_path: str,
        progress_callback=None,
    ) -> List[int]:
        """Scan PDF and return 1-indexed page numbers likely containing structures or bio data.

        Same interface as PageClassifier.detect_structure_pages().

        Args:
            pdf_path: Path to the PDF file.
            progress_callback: Optional callable(current_page, total_pages) for progress.

        Returns:
            Sorted list of 1-indexed page numbers.
        """
        self._load_model()
        detected = []

        doc = pdfium.PdfDocument(pdf_path)
        total_pages = len(doc)

        for page_idx in range(total_pages):
            if progress_callback:
                progress_callback(page_idx + 1, total_pages)

            page = doc[page_idx]
            bitmap = page.render(scale=self.DPI / 72)
            pil_image = bitmap.to_pil()

            classification = self.classify_page(pil_image)
            if classification.should_process:
                detected.append(page_idx + 1)  # 1-indexed

        doc.close()
        return detected
