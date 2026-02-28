"""PP-DocLayout-L rescue pass for catching structures that Docling misses.

Uses PaddleOCR's PP-DocLayout-L model (Apache 2.0, RT-DETR-L, 90.4% mAP@0.5)
to detect figures, formulas, and images on pages that Docling classified as text-only.

Runs inference in a subprocess to avoid mutex crashes when Docling (PyTorch/MPS)
and PaddlePaddle coexist in the same process.
"""

import logging
import multiprocessing as mp
from typing import List

from PIL import Image

logger = logging.getLogger(__name__)

STRUCTURE_LABELS = {"formula", "image", "figure", "figure_title", "figure caption"}
CONFIDENCE_THRESHOLD = 0.25


def _rescue_worker(pdf_path: str, page_numbers: List[int], dpi: int, result_queue: mp.Queue):
    """Subprocess worker: loads PP-DocLayout-L and checks pages for structures."""
    import os
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    import pypdfium2 as pdfium
    import numpy as np
    from paddleocr import LayoutDetection

    model = LayoutDetection(model_name="PP-DocLayout-L")
    doc = pdfium.PdfDocument(pdf_path)

    rescued = []
    for page_no in page_numbers:
        page = doc[page_no - 1]
        bitmap = page.render(scale=dpi / 72)
        pil_image = bitmap.to_pil()
        img_array = np.array(pil_image.convert("RGB"))[:, :, ::-1]  # RGB→BGR

        found = False
        output = model.predict(img_array, batch_size=1)
        for res in output:
            for box in res["boxes"]:
                if (
                    box["score"] >= CONFIDENCE_THRESHOLD
                    and box["label"] in STRUCTURE_LABELS
                ):
                    found = True
                    break
            if found:
                break
        if found:
            rescued.append(page_no)

    doc.close()
    result_queue.put(rescued)


class PPDocLayoutRescue:
    """Run PP-DocLayout-L rescue in a subprocess to avoid Docling/Paddle conflicts."""

    def __init__(self):
        pass

    def rescue_pages(
        self, pdf_path: str, page_numbers: List[int], dpi: int = 200
    ) -> List[int]:
        """Check pages for structures using PP-DocLayout-L in a subprocess.

        Args:
            pdf_path: Path to the PDF file.
            page_numbers: 1-indexed page numbers to check.
            dpi: Rendering resolution.

        Returns:
            List of 1-indexed page numbers where structures were detected.
        """
        if not page_numbers:
            return []

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        proc = ctx.Process(
            target=_rescue_worker,
            args=(pdf_path, page_numbers, dpi, result_queue),
        )
        proc.start()
        proc.join(timeout=600)  # 10-minute timeout

        if proc.exitcode != 0:
            logger.error(
                "PP-DocLayout rescue subprocess failed with exit code %s",
                proc.exitcode,
            )
            return []

        return result_queue.get()
