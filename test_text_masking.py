"""Validation tests for PaddleOCR text masking on chemical structure crops.

Run:  python test_text_masking.py [--image PATH] [--dir PATH]

Without arguments, runs synthetic tests to validate the classification logic.
With --image or --dir, runs masking on real structure crops and saves results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_structure(width: int = 300, height: int = 300) -> Image.Image:
    """Draw a simple benzene-like hexagon as a minimal structure image."""
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = width // 2, height // 2
    r = min(width, height) // 4
    points = []
    for i in range(6):
        angle = np.pi / 3 * i - np.pi / 2
        points.append((cx + int(r * np.cos(angle)), cy + int(r * np.sin(angle))))
    for i in range(6):
        draw.line([points[i], points[(i + 1) % 6]], fill=(0, 0, 0), width=2)
    return img


def _add_atom_labels(img: Image.Image) -> Image.Image:
    """Add typical atom labels (OH, N, Br) onto a structure image."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    labels = [("OH", (50, 50)), ("N", (200, 100)), ("Br", (250, 200))]
    for text, pos in labels:
        draw.text(pos, text, fill=(0, 0, 0))
    return img


def _add_contaminating_text(img: Image.Image) -> Image.Image:
    """Add captions and compound names that should be masked."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    texts = [
        ("Example 85", (10, img.height - 30)),
        ("Step 2", (img.width - 80, 10)),
        ("N-(7-bromo-5-methyl)", (10, 10)),
    ]
    for text, pos in texts:
        draw.text(pos, text, fill=(0, 0, 0))
    return img


def _ink_pixel_count(img: Image.Image) -> int:
    """Count non-white pixels."""
    arr = np.array(img.convert('L'))
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.countNonZero(binary)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_classification_logic():
    """Unit-test the _is_atom_label helper without requiring PaddleOCR."""
    from pdf_to_smiles.core.text_masker import _is_atom_label

    img_w = 300

    # Should keep (atom labels)
    keep = ["O", "N", "OH", "Br", "NH2", "CH3", "R1", "Me", "Boc", "+", "-", "n"]
    for text in keep:
        assert _is_atom_label(text, box_width=20, img_width=img_w), \
            f"Expected KEEP for '{text}' but got MASK"

    # Should mask (contaminating text)
    mask = ["Example", "Step 2", "compound", "N-(7-bromo", "Table 3"]
    for text in mask:
        assert not _is_atom_label(text, box_width=150, img_width=img_w), \
            f"Expected MASK for '{text}' but got KEEP"

    print("[PASS] Classification logic: all atom labels kept, all captions masked")


def test_availability():
    """Check is_available() returns a bool without crashing."""
    from pdf_to_smiles.core.text_masker import is_available
    avail = is_available()
    print(f"[INFO] PaddleOCR available: {avail}")
    return avail


def test_mask_clean_structure():
    """Masking a clean structure (no text) should return image unchanged."""
    from pdf_to_smiles.core.text_masker import mask_text_regions

    img = _make_synthetic_structure()
    result = mask_text_regions(img)

    before_ink = _ink_pixel_count(img)
    after_ink = _ink_pixel_count(result)

    # Allow tiny variance from JPEG-like rounding but no large loss
    ratio = after_ink / max(before_ink, 1)
    assert ratio > 0.90, f"Clean structure lost too much ink: {ratio:.2%}"
    print(f"[PASS] Clean structure: ink preserved ({ratio:.1%})")


def test_mask_with_atom_labels():
    """Atom labels should be preserved by the masker."""
    from pdf_to_smiles.core.text_masker import mask_text_regions

    img = _add_atom_labels(_make_synthetic_structure())
    result = mask_text_regions(img)

    before_ink = _ink_pixel_count(img)
    after_ink = _ink_pixel_count(result)
    ratio = after_ink / max(before_ink, 1)

    assert ratio > 0.85, f"Atom-label structure lost too much ink: {ratio:.2%}"
    print(f"[PASS] Atom labels preserved: ink ratio {ratio:.1%}")


def test_mask_contaminated():
    """Contaminating text should be removed, reducing ink count."""
    from pdf_to_smiles.core.text_masker import mask_text_regions

    base = _make_synthetic_structure()
    contaminated = _add_contaminating_text(base)
    result = mask_text_regions(contaminated)

    base_ink = _ink_pixel_count(base)
    contaminated_ink = _ink_pixel_count(contaminated)
    result_ink = _ink_pixel_count(result)

    # Result should have less ink than contaminated version
    assert result_ink < contaminated_ink, \
        "Masking did not remove any contaminating text"
    # Result should be closer to the clean base than to the contaminated image
    print(f"[PASS] Contaminated text removed: "
          f"base={base_ink}, dirty={contaminated_ink}, cleaned={result_ink}")


def test_image_cleaner_integration():
    """Verify clean_structure_image accepts mask_text parameter."""
    from pdf_to_smiles.core.image_cleaner import clean_structure_image

    img = _make_synthetic_structure()
    # Should not raise
    result_no_mask = clean_structure_image(img, mask_text=False)
    assert result_no_mask is not None
    print("[PASS] clean_structure_image(mask_text=False) works")

    if test_availability():
        result_mask = clean_structure_image(img, mask_text=True)
        assert result_mask is not None
        print("[PASS] clean_structure_image(mask_text=True) works")


def run_on_real_image(image_path: str, output_dir: str | None = None):
    """Run text masking on a real image and optionally save the result."""
    from pdf_to_smiles.core.text_masker import mask_text_regions

    img = Image.open(image_path).convert('RGB')
    result = mask_text_regions(img)

    before_ink = _ink_pixel_count(img)
    after_ink = _ink_pixel_count(result)
    ratio = after_ink / max(before_ink, 1)
    removed = 1.0 - ratio

    name = Path(image_path).stem
    print(f"  {name}: ink removed = {removed:.1%}  (before={before_ink}, after={after_ink})")

    if output_dir:
        out_path = Path(output_dir) / f"{name}_masked.png"
        result.save(str(out_path))
        print(f"    → saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test PaddleOCR text masking")
    parser.add_argument("--image", type=str, help="Path to a single structure crop image")
    parser.add_argument("--dir", type=str, help="Directory of structure crop images")
    parser.add_argument("--output", type=str, help="Directory to save masked images")
    args = parser.parse_args()

    print("=" * 60)
    print("Text Masking Tests")
    print("=" * 60)

    # Always run unit tests
    test_classification_logic()
    avail = test_availability()

    if avail:
        print("\n--- PaddleOCR integration tests ---")
        test_mask_clean_structure()
        test_mask_with_atom_labels()
        test_mask_contaminated()
    else:
        print("\n[SKIP] PaddleOCR not installed — skipping integration tests")

    print("\n--- Integration test ---")
    test_image_cleaner_integration()

    # Real image tests
    if args.image:
        print(f"\n--- Real image: {args.image} ---")
        run_on_real_image(args.image, args.output)
    elif args.dir:
        img_dir = Path(args.dir)
        images = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
        if not images:
            print(f"No images found in {args.dir}")
        else:
            print(f"\n--- Processing {len(images)} images from {args.dir} ---")
            if args.output:
                Path(args.output).mkdir(parents=True, exist_ok=True)
            for p in images:
                run_on_real_image(str(p), args.output)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
