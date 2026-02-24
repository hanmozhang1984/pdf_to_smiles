"""Test script for P3: Claude Vision page layout verification.

Tests:
1. LLM layout analyzer availability and classification
2. Hybrid classifier decision logic
3. get_classifier() factory priority chain
4. Graceful fallback when ANTHROPIC_API_KEY is not set
5. API call count (should be ~10-20% of total pages)

Usage:
    python test_llm_layout.py                          # Run all tests
    python test_llm_layout.py --pdf /path/to/patent.pdf  # Test with a specific PDF
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, "src")

from PIL import Image

from pdf_to_smiles.core.doclayout_classifier import PageClassification


def test_availability():
    """Test is_available() check."""
    print("=" * 60)
    print("TEST 1: LLM Layout Analyzer availability")
    print("=" * 60)

    from pdf_to_smiles.core.llm_layout_analyzer import is_available

    available = is_available()
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    try:
        import anthropic  # noqa: F401
        has_sdk = True
    except ImportError:
        has_sdk = False

    print(f"  anthropic SDK installed: {has_sdk}")
    print(f"  ANTHROPIC_API_KEY set:   {has_key}")
    print(f"  is_available():          {available}")
    print(f"  Expected:                {has_sdk and has_key}")
    assert available == (has_sdk and has_key), "is_available() returned unexpected value"
    print("  PASSED")
    return available


def test_get_classifier():
    """Test get_classifier() factory priority chain."""
    print("\n" + "=" * 60)
    print("TEST 2: get_classifier() factory")
    print("=" * 60)

    from pdf_to_smiles.core.page_classifier import get_classifier

    classifier = get_classifier()
    cls_name = type(classifier).__name__

    # Check YOLO availability
    try:
        from pdf_to_smiles.core.doclayout_classifier import DocLayoutClassifier
        yolo_ok = True
    except ImportError:
        yolo_ok = False

    from pdf_to_smiles.core.llm_layout_analyzer import is_available
    llm_ok = is_available()

    print(f"  YOLO available:    {yolo_ok}")
    print(f"  LLM available:     {llm_ok}")
    print(f"  Classifier chosen: {cls_name}")

    if yolo_ok and llm_ok:
        expected = "HybridClassifier"
    elif yolo_ok:
        expected = "DocLayoutClassifier"
    else:
        expected = "PageClassifier"

    print(f"  Expected:          {expected}")
    assert cls_name == expected, f"Expected {expected}, got {cls_name}"
    print("  PASSED")


def test_llm_classify_single_page():
    """Test Claude Vision classification on a synthetic test image."""
    print("\n" + "=" * 60)
    print("TEST 3: LLM single page classification")
    print("=" * 60)

    from pdf_to_smiles.core.llm_layout_analyzer import LLMLayoutAnalyzer

    analyzer = LLMLayoutAnalyzer()

    # Create a blank white page (should classify as text-only / no structures)
    blank = Image.new("RGB", (800, 1000), "white")

    print("  Classifying blank white page...")
    start = time.time()
    result = analyzer.classify_page(blank)
    elapsed = time.time() - start

    print(f"  Result: structures={result.has_structures}, tables={result.has_tables}")
    print(f"  Categories: {result.categories}")
    print(f"  Confidence: {result.confidence_scores}")
    print(f"  should_process: {result.should_process}")
    print(f"  Time: {elapsed:.1f}s")

    # A blank page should not be classified as having structures
    assert not result.should_process, "Blank page should not be classified as having structures"
    print("  PASSED")


def test_hybrid_classifier_stats():
    """Test hybrid classifier with a PDF and check API call stats."""
    print("\n" + "=" * 60)
    print("TEST 4: Hybrid classifier with PDF")
    print("=" * 60)

    pdf_path = None
    # Try common test patent locations
    candidates = [
        "/Users/hanmozhang/Downloads/WO2021026098A1_kif18pages.pdf",
    ]
    for path in candidates:
        if os.path.exists(path):
            pdf_path = path
            break

    if pdf_path is None:
        print("  SKIPPED: No test PDF found. Use --pdf to specify one.")
        return

    from pdf_to_smiles.core.hybrid_classifier import HybridClassifier

    classifier = HybridClassifier()

    print(f"  PDF: {pdf_path}")
    print("  Scanning pages...")

    start = time.time()
    pages = classifier.detect_structure_pages(
        pdf_path,
        progress_callback=lambda cur, tot: print(f"    Page {cur}/{tot}", end="\r"),
    )
    elapsed = time.time() - start
    print()  # clear \r

    stats = classifier.stats
    print(f"  Detected pages: {pages}")
    print(f"  Total pages scanned: {stats['total']}")
    print(f"  YOLO-only decisions: {stats['yolo_only']}")
    print(f"  LLM-verified pages:  {stats['llm_verified']}")
    print(f"  LLM overrides:       {stats['llm_overridden']}")
    print(f"  Time: {elapsed:.1f}s")

    if stats["total"] > 0:
        llm_pct = stats["llm_verified"] / stats["total"] * 100
        print(f"  LLM call rate: {llm_pct:.1f}% (target: ~10-20%)")

    print("  PASSED")


def test_hybrid_vs_yolo_only(pdf_path: str):
    """Compare hybrid classifier results against YOLO-only baseline."""
    print("\n" + "=" * 60)
    print("TEST 5: Hybrid vs YOLO-only comparison")
    print("=" * 60)

    if not os.path.exists(pdf_path):
        print(f"  SKIPPED: PDF not found at {pdf_path}")
        return

    from pdf_to_smiles.core.doclayout_classifier import DocLayoutClassifier
    from pdf_to_smiles.core.hybrid_classifier import HybridClassifier

    print(f"  PDF: {pdf_path}")

    # YOLO-only
    print("  Running YOLO-only...")
    yolo = DocLayoutClassifier()
    start = time.time()
    yolo_pages = yolo.detect_structure_pages(pdf_path)
    yolo_time = time.time() - start

    # Hybrid
    print("  Running Hybrid (YOLO + Claude Vision)...")
    hybrid = HybridClassifier()
    start = time.time()
    hybrid_pages = hybrid.detect_structure_pages(pdf_path)
    hybrid_time = time.time() - start

    print(f"\n  YOLO-only pages:  {yolo_pages} ({yolo_time:.1f}s)")
    print(f"  Hybrid pages:     {hybrid_pages} ({hybrid_time:.1f}s)")

    # Find differences
    yolo_set = set(yolo_pages)
    hybrid_set = set(hybrid_pages)
    added = hybrid_set - yolo_set
    removed = yolo_set - hybrid_set

    if added:
        print(f"  Pages ADDED by Claude Vision:   {sorted(added)}")
    if removed:
        print(f"  Pages REMOVED by Claude Vision: {sorted(removed)}")
    if not added and not removed:
        print("  No differences (Claude Vision agreed with YOLO on all pages)")

    stats = hybrid.stats
    print(f"\n  Hybrid stats: {stats}")
    print("  PASSED")


def test_fallback_without_api_key():
    """Test graceful fallback when ANTHROPIC_API_KEY is not set."""
    print("\n" + "=" * 60)
    print("TEST 6: Fallback without API key")
    print("=" * 60)

    # Temporarily remove API key
    original_key = os.environ.pop("ANTHROPIC_API_KEY", None)

    try:
        from pdf_to_smiles.core.llm_layout_analyzer import is_available
        assert not is_available(), "is_available() should be False without API key"
        print("  is_available() correctly returns False without API key")

        # Re-import to test get_classifier with modified env
        # (need to reload because is_available is called at import time)
        from pdf_to_smiles.core.page_classifier import get_classifier
        classifier = get_classifier()
        cls_name = type(classifier).__name__
        print(f"  get_classifier() returns: {cls_name}")

        # Should NOT be HybridClassifier
        assert cls_name != "HybridClassifier", "Should not use HybridClassifier without API key"
        print("  PASSED")
    finally:
        # Restore API key
        if original_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = original_key


def main():
    parser = argparse.ArgumentParser(description="Test P3 Claude Vision layout verification")
    parser.add_argument("--pdf", help="Path to a patent PDF for testing")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    print("P3: Claude Vision Page Layout Verification - Test Suite")
    print("=" * 60)

    # Test 1: Availability check (always runs)
    llm_available = test_availability()

    # Test 2: Factory (always runs)
    test_get_classifier()

    # Tests requiring API key
    if llm_available:
        # Test 3: Single page classification
        test_llm_classify_single_page()

        # Test 4: Hybrid classifier with PDF
        test_hybrid_classifier_stats()

        # Test 5: Comparison (needs PDF)
        pdf_path = args.pdf or "/Users/hanmozhang/Downloads/WO2021026098A1_kif18pages.pdf"
        test_hybrid_vs_yolo_only(pdf_path)
    else:
        print("\n  SKIPPING API tests (anthropic SDK or ANTHROPIC_API_KEY not available)")

    # Test 6: Fallback (always runs)
    test_fallback_without_api_key()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
