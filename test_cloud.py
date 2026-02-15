"""Quick test script for the Modal cloud endpoint."""

import sys
sys.path.insert(0, "src")

from pdf_to_smiles.cloud.client import CloudInferenceClient
from PIL import Image
import time

def test_cloud_endpoint():
    print("Testing Modal Cloud Endpoint...")
    print(f"Endpoint: {CloudInferenceClient.DEFAULT_ENDPOINT}")
    print("-" * 50)

    client = CloudInferenceClient(timeout=180)  # Longer timeout for cold start

    # Test 1: Health check with dummy image
    print("\n1. Health check (may take 30-60s on cold start)...")
    start = time.time()
    try:
        is_healthy = client.health_check()
        elapsed = time.time() - start
        print(f"   Result: {'OK' if is_healthy else 'FAILED'} ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - start
        print(f"   Error after {elapsed:.1f}s: {e}")
        return False

    # Test 2: Predict on a simple test image (white background with black square)
    print("\n2. Testing SMILES prediction...")
    test_img = Image.new('RGB', (200, 200), 'white')
    # Draw a simple shape (won't be a valid molecule, but tests the endpoint)
    for x in range(50, 150):
        for y in range(50, 150):
            test_img.putpixel((x, y), (0, 0, 0))

    start = time.time()
    try:
        result = client.predict_smiles(test_img)
        elapsed = time.time() - start
        print(f"   Result: {result} ({elapsed:.1f}s)")
        print("   (None or invalid SMILES expected for non-molecule test image)")
    except Exception as e:
        elapsed = time.time() - start
        print(f"   Error after {elapsed:.1f}s: {e}")

    # Test 3: Segmentation
    print("\n3. Testing segmentation...")
    start = time.time()
    try:
        segments = client.segment_structures(test_img)
        elapsed = time.time() - start
        print(f"   Found {len(segments)} segments ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - start
        print(f"   Error after {elapsed:.1f}s: {e}")

    print("\n" + "-" * 50)
    print("Cloud endpoint test complete!")
    return True

if __name__ == "__main__":
    test_cloud_endpoint()
