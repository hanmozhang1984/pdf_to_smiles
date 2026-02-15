"""Modal.com serverless GPU backend for DECIMER inference.

This module defines the cloud inference endpoint that runs on Modal's
GPU infrastructure. Deploy with:

    modal deploy modal_app.py

Usage from client:
    - Segment structures: POST /segment with image data
    - Predict SMILES: POST /predict with image data
    - Batch predict: POST /predict_batch with list of images
"""

import modal


def download_models():
    """Download DECIMER models at image build time for faster cold starts."""
    import os
    import numpy as np
    from PIL import Image

    print("Pre-downloading DECIMER models...")

    # Import DECIMER to trigger model download
    from DECIMER import predict_SMILES
    from decimer_segmentation import segment_chemical_structures

    # Create a dummy image and run prediction to ensure models are fully downloaded
    dummy_img = Image.new('RGB', (100, 100), 'white')
    dummy_array = np.array(dummy_img)

    # Trigger segmentation model download
    print("Downloading segmentation model...")
    try:
        segment_chemical_structures(dummy_array)
    except Exception as e:
        print(f"Segmentation warmup: {e}")

    # Trigger SMILES prediction model download
    print("Downloading SMILES prediction model...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        dummy_img.save(f.name)
        try:
            predict_SMILES(f.name)
        except Exception as e:
            print(f"Prediction warmup: {e}")
        os.unlink(f.name)

    print("Models pre-downloaded successfully!")


# Define the container image with all dependencies and pre-downloaded models
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")  # OpenCV dependencies
    .pip_install(
        "decimer==2.7.0",
        "decimer-segmentation==1.4.0",
        "tensorflow==2.12.0",
        "numpy==1.23.5",
        "opencv-python==4.9.0.80",
        "Pillow==11.3.0",
        "rdkit==2024.3.3",
        "fastapi[standard]",  # Required for @modal.fastapi_endpoint
    )
    # Pre-download models at build time (baked into image)
    .run_function(download_models)
)

app = modal.App("pdf-to-smiles", image=image)


@app.cls(
    gpu="T4",  # NVIDIA T4 - good balance of cost/performance
    timeout=300,
    scaledown_window=300,  # Keep warm for 5 minutes
)
class DECIMERInference:
    """DECIMER inference class running on Modal GPU."""

    @modal.enter()
    def load_models(self):
        """Load models when container starts (cached across invocations)."""
        # Pre-load DECIMER models
        from DECIMER import predict_SMILES
        from decimer_segmentation import segment_chemical_structures

        self._predict_smiles = predict_SMILES
        self._segment_structures = segment_chemical_structures

        # Warm up with a dummy prediction to fully initialize TensorFlow
        import numpy as np
        from PIL import Image
        dummy_img = Image.new('RGB', (100, 100), 'white')
        dummy_array = np.array(dummy_img)
        try:
            self._segment_structures(dummy_array)
        except Exception:
            pass  # Ignore warmup errors

    @modal.method()
    def segment(self, image_bytes: bytes) -> list[bytes]:
        """Segment chemical structures from a page image.

        Args:
            image_bytes: PNG image data as bytes.

        Returns:
            List of PNG image bytes for each detected structure.
        """
        import io
        import numpy as np
        from PIL import Image

        # Decode input image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(img)

        # Run segmentation
        segments = self._segment_structures(img_array)

        # Convert segments to PNG bytes
        result = []
        for segment in segments:
            if isinstance(segment, np.ndarray):
                if segment.ndim == 2:
                    seg_img = Image.fromarray(segment, mode='L').convert('RGB')
                elif segment.ndim == 3 and segment.shape[2] == 4:
                    seg_img = Image.fromarray(segment, mode='RGBA').convert('RGB')
                else:
                    seg_img = Image.fromarray(segment).convert('RGB')
            else:
                continue

            # Filter small/invalid segments
            w, h = seg_img.size
            if w < 50 or h < 50:
                continue
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 5.0:
                continue

            # Encode to PNG
            buf = io.BytesIO()
            seg_img.save(buf, format='PNG')
            result.append(buf.getvalue())

        return result

    @modal.method()
    def predict(self, image_bytes: bytes) -> str | None:
        """Predict SMILES from a chemical structure image.

        Args:
            image_bytes: PNG image data as bytes.

        Returns:
            SMILES string or None if prediction fails.
        """
        import io
        import tempfile
        import os
        from PIL import Image

        # Decode input image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # DECIMER requires a file path
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "structure.png")
            img.save(input_path, "PNG")

            try:
                smiles = self._predict_smiles(input_path)
                if smiles and isinstance(smiles, str) and len(smiles.strip()) > 0:
                    return smiles.strip()
            except Exception:
                pass

        return None

    @modal.method()
    def predict_batch(self, images_bytes: list[bytes]) -> list[str | None]:
        """Predict SMILES for multiple structure images.

        Args:
            images_bytes: List of PNG image data as bytes.

        Returns:
            List of SMILES strings (None for failed predictions).
        """
        return [self.predict(img_bytes) for img_bytes in images_bytes]

    @modal.method()
    def validate_smiles(self, smiles: str) -> dict:
        """Validate a SMILES string using RDKit.

        Args:
            smiles: SMILES string to validate.

        Returns:
            Dict with 'valid' bool and 'canonical' SMILES if valid.
        """
        from rdkit import Chem

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol)
                return {"valid": True, "canonical": canonical}
        except Exception:
            pass

        return {"valid": False, "canonical": None}


# Global model references for the web endpoint
_predict_smiles = None
_segment_structures = None


def _load_models():
    """Load DECIMER models (called once per container)."""
    global _predict_smiles, _segment_structures
    if _predict_smiles is None:
        from DECIMER import predict_SMILES
        from decimer_segmentation import segment_chemical_structures
        _predict_smiles = predict_SMILES
        _segment_structures = segment_chemical_structures

        # Warm up models
        import numpy as np
        from PIL import Image
        dummy_img = Image.new('RGB', (100, 100), 'white')
        dummy_array = np.array(dummy_img)
        try:
            _segment_structures(dummy_array)
        except Exception:
            pass


# Web endpoint for REST API access
@app.function(
    gpu="T4",
    timeout=300,  # 5 minutes (allows for cold start + inference)
    scaledown_window=300,  # Keep warm for 5 minutes
)
@modal.fastapi_endpoint(method="POST")
def process_image(request: dict) -> dict:
    """REST API endpoint for processing images.

    Request body:
        {
            "action": "segment" | "predict" | "predict_batch",
            "image": "<base64 encoded PNG>" (for segment/predict),
            "images": ["<base64>", ...] (for predict_batch)
        }

    Response:
        {
            "success": bool,
            "result": <action-specific result>,
            "error": <error message if failed>
        }
    """
    import base64
    import io
    import os
    import tempfile
    import numpy as np
    from PIL import Image

    # Load models on first call
    _load_models()

    try:
        action = request.get("action", "predict")

        if action == "segment":
            image_b64 = request.get("image")
            if not image_b64:
                return {"success": False, "error": "Missing 'image' field"}
            image_bytes = base64.b64decode(image_b64)

            # Decode and segment
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_array = np.array(img)
            segments = _segment_structures(img_array)

            # Convert segments to base64 PNG
            result = []
            for segment in segments:
                if isinstance(segment, np.ndarray):
                    if segment.ndim == 2:
                        seg_img = Image.fromarray(segment, mode='L').convert('RGB')
                    elif segment.ndim == 3 and segment.shape[2] == 4:
                        seg_img = Image.fromarray(segment, mode='RGBA').convert('RGB')
                    else:
                        seg_img = Image.fromarray(segment).convert('RGB')
                else:
                    continue

                # Filter small/invalid segments
                w, h = seg_img.size
                if w < 50 or h < 50:
                    continue
                aspect = max(w, h) / max(min(w, h), 1)
                if aspect > 5.0:
                    continue

                buf = io.BytesIO()
                seg_img.save(buf, format='PNG')
                result.append(base64.b64encode(buf.getvalue()).decode())

            return {"success": True, "result": result}

        elif action == "predict":
            image_b64 = request.get("image")
            if not image_b64:
                return {"success": False, "error": "Missing 'image' field"}
            image_bytes = base64.b64decode(image_b64)

            # Decode image
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # DECIMER requires file path
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = os.path.join(temp_dir, "structure.png")
                img.save(input_path, "PNG")

                try:
                    smiles = _predict_smiles(input_path)
                    if smiles and isinstance(smiles, str) and len(smiles.strip()) > 0:
                        return {"success": True, "result": smiles.strip()}
                except Exception:
                    pass

            return {"success": True, "result": None}

        elif action == "predict_batch":
            images_b64 = request.get("images", [])
            if not images_b64:
                return {"success": False, "error": "Missing 'images' field"}

            results = []
            for img_b64 in images_b64:
                image_bytes = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                with tempfile.TemporaryDirectory() as temp_dir:
                    input_path = os.path.join(temp_dir, "structure.png")
                    img.save(input_path, "PNG")

                    try:
                        smiles = _predict_smiles(input_path)
                        if smiles and isinstance(smiles, str) and len(smiles.strip()) > 0:
                            results.append(smiles.strip())
                        else:
                            results.append(None)
                    except Exception:
                        results.append(None)

            return {"success": True, "result": results}

        else:
            return {"success": False, "error": f"Unknown action: {action}"}

    except Exception as e:
        return {"success": False, "error": str(e)}


# CLI for local testing
if __name__ == "__main__":
    # Test locally with: python -m pdf_to_smiles.cloud.modal_app
    print("Modal app defined. Deploy with: modal deploy modal_app.py")
