"""Shared subprocess bridge for running SMILES predictors in isolated venvs.

Manages a persistent worker subprocess so the model stays loaded between
predictions. Communication is via stdin/stdout lines (image path in, SMILES out).
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class SubprocessPredictor:
    """Base class for predictors that run in a separate venv subprocess.

    Subclasses must set:
        VENV_PATH: str  — path to the venv directory
        WORKER_SCRIPT: str — Python source code for the worker process

    The worker script must:
        - Read image paths from stdin (one per line)
        - Write one SMILES result per line to stdout ("NONE" for failures)
        - Print "READY" to stdout once initialization is complete
        - Flush stdout after every write
    """

    VENV_PATH: str = ""
    WORKER_SCRIPT: str = ""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._script_path: Optional[str] = None

    def _get_python(self) -> str:
        return os.path.join(self.VENV_PATH, "bin", "python")

    def _write_worker_script(self) -> str:
        if self._script_path and os.path.exists(self._script_path):
            return self._script_path
        fd, path = tempfile.mkstemp(suffix=".py", prefix="worker_")
        with os.fdopen(fd, "w") as f:
            f.write(self.WORKER_SCRIPT)
        self._script_path = path
        return path

    def _start_process(self) -> None:
        python = self._get_python()
        script = self._write_worker_script()
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        self._process = subprocess.Popen(
            [python, script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit parent's stderr to avoid pipe buffer deadlock
            text=True,
            bufsize=1,
            env=env,
        )
        # Wait for READY signal
        line = self._process.stdout.readline().strip()
        if line != "READY":
            raise RuntimeError(
                f"Worker did not send READY signal. Got: {line!r}\n"
                "Check terminal stderr for details."
            )
        logger.info("Subprocess worker ready: %s", self.__class__.__name__)

    def _ensure_running(self) -> None:
        if self._process is None or self._process.poll() is not None:
            self._start_process()

    def predict_single(self, image: Image.Image) -> Optional[str]:
        """Save image to temp file, send path to worker, read SMILES back."""
        fd, path = tempfile.mkstemp(suffix=".png", prefix="pred_")
        os.close(fd)
        try:
            image.save(path)
            with self._lock:
                self._ensure_running()
                self._process.stdin.write(path + "\n")
                self._process.stdin.flush()
                result = self._process.stdout.readline().strip()
            if result == "NONE" or not result:
                return None
            return result
        except (BrokenPipeError, OSError) as e:
            logger.warning("Subprocess crashed, will restart: %s", e)
            self._process = None
            return None
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def predict_batch(
        self,
        images: List[Image.Image],
        high_accuracy: bool = False,
    ) -> List[Optional[str]]:
        """Predict SMILES for a batch of images, one at a time through the subprocess."""
        return [self.predict_single(img) for img in images]

    def predict(
        self,
        image: Image.Image,
        high_accuracy: bool = False,
    ) -> Optional[str]:
        return self.predict_single(image)

    def check_availability(self) -> tuple[bool, str]:
        python = self._get_python()
        if not os.path.exists(python):
            return False, (
                f"Venv not found at {self.VENV_PATH}\n"
                f"Run the setup script to create it."
            )
        return True, "Venv found, ready to use"

    def close(self) -> None:
        if self._process and self._process.poll() is None:
            try:
                self._process.stdin.close()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None
