"""Inference settings for switching between local and cloud processing."""

from __future__ import annotations

from enum import Enum
from typing import Optional
import json
import os


class InferenceMode(Enum):
    """Available inference modes."""
    LOCAL_LIGHTWEIGHT = "lightweight"  # MolScribe + OpenCV (fast, no TF)
    LOCAL_CPU = "cpu"                 # Local CPU via DECIMER (slow but works everywhere)
    LOCAL_GPU = "gpu"                 # Local GPU (requires NVIDIA + CUDA)
    CLOUD = "cloud"                   # Cloud GPU via Modal.com
    MOLSIGHT = "molsight"             # MolSight (separate venv)


def _auto_detect_best_mode() -> InferenceMode:
    """Auto-detect the best available inference mode.

    Priority: MolScribe (lightweight) > DECIMER (TF) > lightweight recommendation.
    """
    # Check for MolScribe + PyTorch first (lightweight)
    try:
        import molscribe  # noqa: F401
        import torch  # noqa: F401
        return InferenceMode.LOCAL_LIGHTWEIGHT
    except ImportError:
        pass

    # Check for DECIMER + TensorFlow
    try:
        import DECIMER  # noqa: F401
        return InferenceMode.LOCAL_CPU
    except ImportError:
        pass

    # Fall back to lightweight (will prompt user to install on first use)
    return InferenceMode.LOCAL_LIGHTWEIGHT


class InferenceSettings:
    """Singleton settings for inference mode configuration.

    Usage:
        settings = InferenceSettings.get_instance()
        settings.mode = InferenceMode.CLOUD
        settings.cloud_endpoint = "https://..."
    """

    _instance: Optional[InferenceSettings] = None
    _config_file = os.path.join(
        os.path.expanduser("~"),
        ".pdf_to_smiles",
        "inference_settings.json"
    )

    def __init__(self):
        self._mode: InferenceMode = InferenceMode.LOCAL_LIGHTWEIGHT
        self._cloud_endpoint: Optional[str] = None
        self._cloud_timeout: int = 300  # 5 minutes for cold starts
        self._auto_detect_pages: bool = True
        self._anthropic_api_key: Optional[str] = None
        self._load_settings()

    @classmethod
    def get_instance(cls) -> InferenceSettings:
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = InferenceSettings()
        return cls._instance

    @property
    def mode(self) -> InferenceMode:
        """Current inference mode."""
        return self._mode

    @mode.setter
    def mode(self, value: InferenceMode) -> None:
        """Set inference mode."""
        self._mode = value
        self._save_settings()

    @property
    def cloud_endpoint(self) -> Optional[str]:
        """Cloud API endpoint URL."""
        return self._cloud_endpoint

    @cloud_endpoint.setter
    def cloud_endpoint(self, value: Optional[str]) -> None:
        """Set cloud API endpoint."""
        self._cloud_endpoint = value
        self._save_settings()

    @property
    def cloud_timeout(self) -> int:
        """Cloud API timeout in seconds."""
        return self._cloud_timeout

    @cloud_timeout.setter
    def cloud_timeout(self, value: int) -> None:
        """Set cloud API timeout."""
        self._cloud_timeout = value
        self._save_settings()

    @property
    def auto_detect_pages(self) -> bool:
        """Whether to auto-detect structure pages before processing."""
        return self._auto_detect_pages

    @auto_detect_pages.setter
    def auto_detect_pages(self, value: bool) -> None:
        """Set auto-detect pages preference."""
        self._auto_detect_pages = value
        self._save_settings()

    @property
    def anthropic_api_key(self) -> Optional[str]:
        """Anthropic API key for LLM compound classification."""
        return self._anthropic_api_key

    @anthropic_api_key.setter
    def anthropic_api_key(self, value: Optional[str]) -> None:
        """Set Anthropic API key."""
        self._anthropic_api_key = value if value else None
        self._save_settings()

    def apply_api_keys(self) -> None:
        """Apply stored API keys to environment variables."""
        if self._anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self._anthropic_api_key
        elif "ANTHROPIC_API_KEY" not in os.environ:
            pass  # Don't overwrite env var set externally

    @property
    def is_cloud(self) -> bool:
        """Check if using cloud inference."""
        return self._mode == InferenceMode.CLOUD

    @property
    def is_local_gpu(self) -> bool:
        """Check if using local GPU inference."""
        return self._mode == InferenceMode.LOCAL_GPU

    @property
    def is_lightweight(self) -> bool:
        """Check if using lightweight local inference (MolScribe + OpenCV)."""
        return self._mode == InferenceMode.LOCAL_LIGHTWEIGHT

    @property
    def is_molsight(self) -> bool:
        """Check if using MolSight inference."""
        return self._mode == InferenceMode.MOLSIGHT


    def _load_settings(self) -> None:
        """Load settings from config file."""
        try:
            if os.path.exists(self._config_file):
                with open(self._config_file, 'r') as f:
                    data = json.load(f)
                    mode_str = data.get("mode", "lightweight")
                    try:
                        self._mode = InferenceMode(mode_str)
                    except ValueError:
                        self._mode = InferenceMode.LOCAL_LIGHTWEIGHT
                    self._cloud_endpoint = data.get("cloud_endpoint")
                    self._cloud_timeout = data.get("cloud_timeout", 300)
                    self._auto_detect_pages = data.get("auto_detect_pages", True)
                    self._anthropic_api_key = data.get("anthropic_api_key")
        except Exception:
            pass  # Use defaults if loading fails

    def _save_settings(self) -> None:
        """Save settings to config file."""
        try:
            os.makedirs(os.path.dirname(self._config_file), exist_ok=True)
            with open(self._config_file, 'w') as f:
                data = {
                    "mode": self._mode.value,
                    "cloud_endpoint": self._cloud_endpoint,
                    "cloud_timeout": self._cloud_timeout,
                    "auto_detect_pages": self._auto_detect_pages
                }
                if self._anthropic_api_key:
                    data["anthropic_api_key"] = self._anthropic_api_key
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Ignore save errors

    def get_status_text(self) -> str:
        """Get human-readable status text for UI."""
        if self._mode == InferenceMode.CLOUD:
            return "Cloud GPU (Modal.com)"
        elif self._mode == InferenceMode.LOCAL_GPU:
            return "Local GPU (CUDA)"
        elif self._mode == InferenceMode.LOCAL_LIGHTWEIGHT:
            return "Local Lightweight (MolScribe)"
        elif self._mode == InferenceMode.MOLSIGHT:
            return "MolSight (separate venv)"
        else:
            return "Local CPU (DECIMER, slow)"
