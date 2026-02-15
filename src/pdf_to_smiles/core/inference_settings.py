"""Inference settings for switching between local and cloud processing."""

from __future__ import annotations

from enum import Enum
from typing import Optional
import json
import os


class InferenceMode(Enum):
    """Available inference modes."""
    LOCAL_CPU = "cpu"        # Local CPU (slow but works everywhere)
    LOCAL_GPU = "gpu"        # Local GPU (requires NVIDIA + CUDA)
    CLOUD = "cloud"          # Cloud GPU via Modal.com


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
        self._mode: InferenceMode = InferenceMode.LOCAL_CPU
        self._cloud_endpoint: Optional[str] = None
        self._cloud_timeout: int = 300  # 5 minutes for cold starts
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
    def is_cloud(self) -> bool:
        """Check if using cloud inference."""
        return self._mode == InferenceMode.CLOUD

    @property
    def is_local_gpu(self) -> bool:
        """Check if using local GPU inference."""
        return self._mode == InferenceMode.LOCAL_GPU

    def _load_settings(self) -> None:
        """Load settings from config file."""
        try:
            if os.path.exists(self._config_file):
                with open(self._config_file, 'r') as f:
                    data = json.load(f)
                    mode_str = data.get("mode", "cpu")
                    self._mode = InferenceMode(mode_str)
                    self._cloud_endpoint = data.get("cloud_endpoint")
                    self._cloud_timeout = data.get("cloud_timeout", 300)
        except Exception:
            pass  # Use defaults if loading fails

    def _save_settings(self) -> None:
        """Save settings to config file."""
        try:
            os.makedirs(os.path.dirname(self._config_file), exist_ok=True)
            with open(self._config_file, 'w') as f:
                json.dump({
                    "mode": self._mode.value,
                    "cloud_endpoint": self._cloud_endpoint,
                    "cloud_timeout": self._cloud_timeout
                }, f, indent=2)
        except Exception:
            pass  # Ignore save errors

    def get_status_text(self) -> str:
        """Get human-readable status text for UI."""
        if self._mode == InferenceMode.CLOUD:
            return "Cloud GPU (Modal.com)"
        elif self._mode == InferenceMode.LOCAL_GPU:
            return "Local GPU (CUDA)"
        else:
            return "Local CPU (slow)"
