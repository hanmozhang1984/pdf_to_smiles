"""Dialog for configuring inference settings (local vs cloud)."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton,
    QLabel, QLineEdit, QPushButton, QDialogButtonBox, QMessageBox,
    QApplication
)
from PySide6.QtCore import Qt

from ..core.inference_settings import InferenceSettings, InferenceMode


class InferenceSettingsDialog(QDialog):
    """Dialog for configuring inference mode and cloud settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inference Settings")
        self.setMinimumWidth(450)

        self._settings = InferenceSettings.get_instance()

        layout = QVBoxLayout(self)

        # Inference Mode group
        mode_group = QGroupBox("Inference Mode")
        mode_layout = QVBoxLayout(mode_group)

        self._radio_lightweight = QRadioButton("Local Lightweight (recommended)")
        self._radio_lightweight.setToolTip(
            "Fast CPU inference using MolScribe.\n"
            "Works on any machine. No GPU or cloud required.\n"
            "~1-2 sec per structure."
        )
        mode_layout.addWidget(self._radio_lightweight)

        self._radio_cloud = QRadioButton("Cloud GPU (Modal.com)")
        self._radio_cloud.setToolTip(
            "Use cloud GPU for fast inference.\n"
            "Requires internet connection.\n"
            "Cost: ~$0.0001 per structure."
        )
        mode_layout.addWidget(self._radio_cloud)

        self._radio_local_gpu = QRadioButton("Local GPU (CUDA)")
        self._radio_local_gpu.setToolTip(
            "Use your local NVIDIA GPU.\n"
            "Requires CUDA drivers installed.\n"
            "First run downloads ~2GB of models."
        )
        mode_layout.addWidget(self._radio_local_gpu)

        self._radio_local_cpu = QRadioButton("Local CPU (DECIMER, slow)")
        self._radio_local_cpu.setToolTip(
            "Use CPU for inference via DECIMER/TensorFlow.\n"
            "Works everywhere but is slow.\n"
            "~5-10 seconds per structure."
        )
        mode_layout.addWidget(self._radio_local_cpu)

        layout.addWidget(mode_group)

        # Cloud Settings group
        cloud_group = QGroupBox("Cloud Settings")
        cloud_layout = QVBoxLayout(cloud_group)

        endpoint_layout = QHBoxLayout()
        endpoint_layout.addWidget(QLabel("Endpoint URL:"))
        self._txt_endpoint = QLineEdit()
        self._txt_endpoint.setPlaceholderText("https://your-username--pdf-to-smiles-process-image.modal.run")
        endpoint_layout.addWidget(self._txt_endpoint)
        cloud_layout.addLayout(endpoint_layout)

        # Test connection button
        test_layout = QHBoxLayout()
        self._btn_test = QPushButton("Test Connection")
        self._btn_test.clicked.connect(self._on_test_connection)
        test_layout.addWidget(self._btn_test)

        self._lbl_test_status = QLabel("")
        test_layout.addWidget(self._lbl_test_status)
        test_layout.addStretch()

        cloud_layout.addLayout(test_layout)

        layout.addWidget(cloud_group)

        # Anthropic API Key group
        api_key_group = QGroupBox("Anthropic API Key (Structure Classification)")
        api_key_layout = QVBoxLayout(api_key_group)

        key_input_layout = QHBoxLayout()
        self._txt_api_key = QLineEdit()
        self._txt_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._txt_api_key.setPlaceholderText("sk-ant-...")
        key_input_layout.addWidget(self._txt_api_key)

        self._btn_toggle_key = QPushButton("Show")
        self._btn_toggle_key.setFixedWidth(50)
        self._btn_toggle_key.clicked.connect(self._on_toggle_key_visibility)
        key_input_layout.addWidget(self._btn_toggle_key)

        api_key_layout.addLayout(key_input_layout)

        self._lbl_api_key_status = QLabel()
        api_key_layout.addWidget(self._lbl_api_key_status)

        helper_label = QLabel("Required for automatic structure classification (example vs. other)")
        helper_label.setStyleSheet("color: gray; font-size: 11px;")
        helper_label.setWordWrap(True)
        api_key_layout.addWidget(helper_label)

        layout.addWidget(api_key_group)

        # Status info
        status_group = QGroupBox("Current Status")
        status_layout = QVBoxLayout(status_group)

        self._lbl_status = QLabel("Checking...")
        self._lbl_status.setWordWrap(True)
        status_layout.addWidget(self._lbl_status)

        self._btn_check_status = QPushButton("Refresh Status")
        self._btn_check_status.clicked.connect(self._check_current_status)
        status_layout.addWidget(self._btn_check_status)

        layout.addWidget(status_group)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Load current settings
        self._load_settings()

        # Connect radio buttons to update cloud settings visibility
        self._radio_lightweight.toggled.connect(self._on_mode_changed)
        self._radio_cloud.toggled.connect(self._on_mode_changed)
        self._radio_local_gpu.toggled.connect(self._on_mode_changed)
        self._radio_local_cpu.toggled.connect(self._on_mode_changed)

        # Don't auto-check status on open (can be slow due to TensorFlow import)
        self._lbl_status.setText("Click 'Refresh Status' to check backend availability")

    def _load_settings(self) -> None:
        """Load current settings into the dialog."""
        mode = self._settings.mode

        if mode == InferenceMode.CLOUD:
            self._radio_cloud.setChecked(True)
        elif mode == InferenceMode.LOCAL_GPU:
            self._radio_local_gpu.setChecked(True)
        elif mode == InferenceMode.LOCAL_LIGHTWEIGHT:
            self._radio_lightweight.setChecked(True)
        else:
            self._radio_local_cpu.setChecked(True)

        if self._settings.cloud_endpoint:
            self._txt_endpoint.setText(self._settings.cloud_endpoint)

        if self._settings.anthropic_api_key:
            self._txt_api_key.setText(self._settings.anthropic_api_key)

        self._txt_api_key.textChanged.connect(lambda: self._update_api_key_status())
        self._update_api_key_status()
        self._on_mode_changed()

    def _on_mode_changed(self) -> None:
        """Handle inference mode change."""
        is_cloud = self._radio_cloud.isChecked()
        self._txt_endpoint.setEnabled(is_cloud)
        self._btn_test.setEnabled(is_cloud)

    def _check_current_status(self) -> None:
        """Check and display current inference backend status."""
        self._lbl_status.setText("Checking...")
        QApplication.processEvents()

        try:
            from ..core.inference_provider import InferenceProvider
            provider = InferenceProvider()
            is_available, message = provider.check_availability()

            if is_available:
                self._lbl_status.setText(f"Ready: {message}")
                self._lbl_status.setStyleSheet("color: green;")
            else:
                self._lbl_status.setText(f"Not ready: {message}")
                self._lbl_status.setStyleSheet("color: orange;")

            provider.close()

        except Exception as e:
            self._lbl_status.setText(f"Error: {e}")
            self._lbl_status.setStyleSheet("color: red;")

    def _on_test_connection(self) -> None:
        """Test the cloud connection."""
        endpoint = self._txt_endpoint.text().strip()
        if not endpoint:
            QMessageBox.warning(
                self,
                "No Endpoint",
                "Please enter the cloud endpoint URL."
            )
            return

        self._lbl_test_status.setText("Testing...")
        self._btn_test.setEnabled(False)
        QApplication.processEvents()

        try:
            from ..cloud import CloudInferenceClient

            client = CloudInferenceClient(endpoint=endpoint, timeout=30)
            if client.health_check():
                self._lbl_test_status.setText("Connected!")
                self._lbl_test_status.setStyleSheet("color: green;")
            else:
                self._lbl_test_status.setText("Failed")
                self._lbl_test_status.setStyleSheet("color: red;")
            client.close()

        except Exception as e:
            self._lbl_test_status.setText("Error")
            self._lbl_test_status.setStyleSheet("color: red;")
            QMessageBox.warning(
                self,
                "Connection Failed",
                f"Could not connect to cloud API:\n{e}"
            )

        finally:
            self._btn_test.setEnabled(True)

    def _on_toggle_key_visibility(self) -> None:
        """Toggle API key visibility."""
        if self._txt_api_key.echoMode() == QLineEdit.EchoMode.Password:
            self._txt_api_key.setEchoMode(QLineEdit.EchoMode.Normal)
            self._btn_toggle_key.setText("Hide")
        else:
            self._txt_api_key.setEchoMode(QLineEdit.EchoMode.Password)
            self._btn_toggle_key.setText("Show")

    def _update_api_key_status(self) -> None:
        """Update the API key status label."""
        if self._txt_api_key.text().strip():
            self._lbl_api_key_status.setText("Status: Active")
            self._lbl_api_key_status.setStyleSheet("color: green;")
        else:
            self._lbl_api_key_status.setText("Status: Not set")
            self._lbl_api_key_status.setStyleSheet("color: gray;")

    def _on_accept(self) -> None:
        """Save settings and close dialog."""
        # Determine selected mode
        if self._radio_lightweight.isChecked():
            mode = InferenceMode.LOCAL_LIGHTWEIGHT
        elif self._radio_cloud.isChecked():
            mode = InferenceMode.CLOUD
        elif self._radio_local_gpu.isChecked():
            mode = InferenceMode.LOCAL_GPU
        else:
            mode = InferenceMode.LOCAL_CPU

        # Validate cloud settings if cloud mode selected
        if mode == InferenceMode.CLOUD:
            endpoint = self._txt_endpoint.text().strip()
            if not endpoint:
                QMessageBox.warning(
                    self,
                    "Missing Endpoint",
                    "Please enter the cloud endpoint URL for cloud mode."
                )
                return
            self._settings.cloud_endpoint = endpoint

        # Save API key
        api_key = self._txt_api_key.text().strip()
        self._settings.anthropic_api_key = api_key
        self._settings.apply_api_keys()

        # Save mode
        self._settings.mode = mode

        self.accept()

    def get_selected_mode(self) -> InferenceMode:
        """Get the selected inference mode."""
        if self._radio_lightweight.isChecked():
            return InferenceMode.LOCAL_LIGHTWEIGHT
        elif self._radio_cloud.isChecked():
            return InferenceMode.CLOUD
        elif self._radio_local_gpu.isChecked():
            return InferenceMode.LOCAL_GPU
        else:
            return InferenceMode.LOCAL_CPU
