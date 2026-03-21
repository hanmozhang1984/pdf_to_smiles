"""Dialog for configuring inference settings (local vs cloud)."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton,
    QLabel, QLineEdit, QPushButton, QDialogButtonBox, QMessageBox,
    QApplication, QComboBox
)
from PySide6.QtCore import Qt

from ..core.inference_settings import InferenceSettings, InferenceMode, ClassifierMode


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

        self._radio_molsight = QRadioButton("MolSight (best stereochemistry)")
        self._radio_molsight.setToolTip(
            "EfficientViT + transformer decoder.\n"
            "Runs in separate venv (~/Documents/Projects/MolSight/venv).\n"
            "Best for stereochemistry-heavy structures."
        )
        mode_layout.addWidget(self._radio_molsight)

        # MolSight checkpoint selector
        self._molsight_settings_widget = QGroupBox()
        self._molsight_settings_widget.setFlat(True)
        self._molsight_settings_widget.setStyleSheet(
            "QGroupBox { border: none; margin-left: 20px; }"
        )
        molsight_inner_layout = QHBoxLayout(self._molsight_settings_widget)
        molsight_inner_layout.addWidget(QLabel("Checkpoint:"))
        self._combo_molsight_checkpoint = QComboBox()
        self._combo_molsight_checkpoint.setMinimumWidth(250)
        self._populate_molsight_checkpoints()
        molsight_inner_layout.addWidget(self._combo_molsight_checkpoint)
        molsight_inner_layout.addStretch()
        mode_layout.addWidget(self._molsight_settings_widget)

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

        # Compound Classifier group
        classifier_group = QGroupBox("Compound Classifier")
        classifier_layout = QVBoxLayout(classifier_group)

        self._radio_classifier_claude = QRadioButton("Claude Haiku (API)")
        self._radio_classifier_claude.setToolTip(
            "Use Claude Haiku Vision for compound classification.\n"
            "Requires Anthropic API key. ~$0.01-0.05 per patent."
        )
        classifier_layout.addWidget(self._radio_classifier_claude)

        self._radio_classifier_ollama = QRadioButton("Ollama Local (free)")
        self._radio_classifier_ollama.setToolTip(
            "Use a local vision model (Qwen3.5) via Ollama.\n"
            "Free, runs locally. Requires Ollama installed."
        )
        classifier_layout.addWidget(self._radio_classifier_ollama)

        self._radio_classifier_mlx = QRadioButton("MLX-VLM Local (Apple Silicon)")
        self._radio_classifier_mlx.setToolTip(
            "Use MLX-VLM for compound classification.\n"
            "~2x faster than Ollama on Apple Silicon.\n"
            "Requires mlx_vlm.server running locally."
        )
        classifier_layout.addWidget(self._radio_classifier_mlx)

        self._radio_classifier_none = QRadioButton("None (skip classification)")
        self._radio_classifier_none.setToolTip(
            "Skip compound classification entirely.\n"
            "All detected structures will be treated as example compounds."
        )
        classifier_layout.addWidget(self._radio_classifier_none)

        # Ollama settings (model + prompt path + test button)
        self._ollama_settings_widget = QGroupBox()
        self._ollama_settings_widget.setFlat(True)
        self._ollama_settings_widget.setStyleSheet(
            "QGroupBox { border: none; margin-left: 20px; }"
        )
        ollama_inner_layout = QVBoxLayout(self._ollama_settings_widget)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self._txt_ollama_model = QLineEdit()
        self._txt_ollama_model.setPlaceholderText("qwen3.5:9b")
        model_layout.addWidget(self._txt_ollama_model)
        ollama_inner_layout.addLayout(model_layout)

        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel("Prompt file:"))
        self._txt_prompt_path = QLineEdit()
        self._txt_prompt_path.setPlaceholderText("(optional) path to optimized prompt")
        prompt_layout.addWidget(self._txt_prompt_path)
        ollama_inner_layout.addLayout(prompt_layout)

        test_ollama_layout = QHBoxLayout()
        self._btn_test_ollama = QPushButton("Test Connection")
        self._btn_test_ollama.clicked.connect(self._on_test_ollama)
        test_ollama_layout.addWidget(self._btn_test_ollama)
        self._lbl_ollama_status = QLabel("")
        test_ollama_layout.addWidget(self._lbl_ollama_status)
        test_ollama_layout.addStretch()
        ollama_inner_layout.addLayout(test_ollama_layout)

        classifier_layout.addWidget(self._ollama_settings_widget)

        # MLX-VLM settings (endpoint + model + test button)
        self._mlx_settings_widget = QGroupBox()
        self._mlx_settings_widget.setFlat(True)
        self._mlx_settings_widget.setStyleSheet(
            "QGroupBox { border: none; margin-left: 20px; }"
        )
        mlx_inner_layout = QVBoxLayout(self._mlx_settings_widget)

        mlx_endpoint_layout = QHBoxLayout()
        mlx_endpoint_layout.addWidget(QLabel("Endpoint:"))
        self._txt_mlx_endpoint = QLineEdit()
        self._txt_mlx_endpoint.setPlaceholderText("http://localhost:8000")
        mlx_endpoint_layout.addWidget(self._txt_mlx_endpoint)
        mlx_inner_layout.addLayout(mlx_endpoint_layout)

        mlx_model_layout = QHBoxLayout()
        mlx_model_layout.addWidget(QLabel("Model:"))
        self._txt_mlx_model = QLineEdit()
        self._txt_mlx_model.setPlaceholderText("mlx-community/Qwen3-VL-8B-Instruct-4bit")
        mlx_model_layout.addWidget(self._txt_mlx_model)
        mlx_inner_layout.addLayout(mlx_model_layout)

        test_mlx_layout = QHBoxLayout()
        self._btn_test_mlx = QPushButton("Test Connection")
        self._btn_test_mlx.clicked.connect(self._on_test_mlx)
        test_mlx_layout.addWidget(self._btn_test_mlx)
        self._lbl_mlx_status = QLabel("")
        test_mlx_layout.addWidget(self._lbl_mlx_status)
        test_mlx_layout.addStretch()
        mlx_inner_layout.addLayout(test_mlx_layout)

        classifier_layout.addWidget(self._mlx_settings_widget)

        layout.addWidget(classifier_group)

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
        self._radio_molsight.toggled.connect(self._on_mode_changed)
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
        elif mode == InferenceMode.MOLSIGHT:
            self._radio_molsight.setChecked(True)
        else:
            self._radio_local_cpu.setChecked(True)

        if self._settings.cloud_endpoint:
            self._txt_endpoint.setText(self._settings.cloud_endpoint)

        # MolSight checkpoint
        idx = self._combo_molsight_checkpoint.findText(self._settings.molsight_checkpoint)
        if idx >= 0:
            self._combo_molsight_checkpoint.setCurrentIndex(idx)

        if self._settings.anthropic_api_key:
            self._txt_api_key.setText(self._settings.anthropic_api_key)

        # Classifier mode
        cmode = self._settings.classifier_mode
        if cmode == ClassifierMode.OLLAMA:
            self._radio_classifier_ollama.setChecked(True)
        elif cmode == ClassifierMode.MLX:
            self._radio_classifier_mlx.setChecked(True)
        elif cmode == ClassifierMode.NONE:
            self._radio_classifier_none.setChecked(True)
        else:
            self._radio_classifier_claude.setChecked(True)

        self._txt_ollama_model.setText(self._settings.ollama_model or "qwen3.5:9b")
        if self._settings.classifier_prompt_path:
            self._txt_prompt_path.setText(self._settings.classifier_prompt_path)

        self._txt_mlx_endpoint.setText(self._settings.mlx_endpoint or "http://localhost:8000")
        self._txt_mlx_model.setText(
            self._settings.mlx_model or "mlx-community/Qwen3-VL-8B-Instruct-4bit"
        )

        self._radio_classifier_claude.toggled.connect(self._on_classifier_mode_changed)
        self._radio_classifier_ollama.toggled.connect(self._on_classifier_mode_changed)
        self._radio_classifier_mlx.toggled.connect(self._on_classifier_mode_changed)
        self._radio_classifier_none.toggled.connect(self._on_classifier_mode_changed)

        self._txt_api_key.textChanged.connect(lambda: self._update_api_key_status())
        self._update_api_key_status()
        self._on_mode_changed()
        self._on_classifier_mode_changed()

    def _populate_molsight_checkpoints(self) -> None:
        """Scan MolSight directory for .pth checkpoint files."""
        import os
        molsight_dir = os.path.join(
            os.path.expanduser("~"), "Documents", "Projects", "MolSight"
        )
        self._combo_molsight_checkpoint.clear()
        if os.path.isdir(molsight_dir):
            pth_files = sorted(
                f for f in os.listdir(molsight_dir) if f.endswith(".pth")
            )
            for f in pth_files:
                self._combo_molsight_checkpoint.addItem(f)
        # Ensure default is present even if directory doesn't exist
        if self._combo_molsight_checkpoint.count() == 0:
            self._combo_molsight_checkpoint.addItem("pubchem_uspto_smiles_edges_30.pth")

    def _on_mode_changed(self) -> None:
        """Handle inference mode change."""
        is_cloud = self._radio_cloud.isChecked()
        is_molsight = self._radio_molsight.isChecked()
        self._txt_endpoint.setEnabled(is_cloud)
        self._btn_test.setEnabled(is_cloud)
        self._molsight_settings_widget.setVisible(is_molsight)

    def _on_classifier_mode_changed(self) -> None:
        """Show/hide classifier-specific settings based on classifier mode."""
        is_ollama = self._radio_classifier_ollama.isChecked()
        is_mlx = self._radio_classifier_mlx.isChecked()
        self._ollama_settings_widget.setVisible(is_ollama)
        self._mlx_settings_widget.setVisible(is_mlx)

    def _on_test_ollama(self) -> None:
        """Test the Ollama connection and model availability."""
        model = self._txt_ollama_model.text().strip() or "qwen3.5:9b"
        self._lbl_ollama_status.setText("Testing...")
        self._btn_test_ollama.setEnabled(False)
        QApplication.processEvents()

        try:
            from ..core.ollama_compound_classifier import check_ollama_status
            status = check_ollama_status(model)
            if status == "ready":
                self._lbl_ollama_status.setText("Ready!")
                self._lbl_ollama_status.setStyleSheet("color: green;")
            elif status == "model_not_found":
                self._lbl_ollama_status.setText(f"Model '{model}' not found")
                self._lbl_ollama_status.setStyleSheet("color: orange;")
                QMessageBox.information(
                    self, "Model Not Found",
                    f"Ollama is running but model '{model}' is not downloaded.\n\n"
                    f"Run: ollama pull {model}"
                )
            elif status == "not_running":
                self._lbl_ollama_status.setText("Ollama not running")
                self._lbl_ollama_status.setStyleSheet("color: red;")
                QMessageBox.warning(
                    self, "Ollama Not Running",
                    "Ollama server is not running.\n\n"
                    "Start it with: ollama serve"
                )
            else:
                self._lbl_ollama_status.setText("Not installed")
                self._lbl_ollama_status.setStyleSheet("color: red;")
                QMessageBox.warning(
                    self, "Ollama Not Found",
                    "Ollama does not appear to be installed.\n\n"
                    "Install with: brew install ollama"
                )
        except Exception as e:
            self._lbl_ollama_status.setText("Error")
            self._lbl_ollama_status.setStyleSheet("color: red;")
            QMessageBox.warning(self, "Error", f"Failed to check Ollama: {e}")
        finally:
            self._btn_test_ollama.setEnabled(True)

    def _on_test_mlx(self) -> None:
        """Test the MLX-VLM server connection."""
        endpoint = self._txt_mlx_endpoint.text().strip() or "http://localhost:8000"
        self._lbl_mlx_status.setText("Testing...")
        self._btn_test_mlx.setEnabled(False)
        QApplication.processEvents()

        try:
            from ..core.mlx_compound_classifier import check_mlx_status
            status = check_mlx_status(endpoint)
            if status == "ready":
                self._lbl_mlx_status.setText("Ready!")
                self._lbl_mlx_status.setStyleSheet("color: green;")
            else:
                self._lbl_mlx_status.setText("Server not running")
                self._lbl_mlx_status.setStyleSheet("color: red;")
                QMessageBox.warning(
                    self, "MLX-VLM Not Running",
                    f"Could not connect to MLX-VLM server at {endpoint}.\n\n"
                    "Start with:\n"
                    "  python -m mlx_vlm.server --model mlx-community/Qwen3-VL-8B-Instruct-4bit"
                )
        except Exception as e:
            self._lbl_mlx_status.setText("Error")
            self._lbl_mlx_status.setStyleSheet("color: red;")
            QMessageBox.warning(self, "Error", f"Failed to check MLX-VLM: {e}")
        finally:
            self._btn_test_mlx.setEnabled(True)

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
        elif self._radio_molsight.isChecked():
            mode = InferenceMode.MOLSIGHT
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

        # Save classifier mode
        if self._radio_classifier_ollama.isChecked():
            self._settings.classifier_mode = ClassifierMode.OLLAMA
        elif self._radio_classifier_mlx.isChecked():
            self._settings.classifier_mode = ClassifierMode.MLX
        elif self._radio_classifier_none.isChecked():
            self._settings.classifier_mode = ClassifierMode.NONE
        else:
            self._settings.classifier_mode = ClassifierMode.CLAUDE

        ollama_model = self._txt_ollama_model.text().strip()
        if ollama_model:
            self._settings.ollama_model = ollama_model

        prompt_path = self._txt_prompt_path.text().strip()
        self._settings.classifier_prompt_path = prompt_path or None

        # Save MLX settings
        mlx_endpoint = self._txt_mlx_endpoint.text().strip()
        if mlx_endpoint:
            self._settings.mlx_endpoint = mlx_endpoint
        mlx_model = self._txt_mlx_model.text().strip()
        if mlx_model:
            self._settings.mlx_model = mlx_model

        # Save MolSight checkpoint
        checkpoint = self._combo_molsight_checkpoint.currentText()
        if checkpoint:
            self._settings.molsight_checkpoint = checkpoint

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
        elif self._radio_molsight.isChecked():
            return InferenceMode.MOLSIGHT
        else:
            return InferenceMode.LOCAL_CPU
