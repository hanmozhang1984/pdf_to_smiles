"""Entry point for PDF Chemical Structure to SMILES Converter."""

import logging
import sys


def main():
    """Main entry point for the application."""
    # Configure logging — DEBUG level for our modules, WARNING for third-party
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("pdf_to_smiles").setLevel(logging.DEBUG)

    # Import here to avoid slow startup from TensorFlow imports
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    from .gui.main_window import MainWindow

    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("PDF to SMILES Converter")
    app.setOrganizationName("ChemCipher")

    # Apply stored API keys to environment before any processing
    from .core.inference_settings import InferenceSettings
    InferenceSettings.get_instance().apply_api_keys()

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
