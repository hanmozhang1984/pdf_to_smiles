"""Entry point for PDF Chemical Structure to SMILES Converter."""

import sys


def main():
    """Main entry point for the application."""
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

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
