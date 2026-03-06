"""Main application window for PDF to SMILES converter."""

from typing import List, Optional
from pathlib import Path
import io

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QProgressBar,
    QFileDialog, QMessageBox, QHeaderView, QApplication, QGroupBox,
    QToolBar, QStatusBar, QInputDialog, QMenu, QLineEdit, QDialog,
    QDialogButtonBox, QCheckBox
)
from PySide6.QtCore import Qt, Slot, QUrl
from PySide6.QtGui import QPixmap, QImage, QAction, QIcon, QColor

# Try to import WebEngine for chemical drawing
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebChannel import QWebChannel
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False


class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically instead of alphabetically."""

    def __init__(self, display_text: str, sort_value: float = None):
        super().__init__(display_text)
        self._sort_value = sort_value if sort_value is not None else float('-inf')

    def __lt__(self, other):
        if isinstance(other, NumericTableWidgetItem):
            return self._sort_value < other._sort_value
        return super().__lt__(other)


class AutoTypeTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that auto-detects numeric values for sorting."""

    def __init__(self, text: str = ""):
        super().__init__(text)
        self._sort_value = None
        self._is_numeric = False
        self._update_sort_value(text)

    def _update_sort_value(self, text: str) -> None:
        """Update sort value based on text content."""
        try:
            self._sort_value = float(text)
            self._is_numeric = True
        except (ValueError, TypeError):
            self._sort_value = None
            self._is_numeric = False

    def setData(self, role, value):
        """Override to update sort value when data changes."""
        super().setData(role, value)
        if role == Qt.EditRole or role == Qt.DisplayRole:
            self._update_sort_value(str(value) if value else "")

    def __lt__(self, other):
        if isinstance(other, AutoTypeTableWidgetItem):
            # Both numeric: compare numerically
            if self._is_numeric and other._is_numeric:
                return self._sort_value < other._sort_value
            # Both non-numeric: compare as strings
            if not self._is_numeric and not other._is_numeric:
                return self.text().lower() < other.text().lower()
            # Mixed: numbers come before strings
            return self._is_numeric
        if isinstance(other, NumericTableWidgetItem):
            if self._is_numeric:
                return self._sort_value < other._sort_value
            return False  # Strings after numbers
        return super().__lt__(other)
from PIL import Image

from ..models.extraction_result import ExtractionResult, ProcessingProgress, PDFInfo
from ..workers.processing_worker import ProcessingWorker, parse_page_range
from ..core.export_handler import ExportHandler
from ..core.pdf_processor import PDFProcessor
from ..core.inference_settings import InferenceSettings, InferenceMode


# HTML template for JSME molecule editor
JSME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://jsme-editor.github.io/dist/jsme/jsme.nocache.js"></script>
    <style>
        body { margin: 0; padding: 10px; font-family: Arial, sans-serif; }
        #jsme_container { width: 100%; height: 400px; }
        .button-row { margin-top: 10px; text-align: center; }
        button { padding: 8px 20px; margin: 0 5px; font-size: 14px; cursor: pointer; }
        #smiles_output { margin-top: 10px; padding: 5px; background: #f0f0f0;
                         font-family: monospace; word-break: break-all; }
    </style>
</head>
<body>
    <div id="jsme_container"></div>
    <div class="button-row">
        <button onclick="getSmiles()">Get SMILES</button>
        <button onclick="clearEditor()">Clear</button>
    </div>
    <div id="smiles_output">SMILES: (draw a structure)</div>

    <script>
        var jsmeApp;

        function jsmeOnLoad() {
            jsmeApp = new JSApplet.JSME("jsme_container", "100%", "400px", {
                "options": "query,hydrogens"
            });
        }

        function getSmiles() {
            if (jsmeApp) {
                var smiles = jsmeApp.smiles();
                document.getElementById("smiles_output").innerText = "SMILES: " + (smiles || "(empty)");
                // Store in a hidden element for retrieval
                document.title = smiles || "";
            }
        }

        function clearEditor() {
            if (jsmeApp) {
                jsmeApp.reset();
                document.getElementById("smiles_output").innerText = "SMILES: (draw a structure)";
                document.title = "";
            }
        }

        function setSmiles(smiles) {
            if (jsmeApp && smiles) {
                jsmeApp.readGenericMolecularInput(smiles);
            }
        }
    </script>
</body>
</html>
"""


class ChemicalDrawingDialog(QDialog):
    """Dialog with embedded JSME molecule editor for drawing chemical structures."""

    def __init__(self, parent=None, initial_smiles: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Draw Chemical Structure")
        self.setMinimumSize(600, 550)
        self._smiles = ""
        self._initial_smiles = initial_smiles

        layout = QVBoxLayout(self)

        if HAS_WEBENGINE:
            # Create web view with JSME
            self._web_view = QWebEngineView()
            self._web_view.setHtml(JSME_HTML)
            layout.addWidget(self._web_view)

            # Instructions
            instructions = QLabel("Draw a structure and click 'Get SMILES', then click OK to search.")
            instructions.setWordWrap(True)
            layout.addWidget(instructions)
        else:
            # Fallback if WebEngine not available
            fallback_label = QLabel(
                "Chemical drawing requires PyQtWebEngine.\n\n"
                "Install it with: pip install PyQt6-WebEngine\n\n"
                "For now, enter SMILES/SMARTS directly in the text field."
            )
            fallback_label.setWordWrap(True)
            layout.addWidget(fallback_label)

        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_accept(self):
        """Get SMILES from the editor before accepting."""
        if HAS_WEBENGINE:
            # Get SMILES from page title (set by JavaScript)
            self._smiles = self._web_view.page().title()
        self.accept()

    def get_smiles(self) -> str:
        """Return the drawn SMILES string."""
        return self._smiles


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("PDF Chemical Structure to SMILES Converter")
        self.setMinimumSize(1000, 700)

        # State
        self._pdf_infos: List[PDFInfo] = []  # Multiple PDF support
        self._results: List[ExtractionResult] = []
        self._worker: Optional[ProcessingWorker] = None
        self._custom_columns: List[str] = []  # Track custom column names
        self._base_column_count = 15  # Number of built-in columns (without bio data columns)
        self._bio_columns: List[str] = []  # Track dynamic bio data column names
        self._custom_column_data: dict = {}  # Store custom column values by (row, col_name)
        self._last_bio_data: dict = {}  # Store last extracted bio data for unmatched display

        # Setup UI
        self._setup_toolbar()
        self._setup_central_widget()
        self._setup_statusbar()

        # Update UI state
        self._update_button_states()

    def _setup_toolbar(self) -> None:
        """Setup the toolbar with actions."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Open action (multi-select)
        self._action_open = QAction("Open PDFs", self)
        self._action_open.setShortcut("Ctrl+O")
        self._action_open.triggered.connect(self._on_open_pdf)
        toolbar.addAction(self._action_open)

        # Add PDF action
        self._action_add_pdf = QAction("Add PDF...", self)
        self._action_add_pdf.setShortcut("Ctrl+Shift+O")
        self._action_add_pdf.triggered.connect(self._on_add_pdf)
        toolbar.addAction(self._action_add_pdf)

        # Clear All action
        self._action_clear_all = QAction("Clear All", self)
        self._action_clear_all.triggered.connect(self._on_clear_all)
        self._action_clear_all.setEnabled(False)
        toolbar.addAction(self._action_clear_all)

        toolbar.addSeparator()

        # Process action
        self._action_process = QAction("Extract Structures", self)
        self._action_process.setShortcut("Ctrl+E")
        self._action_process.triggered.connect(self._on_extract)
        self._action_process.setEnabled(False)
        toolbar.addAction(self._action_process)

        # Cancel action
        self._action_cancel = QAction("Cancel", self)
        self._action_cancel.setShortcut("Escape")
        self._action_cancel.triggered.connect(self._on_cancel)
        self._action_cancel.setEnabled(False)
        toolbar.addAction(self._action_cancel)

        toolbar.addSeparator()

        # Export CSV action
        self._action_export_csv = QAction("Export CSV", self)
        self._action_export_csv.triggered.connect(self._on_export_csv)
        self._action_export_csv.setEnabled(False)
        toolbar.addAction(self._action_export_csv)

        # Export TXT action
        self._action_export_txt = QAction("Export TXT", self)
        self._action_export_txt.triggered.connect(self._on_export_txt)
        self._action_export_txt.setEnabled(False)
        toolbar.addAction(self._action_export_txt)

        # Export SDF action
        self._action_export_sdf = QAction("Export SDF", self)
        self._action_export_sdf.triggered.connect(self._on_export_sdf)
        self._action_export_sdf.setEnabled(False)
        toolbar.addAction(self._action_export_sdf)

    def _setup_central_widget(self) -> None:
        """Setup the main content area."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - file info and controls
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - results table
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)

        # Set initial sizes (1:3 ratio)
        splitter.setSizes([250, 750])

    def _create_left_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(4)  # Reduce spacing between widgets
        layout.setContentsMargins(4, 4, 4, 4)  # Reduce margins

        # Set smaller font for the entire panel
        from PySide6.QtGui import QFont
        small_font = QFont()
        small_font.setPointSize(8)
        panel.setFont(small_font)

        # Export Checked group
        checked_group = QGroupBox("Export Checked")
        checked_layout = QVBoxLayout(checked_group)
        checked_layout.setSpacing(2)
        checked_layout.setContentsMargins(4, 8, 4, 4)

        self._btn_check_all = QPushButton("Check All")
        self._btn_check_all.clicked.connect(self._on_check_all)
        self._btn_check_all.setEnabled(False)
        self._btn_check_all.setFixedHeight(22)
        checked_layout.addWidget(self._btn_check_all)

        self._btn_uncheck_all = QPushButton("Uncheck All")
        self._btn_uncheck_all.clicked.connect(self._on_uncheck_all)
        self._btn_uncheck_all.setEnabled(False)
        self._btn_uncheck_all.setFixedHeight(22)
        checked_layout.addWidget(self._btn_uncheck_all)

        self._btn_export_checked_csv = QPushButton("CSV...")
        self._btn_export_checked_csv.clicked.connect(self._on_export_checked_csv)
        self._btn_export_checked_csv.setEnabled(False)
        self._btn_export_checked_csv.setFixedHeight(22)
        checked_layout.addWidget(self._btn_export_checked_csv)

        self._btn_export_checked_sdf = QPushButton("SDF...")
        self._btn_export_checked_sdf.clicked.connect(self._on_export_checked_sdf)
        self._btn_export_checked_sdf.setEnabled(False)
        self._btn_export_checked_sdf.setFixedHeight(22)
        checked_layout.addWidget(self._btn_export_checked_sdf)

        self._btn_move_checked_top = QPushButton("Move to Top")
        self._btn_move_checked_top.clicked.connect(self._on_move_checked_to_top)
        self._btn_move_checked_top.setEnabled(False)
        self._btn_move_checked_top.setFixedHeight(22)
        checked_layout.addWidget(self._btn_move_checked_top)

        layout.addWidget(checked_group)

        # Substructure Search group
        search_group = QGroupBox("Substructure Search")
        search_layout = QVBoxLayout(search_group)
        search_layout.setSpacing(2)
        search_layout.setContentsMargins(4, 8, 4, 4)

        self._txt_search_smarts = QLineEdit()
        self._txt_search_smarts.setPlaceholderText("SMILES/SMARTS...")
        self._txt_search_smarts.returnPressed.connect(self._on_substructure_search)
        self._txt_search_smarts.setFixedHeight(22)
        search_layout.addWidget(self._txt_search_smarts)

        self._btn_draw_structure = QPushButton("Draw...")
        self._btn_draw_structure.clicked.connect(self._on_draw_structure)
        self._btn_draw_structure.setFixedHeight(22)
        search_layout.addWidget(self._btn_draw_structure)

        self._btn_search = QPushButton("Search")
        self._btn_search.clicked.connect(self._on_substructure_search)
        self._btn_search.setEnabled(False)
        self._btn_search.setFixedHeight(22)
        search_layout.addWidget(self._btn_search)

        self._btn_clear_search = QPushButton("Clear")
        self._btn_clear_search.clicked.connect(self._on_clear_search)
        self._btn_clear_search.setEnabled(False)
        self._btn_clear_search.setFixedHeight(22)
        search_layout.addWidget(self._btn_clear_search)

        self._lbl_search_results = QLabel("Matches: -")
        search_layout.addWidget(self._lbl_search_results)

        layout.addWidget(search_group)

        # Table customization group
        custom_group = QGroupBox("Columns")
        custom_layout = QVBoxLayout(custom_group)
        custom_layout.setSpacing(2)
        custom_layout.setContentsMargins(4, 8, 4, 4)

        self._btn_add_column = QPushButton("Add...")
        self._btn_add_column.clicked.connect(self._on_add_column)
        self._btn_add_column.setEnabled(False)
        self._btn_add_column.setFixedHeight(22)
        custom_layout.addWidget(self._btn_add_column)

        self._btn_remove_column = QPushButton("Remove...")
        self._btn_remove_column.clicked.connect(self._on_remove_column)
        self._btn_remove_column.setEnabled(False)
        self._btn_remove_column.setFixedHeight(22)
        custom_layout.addWidget(self._btn_remove_column)

        layout.addWidget(custom_group)

        # Processing Settings group
        settings_group = QGroupBox("Processing")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(2)
        settings_layout.setContentsMargins(4, 8, 4, 4)

        self._chk_high_accuracy = QCheckBox("High Accuracy Mode")
        self._chk_high_accuracy.setToolTip(
            "Use higher resolution and image preprocessing for complex structures.\n"
            "Recommended for macrocycles and detailed structures.\n"
            "Processing will be slower."
        )
        settings_layout.addWidget(self._chk_high_accuracy)

        # Page range selection
        page_range_label = QLabel("Pages to process:")
        settings_layout.addWidget(page_range_label)

        self._txt_page_range = QLineEdit()
        self._txt_page_range.setPlaceholderText("All pages (e.g., 45-52, 120-125)")
        self._txt_page_range.setFixedHeight(22)
        self._txt_page_range.setToolTip(
            "Specify which pages to process.\n"
            "Formats: 1-10, 5,8,12, or 1-5, 20-25\n"
            "Leave blank to process all pages."
        )
        settings_layout.addWidget(self._txt_page_range)

        detect_row = QHBoxLayout()

        self._btn_detect_pages = QPushButton("Detect Pages")
        self._btn_detect_pages.setFixedHeight(22)
        self._btn_detect_pages.setToolTip(
            "Scan the PDF to auto-detect pages containing\n"
            "chemical structures or biological data tables."
        )
        self._btn_detect_pages.clicked.connect(self._on_detect_pages)
        self._btn_detect_pages.setEnabled(False)
        detect_row.addWidget(self._btn_detect_pages)

        self._chk_auto_detect = QCheckBox("Auto-detect")
        self._chk_auto_detect.setChecked(self._get_auto_detect_setting())
        self._chk_auto_detect.setToolTip(
            "Automatically skip text-only pages before processing.\n"
            "Fast: ~0.1 sec/page."
        )
        detect_row.addWidget(self._chk_auto_detect)

        settings_layout.addLayout(detect_row)

        layout.addWidget(settings_group)

        # Inference Settings group
        inference_group = QGroupBox("Inference")
        inference_layout = QVBoxLayout(inference_group)
        inference_layout.setSpacing(2)
        inference_layout.setContentsMargins(4, 8, 4, 4)

        # Current mode label
        self._inference_settings = InferenceSettings.get_instance()
        self._lbl_inference_mode = QLabel(self._get_inference_mode_text())
        self._lbl_inference_mode.setWordWrap(True)
        inference_layout.addWidget(self._lbl_inference_mode)

        # Classifier status label
        self._lbl_classifier_status = QLabel(self._get_classifier_status_text())
        inference_layout.addWidget(self._lbl_classifier_status)

        # Settings button
        self._btn_inference_settings = QPushButton("Settings...")
        self._btn_inference_settings.clicked.connect(self._on_inference_settings)
        self._btn_inference_settings.setFixedHeight(22)
        self._btn_inference_settings.setToolTip("Configure local or cloud inference")
        inference_layout.addWidget(self._btn_inference_settings)

        layout.addWidget(inference_group)

        # Results summary
        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.setSpacing(1)
        summary_layout.setContentsMargins(4, 8, 4, 4)

        self._lbl_file_count = QLabel("Files: 0")
        self._lbl_file_count.setToolTip("No files loaded")
        summary_layout.addWidget(self._lbl_file_count)

        self._lbl_page_count = QLabel("Pages: -")
        summary_layout.addWidget(self._lbl_page_count)

        self._lbl_total = QLabel("Structures: -")
        summary_layout.addWidget(self._lbl_total)

        self._lbl_valid = QLabel("Valid: -")
        summary_layout.addWidget(self._lbl_valid)

        self._lbl_invalid = QLabel("Invalid: -")
        summary_layout.addWidget(self._lbl_invalid)

        layout.addWidget(summary_group)

        # Stretch to push everything to top
        layout.addStretch()

        return panel

    def _create_right_panel(self) -> QWidget:
        """Create the right results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Results table - base columns only (bio data columns added dynamically)
        self._table = QTableWidget()
        self._table.setColumnCount(15)
        self._table.setHorizontalHeaderLabels([
            "", "Source", "Compound", "Type", "Page", "Original", "SMILES", "Valid", "RDKit Rendering",
            "MW (Da)", "cLogP", "TPSA", "Rot. Bonds", "Stereo", "Formula"
        ])

        # Configure table - allow interactive column resizing by mouse drag
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)  # Last column stretches to fill

        # Make column headers bold
        header_font = header.font()
        header_font.setBold(True)
        header.setFont(header_font)

        # Set initial column widths for base columns
        self._table.setColumnWidth(0, 30)   # Checkbox
        self._table.setColumnWidth(1, 120)  # Source
        self._table.setColumnWidth(2, 100)  # Compound
        self._table.setColumnWidth(3, 70)   # Type
        self._table.setColumnWidth(4, 50)   # Page
        self._table.setColumnWidth(5, 150)  # Original image
        self._table.setColumnWidth(6, 200)  # SMILES
        self._table.setColumnWidth(7, 50)   # Valid
        self._table.setColumnWidth(8, 150)  # RDKit image
        self._table.setColumnWidth(9, 70)   # MW
        self._table.setColumnWidth(10, 60)  # cLogP
        self._table.setColumnWidth(11, 60)  # TPSA
        self._table.setColumnWidth(12, 80)  # Rot. Bonds
        self._table.setColumnWidth(13, 55)  # Stereo
        # Column 14 (Formula) stretches to fill remaining space initially
        # Bio data columns are added dynamically after processing

        # Enable row resizing by mouse drag
        v_header = self._table.verticalHeader()
        v_header.setSectionResizeMode(QHeaderView.Interactive)
        v_header.setDefaultSectionSize(150)  # Default row height for images

        # Make row numbers bold
        v_header_font = v_header.font()
        v_header_font.setBold(True)
        v_header.setFont(v_header_font)

        # Enable sorting
        self._table.setSortingEnabled(True)

        # Enable copy
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.cellDoubleClicked.connect(self._on_cell_double_clicked)

        # Track changes in custom columns
        self._table.cellChanged.connect(self._on_cell_changed)

        layout.addWidget(self._table)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._lbl_status = QLabel("")
        self._lbl_status.setVisible(False)
        layout.addWidget(self._lbl_status)

        return panel

    def _setup_statusbar(self) -> None:
        """Setup the status bar."""
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready")

    def _update_button_states(self) -> None:
        """Update button enabled states based on current state."""
        has_pdfs = len(self._pdf_infos) > 0
        has_results = len(self._results) > 0
        is_processing = self._worker is not None and self._worker.isRunning()

        self._action_process.setEnabled(has_pdfs and not is_processing)
        self._action_cancel.setEnabled(is_processing)
        self._action_clear_all.setEnabled((has_pdfs or has_results) and not is_processing)

        self._btn_export_checked_csv.setEnabled(has_results)
        self._btn_export_checked_sdf.setEnabled(has_results)
        self._action_export_csv.setEnabled(has_results)
        self._action_export_txt.setEnabled(has_results)
        self._action_export_sdf.setEnabled(has_results)

        # Selection buttons
        self._btn_check_all.setEnabled(has_results and not is_processing)
        self._btn_uncheck_all.setEnabled(has_results and not is_processing)
        self._btn_move_checked_top.setEnabled(has_results and not is_processing)

        # Detect pages button
        self._btn_detect_pages.setEnabled(has_pdfs and not is_processing)

        # Search buttons
        self._btn_search.setEnabled(has_results and not is_processing)
        self._btn_clear_search.setEnabled(has_results and not is_processing)

        # Custom column buttons
        self._btn_add_column.setEnabled(has_results and not is_processing)
        has_custom_columns = len(self._custom_columns) > 0
        self._btn_remove_column.setEnabled(has_results and has_custom_columns and not is_processing)

    def _update_summary(self) -> None:
        """Update the summary labels."""
        if not self._results:
            self._lbl_total.setText("Structures: -")
            self._lbl_valid.setText("Valid: -")
            self._lbl_invalid.setText("Invalid: -")
            return

        summary = ExportHandler.get_summary(self._results)
        self._lbl_total.setText(f"Structures: {summary['total_structures']}")
        self._lbl_valid.setText(f"Valid: {summary['valid_smiles']}")
        invalid = summary['invalid_smiles'] + summary['failed_predictions']
        self._lbl_invalid.setText(f"Invalid: {invalid}")

    def _collect_custom_column_data(self) -> dict:
        """Collect all custom column data from the table.

        Returns:
            Dictionary of {(row, col_name): value} for all custom columns.
        """
        data = {}
        # Custom columns start after base columns + bio columns
        custom_col_start = self._base_column_count + len(self._bio_columns)
        for row in range(self._table.rowCount()):
            for i, col_name in enumerate(self._custom_columns):
                col_index = custom_col_start + i
                item = self._table.item(row, col_index)
                if item:
                    value = item.text()
                    if value:
                        data[(row, col_name)] = value
        return data

    def _pil_to_pixmap(self, pil_image: Image.Image, max_size: int = 140) -> QPixmap:
        """Convert a PIL Image to QPixmap, resized to fit."""
        # Resize to fit
        pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Convert to QImage
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)

        qimage = QImage()
        qimage.loadFromData(buffer.read())

        return QPixmap.fromImage(qimage)

    def _add_result_to_table(self, result: ExtractionResult) -> None:
        """Add a single result to the table."""
        # Temporarily disable sorting while adding row
        self._table.setSortingEnabled(False)

        row = self._table.rowCount()
        self._table.insertRow(row)

        # Set row height for images
        self._table.setRowHeight(row, 150)

        # Checkbox column (column 0)
        checkbox_item = QTableWidgetItem()
        checkbox_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        checkbox_item.setCheckState(Qt.Unchecked)
        self._table.setItem(row, 0, checkbox_item)

        # Source file - column 1
        source_item = QTableWidgetItem(result.source_file or "-")
        source_item.setToolTip(result.source_file or "Unknown source")
        self._table.setItem(row, 1, source_item)

        # Compound ID - column 2
        compound_item = QTableWidgetItem(result.compound_id or "-")
        compound_item.setTextAlignment(Qt.AlignCenter)
        if result.compound_id:
            compound_item.setToolTip(result.compound_id)
        self._table.setItem(row, 2, compound_item)

        # Compound Type - column 3
        if result.compound_type == "example_compound":
            type_text = "Example"
        elif result.compound_type == "other":
            type_text = "Other"
        else:
            type_text = "-"
        type_item = QTableWidgetItem(type_text)
        type_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row, 3, type_item)

        # Page number (numeric sort) - column 4
        page_item = NumericTableWidgetItem(str(result.page_number), float(result.page_number))
        page_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row, 4, page_item)

        # Original image - column 5
        if result.original_image:
            orig_label = QLabel()
            orig_label.setAlignment(Qt.AlignCenter)
            orig_label.setPixmap(self._pil_to_pixmap(result.original_image))
            self._table.setCellWidget(row, 5, orig_label)
        else:
            self._table.setItem(row, 5, QTableWidgetItem("N/A"))

        # SMILES - column 6
        smiles_item = QTableWidgetItem(result.display_smiles)
        smiles_item.setToolTip(result.display_smiles)  # Show full SMILES on hover
        self._table.setItem(row, 6, smiles_item)

        # Valid status with color - column 7
        valid_item = QTableWidgetItem(result.validation_status)
        valid_item.setTextAlignment(Qt.AlignCenter)
        if result.is_valid:
            valid_item.setBackground(Qt.green)
        elif result.smiles:
            valid_item.setBackground(Qt.red)
        self._table.setItem(row, 7, valid_item)

        # RDKit rendering - column 8
        if result.rdkit_image:
            rdkit_label = QLabel()
            rdkit_label.setAlignment(Qt.AlignCenter)
            rdkit_label.setPixmap(self._pil_to_pixmap(result.rdkit_image))
            self._table.setCellWidget(row, 8, rdkit_label)
        else:
            self._table.setItem(row, 8, QTableWidgetItem("N/A"))

        # Molecular Weight (numeric sort) - column 9
        mw_text = f"{result.molecular_weight:.2f}" if result.molecular_weight else "-"
        mw_value = result.molecular_weight if result.molecular_weight else None
        mw_item = NumericTableWidgetItem(mw_text, mw_value)
        mw_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row, 9, mw_item)

        # cLogP (numeric sort) - column 10
        clogp_text = f"{result.clogp:.2f}" if result.clogp is not None else "-"
        clogp_value = result.clogp if result.clogp is not None else None
        clogp_item = NumericTableWidgetItem(clogp_text, clogp_value)
        clogp_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row, 10, clogp_item)

        # TPSA (numeric sort) - column 11
        tpsa_text = f"{result.tpsa:.1f}" if result.tpsa is not None else "-"
        tpsa_value = result.tpsa if result.tpsa is not None else None
        tpsa_item = NumericTableWidgetItem(tpsa_text, tpsa_value)
        tpsa_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row, 11, tpsa_item)

        # Rotatable Bonds (numeric sort) - column 12
        rot_text = str(result.num_rotatable_bonds) if result.num_rotatable_bonds is not None else "-"
        rot_value = float(result.num_rotatable_bonds) if result.num_rotatable_bonds is not None else None
        rot_item = NumericTableWidgetItem(rot_text, rot_value)
        rot_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row, 12, rot_item)

        # Stereocenters (numeric sort) - column 13
        stereo_text = str(result.num_stereocenters) if result.num_stereocenters is not None else "-"
        stereo_value = float(result.num_stereocenters) if result.num_stereocenters is not None else None
        stereo_item = NumericTableWidgetItem(stereo_text, stereo_value)
        stereo_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row, 13, stereo_item)

        # Molecular Formula - column 14
        formula_item = QTableWidgetItem(result.molecular_formula or "-")
        formula_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row, 14, formula_item)

        # Add bio data columns (dynamic, starting at column 14)
        for i, col_name in enumerate(self._bio_columns):
            col_index = self._base_column_count + i
            # Get value from bio_data dict
            value = result.bio_data.get(col_name, "-")
            if not value:
                value = "-"
            bio_item = AutoTypeTableWidgetItem(value)
            bio_item.setTextAlignment(Qt.AlignCenter)
            bio_item.setFlags(bio_item.flags() | Qt.ItemIsEditable)
            bio_item.setToolTip("Double-click to edit")
            self._table.setItem(row, col_index, bio_item)

        # Add editable cells for any custom columns (after bio columns)
        custom_col_start = self._base_column_count + len(self._bio_columns)
        for i, col_name in enumerate(self._custom_columns):
            col_index = custom_col_start + i
            item = AutoTypeTableWidgetItem("")
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, col_index, item)

        # Re-enable sorting
        self._table.setSortingEnabled(True)

    def _update_file_summary(self) -> None:
        """Update the file count and page count labels."""
        file_count = len(self._pdf_infos)
        total_pages = sum(info.page_count for info in self._pdf_infos)

        self._lbl_file_count.setText(f"Files: {file_count}")
        self._lbl_page_count.setText(f"Pages: {total_pages}")

        # Set tooltip with file names
        if self._pdf_infos:
            file_names = [info.file_name for info in self._pdf_infos]
            self._lbl_file_count.setToolTip("\n".join(file_names))
        else:
            self._lbl_file_count.setToolTip("No files loaded")

    @Slot()
    def _on_open_pdf(self) -> None:
        """Handle Open PDFs button click - opens multiple files and replaces existing."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open PDF Files",
            "",
            "PDF Files (*.pdf);;All Files (*)"
        )

        if not file_paths:
            return

        # Clear existing PDFs and results
        self._pdf_infos.clear()
        self._results.clear()
        self._table.setRowCount(0)
        # Remove bio columns and custom columns
        while self._table.columnCount() > self._base_column_count:
            self._table.removeColumn(self._table.columnCount() - 1)
        self._bio_columns.clear()
        self._custom_columns.clear()
        self._custom_column_data.clear()

        # Load all selected PDFs
        failed_files = []
        for file_path in file_paths:
            try:
                processor = PDFProcessor()
                pdf_info = processor.open(file_path)
                processor.close()
                self._pdf_infos.append(pdf_info)
            except Exception as e:
                failed_files.append(f"{Path(file_path).name}: {e}")

        # Update UI
        self._update_file_summary()
        self._update_summary()
        self._update_button_states()

        if failed_files:
            QMessageBox.warning(
                self,
                "Some Files Failed",
                f"Failed to open:\n" + "\n".join(failed_files)
            )

        if self._pdf_infos:
            self._statusbar.showMessage(f"Opened {len(self._pdf_infos)} PDF file(s)")

    @Slot()
    def _on_add_pdf(self) -> None:
        """Handle Add PDF button click - adds files without clearing existing."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add PDF Files",
            "",
            "PDF Files (*.pdf);;All Files (*)"
        )

        if not file_paths:
            return

        # Load selected PDFs (don't clear existing)
        added_count = 0
        failed_files = []
        for file_path in file_paths:
            try:
                processor = PDFProcessor()
                pdf_info = processor.open(file_path)
                processor.close()
                self._pdf_infos.append(pdf_info)
                added_count += 1
            except Exception as e:
                failed_files.append(f"{Path(file_path).name}: {e}")

        # Update UI
        self._update_file_summary()
        self._update_button_states()

        if failed_files:
            QMessageBox.warning(
                self,
                "Some Files Failed",
                f"Failed to open:\n" + "\n".join(failed_files)
            )

        if added_count > 0:
            self._statusbar.showMessage(
                f"Added {added_count} PDF(s). Total: {len(self._pdf_infos)} files"
            )

    @Slot()
    def _on_clear_all(self) -> None:
        """Handle Clear All button click - clears all PDFs and results."""
        self._pdf_infos.clear()
        self._results.clear()
        self._table.setRowCount(0)
        self._txt_page_range.clear()
        # Remove bio columns and custom columns
        while self._table.columnCount() > self._base_column_count:
            self._table.removeColumn(self._table.columnCount() - 1)
        self._bio_columns.clear()
        self._custom_columns.clear()
        self._custom_column_data.clear()

        self._update_file_summary()
        self._update_summary()
        self._update_button_states()
        self._statusbar.showMessage("Cleared all files and results")

    @Slot()
    def _on_detect_pages(self) -> None:
        """Run fast visual page detection to find pages with structures/bio data."""
        if not self._pdf_infos:
            return

        from ..core.page_classifier import PageClassifier
        from ..core.patent_section_detector import PatentSectionDetector

        self._btn_detect_pages.setEnabled(False)
        self._btn_detect_pages.setText("Scanning...")
        self._statusbar.showMessage("Detecting patent sections...")
        QApplication.processEvents()

        try:
            detector = PatentSectionDetector()
            classifier = PageClassifier()
            all_detected = set()
            section_info = ""

            for info in self._pdf_infos:
                # Step 1: Patent section detection (fast)
                self._statusbar.showMessage(
                    f"Detecting sections in {info.file_name}..."
                )
                QApplication.processEvents()

                section_bounds = detector.detect(info.file_path)
                section_pages = section_bounds.get_page_range() if section_bounds.is_valid else set()

                if section_bounds.is_valid:
                    section_info = (
                        f"Examples section: pages {section_bounds.examples_start}"
                        f"-{section_bounds.examples_end}"
                    )

                # Step 2: Visual page classifier
                def progress_cb(current, total, _info=info):
                    self._btn_detect_pages.setText(f"Page {current}/{total}")
                    self._statusbar.showMessage(
                        f"Scanning {_info.file_name}: page {current}/{total}..."
                    )
                    QApplication.processEvents()

                detected = set(classifier.detect_structure_pages(
                    info.file_path, progress_callback=progress_cb
                ))

                # Intersect with section bounds if available
                if detected and section_pages:
                    detected = detected & section_pages
                elif section_pages and not detected:
                    detected = section_pages

                all_detected.update(detected)

            # Format detected pages as compact range string
            if all_detected:
                range_str = self._format_page_set(all_detected)
                self._txt_page_range.setText(range_str)
                status = f"Detected {len(all_detected)} pages with structures/data"
                if section_info:
                    status += f" ({section_info})"
                self._statusbar.showMessage(status)
            else:
                self._txt_page_range.clear()
                self._statusbar.showMessage("No structure pages detected")

        except Exception as e:
            QMessageBox.warning(self, "Detection Error", f"Page detection failed: {e}")
            self._statusbar.showMessage("Page detection failed")
        finally:
            self._btn_detect_pages.setText("Detect Pages")
            self._btn_detect_pages.setEnabled(True)

    @staticmethod
    def _format_page_set(pages: set) -> str:
        """Format a set of page numbers into a compact range string.

        E.g., {1,2,3,5,8,9,10} -> "1-3, 5, 8-10"
        """
        if not pages:
            return ""
        sorted_pages = sorted(pages)
        ranges = []
        start = sorted_pages[0]
        end = start

        for page in sorted_pages[1:]:
            if page == end + 1:
                end = page
            else:
                ranges.append(f"{start}-{end}" if end > start else str(start))
                start = end = page

        ranges.append(f"{start}-{end}" if end > start else str(start))
        return ", ".join(ranges)

    @Slot()
    def _on_extract(self) -> None:
        """Handle Extract button click."""
        if not self._pdf_infos:
            return

        # Clear previous results (but keep PDFs)
        self._results.clear()
        self._table.setRowCount(0)

        # Show progress
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._lbl_status.setVisible(True)

        # Parse page range
        page_range_text = self._txt_page_range.text().strip()
        page_filter = None
        if page_range_text:
            try:
                max_pages = max(info.page_count for info in self._pdf_infos)
                page_filter = parse_page_range(page_range_text, max_pages)
            except ValueError as e:
                QMessageBox.warning(self, "Invalid Page Range", str(e))
                self._progress_bar.setVisible(False)
                self._lbl_status.setVisible(False)
                return

        # Create and start worker with all PDF paths
        self._worker = ProcessingWorker()
        pdf_paths = [info.file_path for info in self._pdf_infos]
        self._worker.set_pdf_paths(pdf_paths)
        self._worker.set_page_filter(page_filter)
        self._worker.set_high_accuracy_mode(self._chk_high_accuracy.isChecked())
        self._worker.set_auto_detect_pages(self._chk_auto_detect.isChecked())

        # Persist auto-detect setting
        self._inference_settings.auto_detect_pages = self._chk_auto_detect.isChecked()

        # Connect signals
        self._worker.progress_updated.connect(self._on_progress_updated)
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.processing_complete.connect(self._on_processing_complete)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.warning_occurred.connect(self._on_worker_warning)

        self._worker.start()
        self._update_button_states()

    @Slot()
    def _on_cancel(self) -> None:
        """Handle Cancel button click."""
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self._statusbar.showMessage("Cancelling...")

    @Slot()
    def _on_export_csv(self) -> None:
        """Handle Export CSV button click."""
        if not self._results:
            return

        # Generate default name based on first file or generic
        default_name = "extraction_results.csv"
        if len(self._pdf_infos) == 1:
            default_name = Path(self._pdf_infos[0].file_path).stem + "_smiles.csv"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to CSV",
            default_name,
            "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return

        try:
            custom_data = self._collect_custom_column_data()
            ExportHandler.export_to_csv(
                self._results, file_path,
                custom_columns=self._custom_columns, custom_data=custom_data
            )
            self._statusbar.showMessage(f"Exported to: {file_path}")
            QMessageBox.information(self, "Export Complete", f"Results exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")

    @Slot()
    def _on_export_txt(self) -> None:
        """Handle Export TXT button click."""
        if not self._results:
            return

        # Generate default name based on first file or generic
        default_name = "extraction_results.txt"
        if len(self._pdf_infos) == 1:
            default_name = Path(self._pdf_infos[0].file_path).stem + "_smiles.txt"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to TXT",
            default_name,
            "Text Files (*.txt);;All Files (*)"
        )

        if not file_path:
            return

        try:
            custom_data = self._collect_custom_column_data()
            ExportHandler.export_to_txt(
                self._results, file_path,
                custom_columns=self._custom_columns, custom_data=custom_data
            )
            self._statusbar.showMessage(f"Exported to: {file_path}")
            QMessageBox.information(self, "Export Complete", f"Results exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")

    @Slot()
    def _on_export_sdf(self) -> None:
        """Handle Export SDF button click."""
        if not self._results:
            return

        # Check if there are any valid results
        valid_count = sum(1 for r in self._results if r.is_valid)
        if valid_count == 0:
            QMessageBox.warning(
                self,
                "No Valid Structures",
                "SDF export requires valid SMILES structures.\n"
                "No valid structures found in results."
            )
            return

        # Generate default name based on first file or generic
        default_name = "extraction_results.sdf"
        if len(self._pdf_infos) == 1:
            default_name = Path(self._pdf_infos[0].file_path).stem + "_structures.sdf"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to SDF",
            default_name,
            "SDF Files (*.sdf);;All Files (*)"
        )

        if not file_path:
            return

        try:
            molecules_written = ExportHandler.export_to_sdf(self._results, file_path)
            self._statusbar.showMessage(f"Exported {molecules_written} molecules to: {file_path}")
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {molecules_written} molecules to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export SDF:\n{e}")

    def _get_checked_row_indices(self) -> List[int]:
        """Get list of row indices that are checked."""
        checked = []
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)  # Checkbox column
            if item and item.checkState() == Qt.Checked:
                checked.append(row)
        return checked

    @Slot()
    def _on_check_all(self) -> None:
        """Check all rows."""
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item:
                item.setCheckState(Qt.Checked)
        self._statusbar.showMessage(f"Checked all {self._table.rowCount()} rows")

    @Slot()
    def _on_uncheck_all(self) -> None:
        """Uncheck all rows."""
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item:
                item.setCheckState(Qt.Unchecked)
        self._statusbar.showMessage("Unchecked all rows")

    @Slot()
    def _on_move_checked_to_top(self) -> None:
        """Move all checked rows to the top of the table."""
        checked_indices = self._get_checked_row_indices()
        if not checked_indices:
            QMessageBox.information(self, "No Rows Checked", "No rows are checked to move.")
            return

        # Disable sorting temporarily
        self._table.setSortingEnabled(False)

        # Collect all row data
        all_rows_data = []
        for row in range(self._table.rowCount()):
            row_data = {
                'result_index': row,
                'is_checked': row in checked_indices,
                'checkbox_state': self._table.item(row, 0).checkState() if self._table.item(row, 0) else Qt.Unchecked
            }
            all_rows_data.append(row_data)

        # Sort: checked rows first, then unchecked
        all_rows_data.sort(key=lambda x: (not x['is_checked'], x['result_index']))

        # Reorder results list and rebuild table
        new_results = [self._results[d['result_index']] for d in all_rows_data]
        checkbox_states = [d['checkbox_state'] for d in all_rows_data]

        # Clear and rebuild
        self._results = new_results
        self._table.setRowCount(0)

        for i, result in enumerate(self._results):
            self._add_result_to_table(result)
            # Restore checkbox state
            checkbox_item = self._table.item(i, 0)
            if checkbox_item:
                checkbox_item.setCheckState(checkbox_states[i])

        # Re-enable sorting
        self._table.setSortingEnabled(True)

        self._statusbar.showMessage(f"Moved {len(checked_indices)} checked rows to top")

    @Slot()
    def _on_substructure_search(self) -> None:
        """Search for substructure matches in the results."""
        query = self._txt_search_smarts.text().strip()
        if not query:
            QMessageBox.warning(self, "No Query", "Please enter a SMILES or SMARTS pattern.")
            return

        try:
            from rdkit import Chem

            # Try to parse as SMARTS first, then as SMILES
            query_mol = Chem.MolFromSmarts(query)
            if query_mol is None:
                query_mol = Chem.MolFromSmiles(query)
            if query_mol is None:
                QMessageBox.warning(
                    self,
                    "Invalid Pattern",
                    "Could not parse the query as SMILES or SMARTS."
                )
                return

            # Search through results
            match_count = 0
            highlight_color = QColor(255, 255, 150)  # Light yellow

            for row in range(self._table.rowCount()):
                if row >= len(self._results):
                    continue

                result = self._results[row]
                is_match = False

                if result.canonical_smiles:
                    mol = Chem.MolFromSmiles(result.canonical_smiles)
                    if mol and mol.HasSubstructMatch(query_mol):
                        is_match = True
                        match_count += 1

                # Update checkbox and highlighting
                checkbox_item = self._table.item(row, 0)
                if checkbox_item:
                    if is_match:
                        checkbox_item.setCheckState(Qt.Checked)
                        # Highlight the row
                        for col in range(self._table.columnCount()):
                            item = self._table.item(row, col)
                            if item:
                                item.setBackground(highlight_color)
                    else:
                        checkbox_item.setCheckState(Qt.Unchecked)
                        # Remove highlight
                        for col in range(self._table.columnCount()):
                            item = self._table.item(row, col)
                            if item and col != 6:  # Don't change Valid column color
                                item.setBackground(QColor(255, 255, 255))

            self._lbl_search_results.setText(f"Matches: {match_count}")
            self._statusbar.showMessage(f"Found {match_count} substructure matches")

        except Exception as e:
            QMessageBox.critical(self, "Search Error", f"Search failed: {e}")

    @Slot()
    def _on_clear_search(self) -> None:
        """Clear search results and highlighting."""
        self._txt_search_smarts.clear()
        self._lbl_search_results.setText("Matches: -")

        # Remove highlighting from all rows
        for row in range(self._table.rowCount()):
            for col in range(self._table.columnCount()):
                item = self._table.item(row, col)
                if item:
                    # Restore original background (white, except Valid column)
                    if col == 6:  # Valid column - restore green/red based on validity
                        if row < len(self._results):
                            result = self._results[row]
                            if result.is_valid:
                                item.setBackground(Qt.green)
                            elif result.smiles:
                                item.setBackground(Qt.red)
                            else:
                                item.setBackground(QColor(255, 255, 255))
                    else:
                        item.setBackground(QColor(255, 255, 255))

        self._statusbar.showMessage("Search cleared")

    @Slot()
    def _on_draw_structure(self) -> None:
        """Open chemical drawing dialog to draw a structure for search."""
        if not HAS_WEBENGINE:
            QMessageBox.information(
                self,
                "WebEngine Not Available",
                "Chemical drawing requires PySide6-WebEngine.\n\n"
                "Install it with: pip install PySide6-WebEngine\n\n"
                "For now, enter SMILES/SMARTS directly in the text field."
            )
            return

        # Get current SMILES from text field to potentially initialize the editor
        current_smiles = self._txt_search_smarts.text().strip()

        dialog = ChemicalDrawingDialog(self, initial_smiles=current_smiles)
        if dialog.exec() == QDialog.Accepted:
            smiles = dialog.get_smiles()
            if smiles:
                self._txt_search_smarts.setText(smiles)
                self._statusbar.showMessage(f"Structure drawn: {smiles[:50]}...")

    @Slot()
    def _on_export_checked_csv(self) -> None:
        """Export only checked rows to CSV."""
        checked_indices = self._get_checked_row_indices()
        if not checked_indices:
            QMessageBox.warning(
                self,
                "No Rows Selected",
                "Please check at least one row to export."
            )
            return

        # Get the checked results
        checked_results = [self._results[i] for i in checked_indices if i < len(self._results)]

        # Generate default name
        default_name = "checked_results.csv"
        if len(self._pdf_infos) == 1:
            default_name = Path(self._pdf_infos[0].file_path).stem + "_checked.csv"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Checked Rows to CSV",
            default_name,
            "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Collect custom data for checked rows only
            custom_data = {}
            for new_idx, orig_idx in enumerate(checked_indices):
                for i, col_name in enumerate(self._custom_columns):
                    col_index = self._base_column_count + i
                    item = self._table.item(orig_idx, col_index)
                    if item and item.text():
                        custom_data[(new_idx, col_name)] = item.text()

            ExportHandler.export_to_csv(
                checked_results, file_path,
                custom_columns=self._custom_columns, custom_data=custom_data
            )
            self._statusbar.showMessage(f"Exported {len(checked_results)} rows to: {file_path}")
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(checked_results)} checked rows to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")

    @Slot()
    def _on_export_checked_sdf(self) -> None:
        """Export only checked rows to SDF."""
        checked_indices = self._get_checked_row_indices()
        if not checked_indices:
            QMessageBox.warning(
                self,
                "No Rows Selected",
                "Please check at least one row to export."
            )
            return

        # Get the checked results
        checked_results = [self._results[i] for i in checked_indices if i < len(self._results)]

        # Check if there are any valid results
        valid_count = sum(1 for r in checked_results if r.is_valid)
        if valid_count == 0:
            QMessageBox.warning(
                self,
                "No Valid Structures",
                "SDF export requires valid SMILES structures.\n"
                "No valid structures found in checked rows."
            )
            return

        # Generate default name
        default_name = "checked_results.sdf"
        if len(self._pdf_infos) == 1:
            default_name = Path(self._pdf_infos[0].file_path).stem + "_checked.sdf"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Checked Rows to SDF",
            default_name,
            "SDF Files (*.sdf);;All Files (*)"
        )

        if not file_path:
            return

        try:
            molecules_written = ExportHandler.export_to_sdf(checked_results, file_path)
            self._statusbar.showMessage(f"Exported {molecules_written} molecules to: {file_path}")
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {molecules_written} molecules from checked rows to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export SDF:\n{e}")

    @Slot()
    def _on_add_column(self) -> None:
        """Handle Add Column button click."""
        column_name, ok = QInputDialog.getText(
            self,
            "Add Column",
            "Enter column name:",
            text=""
        )

        if not ok or not column_name.strip():
            return

        column_name = column_name.strip()

        # Check if column already exists in bio columns or custom columns
        if column_name in self._custom_columns or column_name in self._bio_columns:
            QMessageBox.warning(
                self,
                "Column Exists",
                f"A column named '{column_name}' already exists."
            )
            return

        # Add the column
        self._custom_columns.append(column_name)
        col_index = self._table.columnCount()
        self._table.insertColumn(col_index)
        self._table.setHorizontalHeaderItem(col_index, QTableWidgetItem(column_name))

        # Add editable cells to existing rows
        for row in range(self._table.rowCount()):
            item = AutoTypeTableWidgetItem("")
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, col_index, item)

        self._update_button_states()
        self._statusbar.showMessage(f"Added column: {column_name}")

    @Slot()
    def _on_remove_column(self) -> None:
        """Handle Remove Column button click."""
        if not self._custom_columns:
            return

        # Show dialog to select column to remove
        column_name, ok = QInputDialog.getItem(
            self,
            "Remove Column",
            "Select column to remove:",
            self._custom_columns,
            0,
            False
        )

        if not ok or not column_name:
            return

        # Find and remove the column (custom columns come after bio columns)
        custom_col_start = self._base_column_count + len(self._bio_columns)
        col_index = custom_col_start + self._custom_columns.index(column_name)
        self._table.removeColumn(col_index)
        self._custom_columns.remove(column_name)

        # Clean up stored data for this column
        keys_to_remove = [k for k in self._custom_column_data if k[1] == column_name]
        for key in keys_to_remove:
            del self._custom_column_data[key]

        self._update_button_states()
        self._statusbar.showMessage(f"Removed column: {column_name}")

    @Slot()
    def _on_show_unmatched(self) -> None:
        """Show biological data that wasn't matched to any structure."""
        # Auto-extract bio data if not already done
        if not self._last_bio_data and self._pdf_infos:
            try:
                from ..core.biological_data_extractor import BiologicalDataExtractor

                self._statusbar.showMessage("Extracting biological data...")
                QApplication.processEvents()

                pdf_info = self._pdf_infos[0]
                extractor = BiologicalDataExtractor()
                self._last_bio_data = extractor.extract_from_pdf_path(pdf_info.file_path)

                self._statusbar.showMessage(f"Extracted bio data for {len(self._last_bio_data)} compounds")
            except Exception as e:
                QMessageBox.warning(self, "Extraction Error", f"Failed to extract bio data: {e}")
                return

        if not self._last_bio_data:
            QMessageBox.information(
                self,
                "No Data",
                "No biological data found in the PDF."
            )
            return

        # Get matched compound IDs from results
        matched_ids = set()
        for result in self._results:
            if result.compound_id:
                matched_ids.add(result.compound_id.strip())
                # Also add simplified version
                import re
                match = re.search(r'(\d+[a-zA-Z]?)', result.compound_id)
                if match:
                    matched_ids.add(match.group(1))

        # Find unmatched bio data
        unmatched = []
        for compound_id, bio_data in self._last_bio_data.items():
            # Check if this compound was matched
            is_matched = compound_id in matched_ids
            if not is_matched:
                # Also check simplified version
                import re
                match = re.search(r'(\d+[a-zA-Z]?)', compound_id)
                if match and match.group(1) in matched_ids:
                    is_matched = True

            if not is_matched:
                unmatched.append((compound_id, bio_data))

        # Build output
        output = f"=== Unmatched Biological Data ===\n\n"
        output += f"Total bio data entries: {len(self._last_bio_data)}\n"
        output += f"Matched to structures: {len(self._last_bio_data) - len(unmatched)}\n"
        output += f"Unmatched: {len(unmatched)}\n\n"

        if unmatched:
            output += "--- Unmatched Compounds ---\n\n"
            for compound_id, bio_data in sorted(unmatched, key=lambda x: x[0]):
                output += f"Compound {compound_id}:\n"
                if bio_data.ic50:
                    output += f"  IC50: {bio_data.ic50}\n"
                if bio_data.ec50:
                    output += f"  EC50: {bio_data.ec50}\n"
                if bio_data.ki:
                    output += f"  Ki: {bio_data.ki}\n"
                if bio_data.kd:
                    output += f"  Kd: {bio_data.kd}\n"
                # Show other assays from dynamic columns
                for assay_name, value in bio_data.other_assays.items():
                    output += f"  {assay_name}: {value}\n"
                output += "\n"

            output += "\n--- Possible Reasons ---\n"
            output += "1. Compound ID from OCR doesn't match structure detection\n"
            output += "2. Structure wasn't detected in the PDF\n"
            output += "3. OCR misread the compound number\n"
            output += "\nYou can manually edit the bio data columns in the table.\n"
        else:
            output += "All biological data was matched to structures!\n"

        # Show in dialog
        from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QDialog, QPushButton

        dialog = QDialog(self)
        dialog.setWindowTitle("Unmatched Biological Data")
        dialog.setMinimumSize(500, 400)

        layout = QVBoxLayout(dialog)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFontFamily("Consolas, Monaco, monospace")
        text_edit.setPlainText(output)
        layout.addWidget(text_edit)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec()

    @Slot()
    def _on_assign_bio_data(self) -> None:
        """Assign unmatched biological data to the selected table row."""
        # Get selected row
        selected_rows = self._table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(
                self,
                "No Selection",
                "Please select a row in the table first."
            )
            return

        row = selected_rows[0].row()
        if row >= len(self._results):
            return

        result = self._results[row]

        # Get unmatched bio data
        if not self._last_bio_data:
            QMessageBox.information(
                self,
                "No Bio Data",
                "No biological data available.\n"
                "Click 'Unmatched Bio Data...' first to extract bio data from the PDF."
            )
            return

        # Build list of compounds with bio data
        choices = []
        for compound_id, bio_data in sorted(self._last_bio_data.items()):
            values = []
            if bio_data.ic50:
                values.append(f"IC50:{bio_data.ic50}")
            if bio_data.ec50:
                values.append(f"EC50:{bio_data.ec50}")
            if bio_data.ki:
                values.append(f"Ki:{bio_data.ki}")
            if bio_data.kd:
                values.append(f"Kd:{bio_data.kd}")
            # Include other assays from dynamic columns
            for assay_name, value in bio_data.other_assays.items():
                # Truncate long names for display
                short_name = assay_name[:20] + "..." if len(assay_name) > 20 else assay_name
                values.append(f"{short_name}:{value}")
            if values:
                choices.append(f"{compound_id}: {', '.join(values)}")

        if not choices:
            QMessageBox.information(
                self,
                "No Bio Data",
                "No biological data with values found."
            )
            return

        # Show selection dialog
        choice, ok = QInputDialog.getItem(
            self,
            "Assign Bio Data",
            f"Select bio data to assign to row {row + 1}\n"
            f"(Current compound: {result.compound_id or 'None'}):",
            choices,
            0,
            False
        )

        if not ok or not choice:
            return

        # Parse selection and assign
        compound_id = choice.split(":")[0].strip()
        if compound_id in self._last_bio_data:
            bio_data = self._last_bio_data[compound_id]

            # Update result - legacy fields
            if bio_data.ic50:
                result.ic50 = bio_data.ic50
                result.bio_data['IC50'] = bio_data.ic50
            if bio_data.ec50:
                result.ec50 = bio_data.ec50
                result.bio_data['EC50'] = bio_data.ec50
            if bio_data.ki:
                result.ki = bio_data.ki
                result.bio_data['Ki'] = bio_data.ki
            if bio_data.kd:
                result.kd = bio_data.kd
                result.bio_data['Kd'] = bio_data.kd

            # Update result - dynamic columns from other_assays
            for assay_name, value in bio_data.other_assays.items():
                result.bio_data[assay_name] = value

            # Update table - dynamic bio columns
            for i, col_name in enumerate(self._bio_columns):
                col_index = self._base_column_count + i
                item = self._table.item(row, col_index)
                if item:
                    value = result.bio_data.get(col_name, "-")
                    item.setText(value if value else "-")

            # Also update compound ID if it was empty
            if not result.compound_id:
                result.compound_id = compound_id
                self._table.item(row, 2).setText(compound_id)

            self._statusbar.showMessage(f"Assigned bio data from compound {compound_id} to row {row + 1}")

    @Slot()
    def _on_debug_detection(self) -> None:
        """Show detailed debug information about compound and bio data detection."""
        from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QDialog, QPushButton

        output = "=== DETECTION DEBUG INFO ===\n\n"

        # Show what compound IDs were detected for each structure
        output += "--- Detected Compound IDs per Structure ---\n\n"
        for i, result in enumerate(self._results):
            output += f"Row {i+1}: Page {result.page_number}, Struct {result.structure_index+1}\n"
            output += f"  Compound ID: '{result.compound_id or 'None'}'\n"
            output += f"  Bounding Box: {result.bounding_box}\n"
            output += f"  Bio Data: {result.bio_data}\n"
            output += "\n"

        # Show extracted bio data (format: {source_file: {compound_id: BiologicalData}})
        output += "\n--- Extracted Bio Data (from PDF) ---\n\n"
        if self._last_bio_data:
            for source_file, file_bio_data in sorted(self._last_bio_data.items()):
                output += f"=== {source_file} ===\n\n"
                for compound_id, bio_data in sorted(file_bio_data.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
                    output += f"Compound '{compound_id}':\n"
                    output += f"  IC50: '{bio_data.ic50 or '-'}'\n"
                    output += f"  EC50: '{bio_data.ec50 or '-'}'\n"
                    output += f"  Ki: '{bio_data.ki or '-'}'\n"
                    output += f"  Kd: '{bio_data.kd or '-'}'\n"
                    if bio_data.other_assays:
                        output += f"  Other Assays: {bio_data.other_assays}\n"
                    output += f"  Matched: {bio_data.matched}\n"
                    output += "\n"
        else:
            output += "(No bio data extracted yet - click 'Unmatched...' first)\n"

        # Try to get debug info from bio extractor
        output += "\n--- Bio Extractor Debug Log ---\n\n"
        try:
            from ..core.biological_data_extractor import BiologicalDataExtractor

            if self._pdf_infos:
                pdf_info = self._pdf_infos[0]
                extractor = BiologicalDataExtractor()
                extractor.extract_from_pdf_path(pdf_info.file_path)
                debug_log = extractor.get_debug_info()
                output += debug_log if debug_log else "(No debug info available)\n"
        except Exception as e:
            output += f"Error getting debug info: {e}\n"

        # Debug compound detection for first page with structures
        output += "\n--- Compound Detection Debug (First Page) ---\n\n"
        try:
            from ..core.compound_detector import CompoundDetector
            from ..core.pdf_processor import PDFProcessor
            from ..core.structure_detector import StructureDetector

            if self._pdf_infos and self._results:
                pdf_info = self._pdf_infos[0]
                pdf_processor = PDFProcessor()
                pdf_processor.open(pdf_info.file_path)

                # Find first page with structures
                first_page = self._results[0].page_number if self._results else 1
                page_image = pdf_processor.get_page_image(first_page)

                # Re-detect structures on this page
                structure_detector = StructureDetector()
                structures = structure_detector.detect_structures(page_image)

                if structures:
                    compound_detector = CompoundDetector()
                    compound_debug = compound_detector.debug_compound_detection(
                        structures, page_image
                    )
                    output += compound_debug
                else:
                    output += f"No structures found on page {first_page}\n"

                pdf_processor.close()
        except Exception as e:
            output += f"Error getting compound detection debug: {e}\n"

        # Show package versions
        output += "\n\n--- Package Versions ---\n\n"
        try:
            import cv2
            output += f"OpenCV: {cv2.__version__}\n"
        except:
            output += "OpenCV: unknown\n"
        try:
            import importlib.metadata
            output += f"pypdfium2: {importlib.metadata.version('pypdfium2')}\n"
        except:
            output += "pypdfium2: unknown\n"
        try:
            import importlib.metadata
            output += f"pdfplumber: {importlib.metadata.version('pdfplumber')}\n"
        except:
            output += "pdfplumber: unknown\n"
        try:
            import pytesseract
            output += f"pytesseract: {pytesseract.__version__}\n"
        except:
            output += "pytesseract: unknown\n"

        # Show dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Detection Debug Info")
        dialog.setMinimumSize(700, 500)

        layout = QVBoxLayout(dialog)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFontFamily("Consolas, Monaco, monospace")
        text_edit.setPlainText(output)
        layout.addWidget(text_edit)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec()

    @Slot(int, int)
    def _on_cell_changed(self, row: int, column: int) -> None:
        """Handle cell value changes for bio data and custom columns."""
        import re

        # Handle bio data columns (dynamic, starting at column 14)
        bio_col_start = self._base_column_count
        bio_col_end = bio_col_start + len(self._bio_columns)

        if row < len(self._results):
            item = self._table.item(row, column)
            if item:
                value = item.text()
                # Convert "-" back to None/empty for storage
                if value == "-" or value == "":
                    value = None

                # Check if this is a bio column
                if bio_col_start <= column < bio_col_end:
                    bio_col_index = column - bio_col_start
                    if bio_col_index < len(self._bio_columns):
                        col_name = self._bio_columns[bio_col_index]
                        if value:
                            self._results[row].bio_data[col_name] = value
                        elif col_name in self._results[row].bio_data:
                            del self._results[row].bio_data[col_name]

                        # Also update legacy fields if column name matches pattern
                        if re.search(r'ic\s*[-_]?\s*50', col_name, re.IGNORECASE):
                            self._results[row].ic50 = value
                        elif re.search(r'ec\s*[-_]?\s*50', col_name, re.IGNORECASE):
                            self._results[row].ec50 = value
                        elif re.search(r'\bki\b', col_name, re.IGNORECASE):
                            self._results[row].ki = value
                        elif re.search(r'\bkd\b', col_name, re.IGNORECASE):
                            self._results[row].kd = value

        # Handle custom columns (after bio columns)
        custom_col_start = bio_col_end
        if column >= custom_col_start:
            col_name_index = column - custom_col_start
            if col_name_index < len(self._custom_columns):
                col_name = self._custom_columns[col_name_index]
                item = self._table.item(row, column)
                if item:
                    value = item.text()
                    self._custom_column_data[(row, col_name)] = value

    @Slot(ProcessingProgress)
    def _on_progress_updated(self, progress: ProcessingProgress) -> None:
        """Handle progress updates from worker."""
        self._progress_bar.setValue(int(progress.overall_progress))
        self._lbl_status.setText(progress.status_message)
        self._statusbar.showMessage(progress.status_message)

    @Slot(ExtractionResult)
    def _on_result_ready(self, result: ExtractionResult) -> None:
        """Handle new result from worker."""
        self._results.append(result)
        self._add_result_to_table(result)
        self._update_summary()

    @Slot(list)
    def _on_processing_complete(self, results: List[ExtractionResult]) -> None:
        """Handle processing completion."""
        self._progress_bar.setVisible(False)
        self._lbl_status.setVisible(False)

        # Get bio data from worker before clearing reference
        # Qt signals across threads create copies, so we need to re-merge in main thread
        if self._worker and hasattr(self._worker, '_extracted_bio_data') and self._worker._extracted_bio_data:
            self._last_bio_data = self._worker._extracted_bio_data.copy()

        # Clear worker reference so is_processing becomes False
        self._worker = None

        # Update results list
        self._results = results

        # IMPORTANT: Re-merge bio data in main thread
        # Qt signals across threads create copies of Python objects,
        # so the worker's merge updates don't affect these results.
        # We need to merge again using the extracted bio_data.
        if self._last_bio_data:
            self._merge_bio_data_to_results()

        # Collect all unique bio_data keys from results and add dynamic columns
        self._add_dynamic_bio_columns(self._results)

        # Update biological data columns in the table
        self._update_bio_data_in_table()

        self._update_button_states()
        self._update_summary()

        if results:
            summary = ExportHandler.get_summary(results)
            # Count results with bio_data for debugging
            bio_count = sum(1 for r in self._results if r.bio_data)
            bio_keys = set()
            for r in self._results:
                bio_keys.update(r.bio_data.keys())
            self._statusbar.showMessage(
                f"Complete: {summary['total_structures']} structures, "
                f"{summary['valid_smiles']} valid, "
                f"{bio_count} with bio data ({len(bio_keys)} columns)"
            )
        else:
            self._statusbar.showMessage("No chemical structures found")

    def _merge_bio_data_to_results(self) -> None:
        """Merge extracted bio data into results by compound ID.

        This is needed because Qt signals across threads create copies of objects,
        so the worker's merge doesn't affect the results received in the main window.

        Bio data format: {source_file: {compound_id: BiologicalData}}
        """
        import re

        if not self._last_bio_data:
            return

        merge_count = 0

        for result in self._results:
            if not result.compound_id or not result.source_file:
                continue

            # Get bio data for this file
            file_bio_data = self._last_bio_data.get(result.source_file, {})
            if not file_bio_data:
                continue

            bio_data_keys = set(file_bio_data.keys())
            compound_id = result.compound_id.strip()
            matched_key = None

            # Try exact match
            if compound_id in file_bio_data:
                matched_key = compound_id
            else:
                # Try matching just the numeric part
                match = re.search(r'(\d+[a-zA-Z]?)', compound_id)
                if match:
                    simple_id = match.group(1)
                    if simple_id in file_bio_data:
                        matched_key = simple_id
                    else:
                        # Fuzzy match
                        for key in bio_data_keys:
                            key_simple = re.search(r'(\d+[a-zA-Z]?)', key)
                            if key_simple:
                                try:
                                    if int(re.sub(r'[a-zA-Z]', '', simple_id)) == int(re.sub(r'[a-zA-Z]', '', key_simple.group(1))):
                                        suffix1 = re.sub(r'\d', '', simple_id).lower()
                                        suffix2 = re.sub(r'\d', '', key_simple.group(1)).lower()
                                        if suffix1 == suffix2:
                                            matched_key = key
                                            break
                                except ValueError:
                                    continue

            if matched_key:
                data = file_bio_data[matched_key]
                # Copy bio data to result
                result.ic50 = data.ic50
                result.ec50 = data.ec50
                result.ki = data.ki
                result.kd = data.kd
                for assay_name, value in data.other_assays.items():
                    result.bio_data[assay_name] = value
                data.matched = True
                merge_count += 1

        self._statusbar.showMessage(f"Merged bio data for {merge_count} compounds")

    def _add_dynamic_bio_columns(self, results: List[ExtractionResult]) -> None:
        """Collect all unique bio_data keys from results and add columns dynamically."""
        import re

        # Collect all unique bio_data keys
        all_bio_keys = set()
        for result in results:
            all_bio_keys.update(result.bio_data.keys())

        # Sort for consistent column order
        # Priority patterns for common assay types
        priority_patterns = [
            re.compile(r'ic\s*[-_]?\s*50', re.IGNORECASE),
            re.compile(r'ec\s*[-_]?\s*50', re.IGNORECASE),
            re.compile(r'gi\s*[-_]?\s*50', re.IGNORECASE),
            re.compile(r'\bki\b', re.IGNORECASE),
            re.compile(r'\bkd\b', re.IGNORECASE),
            re.compile(r'emax', re.IGNORECASE),
        ]

        def get_priority(key):
            """Return priority index (lower = higher priority)."""
            key_lower = key.lower()
            for idx, pattern in enumerate(priority_patterns):
                if pattern.search(key):
                    return idx
            return len(priority_patterns)  # Non-priority items come last

        # Sort: priority items first, then alphabetically within each group
        sorted_keys = sorted(all_bio_keys, key=lambda k: (get_priority(k), k.lower()))

        # Only add columns if there are bio keys
        if not sorted_keys:
            return

        # Store the bio columns list
        self._bio_columns = sorted_keys

        # Block signals while creating cells to prevent cellChanged from deleting bio_data
        self._table.blockSignals(True)
        try:
            # Add columns to the table and create cells for existing rows
            for i, col_name in enumerate(sorted_keys):
                col_index = self._base_column_count + i
                self._table.insertColumn(col_index)
                self._table.setHorizontalHeaderItem(col_index, QTableWidgetItem(col_name))
                # Set reasonable width based on column name length
                width = max(100, min(250, len(col_name) * 7 + 30))
                self._table.setColumnWidth(col_index, width)

                # Create cells for all existing rows
                for row in range(self._table.rowCount()):
                    item = AutoTypeTableWidgetItem("-")
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                    item.setToolTip("Double-click to edit")
                    self._table.setItem(row, col_index, item)
        finally:
            self._table.blockSignals(False)

    def _update_bio_data_in_table(self) -> None:
        """Update biological data columns in the table from results."""
        def normalize_key(key):
            """Normalize key for matching - lowercase, replace spaces with underscores."""
            return key.lower().replace(' ', '_').replace('-', '_')

        # Block signals while updating cells to prevent cellChanged handler
        self._table.blockSignals(True)
        try:
            for row, result in enumerate(self._results):
                if row >= self._table.rowCount():
                    break

                # Build a normalized lookup for this result's bio_data
                bio_data_normalized = {normalize_key(k): v for k, v in result.bio_data.items()}

                # Update dynamic bio columns
                for i, col_name in enumerate(self._bio_columns):
                    col_index = self._base_column_count + i
                    item = self._table.item(row, col_index)
                    if item:
                        # Try exact match first, then normalized match
                        value = result.bio_data.get(col_name)
                        if value is None:
                            value = bio_data_normalized.get(normalize_key(col_name), "-")
                        item.setText(value if value else "-")
        finally:
            self._table.blockSignals(False)


    @Slot(str)
    def _on_error(self, error_message: str) -> None:
        """Handle error from worker."""
        self._progress_bar.setVisible(False)
        self._lbl_status.setVisible(False)
        self._update_button_states()

        QMessageBox.critical(self, "Processing Error", error_message)
        self._statusbar.showMessage("Error occurred")

    @Slot(str)
    def _on_worker_warning(self, message: str) -> None:
        """Handle non-fatal warning from worker."""
        self._statusbar.showMessage(message, 10000)

    @Slot(int, int)
    def _on_cell_double_clicked(self, row: int, column: int) -> None:
        """Handle double-click on table cell to copy SMILES."""
        if column == 6:  # SMILES column (after checkbox, Source, Compound, Type, and Page columns)
            item = self._table.item(row, column)
            if item:
                smiles = item.text()
                if smiles and not smiles.startswith("ERROR"):
                    clipboard = QApplication.clipboard()
                    clipboard.setText(smiles)
                    self._statusbar.showMessage(f"Copied SMILES to clipboard: {smiles[:50]}...")

    def _get_auto_detect_setting(self) -> bool:
        """Get the persisted auto-detect pages setting."""
        return InferenceSettings.get_instance().auto_detect_pages

    def _get_inference_mode_text(self) -> str:
        """Get display text for current inference mode."""
        mode = self._inference_settings.mode
        if mode == InferenceMode.CLOUD:
            return "Mode: Cloud GPU"
        elif mode == InferenceMode.LOCAL_GPU:
            return "Mode: Local GPU"
        elif mode == InferenceMode.LOCAL_LIGHTWEIGHT:
            return "Mode: Lightweight"
        else:
            return "Mode: Local CPU"

    def _get_classifier_status_text(self) -> str:
        """Get display text for classifier API key status."""
        if self._inference_settings.anthropic_api_key:
            return '<span style="color: green;">Classifier: Active</span>'
        return '<span style="color: gray;">Classifier: No API key</span>'

    @Slot()
    def _on_inference_settings(self) -> None:
        """Open inference settings dialog."""
        from .inference_settings_dialog import InferenceSettingsDialog

        dialog = InferenceSettingsDialog(self)
        if dialog.exec() == QDialog.Accepted:
            # Update the mode and classifier labels
            self._lbl_inference_mode.setText(self._get_inference_mode_text())
            self._lbl_classifier_status.setText(self._get_classifier_status_text())
            mode = self._inference_settings.mode
            if mode == InferenceMode.CLOUD:
                self._statusbar.showMessage("Inference mode: Cloud GPU (Modal.com)")
            elif mode == InferenceMode.LOCAL_GPU:
                self._statusbar.showMessage("Inference mode: Local GPU (CUDA)")
            else:
                self._statusbar.showMessage("Inference mode: Local CPU (slow)")
