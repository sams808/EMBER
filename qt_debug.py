"""
qt_debug.py — Debug / Export workspace: a live app log viewer plus a
one-click "export everything" bundle for bug reports. Ported from the old
app's DebugTab (export global debug bundle / open output folder / clear
log), adapted to Ember's StatusLogger (a live-updating Qt log via the
`logged` signal, rather than the old app's static Tk Text widget bound
once at startup).
"""
from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

import overview_science as osci
from data_model import HanfordDataset


class DebugPage(QWidget):
    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.dataset: Optional[HanfordDataset] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        top = QHBoxLayout()
        top.setSpacing(8)
        text = QVBoxLayout()
        text.setSpacing(2)
        title = QLabel("Diagnostics")
        title.setObjectName("PageTitle")
        text.addWidget(title)
        # Wrapping, and no width of its own: a sentence must not set the
        # width of the window it sits in.
        note = QLabel("The session log, live. To report a problem, export the diagnostics bundle and send "
                      "it instead of the source data file.")
        note.setObjectName("PageSubtitle")
        note.setWordWrap(True)
        note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        text.addWidget(note)
        top.addLayout(text, 1)
        clear_btn = QPushButton("Clear log")
        clear_btn.setObjectName("Secondary")
        clear_btn.clicked.connect(self.clear_log)
        top.addWidget(clear_btn, 0, Qt.AlignBottom)
        open_btn = QPushButton("Open output folder")
        open_btn.clicked.connect(self.open_output)
        top.addWidget(open_btn, 0, Qt.AlignBottom)
        export_btn = QPushButton("Export diagnostics bundle")
        export_btn.setObjectName("Primary")
        export_btn.setToolTip("Write the log, the dataset summary and the environment to the output folder.")
        export_btn.clicked.connect(self.export_bundle)
        top.addWidget(export_btn, 0, Qt.AlignBottom)
        root.addLayout(top)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Nothing logged yet in this session.")
        root.addWidget(self.log_view, 1)

        logger = self.app_window.status_logger
        self.log_view.setPlainText(logger.full_text())
        logger.logged.connect(self._append_line)

    def _append_line(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    # ------------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset

    def export_bundle(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Debug / Export", "Load a dataset first.")
            return
        try:
            path = osci.export_global_debug_bundle(self.dataset)
        except Exception as exc:
            QMessageBox.critical(self, "Debug export failed", str(exc))
            return
        self.app_window.status_logger.log(f"Global debug bundle exported: {path}")
        QMessageBox.information(self, "Export complete", f"Debug bundle exported:\n{path}")

    def open_output(self) -> None:
        if self.dataset is None:
            QMessageBox.information(self, "Debug / Export", "Load a dataset first.")
            return
        self.dataset.output_root.mkdir(parents=True, exist_ok=True)
        os.startfile(str(self.dataset.output_root.resolve()))

    def clear_log(self) -> None:
        self.app_window.status_logger.clear()
        self.log_view.clear()
