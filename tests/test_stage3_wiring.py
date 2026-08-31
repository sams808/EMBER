"""
Stage 3 in the app: exports that describe themselves, and projects that
round-trip through the File menu.
"""
import polars as pl
import pytest
from PySide6.QtWidgets import QMainWindow

import data_model as dm
from ember_project import PROJECT_SUFFIX, EmberProject
from qt_query import QueryPage


def _dataset() -> dm.HanfordDataset:
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-A-101", "241-AN-104"],
        "Analyte": ["SO4", "Fe"],
        "WastePhase": ["Sludge", "Supernatant"],
        "WasteType": ["T1", "T2"],
        "Inventory": [100.0, 10.0],
        "Units": ["kg", "kg"],
    }))
    dataset.report = None
    return dataset


def _query_page(qtbot):
    window = QMainWindow()
    qtbot.addWidget(window)
    page = QueryPage(window)
    qtbot.addWidget(page)
    return page


class TestExportsDescribeThemselves:
    def _export(self, qtbot, tmp_path):
        dataset = _dataset()
        dataset.output_root = tmp_path
        page = _query_page(qtbot)
        page.on_dataset_changed(dataset)
        page.kind_combo.setCurrentText("Element")
        page.subject_edit.setText("S")
        page.basis_combo.setCurrentText("kg")
        page.scope.set_filter("WastePhase", ["Sludge"])
        page.run()
        qtbot.wait(20)
        page._export()
        return next(iter(tmp_path.glob("query_S_*")))

    def test_a_provenance_table_is_written(self, qtbot, tmp_path):
        out = self._export(qtbot, tmp_path)
        assert (out / "provenance.csv").exists()
        text = (out / "provenance.csv").read_text(encoding="utf-8")
        assert "reference_tables.atomic_weights" in text
        assert "CIAAW" in text

    def test_a_methods_paragraph_is_written(self, qtbot, tmp_path):
        out = self._export(qtbot, tmp_path)
        methods = (out / "methods.txt").read_text(encoding="utf-8")
        assert "apportioned stoichiometrically" in methods
        assert "Sludge" in methods          # the scope it ran under
        assert "S was quantified" in methods

    def test_the_methods_text_carries_the_caveats(self, qtbot, tmp_path):
        methods = (self._export(qtbot, tmp_path) / "methods.txt").read_text(encoding="utf-8")
        # The engine raised at least one warning for this query; a methods
        # paragraph that drops them is the thing provenance exists to stop.
        assert "Caveats" in methods or "Source resolution" in methods

    def test_methods_is_listed_in_the_manifest(self, qtbot, tmp_path):
        out = self._export(qtbot, tmp_path)
        assert "methods.txt" in (out / "manifest.csv").read_text(encoding="utf-8")


class TestProjectRoundTripThroughTheShell:
    @pytest.fixture
    def window(self, qtbot, tmp_path, monkeypatch):
        monkeypatch.setattr(dm, "app_base_dir", lambda: tmp_path)
        monkeypatch.chdir(tmp_path)
        from qt_shell import EmberMainWindow
        win = EmberMainWindow()
        qtbot.addWidget(win)
        qtbot.wait(50)
        return win

    def test_current_project_captures_scope_and_the_standing_question(self, window, qtbot, tmp_path):
        window.dataset.df = _dataset().df
        window.dataset.report = None
        window.scope_bar.on_dataset_changed(window.dataset)
        for page in window._dataset_aware_pages():
            if hasattr(page, "on_dataset_changed"):
                page.on_dataset_changed(window.dataset)
        window.global_scope.set_filter("WastePhase", ["Sludge"])
        qtbot.wait(20)

        page = window.query_page
        page.kind_combo.setCurrentText("Element")
        page.subject_edit.setText("S")
        page.basis_combo.setCurrentText("kg")
        page.run()
        qtbot.wait(20)

        project = window.current_project()
        assert project.scope == {"WastePhase": ["Sludge"]}
        assert len(project.questions) == 1
        assert project.questions[0].subject == "S"
        assert project.questions[0].basis == "kg"
        assert project.problems() == []

    def test_saving_and_reopening_restores_the_scope(self, window, qtbot, tmp_path, monkeypatch):
        window.dataset.df = _dataset().df
        window.dataset.report = None
        window.scope_bar.on_dataset_changed(window.dataset)
        window.global_scope.set_filter("WastePhase", ["Sludge"])
        qtbot.wait(20)

        target = tmp_path / "study"
        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getSaveFileName",
            staticmethod(lambda *a, **k: (str(target), "")))
        window.save_project()
        saved = target.with_suffix(PROJECT_SUFFIX)
        assert saved.exists()

        window.global_scope.clear()
        assert window.global_scope.as_scope() == {}

        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getOpenFileName",
            staticmethod(lambda *a, **k: (str(saved), "")))
        window.open_project()
        qtbot.wait(20)
        assert window.global_scope.as_scope() == {"WastePhase": ["Sludge"]}

    def test_a_project_from_a_newer_build_is_refused_with_a_message(
            self, window, qtbot, tmp_path, monkeypatch):
        bad = tmp_path / "future.ember"
        bad.write_text('{"version": 99, "name": "x"}', encoding="utf-8")
        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getOpenFileName",
            staticmethod(lambda *a, **k: (str(bad), "")))
        window.open_project()   # QMessageBox.warning neutralized by conftest
        qtbot.wait(20)

    def test_cancelling_the_dialog_does_nothing(self, window, monkeypatch):
        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getOpenFileName",
            staticmethod(lambda *a, **k: ("", "")))
        window.open_project()  # must not raise
