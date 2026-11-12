"""
Qt-level tests for the Cross-correlation sub-tab (qt_correlations_association)
and its wiring into the Correlations page and the shell's scope bar.

The science is association_query's and is tested there. What is tested
here is the UI's own contract: every table is filled, the DRAWN figure
agrees with the table beside it, nothing leaves the figure without being
counted on it, the global scope reaches the engine, and results come back
through bound methods (workers are synchronous under pytest, see conftest).
"""
import re

import numpy as np
import pandas as pd
import polars as pl
import pytest
from PySide6.QtWidgets import QMainWindow, QMessageBox

import association_query as aq
import data_model as dm
import plot_helpers as ph
import qt_correlations_association as qca
from data_model import HanfordDataset
from qt_correlations import CorrelationsPage
from qt_correlations_association import (
    ALL_FAMILIES, DEFAULT_COMPONENTS, DEFAULT_PRESET, PLOT_MATRIX, PLOT_SCATTER,
    TABLE_TITLES, T_COOCCURRENCE, T_COUNTS, T_MATRIX, T_NOTES, T_PAIRS,
    T_QUALITY, T_VALUES, AssociationTab, parse_components,
)
from qt_scope import GlobalScope
from query_engine import BASKET_PRESETS

COMPONENTS = "F, Cl, SO4, Cr, 137Cs"
PHASES = ["Supernatant", "Saltcake Solid", "Sludge Solid"]


@pytest.fixture
def association_dataset() -> HanfordDataset:
    """14 single-layer tanks in two farms (A x 8, AN x 6), kg rows for Na,
    F, Cl, SO4, Cr plus a curie-only nuclide (137Cs), built so that every
    behavior the tab has to show is present in the data:

    - F rises and Cl falls along the series (a positive and a negative pair);
    - two layers report a ZERO chloride (kept by the rank methods, no
      position on a log axis);
    - one layer does not report sulfate at all (a blank, not a zero);
    - tank 241-AN-106 is 241-AN-105's composition written into three times
      the volume (one composition, two layers).
    """
    rng = np.random.default_rng(7)
    tanks = [f"241-A-{101 + i}" for i in range(8)] + [f"241-AN-{101 + i}" for i in range(6)]
    rows = {"WasteSiteId": [], "Analyte": [], "WastePhase": [], "WasteType": [], "Inventory": [], "Units": []}

    def add(tank, phase, analyte, value, units="kg"):
        rows["WasteSiteId"].append(tank)
        rows["Analyte"].append(analyte)
        rows["WastePhase"].append(phase)
        rows["WasteType"].append("T1")
        rows["Inventory"].append(float(value))
        rows["Units"].append(units)

    previous = None
    for i, tank in enumerate(tanks):
        phase = PHASES[i % 3]
        if tank == "241-AN-106":
            composition = {k: 3.0 * v for k, v in previous.items()}
        else:
            na = 1000.0 + 150.0 * i
            composition = {
                "Na": na,
                "F": na * (0.002 + 0.0015 * i) * (1 + 0.2 * rng.random()),
                "Cl": 0.0 if i in (2, 9) else na * (0.03 - 0.0018 * i) * (1 + 0.2 * rng.random()),
                "SO4": na * 0.01 * (1 + rng.random()),
                "Cr": na * 0.004 * (1 + rng.random()),
                "137Cs": 50.0 + 400.0 * rng.random(),
            }
            if i == 5:
                del composition["SO4"]
        previous = composition
        for analyte, value in composition.items():
            add(tank, phase, analyte, value, "Ci" if analyte == "137Cs" else "kg")

    dataset = HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame(rows))
    dataset.report = None
    return dataset


def _make_tab(qtbot, scope=None):
    app_window = QMainWindow()
    qtbot.addWidget(app_window)
    tab = AssociationTab(app_window, scope=scope)
    qtbot.addWidget(tab)
    return tab


def _run(tab, qtbot, dataset, components=COMPONENTS, **settings):
    tab.on_dataset_changed(dataset)
    tab.components_edit.setText(components)
    tab.min_n_spin.setValue(settings.pop("min_n", 4))
    if "basis" in settings:
        tab.basis_combo.setCurrentText(settings.pop("basis"))
    if "level" in settings:
        tab.level_combo.setCurrentIndex(tab.level_combo.findData(settings.pop("level")))
    if "method" in settings:
        index = tab.method_combo.findData(settings.pop("method"))
        assert index >= 0
        tab.method_combo.setCurrentIndex(index)
    if "family" in settings:
        tab.family_combo.setCurrentText(settings.pop("family"))
    if "distinct" in settings:
        tab.distinct_check.setChecked(settings.pop("distinct"))
    assert not settings, f"unknown settings {settings}"
    tab.run()
    qtbot.wait(20)
    return tab


def _table(tab, title):
    return tab._table_views[title].dataframe()


def _cell_texts(tab):
    """{(row component, column component): drawn text} for the heatmap."""
    out = {}
    for text in tab.plot.ax.texts:
        gid = text.get_gid() or ""
        if gid.startswith("cell:"):
            out[tuple(gid[len("cell:"):].split("|"))] = text
    return out


def _capture(monkeypatch, name):
    calls = []
    monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: calls.append(a) or QMessageBox.Ok))
    return calls


# ----------------------------------------------------------------------
class TestConstruction:
    def test_defaults(self, qtbot):
        tab = _make_tab(qtbot)
        assert tab.components_edit.text() == DEFAULT_COMPONENTS
        assert tab.components_edit.text() == "F, Cl, 129I, SO4, PO4, Cr"
        assert tab.basis_combo.currentText() == aq.MOL_PER_MOL_NA
        assert [tab.basis_combo.itemText(i) for i in range(tab.basis_combo.count())] == aq.BASES
        assert tab.level_combo.currentData() == "layer"
        assert [tab.level_combo.itemData(i) for i in range(tab.level_combo.count())] == ["layer", "tank"]
        assert tab.run_btn.objectName() == "Primary" and tab.run_btn.text() == "Run"
        # The notes slot is the shared calm banner, hidden while there is nothing to say.
        assert tab.warnings_label is tab.notes_banner
        assert tab.warnings_label.objectName() == "NotesBanner"
        assert tab.warnings_label.isHidden()
        assert not tab.advanced_section.is_expanded()
        # Methods read as words; the engine's identifier rides as item data.
        assert [tab.method_combo.itemData(i) for i in range(tab.method_combo.count())] == aq.METHODS
        assert [tab.method_combo.itemText(i) for i in range(tab.method_combo.count())] == [
            "Spearman", "Pearson on log10 values", "Kendall"]
        assert [tab.family_combo.itemText(i) for i in range(tab.family_combo.count())] == (
            [ALL_FAMILIES] + aq.PHASE_FAMILIES)
        assert tab.distinct_check.isChecked()
        assert tab.min_n_spin.value() == 8
        assert tab.quantile_spin.value() == pytest.approx(0.75)
        assert [tab.plot_combo.itemText(i) for i in range(tab.plot_combo.count())] == [PLOT_MATRIX, PLOT_SCATTER]

    def test_table_tabs_in_reading_order(self, qtbot):
        tab = _make_tab(qtbot)
        titles = [tab.tables.tabText(i) for i in range(tab.tables.count())]
        assert titles == TABLE_TITLES == [
            "Pairs", "Correlation matrix", "N per pair", "What each component rests on",
            "Co-occurrence", "Values", "Notes",
        ]

    def test_presets_start_with_the_default_list_then_the_engine_baskets(self, qtbot):
        tab = _make_tab(qtbot)
        names = [tab.preset_combo.itemText(i) for i in range(tab.preset_combo.count())]
        assert names == [DEFAULT_PRESET] + list(BASKET_PRESETS)
        assert names[0] == "Salt formers (as reported)"

    def test_choosing_a_preset_fills_the_component_list(self, qtbot):
        tab = _make_tab(qtbot)
        index = tab.preset_combo.findText("Halides")
        tab.preset_combo.setCurrentIndex(index)
        tab.preset_combo.activated.emit(index)
        assert parse_components(tab.components_edit.text()) == list(BASKET_PRESETS["Halides"].members)
        tab.preset_combo.setCurrentIndex(0)
        tab.preset_combo.activated.emit(0)
        assert tab.components_edit.text() == DEFAULT_COMPONENTS

    def test_distinct_count_is_disabled_and_explained_on_the_kg_basis(self, qtbot):
        tab = _make_tab(qtbot)
        assert tab.distinct_check.isEnabled()
        tab.basis_combo.setCurrentText("kg")
        assert not tab.distinct_check.isEnabled()
        assert "kg" in tab.distinct_check.toolTip()
        tab.basis_combo.setCurrentText("wt%")
        assert tab.distinct_check.isEnabled()

    def test_phase_family_is_disabled_at_the_tank_level(self, qtbot):
        tab = _make_tab(qtbot)
        assert tab.family_combo.isEnabled()
        tab.level_combo.setCurrentIndex(tab.level_combo.findData("tank"))
        assert not tab.family_combo.isEnabled()
        assert "tank" in tab.family_combo.toolTip().lower()
        tab.level_combo.setCurrentIndex(tab.level_combo.findData("layer"))
        assert tab.family_combo.isEnabled()

    def test_actions_without_dataset_show_message_not_crash(self, qtbot, monkeypatch):
        info = _capture(monkeypatch, "information")
        tab = _make_tab(qtbot)
        status = tab.app_window.statusBar()
        tab.run()
        assert "no dataset is loaded" in status.currentMessage()
        tab.export_tables()
        assert "nothing to export" in status.currentMessage()
        tab.on_scope_changed()
        tab.plot_combo.setCurrentText(PLOT_SCATTER)
        # Said in the status bar: nothing here needs a dialog to be dismissed.
        assert not info
        assert tab._result is None

    def test_parse_components(self):
        assert parse_components(" F, Cl ,,129I; SO4\nF ") == ["F", "Cl", "129I", "SO4"]
        assert parse_components("") == []


# ----------------------------------------------------------------------
class TestRun:
    def test_run_fills_every_table_and_draws(self, qtbot, association_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset)
        for title in TABLE_TITLES:
            assert not _table(tab, title).empty, f"{title} is empty after a run"

        members = ["F", "Cl", "SO4", "Cr", "137Cs"]
        pairs = _table(tab, T_PAIRS)
        assert len(pairs) == 10                                   # 5 choose 2
        assert {"A", "B", "r", "p_value", "N", "N_both_positive"} <= set(pairs.columns)
        assert list(_table(tab, T_MATRIX).columns) == ["Component"] + members
        assert list(_table(tab, T_MATRIX)["Component"]) == members
        assert list(_table(tab, T_COUNTS)["Component"]) == members
        assert list(_table(tab, T_QUALITY)["Member"]) == members
        assert len(_table(tab, T_VALUES)) == 14                   # every layer, replicate included
        assert len(_table(tab, T_COOCCURRENCE)) == 14
        assert {"N_high", "High_members"} <= set(_table(tab, T_COOCCURRENCE).columns)

        # The figure is the matrix, not the placeholder message.
        assert len(tab.plot.ax.images) == 1
        assert len(_cell_texts(tab)) == 10
        assert tab.run_btn.isEnabled() and tab.run_btn.text() == "Run"

    def test_tables_are_the_engines_own_objects(self, qtbot, association_dataset):
        """The tab displays what association_query returns; it computes no
        numbers of its own."""
        tab = _run(_make_tab(qtbot), qtbot, association_dataset)
        members = ["F", "Cl", "SO4", "Cr", "137Cs"]
        matrix = aq.association_matrix(association_dataset, members)
        result = aq.correlate(matrix, min_n=4)
        pd.testing.assert_frame_equal(_table(tab, T_PAIRS), result.pairs)
        pd.testing.assert_frame_equal(_table(tab, T_VALUES), matrix.values)
        pd.testing.assert_frame_equal(_table(tab, T_MATRIX).set_index("Component").rename_axis(None),
                                      result.correlation)
        pd.testing.assert_frame_equal(_table(tab, T_COUNTS).set_index("Component").rename_axis(None),
                                      result.n_used)
        pd.testing.assert_frame_equal(_table(tab, T_QUALITY), aq.member_quality(association_dataset, members))
        pd.testing.assert_frame_equal(_table(tab, T_COOCCURRENCE), aq.co_occurrence(matrix, quantile=0.75))

    def test_notes_carry_warnings_disclosures_and_result_notes(self, qtbot, association_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset)
        notes = _table(tab, T_NOTES)
        assert list(notes.columns) == ["kind", "text"]
        matrix = aq.association_matrix(association_dataset, parse_components(COMPONENTS))
        result = aq.correlate(matrix, min_n=4)
        texts = set(notes["text"])
        assert set(matrix.warnings) <= texts
        assert set(matrix.disclosures) <= texts
        assert set(result.notes) <= texts
        assert result.notes, "fixture should produce a replicate-composition note"
        # The banner stays one calm line: what bears on this matrix in
        # full, the dataset-wide notes as a count. All of them remain in
        # the tooltip and in Notes (asserted above).
        for warning in matrix.warnings:
            assert warning in tab.warnings_label.toolTip()
        assert tab.warnings_label.isVisibleTo(tab) == bool(matrix.warnings)
        scope_rows = notes[notes["kind"] == "scope"]
        assert len(scope_rows) == 1 and "none" in scope_rows["text"].iloc[0]

    def test_engine_refusal_shows_the_message_without_raising(self, qtbot, association_dataset, monkeypatch):
        warnings = _capture(monkeypatch, "warning")
        criticals = _capture(monkeypatch, "critical")
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(association_dataset)
        tab.components_edit.setText("F")            # one component only
        tab.run()
        qtbot.wait(20)
        assert len(warnings) == 1 and not criticals
        with pytest.raises(ValueError) as refusal:
            aq.association_matrix(association_dataset, ["F"])
        assert warnings[0][2] == str(refusal.value)   # the engine's sentence, no traceback
        assert "Traceback" not in warnings[0][2]
        assert tab._result is None
        assert all(_table(tab, title).empty for title in TABLE_TITLES)
        assert tab.run_btn.isEnabled() and tab.run_btn.text() == "Run"

    def test_refusal_keeps_the_standing_result_and_its_labels(self, qtbot, association_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset)
        before = _table(tab, T_PAIRS).copy()
        title_before = tab.plot.figure.get_suptitle()
        tab.components_edit.setText("F")
        tab.basis_combo.setCurrentText("wt%")
        tab.run()
        qtbot.wait(20)
        pd.testing.assert_frame_equal(_table(tab, T_PAIRS), before)
        assert tab.plot.figure.get_suptitle() == title_before
        assert tab._params["basis"] == aq.MOL_PER_MOL_NA

    def test_unexpected_failure_is_reported_not_raised(self, qtbot, association_dataset, monkeypatch):
        criticals = _capture(monkeypatch, "critical")

        def boom(*args, **kwargs):
            raise RuntimeError("engine fell over")

        monkeypatch.setattr(qca.aq, "association_matrix", boom)
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(association_dataset)
        tab.run()
        qtbot.wait(20)
        assert len(criticals) == 1 and "engine fell over" in criticals[0][2]
        assert tab.run_btn.isEnabled()

    def test_unknown_component_is_named_in_the_warnings(self, qtbot, association_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset, components="F, Cl, Zz")
        assert "Zz" in tab.warnings_label.text()
        notes = _table(tab, T_NOTES)
        assert notes[(notes["kind"] == "warning") & notes["text"].str.contains("Zz")].shape[0] >= 1

    def test_scatter_rows_are_the_rows_the_engine_counted(self, qtbot, association_dataset, monkeypatch):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset)
        assert len(tab._used) == tab._result.n_distinct == 13
        assert "Pair scatter shows" not in tab.warnings_label.text()
        tab = _run(tab, qtbot, association_dataset, distinct=False, family="Sludge", min_n=3)
        assert len(tab._used) == tab._result.n_rows
        assert set(tab._used["PhaseFamily"]) == {"Sludge"}
        # If the UI's selection ever drifts from the engine's, it says so.
        monkeypatch.setattr(qca, "counted_rows", lambda matrix, family, distinct: matrix.values.head(3))
        tab.run()
        qtbot.wait(20)
        assert "Pair scatter shows 3 observations" in tab.warnings_label.text()

    def test_dataset_change_clears_the_previous_result(self, qtbot, association_dataset, sample_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset)
        tab.on_dataset_changed(sample_dataset)
        qtbot.wait(20)
        assert tab._result is None and tab._params == {}
        assert all(_table(tab, title).empty for title in TABLE_TITLES)
        assert not tab.plot.ax.images
        assert not tab.warnings_label.isVisibleTo(tab)

    def test_tank_level_and_kg_basis_run(self, qtbot, association_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset, basis="kg", level="tank")
        values = _table(tab, T_VALUES)
        assert "PhaseFamily" not in values.columns and len(values) == 14
        assert "14 tanks, every one counted" in tab.plot.ax.get_title()
        assert "Phase family" not in tab.plot.ax.get_title()
        assert any("inventory basis" in t for t in _table(tab, T_NOTES)["text"])


# ----------------------------------------------------------------------
class TestWorkerContract:
    def test_results_are_delivered_to_bound_methods_of_the_tab(self, qtbot, association_dataset, monkeypatch):
        """A signal connected to a plain function or lambda runs on the
        worker thread, where touching widgets hangs. The callbacks handed
        to the worker must be bound methods of the (UI-thread) tab."""
        seen = {}

        def fake_run_in_thread(fn, on_done, on_error=None, *args, **kwargs):
            seen.update(fn=fn, on_done=on_done, on_error=on_error)

        monkeypatch.setattr(qca, "run_in_thread", fake_run_in_thread)
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(association_dataset)
        tab.run()
        assert seen["on_done"].__self__ is tab and seen["on_error"].__self__ is tab
        # While the worker is out, Run is disabled and says so.
        assert not tab.run_btn.isEnabled() and "Running" in tab.run_btn.text()
        outcome = seen["fn"]()                     # what the worker thread would execute
        assert outcome.token == tab._run_token and outcome.result is not None
        tab._on_run_done(outcome)
        qtbot.wait(20)
        assert tab.run_btn.isEnabled() and not _table(tab, T_PAIRS).empty

    def test_a_superseded_answer_is_dropped(self, qtbot, association_dataset, monkeypatch):
        pending = []
        monkeypatch.setattr(qca, "run_in_thread",
                            lambda fn, on_done, on_error=None, *a, **k: pending.append(fn))
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(association_dataset)
        tab.components_edit.setText("F, Cl")
        tab.run()
        tab.components_edit.setText("F, Cr, SO4")
        tab._start(tab._read_params())
        first, second = pending[0](), pending[1]()
        tab._on_run_done(second)
        tab._on_run_done(first)                    # arrives late: must not overwrite
        qtbot.wait(20)
        assert list(_table(tab, T_MATRIX)["Component"]) == ["F", "Cr", "SO4"]

    def test_an_answer_for_the_previous_dataset_is_dropped(self, qtbot, association_dataset, sample_dataset,
                                                           monkeypatch):
        pending = []
        monkeypatch.setattr(qca, "run_in_thread",
                            lambda fn, on_done, on_error=None, *a, **k: pending.append(fn))
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(association_dataset)
        tab.run()
        tab.on_dataset_changed(sample_dataset)
        tab._on_run_done(pending[0]())
        qtbot.wait(20)
        assert tab._result is None and _table(tab, T_PAIRS).empty
        assert tab.run_btn.isEnabled()


# ----------------------------------------------------------------------
class TestHeatmapAgreesWithTheTable:
    def test_drawn_cell_texts_equal_the_correlation_table(self, qtbot, association_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset)
        table = _table(tab, T_MATRIX).set_index("Component")
        names = list(table.index)
        cells = _cell_texts(tab)
        expected = {
            (names[i], names[j]) for i in range(len(names)) for j in range(i)
            if np.isfinite(table.iloc[i, j])
        }
        assert set(cells) == expected and len(expected) == 10
        xlabels = [t.get_text() for t in tab.plot.ax.get_xticklabels()]
        ylabels = [t.get_text() for t in tab.plot.ax.get_yticklabels()]
        assert xlabels == names[:-1] and ylabels == names[1:]
        image = tab.plot.ax.images[0].get_array()
        for (row, col), text in cells.items():
            value = float(table.loc[row, col])
            assert float(text.get_text()) == pytest.approx(value, abs=0.005 + 1e-9)
            # ...and it is written IN the cell the axis labels name.
            x, y = text.get_position()
            assert xlabels[int(x)] == col and ylabels[int(y)] == row
            assert float(image[int(y), int(x)]) == pytest.approx(value)
        # The coefficients are not all alike (a constant fixture would pass trivially).
        assert len({t.get_text() for t in cells.values()}) > 3
        assert any(float(t.get_text()) < 0 for t in cells.values())
        assert any(float(t.get_text()) > 0 for t in cells.values())

    def test_pairs_left_blank_are_hatched_and_carry_no_number(self, qtbot, association_dataset):
        # 137Cs/SO4 share 13 layers at most; a floor of 14 blanks every SO4
        # pair (one layer does not report sulfate) and leaves the rest.
        tab = _run(_make_tab(qtbot), qtbot, association_dataset, distinct=False, min_n=14)
        table = _table(tab, T_MATRIX).set_index("Component")
        names = list(table.index)
        blank = {(names[i], names[j]) for i in range(len(names)) for j in range(i)
                 if not np.isfinite(table.iloc[i, j])}
        assert blank and len(blank) < 10
        hatched = {tuple(p.get_gid()[len("blank:"):].split("|"))
                   for p in tab.plot.ax.patches if (p.get_gid() or "").startswith("blank:")}
        assert hatched == blank
        assert not (set(_cell_texts(tab)) & blank)
        assert f"{len(blank)} pair(s) left blank" in tab.plot.ax.get_xlabel()

    def test_subtitle_wraps_between_statements_and_loses_none(self):
        parts = ["Level: waste layer", "Phase family: all layers",
                 "n = 393 distinct compositions of 557 layers", "Scope: AN"]
        wrapped = ph.wrap_subtitle(parts, width=50)
        assert all(len(line) <= 50 for line in wrapped.split("\n")) and "\n" in wrapped
        assert wrapped.replace("\n", ph.SUBTITLE_SEP) == ph.SUBTITLE_SEP.join(parts)
        assert ph.wrap_subtitle(parts, width=500) == ph.SUBTITLE_SEP.join(parts)

    def test_colormap_is_diverging_with_a_neutral_gray_midpoint(self):
        cmap = ph.association_cmap()
        red, green, blue, _ = cmap(0.5)
        assert max(red, green, blue) - min(red, green, blue) < 0.03     # no hue at zero
        assert min(red, green, blue) > 0.85                              # and light
        neg, pos = cmap(0.0), cmap(1.0)
        assert neg[2] > neg[0] + 0.3                                     # blue pole
        assert pos[0] > pos[2] + 0.3                                     # red pole
        # Each arm darkens monotonically away from the midpoint.
        def luminance(c):
            return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
        up = [luminance(cmap(v)) for v in np.linspace(0.5, 1.0, 26)]
        down = [luminance(cmap(v)) for v in np.linspace(0.5, 0.0, 26)]
        assert all(b < a for a, b in zip(up, up[1:]))
        assert all(b < a for a, b in zip(down, down[1:]))

    def test_color_scale_is_fixed_and_centered_on_zero(self, qtbot, association_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset)
        image = tab.plot.ax.images[0]
        assert image.get_clim() == (-1.0, 1.0)
        assert image.get_cmap().name == "ember_association"

    def test_title_and_subtitle_state_what_was_computed(self, qtbot, association_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset)
        title = tab.plot.figure.get_suptitle()
        subtitle = tab.plot.ax.get_title()
        assert "Spearman" in title and "mol per mol Na" in title
        assert "Level: waste layer" in subtitle
        assert "Phase family: all layers" in subtitle
        # 14 layers, one of them a second copy of a composition already counted.
        assert tab._result.n_rows == 14 and tab._result.n_distinct == 13
        assert "n = 13 distinct compositions of 14 layers" in subtitle
        assert "Scope" not in subtitle
        assert tab.plot.suggested_caption() == title

    def test_title_follows_method_basis_and_family(self, qtbot, association_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset, basis="wt%", method="kendall",
                   family="Saltcake", min_n=3)
        assert "Kendall" in tab.plot.figure.get_suptitle()
        assert aq.BASIS_LABELS["wt%"] in tab.plot.figure.get_suptitle()
        subtitle = tab.plot.ax.get_title()
        assert "Phase family: Saltcake" in subtitle
        n_saltcake = int((_table(tab, T_VALUES)["PhaseFamily"] == "Saltcake").sum())
        assert tab._result.n_rows == n_saltcake
        assert f"of {n_saltcake} Saltcake layers" in subtitle

    def test_labels_describe_the_run_not_the_live_controls(self, qtbot, association_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset)
        tab.basis_combo.setCurrentText("kg")
        tab.method_combo.setCurrentIndex(tab.method_combo.findData("kendall"))
        assert tab.run_btn.text() == "Update"             # the controls left the result
        tab.plot_combo.setCurrentText(PLOT_SCATTER)       # forces a redraw
        tab.plot_combo.setCurrentText(PLOT_MATRIX)
        qtbot.wait(20)
        assert "Spearman" in tab.plot.figure.get_suptitle()
        assert "mol per mol Na" in tab.plot.figure.get_suptitle()

    def test_send_to_figure_composer_replays_the_same_drawing(self, qtbot, association_dataset):
        from matplotlib.figure import Figure
        from figure_composer import ComposerCellPanel

        tab = _run(_make_tab(qtbot), qtbot, association_dataset)
        render_fn, args, kwargs = tab.plot._last_recipe
        cell = ComposerCellPanel(Figure())
        render_fn(cell, *args, **kwargs)
        replayed = {t.get_gid(): t.get_text() for t in cell.ax.texts if (t.get_gid() or "").startswith("cell:")}
        shown = {t.get_gid(): t.get_text() for t in tab.plot.ax.texts if (t.get_gid() or "").startswith("cell:")}
        assert replayed == shown and len(shown) == 10


# ----------------------------------------------------------------------
def _left_off(subtitle: str) -> int:
    found = re.search(r"(\d+) left off", subtitle)
    return int(found.group(1)) if found else 0


def _drawn(subtitle: str):
    found = re.search(r"(\d+) of (\d+) [\w ]+ drawn", subtitle)
    return int(found.group(1)), int(found.group(2))


def _n_points(ax) -> int:
    return int(sum(len(c.get_offsets()) for c in ax.collections))


class TestPairScatter:
    def _select(self, tab, a, b):
        shown = _table(tab, T_PAIRS)
        hits = shown.index[((shown["A"] == a) & (shown["B"] == b)) | ((shown["A"] == b) & (shown["B"] == a))]
        view = tab._table_views[T_PAIRS]
        view.setCurrentIndex(view.model().index(int(hits[0]), 0))

    def test_left_off_count_equals_the_non_positive_rows(self, qtbot, association_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset, distinct=False)
        tab.plot_combo.setCurrentText(PLOT_SCATTER)
        self._select(tab, "F", "Cl")
        qtbot.wait(20)
        values = _table(tab, T_VALUES)
        both = values.dropna(subset=["F", "Cl"])
        non_positive = int(((both["F"] <= 0) | (both["Cl"] <= 0)).sum())
        assert non_positive == 2, "fixture should carry two zero-chloride layers"
        subtitle = tab.plot.ax.get_title()
        assert _left_off(subtitle) == non_positive
        drawn, offered = _drawn(subtitle)
        assert offered == len(values) == 14
        assert drawn == len(both) - non_positive == _n_points(tab.plot.ax)
        assert tab.plot.ax.get_xscale() == "log" and tab.plot.ax.get_yscale() == "log"
        # Every drawn point is one of the table's positive rows.
        drawn_xy = sorted(tuple(np.round(p, 12)) for c in tab.plot.ax.collections for p in c.get_offsets())
        positive = both[(both["F"] > 0) & (both["Cl"] > 0)]
        assert drawn_xy == sorted(tuple(np.round(p, 12)) for p in positive[["F", "Cl"]].to_numpy())

    def test_rows_not_reporting_both_are_counted_too(self, qtbot, association_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset, distinct=False)
        tab.plot_combo.setCurrentText(PLOT_SCATTER)
        self._select(tab, "SO4", "Cr")
        qtbot.wait(20)
        subtitle = tab.plot.ax.get_title()
        assert "1 do not report both" in subtitle
        assert _left_off(subtitle) == 0
        assert _drawn(subtitle) == (13, 14)

    def test_scatter_draws_the_observations_the_coefficient_used(self, qtbot, association_dataset):
        """With distinct counting on, the replicate layer is not a second
        observation, so it is not a second point either."""
        tab = _run(_make_tab(qtbot), qtbot, association_dataset)
        tab.plot_combo.setCurrentText(PLOT_SCATTER)
        self._select(tab, "F", "Cr")
        qtbot.wait(20)
        subtitle = tab.plot.ax.get_title()
        assert _drawn(subtitle) == (13, 13)
        assert "distinct compositions drawn" in subtitle
        pairs = tab._result.pairs
        row = pairs[((pairs["A"] == "F") & (pairs["B"] == "Cr")) | ((pairs["A"] == "Cr") & (pairs["B"] == "F"))].iloc[0]
        assert int(row["N"]) == 13 == _n_points(tab.plot.ax)
        title = tab.plot.figure.get_suptitle()
        assert f"= {row['r']:.2f}" in title and "N = 13" in title

    def test_top_pair_is_drawn_when_nothing_is_selected(self, qtbot, association_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset)
        tab.plot_combo.setCurrentText(PLOT_SCATTER)
        qtbot.wait(20)
        top = tab._result.pairs.iloc[0]
        assert tab.selected_pair() == (top["A"], top["B"])
        assert tab.plot.figure.get_suptitle().startswith(f"{top['A']} vs {top['B']}")
        assert tab.plot.ax.get_xlabel().startswith(top["A"])
        assert tab.plot.ax.get_ylabel().startswith(top["B"])
        assert "mol per mol Na" in tab.plot.ax.get_xlabel()

    def test_selecting_a_pair_redraws_it_even_after_the_table_is_sorted(self, qtbot, association_dataset):
        from PySide6.QtCore import Qt
        tab = _run(_make_tab(qtbot), qtbot, association_dataset)
        tab.plot_combo.setCurrentText(PLOT_SCATTER)
        view = tab._table_views[T_PAIRS]
        view.sortByColumn(list(_table(tab, T_PAIRS).columns).index("r"), Qt.AscendingOrder)
        shown = _table(tab, T_PAIRS)
        view.setCurrentIndex(view.model().index(0, 0))
        qtbot.wait(20)
        a, b = shown.iloc[0]["A"], shown.iloc[0]["B"]
        assert tab.plot.figure.get_suptitle().startswith(f"{a} vs {b}")
        assert float(shown.iloc[0]["r"]) == pytest.approx(float(shown["r"].min()))

    def test_layers_are_colored_by_phase_family(self, qtbot, association_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset)
        tab.plot_combo.setCurrentText(PLOT_SCATTER)
        qtbot.wait(20)
        legend = tab.plot.ax.get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        assert [label.split(" (")[0] for label in labels] == aq.PHASE_FAMILIES
        assert sum(int(label.split("(")[1].rstrip(")")) for label in labels) == _n_points(tab.plot.ax)

    def test_tanks_have_no_phase_family_to_color_by(self, qtbot, association_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset, level="tank")
        tab.plot_combo.setCurrentText(PLOT_SCATTER)
        qtbot.wait(20)
        assert tab.plot.ax.get_legend() is None
        assert "Phase family" not in tab.plot.ax.get_title()

    def test_nothing_positive_says_so_with_the_counts(self, qtbot):
        from qt_widgets import PlotWidget
        panel = PlotWidget()
        qtbot.addWidget(panel)
        values = pd.DataFrame({"A": [0.0, 0.0, np.nan], "B": [1.0, 2.0, 3.0]})
        ph.plot_association_scatter(panel, values, "A", "B", unit_word="layer")
        qtbot.wait(20)
        message = panel.ax.texts[0].get_text()
        assert "0 of 3 layers drawn" in message and "2 left off" in message and "1 do not report both" in message
        assert ph.association_scatter_counts(values, "A", "B") == {
            "offered": 3, "both": 2, "drawn": 0, "left_off": 2, "missing": 1}


# ----------------------------------------------------------------------
class TestScope:
    def test_scope_reaches_the_engine_and_is_stated(self, qtbot, association_dataset):
        scope = GlobalScope()
        tab = _run(_make_tab(qtbot, scope=scope), qtbot, association_dataset)
        assert tab.scope is scope
        assert len(_table(tab, T_VALUES)) == 14

        scope.set_filter("TankFarm", ["AN"])
        tab.on_scope_changed()
        qtbot.wait(20)
        values = _table(tab, T_VALUES)
        assert len(values) == 6 and set(values["TankFarm"]) == {"AN"}
        expected = aq.correlate(
            aq.association_matrix(association_dataset, parse_components(COMPONENTS), scope={"TankFarm": ["AN"]}),
            min_n=4)
        pd.testing.assert_frame_equal(_table(tab, T_PAIRS), expected.pairs)
        assert "Scope: AN" in tab.plot.ax.get_title()
        notes = _table(tab, T_NOTES)
        assert list(notes[notes["kind"] == "scope"]["text"]) == ["Scope: AN"]
        assert set(_table(tab, T_QUALITY)["N_tanks_in_scope"]) == {6}

        scope.clear()
        tab.on_scope_changed()
        qtbot.wait(20)
        assert len(_table(tab, T_VALUES)) == 14
        assert "Scope" not in tab.plot.ax.get_title()

    def test_scope_change_re_asks_the_standing_question_not_the_live_controls(self, qtbot, association_dataset):
        scope = GlobalScope()
        tab = _run(_make_tab(qtbot, scope=scope), qtbot, association_dataset)
        tab.components_edit.setText("F, Cl")       # edited, not run
        tab.basis_combo.setCurrentText("kg")
        scope.set_filter("TankFarm", ["A"])
        tab.on_scope_changed()
        qtbot.wait(20)
        assert list(_table(tab, T_MATRIX)["Component"]) == ["F", "Cl", "SO4", "Cr", "137Cs"]
        assert tab._params["basis"] == aq.MOL_PER_MOL_NA and tab._params["scope"] == "A"
        assert len(_table(tab, T_VALUES)) == 8

    def test_scope_change_before_any_run_computes_nothing(self, qtbot, association_dataset, monkeypatch):
        started = []
        monkeypatch.setattr(qca, "run_in_thread", lambda *a, **k: started.append(a))
        scope = GlobalScope()
        tab = _make_tab(qtbot, scope=scope)
        tab.on_dataset_changed(association_dataset)
        scope.set_filter("TankFarm", ["AN"])
        tab.on_scope_changed()
        assert not started

    def test_a_scope_matching_nothing_empties_the_view_and_recovers(self, qtbot, association_dataset):
        scope = GlobalScope()
        tab = _run(_make_tab(qtbot, scope=scope), qtbot, association_dataset)
        scope.set_filter("TankFarm", ["ZZ"])
        tab.on_scope_changed()
        qtbot.wait(20)
        assert _table(tab, T_VALUES).empty
        assert not tab.plot.ax.images, "the previous figure was left standing beside empty tables"
        assert "nothing to correlate" in tab.warnings_label.text()
        scope.set_filter("TankFarm", ["AN"])
        tab.on_scope_changed()
        qtbot.wait(20)
        assert len(_table(tab, T_VALUES)) == 6 and len(tab.plot.ax.images) == 1

    def test_export_states_the_scope_and_the_run(self, qtbot, association_dataset, tmp_path):
        scope = GlobalScope()
        association_dataset.output_root = tmp_path
        tab = _run(_make_tab(qtbot, scope=scope), qtbot, association_dataset)
        scope.set_filter("TankFarm", ["AN"])
        tab.on_scope_changed()
        qtbot.wait(20)
        tab.export_tables()
        bundles = list(tmp_path.glob("cross_correlation_*"))
        assert len(bundles) == 1
        for name in ("association_pairs", "association_correlation_matrix", "association_n_per_pair",
                     "association_component_quality", "association_co_occurrence", "association_values",
                     "association_notes", "association_parameters", "context", "manifest"):
            assert (bundles[0] / f"{name}.csv").exists(), name
        assert (bundles[0] / "cross_correlation_plot.png").exists()
        context = pd.read_csv(bundles[0] / "context.csv").set_index("key")["value"]
        assert context["global_scope"] == "AN"
        assert context["basis"] == aq.MOL_PER_MOL_NA and context["method"] == "spearman"
        exported = pd.read_csv(bundles[0] / "association_correlation_matrix.csv")
        assert list(exported["Component"]) == ["F", "Cl", "SO4", "Cr", "137Cs"]
        pd.testing.assert_frame_equal(exported, _table(tab, T_MATRIX), check_dtype=False)


# ----------------------------------------------------------------------
class TestCorrelationsPageWiring:
    def _page(self, qtbot, scope=None):
        app_window = QMainWindow()
        qtbot.addWidget(app_window)
        page = CorrelationsPage(app_window, scope=scope)
        qtbot.addWidget(page)
        return page

    def test_cross_correlation_is_the_first_tab(self, qtbot):
        page = self._page(qtbot)
        assert page.tabs.tabText(0) == "Cross-correlation"
        assert page.tabs.widget(0) is page.association_tab
        assert page.tabs.currentWidget() is page.association_tab

    def test_dataset_and_scope_are_forwarded(self, qtbot, association_dataset):
        scope = GlobalScope()
        page = self._page(qtbot, scope=scope)
        assert page.association_tab.scope is scope
        page.on_dataset_changed(association_dataset)
        assert page.association_tab.dataset is association_dataset
        tab = page.association_tab
        tab.components_edit.setText(COMPONENTS)
        tab.min_n_spin.setValue(4)
        tab.run()
        qtbot.wait(20)
        scope.set_filter("TankFarm", ["A"])
        page.on_scope_changed()
        qtbot.wait(20)
        assert set(_table(tab, T_VALUES)["TankFarm"]) == {"A"}

    def test_legacy_tabs_are_not_told_about_the_scope(self, qtbot, association_dataset):
        """They compute on the whole dataset; nothing on them may claim a
        scope, and a scope change must not disturb them."""
        scope = GlobalScope()
        page = self._page(qtbot, scope=scope)
        page.on_dataset_changed(association_dataset)
        for legacy in (page.quick_scan_tab, page.workbench_tab, page.structure_tab):
            assert not hasattr(legacy, "on_scope_changed")
            assert not hasattr(legacy, "scope")
        scope.set_filter("TankFarm", ["A"])
        page.on_scope_changed()                    # must not raise on the legacy tabs
        qtbot.wait(20)
        assert page.quick_scan_tab.dataset is association_dataset

    def test_scope_applies_only_while_the_scoped_tab_is_showing(self, qtbot):
        page = self._page(qtbot)
        changes = []
        page.scope_relevance_changed.connect(lambda: changes.append(page.scope_applies()))
        assert page.scope_applies()
        page.tabs.setCurrentWidget(page.quick_scan_tab)
        page.tabs.setCurrentWidget(page.association_tab)
        assert changes == [False, True]


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    from qt_shell import EmberMainWindow
    monkeypatch.setattr(dm, "app_base_dir", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    win = EmberMainWindow()
    qtbot.addWidget(win)
    qtbot.wait(50)
    return win


class TestShellWiring:
    def test_shell_hands_the_shared_scope_to_the_tab(self, window):
        assert window.correlations_page.association_tab.scope is window.global_scope

    def test_scope_bar_shows_above_the_scoped_tab_only(self, window, qtbot):
        from qt_shell import NAV_CORRELATIONS, NAV_ITEMS, NAV_QUERY
        page = window.correlations_page
        window.show()
        qtbot.waitExposed(window)
        try:
            window.nav.setCurrentRow(NAV_ITEMS.index(NAV_CORRELATIONS))
            assert page.tabs.currentWidget() is page.association_tab
            assert window.scope_bar.isVisible()
            for legacy in (page.quick_scan_tab, page.workbench_tab, page.structure_tab):
                page.tabs.setCurrentWidget(legacy)
                assert not window.scope_bar.isVisible(), "the scope bar hovers over a tab that ignores it"
            page.tabs.setCurrentWidget(page.association_tab)
            assert window.scope_bar.isVisible()
            # Leaving on a legacy tab and coming back must not resurrect the bar.
            page.tabs.setCurrentWidget(page.quick_scan_tab)
            window.nav.setCurrentRow(NAV_ITEMS.index(NAV_QUERY))
            assert window.scope_bar.isVisible()
            window.nav.setCurrentRow(NAV_ITEMS.index(NAV_CORRELATIONS))
            assert not window.scope_bar.isVisible()
        finally:
            window.hide()

    def test_a_scope_change_in_the_shell_reaches_the_tab(self, window, qtbot, association_dataset):
        tab = window.correlations_page.association_tab
        _run(tab, qtbot, association_dataset)
        window.global_scope.set_filter("TankFarm", ["AN"])
        qtbot.wait(20)
        assert set(_table(tab, T_VALUES)["TankFarm"]) == {"AN"}
        assert "Scope: AN" in tab.plot.ax.get_title()


class TestPagePass:
    """What the page pass added: the shared notes banner, the Run / Update
    label, completions, chart-to-table selection, the busy state."""

    def test_banner_shows_one_calm_line_and_keeps_every_note(self, qtbot, association_dataset):
        from qt_correlations_association import banner_text
        tab = _run(_make_tab(qtbot), qtbot, association_dataset, components="F, Cl, Zz")
        banner = tab.notes_banner
        engine = list(tab._matrix.warnings)
        own = [note for note in banner.notes() if note not in engine]
        assert any("Zz" in note for note in own)
        # The visible line is the designed one-line text; the full list is one click away.
        assert banner.summary_label.text() == banner_text(own, engine)
        assert "Zz" in banner.summary_label.text()
        assert list(banner.notes()) == own + engine
        for note in banner.notes():
            assert note in banner.detail_label.text() and note in banner.toolTip()
        assert banner.detail_label.isHidden() and not banner.toggle.isHidden()
        banner.toggle.setChecked(True)
        assert not banner.detail_label.isHidden()

    def test_run_button_says_update_when_the_controls_left_the_result(self, qtbot, association_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(association_dataset)
        tab.components_edit.setText("F, Cl")
        assert tab.run_btn.text() == "Run"              # nothing standing yet
        _run(tab, qtbot, association_dataset)
        assert tab.run_btn.text() == "Run"
        tab.components_edit.setText("F, Cl, Cr")
        assert tab.run_btn.text() == "Update"
        tab.components_edit.setText(COMPONENTS)
        assert tab.run_btn.text() == "Run"              # back to what produced the result
        tab.min_n_spin.setValue(6)
        assert tab.run_btn.text() == "Update"
        tab.run()
        qtbot.wait(20)
        assert tab.run_btn.text() == "Run"

    def test_component_field_offers_what_the_dataset_holds(self, qtbot, association_dataset):
        from qt_correlations_workbench import completion_items
        tab = _make_tab(qtbot)
        assert completion_items(tab.components_edit) == []
        tab.on_dataset_changed(association_dataset)
        offered = set(completion_items(tab.components_edit))
        assert {"F", "Cl", "SO4", "137Cs", "Cs"} <= offered
        # The entry being typed is completed; the entries before it are kept.
        completer = tab.components_edit.completer()
        tab.components_edit.setText("F, Cl, SO")
        assert completer.splitPath("F, Cl, SO") == ["SO"]
        completer.setCompletionPrefix("F, Cl, SO")
        assert completer.currentCompletion() == "F, Cl, SO4"

    def test_clicking_a_cell_selects_its_pair(self, qtbot, association_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset)
        tab.tables.setCurrentWidget(tab._table_views[T_VALUES])
        tab._on_mark_clicked({"row": "Cr", "column": "Cl", "value": 0.1})
        view = tab._table_views[T_PAIRS]
        assert tab.tables.currentWidget() is view
        row = _table(tab, T_PAIRS).iloc[view.selectedIndexes()[0].row()]
        assert {row["A"], row["B"]} == {"Cl", "Cr"}
        # ...and that pair is the one the scatter then draws.
        tab.plot_combo.setCurrentText(PLOT_SCATTER)
        qtbot.wait(20)
        assert set(tab.selected_pair()) == {"Cl", "Cr"}
        tab._on_mark_clicked({"label": "nothing by that name"})       # no match: no error

    def test_clicking_a_point_selects_its_layer(self, qtbot, association_dataset):
        tab = _run(_make_tab(qtbot), qtbot, association_dataset)
        tab.plot_combo.setCurrentText(PLOT_SCATTER)
        qtbot.wait(20)
        values = _table(tab, T_VALUES)
        tank = str(values["WasteSiteId"].iloc[7])
        phase = str(values["WastePhase"].iloc[7])
        tab._on_mark_clicked({"label": f"{tank}, {phase}", "tank_id": tank})
        view = tab._table_views[T_VALUES]
        assert tab.tables.currentWidget() is view
        assert values["WasteSiteId"].iloc[view.selectedIndexes()[0].row()] == tank

    def test_busy_state_is_shown_while_the_worker_is_out(self, qtbot, association_dataset, monkeypatch):
        class Window(QMainWindow):
            def __init__(self):
                super().__init__()
                self.busy = []

            def set_busy(self, text):
                self.busy.append(text)

        window = Window()
        qtbot.addWidget(window)
        tab = AssociationTab(window)
        qtbot.addWidget(tab)
        tab.on_dataset_changed(association_dataset)
        assert window.busy == []                         # never cleared before it was set
        pending = []
        monkeypatch.setattr(qca, "run_in_thread",
                            lambda fn, on_done, on_error=None, *a, **k: pending.append(fn))
        tab.run()
        assert len(window.busy) == 1 and "cross-correlation" in window.busy[0].lower()
        tab._on_run_done(pending[0]())
        assert window.busy[-1] is None

    def test_idle_chart_says_what_to_do(self, qtbot):
        tab = _make_tab(qtbot)
        texts = [t.get_text() for t in tab.plot.ax.texts]
        assert texts == ["No result yet", "Choose components and press Run."]
        assert tab.header.subtitle_label.wordWrap()


class TestBannerText:
    def test_result_specific_warnings_are_shown_in_full_and_the_rest_counted(self):
        from qt_correlations_association import banner_text
        engine = [
            "Not one quantity across layers: Se (153 layer(s) with a reported mass, 383 nuclide-only).",
            "18 waste layer(s) report no sodium — mol/mol Na left blank there.",
            "Ci values mix decay baselines 2015/07/01 and 2022/01/01.",
            "1893 of 4462 contributing row(s) have no assigned uncertainty.",
        ]
        text = banner_text(["Zz: no value in any waste layer."], engine)
        assert "Zz: no value" in text and engine[0] in text and engine[1] in text
        assert engine[2] not in text and engine[3] not in text
        assert "2 dataset note(s)" in text

    def test_nothing_to_say_is_an_empty_banner(self):
        from qt_correlations_association import banner_text
        assert banner_text([], []) == ""
        assert banner_text([], ["Ci values mix decay baselines."]).startswith("1 dataset note(s)")
