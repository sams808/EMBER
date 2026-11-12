"""
Qt-level tests for the Loading rules sub-tab (qt_vitrification_rules) and
its wiring into the Vitrification page and the shell's scope bar.

The rules are glass_rules' and are tested there. What is tested here is
the interface's own contract: the tiles and tables state what the screen
computed, every drawing agrees with the table beside it, a mark and its
row select each other, the scope reaches the engine and is stated, a
hidden tab does not recompute, and results come back through bound
methods (workers are synchronous under pytest, see conftest).

Widgets are never put on the real screen: a tab that has to be "shown"
is shown with WA_DontShowOnScreen.
"""
import math

import numpy as np
import pandas as pd
import polars as pl
import pytest
from matplotlib.backend_bases import MouseEvent
from matplotlib.collections import PathCollection
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QMessageBox

import data_model as dm
import glass_rules as gr
import qt_vitrification_rules as qvr
from conftest import REAL_ATTRS_PATH, REAL_CSV_PATH, requires_real_data
from data_model import HanfordDataset
from qt_scope import GlobalScope
from qt_vitrification import VitrificationPage
from qt_vitrification_rules import (
    PLOT_DIAGRAM, PLOT_EXTRA, PLOT_KINDS, PLOT_SHARES, TAB_TITLE, TABLE_TITLES, T_GROUPS, T_NOTES,
    T_SUMMARY, T_UNITS, LoadingRulesTab,
)

TANKS = {
    # tank: (tank type, [(phase, waste type, {species: kg})])
    "241-A-101": ("SST-4", [
        ("Supernatant", "NA (Liquid)", dict(Na=2000, K=40, SO4=20, Cl=10, F=2, Cr=1)),
        ("Saltcake Solid", "A1-SltCk (Solid)", dict(Na=1000, K=5, SO4=400, Cl=2, F=30, Cr=3)),
        ("Sludge Solid", "R1 (Solid)", dict(Na=300, K=1, SO4=5, Cl=1, F=1, Cr=40)),
    ]),
    "241-A-102": ("SST-4", [("Sludge Solid", "R1 (Solid)", dict(Na=500, K=2, SO4=8, Cl=1, F=2, Cr=60))]),
    "241-A-103": ("SST-4", [("Saltcake Solid", "A1-SltCk (Solid)", dict(Na=1500, K=4, SO4=10, Cl=3, F=1, Cr=1))]),
    "241-B-101": ("SST-4", [("Saltcake Solid", "B-SltCk (Solid)", dict(Na=800, SO4=30, Cl=120, F=5, Cr=2))]),
    "241-B-102": ("SST-4", [("Saltcake Solid", "B-SltCk (Solid)", dict(Na=600, K=1, SO4=500, Cl=2, F=40, Cr=1))]),
    "241-AN-101": ("DST", [("Supernatant", "NA (Liquid)", dict(Na=3000, K=100, SO4=60, Cl=30, F=5, Cr=4))]),
    "241-AN-102": ("DST", [("Supernatant", "NA (Liquid)", dict(Na=2500, K=60, SO4=300, Cl=20, F=4, Cr=3))]),
}
SALT_TANKS = ["241-A-101", "241-A-103", "241-B-101", "241-B-102", "241-AN-101", "241-AN-102"]


@pytest.fixture
def rules_dataset() -> HanfordDataset:
    """Seven tanks in three farms, single- and double-shell, built so that
    every rule of the advanced set binds somewhere: alkali (A-103, AN-101),
    alkali + sulfur (AN-102), sulfur (B-102), halide (B-101); one tank
    holds sludge only (A-102, no salt waste) and one does not report
    potassium (B-101)."""
    rows = {"WasteSiteId": [], "Analyte": [], "WastePhase": [], "WasteType": [], "Inventory": [], "Units": []}
    for tank, (_type, layers) in TANKS.items():
        for phase, waste_type, kg in layers:
            for analyte, value in kg.items():
                rows["WasteSiteId"].append(tank)
                rows["Analyte"].append(analyte)
                rows["WastePhase"].append(phase)
                rows["WasteType"].append(waste_type)
                rows["Inventory"].append(float(value))
                rows["Units"].append("kg")
    attrs = pl.DataFrame({
        "Name": list(TANKS),
        "TankType": [tank_type for tank_type, _layers in TANKS.values()],
        "TankStatus": [""] * len(TANKS),
        "TankIntegrity": ["Sound"] * len(TANKS),
        "Capacity": ["1000"] * len(TANKS),
        "DIL_Gal": ["0"] * len(TANKS),
    })
    dataset = HanfordDataset()
    df = dataset._clean_dataframe(pl.DataFrame(rows))
    cleaned = dataset._clean_attributes_dataframe(attrs)
    dataset.attrs_df = cleaned
    dataset.df = dataset._merge_tank_attributes(df, cleaned)
    dataset.report = None
    return dataset


class ShellWindow(QMainWindow):
    """A bare window that can open a tank, as the real shell can."""

    def __init__(self):
        super().__init__()
        self.opened = []

    def open_tank(self, tank_id):
        self.opened.append(tank_id)
        return True


def _make_tab(qtbot, scope=None, window=None):
    app_window = window if window is not None else ShellWindow()
    qtbot.addWidget(app_window)
    tab = LoadingRulesTab(app_window, scope=scope)
    qtbot.addWidget(tab)
    return tab


def _run(tab, qtbot, dataset, **settings):
    tab.on_dataset_changed(dataset)
    if "rule_set" in settings:
        tab.rule_set_combo.setCurrentText(settings["rule_set"])
    if "level" in settings:
        tab.unit_combo.setCurrentIndex(tab.unit_combo.findData(settings["level"]))
    if "waste" in settings:
        tab.waste_combo.setCurrentText(settings["waste"])
    if "plot" in settings:
        tab.plot_combo.setCurrentText(settings["plot"])
    tab.run()
    qtbot.wait(20)
    return tab


def _show_off_screen(widget, qtbot):
    widget.setAttribute(Qt.WA_DontShowOnScreen, True)
    widget.show()
    qtbot.wait(40)


def _table(tab, title):
    return tab._table_views[title].dataframe()


def _capture(monkeypatch, name):
    calls = []
    monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: calls.append(a) or QMessageBox.Ok))
    return calls


def _pointer(tab, kind, unit, **details):
    """A real matplotlib pointer event on the mark that stands for `unit`."""
    canvas = tab.plot.canvas
    canvas.draw()
    spot = tab.ring.spot(unit)
    assert spot is not None, f"{unit} is not on the drawing"
    if spot[0] == "bar":
        x, y = (spot[1] + spot[3]) / 2.0, (spot[2] + spot[4]) / 2.0
    else:
        x, y = spot[1], spot[2]
    px, py = tab.plot.ax.transData.transform((x, y))
    canvas.callbacks.process(kind, MouseEvent(kind, canvas, px, py, **details))


def _count_screens(monkeypatch):
    calls = []
    original = gr.screen

    def counting(dataset, **kwargs):
        calls.append(kwargs)
        return original(dataset, **kwargs)

    monkeypatch.setattr(gr, "screen", counting)
    return calls


# ----------------------------------------------------------------------
class TestConstruction:
    def test_defaults(self, qtbot):
        tab = _make_tab(qtbot)
        assert [tab.rule_set_combo.itemText(i) for i in range(tab.rule_set_combo.count())] == list(gr.RULE_SETS)
        assert tab.rule_set_combo.currentText() == "Advanced LAW (2013)"
        assert [tab.unit_combo.itemText(i) for i in range(tab.unit_combo.count())] == ["Tanks", "Waste layers"]
        assert tab.unit_combo.currentData() == "tank"
        assert [tab.waste_combo.itemText(i) for i in range(tab.waste_combo.count())] == [
            "Supernatant + saltcake", "Saltcake", "Supernatant", "All phases"]
        assert tab.waste_combo.currentText() == "Supernatant + saltcake"
        assert [tab.plot_combo.itemText(i) for i in range(tab.plot_combo.count())] == PLOT_KINDS
        assert tab.run_btn.text() == "Run" and tab.run_btn.objectName() == "Primary"
        assert tab.export_btn.text() == "Export..."
        assert tab.header.title_label.text() == TAB_TITLE

    def test_table_tabs(self, qtbot):
        tab = _make_tab(qtbot)
        assert [tab.tables.tabText(i) for i in range(tab.tables.count())] == TABLE_TITLES
        assert TABLE_TITLES == ["Units", "Summary", "Groups", "Notes"]
        assert all(_table(tab, title).empty for title in TABLE_TITLES)

    def test_starts_on_an_empty_state_with_blank_tiles(self, qtbot):
        tab = _make_tab(qtbot)
        assert tab.plot_stack.currentWidget() is tab.empty_state
        assert tab.empty_state.title_label.text() == "No result yet"
        assert [tile.value() for tile in tab.tiles] == ["—"] * 5
        assert not tab.notes_banner.notes()

    def test_each_rule_set_states_its_equations_in_the_tooltip(self, qtbot):
        tab = _make_tab(qtbot)
        tips = [tab.rule_set_combo.itemData(i, Qt.ToolTipRole) for i in range(tab.rule_set_combo.count())]
        assert "33.94 - 11.69 SO₃" in tips[0]
        assert "0.607 F" in tips[1]
        assert "Na₂O ≤ 21" in tips[2]

    def test_actions_without_a_dataset_say_so(self, qtbot, monkeypatch):
        tab = _make_tab(qtbot)
        infos = _capture(monkeypatch, "information")
        bar = tab.app_window.statusBar()
        tab.run()
        assert "no dataset is loaded" in bar.currentMessage()
        tab.export_tables()
        assert "nothing to export" in bar.currentMessage()
        assert infos == []          # a sentence in the status bar, not a popup
        assert tab._result is None

    def test_help_opens_the_section_about_this_tab(self, qtbot):
        tab = _make_tab(qtbot)
        text = tab.help_markdown()
        assert text.startswith("### 12.5")
        assert "PNNL-22631" in text
        assert "### 12.4" not in text and "## 13." not in text


class TestRun:
    def test_run_fills_the_tables_and_shows_the_diagram(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        assert tab._result is not None
        assert tab.plot_stack.currentWidget() is tab.plot
        units = _table(tab, T_UNITS)
        assert sorted(units["Tank"]) == sorted(SALT_TANKS)
        assert not _table(tab, T_SUMMARY).empty and not _table(tab, T_GROUPS).empty and not _table(tab, T_NOTES).empty
        assert "Loading rules" in tab.plot.figure.get_suptitle()
        assert tab.app_window.statusBar().currentMessage().startswith("Loading rules: 6 tanks screened")

    def test_units_table_states_the_screen_and_is_sorted_by_extra_glass(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        units = _table(tab, T_UNITS)
        assert list(units.columns[:5]) == ["Tank", "Loading (wt% Na₂O)", "Binding rule", "Extra glass (t)", "Glass (t)"]
        assert units["Extra glass (t)"].is_monotonic_decreasing
        expected = gr.screen(rules_dataset).table.set_index("WasteSiteId")
        shown = units.set_index("Tank")
        for tank in SALT_TANKS:
            assert shown.loc[tank, "Loading (wt% Na₂O)"] == pytest.approx(expected.loc[tank, "Na2O_loading_wtpct"])
            assert shown.loc[tank, "Binding rule"] == expected.loc[tank, "Binding_rule"]
            assert shown.loc[tank, "Extra glass (t)"] == pytest.approx(expected.loc[tank, "Extra_glass_t"])
        assert set(shown["Binding rule"]) == {"Alkali", "Alkali + sulfur", "Sulfur", "Halide"}

    def test_tiles_state_the_totals_of_the_screen(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        s = gr.screen(rules_dataset).summary
        assert tab.tile_sodium.value() == f"{qvr.fmt(s.na2o_kt)} kt"
        assert tab.tile_sodium.hint_label.text() == "Na₂O in 6 tanks"
        assert tab.tile_not_alkali.value() == f"{s.not_alkali_limited_pct:.1f} %"
        assert tab.tile_glass.value() == f"{qvr.fmt(s.glass_unit_by_unit_kt, 4)} kt"
        assert tab.tile_glass.hint_label.text() == f"+{s.extra_glass_pct:.1f} % over alkali only"
        assert f"{qvr.fmt(s.glass_alkali_only_kt, 4)} kt" in tab.tile_glass.toolTip()      # the alkali-only figure
        assert tab.tile_blend.value() == f"{qvr.fmt(s.glass_blended_kt, 4)} kt"
        assert tab.tile_blend.hint_label.text() == (
            f"{s.blend_loading_wtpct:.1f} wt% Na₂O, {str(s.blend_binding_rule).lower()}")
        assert tab.tile_low.value() == str(s.n_below_low_loading)
        assert tab.tile_low.label_label.text() == "Tanks below 15 wt% Na₂O"
        assert tab.tile_low.hint_label.text() == "of 6 holding sodium"
        # no hint is long enough to be cut on a 1440 px window
        assert all(len(tile.hint_label.text()) <= 36 for tile in tab.tiles)

    def test_summary_tab_writes_each_quantity_with_its_unit(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        s = gr.screen(rules_dataset).summary
        shown = _table(tab, T_SUMMARY).set_index("Quantity")["Value"]
        assert shown["Tanks screened"] == "6" and shown["Tanks holding sodium"] == "6"
        assert shown["Sodium screened"] == f"{qvr.fmt(s.na2o_kt, 4)} kt Na₂O"
        assert shown["Glass, tank by tank"] == f"{qvr.fmt(s.glass_unit_by_unit_kt, 4)} kt"
        assert shown["Sodium not alkali-limited"] == f"{qvr.fmt(s.not_alkali_limited_pct, 4)} % of Na₂O"
        blend_row = f"Loading of the blended feed ({s.blend_binding_rule.lower()} rule)"
        assert shown[blend_row] == f"{qvr.fmt(s.blend_loading_wtpct, 4)} wt% Na₂O"

    def test_notes_are_calm_and_complete(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        banner = tab.notes_banner
        assert banner.summary_label.text().startswith("Screening index, not a forecast of glass production")
        assert tab.warnings_label is banner
        assert "do not report K" in banner.text()
        assert gr.SCOPE_OF_VALIDITY in banner.notes()
        # the banner stays short: the engine's dataset notes are counted there and listed in the Notes tab
        assert len(banner.notes()) <= 5
        assert not any("SPECIES_FORMULAS" in note or "Inventory_RSD" in note for note in banner.notes())
        for warning in tab._result.warnings:
            assert warning in set(_table(tab, T_NOTES)["Text"])
        notes = _table(tab, T_NOTES)
        assert list(notes.columns) == ["Kind", "Text"]
        assert notes.iloc[0]["Text"].startswith("Scope: none")
        text = " ".join(notes["Text"])
        assert "PNNL-22631 Rev. 1" in text
        assert "SO₃ ≤ 1.5" in text and "Table S.12" in text
        assert "screening index" in text
        assert set(notes["Kind"]) >= {"Scope", "Method", "Rule", "Source", "Disclosure"}

    def test_waste_layers_and_other_phases(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset, level="layer")
        units = _table(tab, T_UNITS)
        assert units.columns[0] == "Waste layer" and len(units) == 7
        assert "Tank" in units.columns                        # a layer row still names its tank
        assert tab.tile_glass.label_label.text() == "Glass, layer by layer"
        assert tab.tile_low.label_label.text() == "Layers below 15 wt% Na₂O"
        tab = _run(tab, qtbot, rules_dataset, level="tank", waste="All phases")
        assert sorted(_table(tab, T_UNITS)["Tank"]) == sorted(TANKS)
        assert "All phases of each tank" in tab.plot.ax.get_title()

    def test_run_button_offers_update_when_the_inputs_have_moved(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        assert tab.run_btn.text() == "Run"
        tab.waste_combo.setCurrentText("Saltcake")
        assert tab.run_btn.text() == "Update"
        tab.run()
        qtbot.wait(20)
        assert tab.run_btn.text() == "Run"
        assert tab._result.phases == ("Saltcake",)

    def test_labels_describe_the_run_not_the_live_controls(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        tab.rule_set_combo.setCurrentText("WTP baseline")
        tab.plot_combo.setCurrentText(PLOT_SHARES)             # a redraw of the standing result
        assert "Advanced LAW (2013)" in tab.plot.ax.get_title()
        assert "WTP baseline" not in tab.plot.ax.get_title()

    def test_dataset_change_clears_the_previous_result(self, qtbot, rules_dataset, sample_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        tab.on_dataset_changed(sample_dataset)
        assert tab._result is None
        assert all(_table(tab, title).empty for title in TABLE_TITLES)
        assert tab.plot_stack.currentWidget() is tab.empty_state
        assert tab.tile_sodium.value() == "—"
        assert tab.is_stale()

    def test_a_dataset_without_salt_waste_is_an_empty_state_not_an_error(self, qtbot, sample_dataset, monkeypatch):
        criticals = _capture(monkeypatch, "critical")
        tab = _run(_make_tab(qtbot), qtbot, sample_dataset)
        assert not criticals
        assert tab._result is not None and tab._result.summary.n_with_sodium == 0
        assert tab.plot_stack.currentWidget() is tab.empty_state
        assert tab.empty_state.title_label.text() == "Nothing to screen in this selection"
        assert tab.tile_sodium.value() == "—"

    def test_unexpected_failure_is_reported_not_raised(self, qtbot, rules_dataset, monkeypatch):
        criticals = _capture(monkeypatch, "critical")
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(rules_dataset)

        def boom(*args, **kwargs):
            raise RuntimeError("engine fell over")

        monkeypatch.setattr(gr, "screen", boom)
        tab.run()
        qtbot.wait(20)
        assert len(criticals) == 1 and "engine fell over" in criticals[0][2]
        assert tab._result is None and tab.run_btn.isEnabled()


class TestWorkerContract:
    def test_results_are_delivered_to_bound_methods_of_the_tab(self, qtbot, rules_dataset, monkeypatch):
        seen = {}

        def fake_run_in_thread(fn, on_done, on_error=None, *args, **kwargs):
            seen["on_done"], seen["on_error"] = on_done, on_error
            seen["fn"] = fn

        monkeypatch.setattr(qvr, "run_in_thread", fake_run_in_thread)
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(rules_dataset)
        tab.run()
        # A bound method of a QObject is delivered on that object's thread; a
        # plain function or lambda would run on the worker thread.
        assert seen["on_done"].__self__ is tab and seen["on_error"].__self__ is tab
        assert tab.run_btn.text() == "Running…" and not tab.run_btn.isEnabled()
        seen["on_done"](seen["fn"]())
        assert tab._result is not None and tab.run_btn.isEnabled()

    def test_a_superseded_answer_is_dropped(self, qtbot, rules_dataset, monkeypatch):
        pending = []
        monkeypatch.setattr(qvr, "run_in_thread",
                            lambda fn, on_done, on_error=None, *a, **k: pending.append((fn, on_done)))
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(rules_dataset)
        tab.run()
        tab.waste_combo.setCurrentText("Saltcake")
        tab.run()
        (first_fn, first_done), (second_fn, second_done) = pending
        second_done(second_fn())
        first_done(first_fn())                      # the older answer arrives last
        assert tab._result.phases == ("Saltcake",)

    def test_an_answer_for_the_previous_dataset_is_dropped(self, qtbot, rules_dataset, sample_dataset, monkeypatch):
        pending = []
        monkeypatch.setattr(qvr, "run_in_thread",
                            lambda fn, on_done, on_error=None, *a, **k: pending.append((fn, on_done)))
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(rules_dataset)
        tab.run()
        tab.on_dataset_changed(sample_dataset)
        fn, done = pending[0]
        done(fn())
        assert tab._result is None


class TestStaleWhenHidden:
    def test_runs_once_by_itself_when_first_shown_with_data(self, qtbot, rules_dataset, monkeypatch):
        calls = _count_screens(monkeypatch)
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(rules_dataset)
        assert tab._result is None and not calls           # hidden: nothing computed
        _show_off_screen(tab, qtbot)
        assert len(calls) == 1 and tab._result is not None
        tab.hide()
        _show_off_screen(tab, qtbot)
        assert len(calls) == 1                               # shown again: nothing owed

    def test_shown_without_data_computes_nothing(self, qtbot, monkeypatch):
        calls = _count_screens(monkeypatch)
        tab = _make_tab(qtbot)
        _show_off_screen(tab, qtbot)
        assert not calls and tab._result is None

    def test_a_hidden_tab_does_not_recompute_on_a_scope_change(self, qtbot, rules_dataset, monkeypatch):
        scope = GlobalScope()
        tab = _make_tab(qtbot, scope=scope)
        _show_off_screen(tab, qtbot)
        _run(tab, qtbot, rules_dataset)
        calls = _count_screens(monkeypatch)
        tab.hide()
        scope.set_filter("TankFarm", ["B"])
        tab.on_scope_changed()
        qtbot.wait(20)
        assert not calls and tab.is_stale()
        assert sorted(_table(tab, T_UNITS)["Tank"]) == sorted(SALT_TANKS)      # the old result stands meanwhile
        _show_off_screen(tab, qtbot)
        assert len(calls) == 1 and not tab.is_stale()
        assert sorted(_table(tab, T_UNITS)["Tank"]) == ["241-B-101", "241-B-102"]

    def test_a_dataset_arriving_while_shown_is_screened(self, qtbot, rules_dataset):
        tab = _make_tab(qtbot)
        _show_off_screen(tab, qtbot)
        tab.on_dataset_changed(rules_dataset)
        qtbot.wait(20)
        assert tab._result is not None


class TestScope:
    def test_scope_reaches_the_engine_and_is_stated(self, qtbot, rules_dataset):
        scope = GlobalScope()
        scope.set_filter("TankFarm", ["B"])
        tab = _run(_make_tab(qtbot, scope=scope), qtbot, rules_dataset)
        assert sorted(_table(tab, T_UNITS)["Tank"]) == ["241-B-101", "241-B-102"]
        assert "Scope: B" in tab.plot.ax.get_title()
        assert _table(tab, T_NOTES).iloc[0]["Text"] == "Scope: B"
        assert tab.tile_sodium.hint_label.text() == "Na₂O in 2 tanks"
        assert "scope: B" in tab.app_window.statusBar().currentMessage()

    def test_scope_change_re_asks_the_standing_question_not_the_live_controls(self, qtbot, rules_dataset):
        scope = GlobalScope()
        tab = _make_tab(qtbot, scope=scope)
        _show_off_screen(tab, qtbot)
        _run(tab, qtbot, rules_dataset, waste="Saltcake")
        tab.waste_combo.setCurrentText("All phases")           # moved, not run
        scope.set_filter("TankFarm", ["A"])
        tab.on_scope_changed()
        qtbot.wait(20)
        assert tab._result.phases == ("Saltcake",)
        assert sorted(_table(tab, T_UNITS)["Tank"]) == ["241-A-101", "241-A-103"]

    def test_a_scope_matching_nothing_is_an_empty_state_and_recovers(self, qtbot, rules_dataset, monkeypatch):
        criticals = _capture(monkeypatch, "critical")
        warnings = _capture(monkeypatch, "warning")
        scope = GlobalScope()
        tab = _make_tab(qtbot, scope=scope)
        _show_off_screen(tab, qtbot)
        _run(tab, qtbot, rules_dataset)
        scope.set_filter("WastePhase", ["Sludge Solid"])
        tab.on_scope_changed()
        qtbot.wait(20)
        assert not criticals and not warnings
        assert _table(tab, T_UNITS).empty
        assert tab.plot_stack.currentWidget() is tab.empty_state
        assert "Sludge Solid" in tab.empty_state.hint_label.text()
        assert tab.tile_sodium.value() == "—"
        scope.clear()
        tab.on_scope_changed()
        qtbot.wait(20)
        assert len(_table(tab, T_UNITS)) == 6
        assert tab.plot_stack.currentWidget() is tab.plot


class TestRuleDiagram:
    def _points(self, tab):
        """(unit -> (x, y, area)) of the unit markers as drawn."""
        scatters = [c for c in tab.plot.ax.collections if isinstance(c, PathCollection)]
        units = scatters[0]
        marks = getattr(tab.plot.ax, "_ember_marks", {})
        labels = marks["points"][0]["labels"]
        offsets = np.asarray(units.get_offsets())
        sizes = np.asarray(units.get_sizes())
        return {label: (offsets[i, 0], offsets[i, 1], sizes[i]) for i, label in enumerate(labels)}, scatters

    def test_one_marker_per_unit_at_its_ratios(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        points, scatters = self._points(tab)
        table = tab._result.table.set_index("Unit")
        assert sorted(points) == sorted(SALT_TANKS)
        x_lo, x_hi = tab.plot.ax.get_xlim()
        y_lo, y_hi = tab.plot.ax.get_ylim()
        for unit, (x, y, _area) in points.items():
            assert x == pytest.approx(min(max(table.loc[unit, "SO3_per_Na2O"], x_lo), x_hi))
            assert y == pytest.approx(min(max(table.loc[unit, "Halide_term_per_Na2O"], y_lo), y_hi))
        assert len(scatters) == 2 and len(scatters[1].get_offsets()) == 1          # the blended feed
        blend = tab._result.blend
        assert tuple(scatters[1].get_offsets()[0]) == pytest.approx(
            (blend["SO3_per_Na2O"], blend["Halide_term_per_Na2O"]))

    def test_marker_area_is_proportional_to_sodium(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        points, _ = self._points(tab)
        table = tab._result.table.set_index("Unit")
        largest = table["Na2O_t"].max()
        for unit, (_x, _y, area) in points.items():
            assert area - 9.0 == pytest.approx(200.0 * table.loc[unit, "Na2O_t"] / largest)

    def test_regions_are_named_in_place_and_axes_are_logarithmic(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        ax = tab.plot.ax
        texts = {t.get_text() for t in ax.texts}
        assert {qvr.region_label(name) for name in gr.RULE_CLASSES} <= texts
        assert qvr.region_label("Alkali + sulfur") == "Alkali +\nsulfur rule"
        assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
        assert "SO₃ / Na₂O" in ax.get_xlabel()
        assert "(Cl + 0.3 F + 0.4 Cr₂O₃) / Na₂O" in ax.get_ylabel()
        assert ax.get_legend() is None                       # nothing sits on top of the units
        legend = [t.get_text() for t in tab.plot.figure.legends[0].get_texts()]
        assert legend == ["Single-shell tank (4)", "Double-shell tank (2)", "Blended feed",
                          "Na₂O loading allowed (wt% of glass)"]
        assert "Marker area: Na₂O in the tank" in ax.get_title()
        assert "Region boundaries for K₂O / Na₂O" in ax.get_title()

    def test_loading_contours_are_labelled_on_their_own_line(self, qtbot, rules_dataset):
        """A contour label says what the rules allow AT the place it is written."""
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        rules = tab._result.rule_set
        k = tab._result.summary.k2o_per_na2o
        labelled = {}
        for text in tab.plot.ax.texts:
            gid = text.get_gid() or ""
            if gid.startswith("loading:"):
                x, y = text.get_position()
                limits = rules.limits(k, x, y)
                labelled[float(text.get_text())] = min(float(limits[name]) for name in rules.rule_names)
        assert sorted(labelled) == [5.0, 10.0, 15.0, 20.0, 23.0]
        for level, allowed in labelled.items():
            assert allowed == pytest.approx(level, rel=0.01)

    def test_region_names_sit_inside_their_own_region(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        rules = tab._result.rule_set
        k = tab._result.summary.k2o_per_na2o
        named = 0
        for text in tab.plot.ax.texts:
            label = text.get_text()
            if not label.endswith(" rule"):
                continue
            x, y = text.get_position()
            limits = rules.limits(k, x, y)
            binding = min(rules.rule_names, key=lambda name: float(limits[name]))
            assert qvr.region_label(rules.rule_classes[binding]) == label
            named += 1
        assert named == 4

    def test_each_rule_set_draws_its_own_halide_term(self, qtbot, rules_dataset):
        tab = _make_tab(qtbot)
        expected = {
            "Advanced LAW, conservative halide": "(Cl + 0.607 F + 0.542 Cr₂O₃ + K₂O) / Na₂O",
            "WTP baseline": "(Cl + 0.3 F) / Na₂O",
        }
        for name, label in expected.items():
            _run(tab, qtbot, rules_dataset, rule_set=name)
            assert label in tab.plot.ax.get_ylabel()
            assert name in tab.plot.ax.get_title()
            points, _ = self._points(tab)
            table = tab._result.table.set_index("Unit")
            y_lo, y_hi = tab.plot.ax.get_ylim()
            for unit, (_x, y, _area) in points.items():
                assert y == pytest.approx(min(max(table.loc[unit, "Halide_term_per_Na2O"], y_lo), y_hi))

    def test_a_unit_beyond_the_axes_is_drawn_at_the_edge_and_counted(self):
        (x_lo, x_hi), (y_lo, y_hi) = qvr.diagram_ranges([1e-9, 0.05, 2.0], [0.02])
        assert x_lo == pytest.approx(gr.DIAGRAM_X_RANGE[0] / 20.0)       # never more than 20 times lower
        assert x_hi == pytest.approx(2.5)
        assert (y_lo, y_hi) == gr.DIAGRAM_Y_RANGE

    def test_hover_names_the_tank_and_states_its_loading(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        _pointer(tab, "motion_notify_event", "241-B-101")
        mark = tab.plot.interaction.current_mark()
        assert mark is not None and mark["title"] == "241-B-101" and mark["tank_id"] == "241-B-101"
        rows = dict(mark["rows"])
        row = tab._result.table.set_index("Unit").loc["241-B-101"]
        assert rows["Na₂O loading"] == f"{qvr.fmt(row['Na2O_loading_wtpct'])} wt%"
        assert rows["Binding rule"] == "Halide"
        assert "SO₃ / Na₂O" in rows and "(Cl + 0.3 F + 0.4 Cr₂O₃) / Na₂O" in rows

    def test_the_background_is_not_a_mark(self, qtbot, rules_dataset):
        """Only units and the blended feed answer to the pointer: the
        tinted regions must not raise a tooltip of their own."""
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        canvas = tab.plot.canvas
        canvas.draw()
        ax = tab.plot.ax
        box = ax.bbox
        px, py = box.x0 + 0.5 * box.width, box.y0 + 0.02 * box.height
        canvas.callbacks.process("motion_notify_event", MouseEvent("motion_notify_event", canvas, px, py))
        assert tab.plot.interaction.current_mark() is None


class TestSelectionLink:
    def test_clicking_a_marker_selects_its_row(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        tab.tables.setCurrentIndex(TABLE_TITLES.index(T_NOTES))
        _pointer(tab, "button_press_event", "241-B-102", button=1)
        assert tab.selected_unit() == "241-B-102"
        assert tab.tables.currentWidget() is tab._table_views[T_UNITS]
        assert tab.ring.key() == "241-B-102" and tab.ring.is_drawn()

    def test_selecting_a_row_rings_its_marker_even_after_a_sort(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        view = tab._table_views[T_UNITS]
        view.sortByColumn(0, Qt.AscendingOrder)                 # by tank name
        row = list(view.dataframe()["Tank"]).index("241-AN-102")
        view.setCurrentIndex(view.model().index(row, 0))
        assert tab.selected_unit() == "241-AN-102"
        assert tab.ring.key() == "241-AN-102" and tab.ring.is_drawn()
        spot = tab.ring.spot("241-AN-102")
        ring_line = tab.ring._artist
        assert (ring_line.get_xdata()[0], ring_line.get_ydata()[0]) == pytest.approx((spot[1], spot[2]))

    def test_the_selection_follows_the_unit_across_drawings(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        _pointer(tab, "button_press_event", "241-B-102", button=1)
        tab.plot_combo.setCurrentText(PLOT_EXTRA)
        assert tab.ring.is_drawn() and tab.ring.spot("241-B-102")[0] == "bar"
        tab.plot_combo.setCurrentText(PLOT_SHARES)              # no unit on this drawing
        assert not tab.ring.is_drawn() and tab.ring.key() == "241-B-102"
        tab.plot_combo.setCurrentText(PLOT_DIAGRAM)
        assert tab.ring.is_drawn()

    def test_a_click_wins_over_a_filter_that_hides_the_unit(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        view = tab._table_views[T_UNITS]
        view.set_filter_text("AN-101")
        _pointer(tab, "button_press_event", "241-B-101", button=1)
        assert tab.selected_unit() == "241-B-101"
        assert view.filter_text() == ""

    def test_clicking_a_bar_of_the_extra_glass_chart_selects_the_unit(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset, plot=PLOT_EXTRA)
        _pointer(tab, "button_press_event", "241-A-101", button=1)
        assert tab.selected_unit() == "241-A-101"

    def test_double_click_opens_the_tank_through_the_shell(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        _pointer(tab, "button_press_event", "241-B-101", button=1, dblclick=True)
        assert tab.app_window.opened == ["241-B-101"]
        view = tab._table_views[T_UNITS]
        row = list(view.dataframe()["Tank"]).index("241-AN-101")
        tab._on_unit_double_clicked(view.model().index(row, 2))
        assert tab.app_window.opened == ["241-B-101", "241-AN-101"]

    def test_a_layer_opens_its_tank(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset, level="layer")
        view = tab._table_views[T_UNITS]
        layer = "241-A-101 | Saltcake Solid | A1-SltCk (Solid)"
        row = list(view.dataframe()["Waste layer"]).index(layer)
        tab._on_unit_double_clicked(view.model().index(row, 0))
        assert tab.app_window.opened == ["241-A-101"]

    def test_a_window_that_cannot_open_tanks_is_left_alone(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot, window=QMainWindow()), qtbot, rules_dataset)
        assert tab.open_tank("241-B-101") is False
        _pointer(tab, "button_press_event", "241-B-101", button=1, dblclick=True)      # must not raise


class TestOtherDrawings:
    def test_stacked_bars_equal_the_groups_table(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset, plot=PLOT_SHARES)
        ax = tab.plot.ax
        groups = tab._result.groups
        names = list(groups["Group"])
        assert names == ["All tanks", "Single-shell tanks", "Double-shell tanks", "Saltcake layers",
                         "Supernatant layers"]
        totals = {name: 0.0 for name in names}
        by_position = {len(names) - 1 - i: name for i, name in enumerate(names)}
        for container in ax.containers:
            rule = container.get_label().replace(" rule", "")
            for patch in container.patches:
                group = by_position[int(round(patch.get_y() + patch.get_height() / 2.0))]
                expected = float(groups.set_index("Group").loc[group, f"{rule}_pct"])
                assert patch.get_width() == pytest.approx(expected)
                totals[group] += patch.get_width()
        assert all(total == pytest.approx(100.0) for total in totals.values())
        assert [t.get_text() for t in ax.get_yticklabels()] == names[::-1]

    def test_hovering_a_segment_names_the_group_the_rule_and_the_share(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset, plot=PLOT_SHARES)
        canvas = tab.plot.canvas
        canvas.draw()
        ax = tab.plot.ax
        container = ax.containers[0]
        patch = container.patches[0]
        centre = (patch.get_x() + patch.get_width() / 2.0, patch.get_y() + patch.get_height() / 2.0)
        px, py = ax.transData.transform(centre)
        canvas.callbacks.process("motion_notify_event", MouseEvent("motion_notify_event", canvas, px, py))
        mark = tab.plot.interaction.current_mark()
        assert mark["title"] == "All tanks"
        rows = dict(mark["rows"])
        assert rows["Binding rule"] == container.get_label()
        assert rows["Share of the sodium"].startswith(f"{patch.get_width():.4g}"[:4])
        assert "Sodium in the group" in rows

    def test_extra_glass_bars_are_the_largest_units_in_order(self, qtbot, rules_dataset):
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset, plot=PLOT_EXTRA)
        ax = tab.plot.ax
        table = tab._result.table
        expected = table[table["Extra_glass_t"] > 0].sort_values("Extra_glass_t", ascending=False)
        patches = sorted(ax.containers[0].patches, key=lambda p: -p.get_y())           # top to bottom
        assert [p.get_width() for p in patches] == pytest.approx(list(expected["Extra_glass_t"]))
        assert [t.get_text() for t in ax.get_yticklabels()][::-1] == list(expected["Unit"])
        labels = [t.get_text().strip() for t in ax.texts if t.get_text().strip().endswith("wt%")]
        assert len(labels) == len(expected)
        assert labels == [f"{qvr.fmt(v)} wt%" for v in expected["Na2O_loading_wtpct"]]
        # bars are coloured by the rule that binds the unit
        for patch, rule in zip(patches, expected["Binding_rule"]):
            assert patch.get_facecolor()[:3] == pytest.approx(
                tuple(int(qvr.rule_color(tab._result.rule_set, rule)[i:i + 2], 16) / 255 for i in (1, 3, 5)))

    def test_rule_colours(self):
        advanced, baseline = gr.RULE_SETS[gr.ADVANCED], gr.RULE_SETS[gr.BASELINE]
        assert [qvr.rule_color(advanced, name) for name in advanced.rule_names] == [
            "#c3c2b7", "#6250d6", "#e34948", "#eda100"]
        # the baseline has two alkali-class rules: they must not share a colour
        assert qvr.rule_color(baseline, "Sodium") != qvr.rule_color(baseline, "Alkali")
        assert set(qvr.RULE_TINTS.values()) == {"#f6f5f2", "#eceafb", "#fce6e6", "#fdf3d9"}

    def test_drawings_can_be_sent_to_the_figure_composer(self, qtbot, rules_dataset):
        from matplotlib.figure import Figure

        from figure_composer import ComposerCellPanel
        tab = _run(_make_tab(qtbot), qtbot, rules_dataset)
        for kind in PLOT_KINDS:
            tab.plot_combo.setCurrentText(kind)
            render_fn, args, kwargs = tab.plot._last_recipe
            panel = ComposerCellPanel(Figure(figsize=(6, 4)))
            render_fn(panel, *args, **kwargs)                    # the same drawing, replayed
            assert panel.ax.get_xlabel() == tab.plot.ax.get_xlabel()


class TestExport:
    def test_export_writes_the_tables_the_figure_and_the_methods_note(self, qtbot, rules_dataset, tmp_path,
                                                                    monkeypatch):
        infos = _capture(monkeypatch, "information")
        rules_dataset.output_root = tmp_path
        scope = GlobalScope()
        scope.set_filter("TankFarm", ["B"])
        tab = _run(_make_tab(qtbot, scope=scope), qtbot, rules_dataset)
        tab.export_tables()
        folders = list(tmp_path.glob("loading_rules_*"))
        assert len(folders) == 1 and len(infos) == 1
        out = folders[0]
        names = {p.name for p in out.iterdir()}
        assert names >= {
            "loading_rules_units.csv", "loading_rules_layers.csv", "loading_rules_summary.csv",
            "loading_rules_groups.csv", "loading_rules_rule_set.csv", "loading_rules_notes.csv",
            "loading_rules_parameters.csv", "context.csv", "manifest.csv", "loading_rules_methods.txt",
            "loading_rules_plot.png",
        }
        units = pd.read_csv(out / "loading_rules_units.csv")
        assert sorted(units["WasteSiteId"]) == ["241-B-101", "241-B-102"]
        assert "Na2O_loading_wtpct" in units.columns            # raw names, full precision
        context = dict(pd.read_csv(out / "context.csv").values)
        assert context["global_scope"] == "B" and context["rule_set"] == "Advanced LAW (2013)"
        methods = (out / "loading_rules_methods.txt").read_text(encoding="utf-8")
        assert "PNNL-22631 Rev. 1" in methods and "Scope: B" in methods
        assert "SO3 <= 1.65 - 0.725 (Cl + 0.3 F + 0.4 Cr2O3)" in methods
        assert "not a forecast of glass production" in methods
        assert (out / "loading_rules_plot.png").stat().st_size > 5000


# ----------------------------------------------------------------------
class TestVitrificationPageWiring:
    def _page(self, qtbot, scope=None):
        app_window = QMainWindow()
        qtbot.addWidget(app_window)
        page = VitrificationPage(app_window, scope=scope)
        qtbot.addWidget(page)
        return page

    def test_loading_rules_is_appended_after_the_four_existing_tabs(self, qtbot):
        page = self._page(qtbot)
        titles = [page.tabs.tabText(i) for i in range(page.tabs.count())]
        assert titles == ["Screening", "Oxide Chemistry", "Candidate Search", "Blend Partners", "Loading rules"]
        assert page.tabs.widget(4) is page.rules_tab
        assert page.tabs.currentWidget() is page.screening_tab

    def test_dataset_and_scope_are_forwarded(self, qtbot, rules_dataset):
        scope = GlobalScope()
        page = self._page(qtbot, scope=scope)
        assert page.rules_tab.scope is scope
        page.on_dataset_changed(rules_dataset)
        assert page.rules_tab.dataset is rules_dataset
        scope.set_filter("TankFarm", ["A"])
        page.on_scope_changed()                                 # hidden: marked stale, nothing raised
        assert page.rules_tab.is_stale()
        for legacy in (page.screening_tab, page.oxide_tab, page.candidate_tab, page.blend_tab):
            assert not hasattr(legacy, "on_scope_changed")

    def test_scope_applies_only_while_loading_rules_is_showing(self, qtbot):
        page = self._page(qtbot)
        changes = []
        page.scope_relevance_changed.connect(lambda: changes.append(page.scope_applies()))
        assert not page.scope_applies()
        page.tabs.setCurrentWidget(page.rules_tab)
        page.tabs.setCurrentWidget(page.oxide_tab)
        assert changes == [True, False]

    def test_the_shared_scope_is_taken_from_the_window_when_not_passed(self, qtbot):
        app_window = QMainWindow()
        qtbot.addWidget(app_window)
        app_window.global_scope = GlobalScope(app_window)
        page = VitrificationPage(app_window)
        qtbot.addWidget(page)
        assert page.rules_tab.scope is app_window.global_scope


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
        assert window.vitrification_page.rules_tab.scope is window.global_scope

    def test_scope_bar_shows_above_loading_rules_only(self, window, qtbot):
        from qt_shell import NAV_ITEMS
        page = window.vitrification_page
        _show_off_screen(window, qtbot)
        try:
            window.nav.setCurrentRow(NAV_ITEMS.index("Vitrification"))
            assert page.tabs.currentWidget() is page.screening_tab
            assert not window.scope_bar.isVisible()
            page.tabs.setCurrentWidget(page.rules_tab)
            assert window.scope_bar.isVisible()
            for legacy in (page.screening_tab, page.oxide_tab, page.candidate_tab, page.blend_tab):
                page.tabs.setCurrentWidget(legacy)
                assert not window.scope_bar.isVisible(), "the scope bar hovers over a tab that ignores it"
        finally:
            window.hide()

    def test_a_scope_change_in_the_shell_reaches_the_visible_tab(self, window, qtbot, rules_dataset):
        from qt_shell import NAV_ITEMS
        page = window.vitrification_page
        tab = page.rules_tab
        _show_off_screen(window, qtbot)
        try:
            window.nav.setCurrentRow(NAV_ITEMS.index("Vitrification"))
            page.tabs.setCurrentWidget(tab)
            _run(tab, qtbot, rules_dataset)
            window.global_scope.set_filter("TankFarm", ["AN"])
            qtbot.wait(30)
            assert sorted(_table(tab, T_UNITS)["Tank"]) == ["241-AN-101", "241-AN-102"]
            assert "Scope: AN" in tab.plot.ax.get_title()
            # ... and a hidden tab waits
            page.tabs.setCurrentWidget(page.screening_tab)
            window.global_scope.set_filter("TankFarm", ["B"])
            qtbot.wait(30)
            assert tab.is_stale()
            assert sorted(_table(tab, T_UNITS)["Tank"]) == ["241-AN-101", "241-AN-102"]
        finally:
            window.hide()


# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_dataset():
    dataset = HanfordDataset()
    dataset.load(REAL_CSV_PATH, use_cache=False, attributes_path=REAL_ATTRS_PATH)
    return dataset


@requires_real_data
class TestRealDataset:
    def test_tiles_table_and_hover_on_the_shipped_dataset(self, qtbot, real_dataset):
        tab = _run(_make_tab(qtbot), qtbot, real_dataset)
        s = gr.screen(real_dataset).summary
        assert tab.tile_sodium.value() == f"{qvr.fmt(s.na2o_kt)} kt" == "54.6 kt"
        assert tab.tile_sodium.hint_label.text() == "Na₂O in 133 tanks"
        assert tab.tile_not_alkali.value() == "46.8 %"
        assert tab.tile_glass.value() == "276.9 kt"
        assert tab.tile_glass.hint_label.text() == "+20.3 % over alkali only"
        assert "230.3 kt" in tab.tile_glass.toolTip()
        assert tab.tile_blend.value() == "248.4 kt"
        assert tab.tile_blend.hint_label.text() == "22.0 wt% Na₂O, alkali + sulfur"
        assert tab.tile_low.value() == str(s.n_below_low_loading)
        units = _table(tab, T_UNITS)
        assert len(units) == 142 and units.iloc[0]["Tank"] == "241-AW-106"       # the largest extra glass
        _pointer(tab, "motion_notify_event", "241-AW-106")
        mark = tab.plot.interaction.current_mark()
        assert mark["title"] == "241-AW-106"
        rows = dict(mark["rows"])
        assert rows["Na₂O loading"] == "11.2 wt%" and rows["Binding rule"] == "Halide"

    def test_farm_scope_on_the_shipped_dataset(self, qtbot, real_dataset):
        scope = GlobalScope()
        scope.set_filter("TankFarm", ["B"])
        tab = _run(_make_tab(qtbot, scope=scope), qtbot, real_dataset)
        units = _table(tab, T_UNITS)
        assert set(units["Farm"]) == {"B"}
        assert tab._result.summary.n_with_sodium == 15
        assert "Scope: B" in tab.plot.ax.get_title()
        assert math.isclose(float(units.set_index("Tank").loc["241-B-107", "Loading (wt% Na₂O)"]), 3.27,
                            abs_tol=0.005)
