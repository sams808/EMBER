"""
Global scope bar tests: the shared GlobalScope state object and the
GlobalScopeBar widget's dataset reconciliation (including the mid-load
silence contract the shell relies on).
"""
import polars as pl
import pytest

import data_model as dm
from qt_scope import GlobalScope, GlobalScopeBar


def _text_scale() -> float:
    """How wide UI text is on this Qt platform, relative to the native one.

    The offscreen platform has no Segoe UI and falls back to a font about
    twice as wide, so a pixel bound that holds on screen is scaled by this
    factor before it is asserted (1.0 on the native platform)."""
    from PySide6.QtGui import QFontMetrics
    from PySide6.QtWidgets import QApplication
    reference = "Scope Farm Waste phase Waste type SST/DST Clear"
    return max(1.0, QFontMetrics(QApplication.font()).horizontalAdvance(reference) / 272.0)



def _dataset(rows: dict) -> dm.HanfordDataset:
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame(rows))
    dataset.report = None
    return dataset


def _bare_dataset(farms=("241-A-101", "241-AN-104")) -> dm.HanfordDataset:
    n = len(farms)
    return _dataset({
        "WasteSiteId": list(farms),
        "Analyte": ["Fe"] * n,
        "WastePhase": ["Liquid"] * n,
        "WasteType": ["T1"] * n,
        "Inventory": [1.0] * n,
        "Units": ["kg"] * n,
    })


class TestGlobalScope:
    def test_starts_inactive_and_empty(self):
        scope = GlobalScope()
        assert not scope.is_active()
        assert scope.as_scope() == {}
        assert scope.summary() == ""

    def test_set_filter_emits_and_round_trips(self, qtbot):
        scope = GlobalScope()
        received = []
        scope.changed.connect(lambda: received.append(1))
        scope.set_filter("TankFarm", ["AN"])
        assert scope.as_scope() == {"TankFarm": ["AN"]}
        assert scope.is_active()
        assert received == [1]

    def test_no_op_set_does_not_emit(self, qtbot):
        scope = GlobalScope()
        scope.set_filter("TankFarm", ["AN"])
        received = []
        scope.changed.connect(lambda: received.append(1))
        scope.set_filter("TankFarm", ["AN"])   # unchanged
        scope.set_filter("WastePhase", None)   # already absent
        assert received == []

    def test_coalesce_defers_to_one_emission_at_the_end(self, qtbot):
        scope = GlobalScope()
        received = []
        scope.changed.connect(lambda: received.append(1))
        with scope.coalesce():
            scope.set_filter("TankFarm", ["AN"])
            scope.set_filter("WastePhase", ["Sludge"])
            assert received == []          # nothing announced mid-block
        assert received == [1]             # exactly once, after
        assert scope.as_scope() == {"TankFarm": ["AN"], "WastePhase": ["Sludge"]}

    def test_coalesce_with_no_change_emits_nothing(self, qtbot):
        scope = GlobalScope()
        received = []
        scope.changed.connect(lambda: received.append(1))
        with scope.coalesce():
            scope.set_filter("TankFarm", None)  # already absent
        assert received == []

    def test_coalesce_emits_even_if_the_block_raises(self, qtbot):
        scope = GlobalScope()
        received = []
        scope.changed.connect(lambda: received.append(1))
        with pytest.raises(RuntimeError):
            with scope.coalesce():
                scope.set_filter("TankFarm", ["AN"])
                raise RuntimeError("load blew up")
        assert received == [1]  # listeners must not be left stale

    def test_clear_announces_once_not_per_filter(self, qtbot):
        scope = GlobalScope()
        scope.set_filter("TankFarm", ["AN"])
        scope.set_filter("WastePhase", ["Sludge"])
        received = []
        scope.changed.connect(lambda: received.append(1))
        scope.clear()
        assert scope.as_scope() == {}
        assert received == [1]

    def test_summary_stamps_values_in_bar_order(self):
        scope = GlobalScope()
        scope.set_filter("WastePhase", ["Supernatant"])
        scope.set_filter("TankFarm", ["AN"])
        assert scope.summary() == "AN · Supernatant"

    def test_as_scope_returns_a_copy(self):
        scope = GlobalScope()
        scope.set_filter("TankFarm", ["AN"])
        scope.as_scope()["TankFarm"].append("A")
        assert scope.as_scope() == {"TankFarm": ["AN"]}


class TestGlobalScopeBar:
    def test_combo_selection_drives_the_shared_state(self, qtbot, sample_dataset):
        scope = GlobalScope()
        bar = GlobalScopeBar(scope)
        qtbot.addWidget(bar)
        bar.on_dataset_changed(sample_dataset)
        bar._combos["TankFarm"].setCurrentText("AN")
        assert scope.as_scope() == {"TankFarm": ["AN"]}
        assert "AN" in bar.stamp_label.text()

    def test_clear_resets_state_and_combos(self, qtbot, sample_dataset):
        scope = GlobalScope()
        bar = GlobalScopeBar(scope)
        qtbot.addWidget(bar)
        bar.on_dataset_changed(sample_dataset)
        bar._combos["TankFarm"].setCurrentText("AN")
        bar._clear()
        assert scope.as_scope() == {}
        assert bar._combos["TankFarm"].currentText() == "All"
        assert bar.stamp_label.text() == ""

    def test_dataset_reload_drops_a_filter_the_new_data_cannot_honor(self, qtbot, sample_dataset):
        scope = GlobalScope()
        bar = GlobalScopeBar(scope)
        qtbot.addWidget(bar)
        bar.on_dataset_changed(sample_dataset)
        bar._combos["TankFarm"].setCurrentText("AN")
        bar.on_dataset_changed(_bare_dataset(("241-B-101",)))  # AN farm gone
        assert scope.as_scope() == {}  # stale filter dropped from state, not just the UI
        assert bar._combos["TankFarm"].currentText() == "All"

    def test_shell_style_coalesced_reload_announces_once_at_the_end(self, qtbot, sample_dataset):
        # How the shell actually calls it: the coalesce block spans the
        # page dispatch, so listeners hear the dropped filter exactly once,
        # after every page already has the new dataset.
        scope = GlobalScope()
        bar = GlobalScopeBar(scope)
        qtbot.addWidget(bar)
        bar.on_dataset_changed(sample_dataset)
        bar._combos["TankFarm"].setCurrentText("AN")
        received = []
        scope.changed.connect(lambda: received.append(1))
        with scope.coalesce():
            bar.on_dataset_changed(_bare_dataset(("241-B-101",)))
            assert received == []  # nothing fires mid-load
        assert received == [1]

    def test_attribute_dim_disabled_without_attributes_file(self, qtbot):
        scope = GlobalScope()
        bar = GlobalScopeBar(scope)
        qtbot.addWidget(bar)
        scope.set_filter("TankSystem", ["SST"])
        bar.on_dataset_changed(_bare_dataset())  # no TankSystem column
        assert not bar._combos["TankSystem"].isEnabled()
        assert scope.as_scope() == {}  # engine would refuse it -- state follows

    def test_selection_survives_a_reload_when_still_valid(self, qtbot, sample_dataset):
        scope = GlobalScope()
        bar = GlobalScopeBar(scope)
        qtbot.addWidget(bar)
        bar.on_dataset_changed(sample_dataset)
        bar._combos["WastePhase"].setCurrentText("Sludge")
        bar.on_dataset_changed(sample_dataset)
        assert bar._combos["WastePhase"].currentText() == "Sludge"
        assert scope.as_scope() == {"WastePhase": ["Sludge"]}


class TestScopeStrip:
    """The restyled strip: field labels, an active-filter count, a Clear
    that is only available when there is something to clear, and a stamp
    that never widens the window."""

    def _bar(self, qtbot, dataset):
        scope = GlobalScope()
        bar = GlobalScopeBar(scope)
        qtbot.addWidget(bar)
        bar.on_dataset_changed(dataset)
        return scope, bar

    def test_every_selector_has_its_field_label(self, qtbot, sample_dataset):
        from PySide6.QtWidgets import QLabel
        from qt_scope import SCOPE_DIMS
        _scope, bar = self._bar(qtbot, sample_dataset)
        fields = [w.text() for w in bar.findChildren(QLabel) if w.objectName() == "FieldLabel"]
        assert fields == [label for _column, label, _fixed in SCOPE_DIMS]

    def test_count_and_clear_follow_the_number_of_filters(self, qtbot, sample_dataset):
        scope, bar = self._bar(qtbot, sample_dataset)
        assert bar.active_count() == 0
        assert not bar.clear_btn.isEnabled() and bar.count_label.isHidden()
        bar._combos["TankFarm"].setCurrentText("AN")
        assert bar.active_count() == 1
        assert bar.clear_btn.isEnabled() and bar.count_label.text() == "1 filter"
        bar._combos["WastePhase"].setCurrentText("Sludge")
        assert bar.count_label.text() == "2 filters" and not bar.count_label.isHidden()
        bar.clear_btn.click()
        assert scope.as_scope() == {}
        assert not bar.clear_btn.isEnabled() and bar.count_label.isHidden()

    def test_stamp_text_is_the_whole_stamp_however_narrow_the_strip(self, qtbot, sample_dataset):
        from PySide6.QtCore import Qt
        _scope, bar = self._bar(qtbot, sample_dataset)
        bar._combos["TankFarm"].setCurrentText("AN")
        bar._combos["WastePhase"].setCurrentText("Sludge")
        bar.setAttribute(Qt.WA_DontShowOnScreen, True)
        try:
            for width in (bar.minimumSizeHint().width() + 30, 1600):    # too little room, then plenty
                bar.resize(width, 46)
                bar.show()
                qtbot.wait(10)
                assert bar.stamp_label.text() == "Scope: AN · Sludge"   # drawn shortened, never stored shortened
                assert bar.stamp_label.toolTip() == "Scope: AN · Sludge"
                assert not bar.grab().isNull()                          # the paint path runs at both widths
            assert bar.stamp_label.width() > 0                          # with room, the stamp is on screen
        finally:
            bar.hide()

    def test_the_strip_does_not_set_a_wide_minimum(self, qtbot, sample_dataset):
        scope, bar = self._bar(qtbot, sample_dataset)
        assert bar.minimumSizeHint().width() < 800 * _text_scale()
        scope.set_filter("TankFarm", ["AN"])                 # the count appears beside Clear
        with_one_filter = bar.minimumSizeHint().width()
        assert with_one_filter < 900 * _text_scale()
        # A long filter value, hence a long stamp, asks for nothing more.
        scope.set_filter("TankFarm", ["A very long farm name that would not fit in any selector at all"])
        assert scope.summary().startswith("A very long farm name")
        assert bar.minimumSizeHint().width() == with_one_filter

    def test_a_filter_set_from_outside_is_shown_by_its_selector(self, qtbot, sample_dataset):
        scope, bar = self._bar(qtbot, sample_dataset)
        scope.set_filter("TankFarm", ["AN"])
        assert bar._combos["TankFarm"].currentText() == "AN"
        scope.set_filter("TankFarm", None)
        assert bar._combos["TankFarm"].currentText() == "All"

    def test_a_filter_the_selector_cannot_express_leaves_it_alone(self, qtbot, sample_dataset):
        scope, bar = self._bar(qtbot, sample_dataset)
        scope.set_filter("TankFarm", ["A", "AN"])        # two values: no single entry says that
        assert bar._combos["TankFarm"].currentText() == "All"
        assert scope.as_scope() == {"TankFarm": ["A", "AN"]}     # the state is untouched
        assert bar.count_label.text() == "1 filter"
        assert "A, AN" in bar.stamp_label.text()
