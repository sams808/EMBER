"""
Global scope bar tests: the shared GlobalScope state object and the
GlobalScopeBar widget's dataset reconciliation (including the mid-load
silence contract the shell relies on).
"""
import polars as pl
import pytest

import data_model as dm
from qt_scope import GlobalScope, GlobalScopeBar


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
