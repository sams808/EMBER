"""
L1->L2 bridge tests (query_plots.plan_plot): which table backs the chart,
which dimension becomes the bar axis versus the color series, and which
(mark, grouping) combinations are refused. This policy is shared by every
workspace migrating onto the engine, so it is pinned here rather than
inside one page's tests.
"""
import polars as pl
import pytest

import data_model as dm
from plot_grammar import MARKS, validate_spec
from query_engine import BASKET_PRESETS, run_query
from query_plots import SOURCE_MEMBER_TABLE, SOURCE_TABLE, plan_plot


def _dataset() -> dm.HanfordDataset:
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-A-101"] * 3 + ["241-AN-104"],
        "Analyte": ["SO4", "Cl", "Cr", "Cl"],
        "WastePhase": ["Supernatant"] * 2 + ["Sludge"] * 2,
        "WasteType": ["T1"] * 2 + ["T2"] * 2,
        "Inventory": [100.0, 3.0, 4.0, 6.0],
        "Units": ["kg"] * 4,
        "Volume": [2.0] * 2 + [1.0] * 2,
        "ComponentDensity": [1.5] * 2 + [1.0] * 2,
    }))
    dataset.report = None
    return dataset


def _result(group_by=(), basis="kg"):
    return run_query(_dataset(), subject=("basket", BASKET_PRESETS["Salt formers"]),
                     basis=basis, group_by=list(group_by))


class TestTableSelection:
    def test_ungrouped_bars_draw_the_per_member_split(self):
        result = _result()
        plan = plan_plot(result, [], "bar", subject_label="Salt formers")
        assert plan.is_drawable
        assert plan.source == SOURCE_MEMBER_TABLE
        assert plan.spec.y == "Member"
        assert plan.data is result.member_table

    def test_one_dim_bars_draw_the_grouped_table(self):
        result = _result(["WasteSiteId"])
        plan = plan_plot(result, ["WasteSiteId"], "bar", subject_label="Salt formers")
        assert plan.source == SOURCE_TABLE
        assert plan.data is result.table
        assert plan.spec.y == "WasteSiteId" and plan.spec.hue is None

    def test_one_dim_stacked_and_heatmap_split_by_member(self):
        result = _result(["WasteSiteId"])
        for mark in ("stackedbar", "heatmap"):
            plan = plan_plot(result, ["WasteSiteId"], mark, subject_label="Salt formers")
            assert plan.source == SOURCE_MEMBER_TABLE, mark
            assert plan.spec.hue == "Member", mark

    def test_two_dims_map_to_bar_axis_and_color_series(self):
        result = _result(["WasteSiteId", "WastePhase"])
        plan = plan_plot(result, ["WasteSiteId", "WastePhase"], "bar", subject_label="Salt formers")
        assert plan.spec.y == "WasteSiteId"
        assert plan.spec.hue == "WastePhase"

    def test_box_uses_the_second_dimension_as_the_category(self):
        result = _result(["WasteSiteId", "WastePhase"])
        plan = plan_plot(result, ["WasteSiteId", "WastePhase"], "box", subject_label="Salt formers")
        assert plan.spec.mark == "box"
        assert plan.spec.y == "WastePhase"   # distribution of per-tank values within each phase
        assert plan.spec.hue is None


class TestRefusals:
    def test_ungrouped_non_bar_marks_are_refused_with_a_reason(self):
        result = _result()
        for mark in ("stackedbar", "heatmap", "box"):
            plan = plan_plot(result, [], mark, subject_label="Salt formers")
            assert not plan.is_drawable, mark
            assert plan.refusal, mark

    def test_box_with_one_dimension_is_refused(self):
        result = _result(["WastePhase"])
        plan = plan_plot(result, ["WastePhase"], "box", subject_label="Salt formers")
        assert not plan.is_drawable
        assert "two grouping dimensions" in plan.refusal

    def test_scatter_is_refused_a_single_measure_result(self):
        result = _result(["WasteSiteId", "WastePhase"])
        plan = plan_plot(result, ["WasteSiteId", "WastePhase"], "scatter", subject_label="X")
        assert not plan.is_drawable
        assert "two value columns" in plan.refusal

    def test_every_mark_yields_a_plan_or_a_reason_never_a_crash(self):
        # No (mark, grouping) combination may fall through silently.
        for dims in ([], ["WasteSiteId"], ["WasteSiteId", "WastePhase"]):
            result = _result(dims)
            for mark in MARKS:
                plan = plan_plot(result, dims, mark, subject_label="Salt formers")
                assert plan.is_drawable or plan.refusal, (mark, dims)


class TestSpecContent:
    def test_every_drawable_plan_validates_against_its_own_data(self):
        for dims in ([], ["WasteSiteId"], ["WasteSiteId", "WastePhase"]):
            result = _result(dims)
            for mark in MARKS:
                plan = plan_plot(result, dims, mark, subject_label="Salt formers")
                if plan.is_drawable:
                    validate_spec(plan.spec, list(plan.data.columns))  # must not raise

    def test_mass_bases_use_a_log_axis_and_concentrations_do_not(self):
        for basis, expected in (("kg", True), ("Ci", True), ("g", True), ("wt%", False)):
            result = _result(["WasteSiteId"], basis=basis)
            plan = plan_plot(result, ["WasteSiteId"], "bar", subject_label="Salt formers")
            assert plan.spec.log_x is expected, basis

    def test_subtitle_carries_the_scope_stamp(self):
        result = _result(["WasteSiteId"])
        plan = plan_plot(result, ["WasteSiteId"], "bar", subject_label="Salt formers",
                         subtitle="Scope: AN")
        assert plan.spec.subtitle == "Scope: AN"
        assert "Salt formers" in plan.spec.title

    def test_value_column_is_taken_from_the_result_not_guessed(self):
        result = _result(["WasteSiteId"], basis="wt%")
        plan = plan_plot(result, ["WasteSiteId"], "bar", subject_label="Salt formers")
        assert plan.spec.x == result.value_column == "wt_pct"
