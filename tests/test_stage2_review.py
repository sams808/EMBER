"""
Repro tests for the Stage-2 adversarial correctness review (commit
8fe0c49). Every test here failed against the pre-fix code; each pins one
way a figure could disagree with the numbers beside it.

The unifying rule: a chart may show less than the query returned, but it
must never do so SILENTLY, and it must never draw lengths that are not
proportional to the values they represent.
"""
import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QMainWindow

import data_model as dm
import polars as pl
from plot_grammar import PlotSpec, render_plot
from qt_query import QueryPage
from qt_widgets import PlotWidget
from query_engine import BASKET_PRESETS, run_query
from query_plots import plan_plot

from conftest import requires_real_data


def _panel(qtbot):
    panel = PlotWidget()
    qtbot.addWidget(panel)
    return panel


def _page(qtbot):
    app_window = QMainWindow()
    qtbot.addWidget(app_window)
    page = QueryPage(app_window)
    qtbot.addWidget(page)
    return page


def _segment_widths(ax):
    """Drawn width of each patch in DATA space per bar row -- what the eye
    actually compares."""
    by_row = {}
    for patch in ax.patches:
        key = round(patch.get_y() + patch.get_height() / 2, 4)
        by_row.setdefault(key, []).append(patch.get_width())
    return by_row


def _pixel_widths(ax):
    """Drawn width of each patch in PIXELS -- on a log axis this is what
    diverges from the value it claims to represent."""
    out = []
    for patch in ax.patches:
        x0 = patch.get_x()
        x1 = x0 + patch.get_width()
        (px0, _), (px1, _) = ax.transData.transform([(max(x0, 1e-300), 0), (max(x1, 1e-300), 0)])
        out.append(px1 - px0)
    return out


class TestStackedBarsAreNeverLogScaled:
    """THE critical finding: stacked segments were laid out at linear
    cumulative offsets and then the axis was switched to log, so a
    segment's drawn length became log(left+v) - log(left) -- which depends
    on its position in the stack, not on its value. On the real dataset
    this rank-REVERSED a composition (F drawn as 65% of a bar where it is
    3.7%; Cr drawn as 6.7% where it is 47%)."""

    def _stacked_df(self):
        return pd.DataFrame({
            "TankFarm": ["SX"] * 2,
            "Member": ["Na", "Cl"],
            "Total_kg": [999.0, 1.0],
        })

    def test_pixel_lengths_stay_proportional_to_values(self, qtbot):
        panel = _panel(qtbot)
        render_plot(panel, self._stacked_df(),
                    PlotSpec(mark="stackedbar", x="Total_kg", y="TankFarm", hue="Member",
                             title="T", log_x=True))
        qtbot.wait(20)
        widths = _pixel_widths(panel.ax)
        assert len(widths) == 2
        big, small = max(widths), min(widths)
        # 999:1 in value must not render as 1:500 in pixels.
        assert big / max(small, 1e-9) > 100, f"segment pixels not proportional: {widths}"

    def test_a_stacked_bar_is_drawn_on_a_linear_axis(self, qtbot):
        panel = _panel(qtbot)
        render_plot(panel, self._stacked_df(),
                    PlotSpec(mark="stackedbar", x="Total_kg", y="TankFarm", hue="Member",
                             title="T", log_x=True))
        qtbot.wait(20)
        assert panel.ax.get_xscale() == "linear"

    def test_the_planner_never_asks_for_a_log_stacked_bar(self):
        dataset = _chem_dataset()
        result = run_query(dataset, subject=("basket", BASKET_PRESETS["Salt formers"]),
                           basis="kg", group_by=["WasteSiteId"])
        plan = plan_plot(result, ["WasteSiteId"], "stackedbar", subject_label="Salt formers")
        assert plan.is_drawable
        assert plan.spec.log_x is False

    def test_plain_bars_still_get_the_log_axis(self):
        dataset = _chem_dataset()
        result = run_query(dataset, subject=("element", "Cl"), basis="kg", group_by=["WasteSiteId"])
        plan = plan_plot(result, ["WasteSiteId"], "bar", subject_label="Cl")
        assert plan.spec.log_x is True

    def test_a_log_axis_says_so_on_the_figure(self, qtbot):
        panel = _panel(qtbot)
        df = pd.DataFrame({"WasteSiteId": ["A", "B"], "Total_kg": [1.0, 1000.0]})
        render_plot(panel, df, PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId",
                                        title="T", log_x=True))
        qtbot.wait(20)
        assert "log" in panel.ax.get_xlabel().lower()


class TestNullGroupsAreShownNotDropped:
    """Rows whose grouping dimension is null were silently deleted at four
    sites. On the real dataset that removed 85.3% of the sulfur mass from a
    chart while the Result tab beside it showed the full total."""

    def _null_df(self):
        return pd.DataFrame({
            "TankStatus": [None, "Active", "Active"],
            "WastePhase": ["Liquid", "Liquid", "Solid"],
            "Total_kg": [80.0, 15.0, 5.0],
        })

    def test_plain_bar_keeps_the_null_category(self, qtbot):
        panel = _panel(qtbot)
        render_plot(panel, self._null_df(),
                    PlotSpec(mark="bar", x="Total_kg", y="TankStatus", title="T"))
        qtbot.wait(20)
        drawn = sum(p.get_width() for p in panel.ax.patches)
        assert drawn == pytest.approx(100.0), "the null-status 80 kg vanished from the chart"
        labels = [t.get_text() for t in panel.ax.get_yticklabels()]
        assert any("not reported" in label for label in labels)

    def test_stacked_bar_keeps_null_hue_levels(self, qtbot):
        panel = _panel(qtbot)
        render_plot(panel, self._null_df(),
                    PlotSpec(mark="stackedbar", x="Total_kg", y="WastePhase",
                             hue="TankStatus", title="T"))
        qtbot.wait(20)
        assert sum(p.get_width() for p in panel.ax.patches) == pytest.approx(100.0)

    def test_heatmap_keeps_null_rows_and_columns(self, qtbot):
        panel = _panel(qtbot)
        render_plot(panel, self._null_df(),
                    PlotSpec(mark="heatmap", x="Total_kg", y="WastePhase",
                             hue="TankStatus", title="T"))
        qtbot.wait(20)
        image = panel.ax.get_images()[0].get_array()
        assert np.nansum(np.asarray(image, dtype=float)) == pytest.approx(100.0)

    def test_box_keeps_the_null_category(self, qtbot):
        panel = _panel(qtbot)
        render_plot(panel, self._null_df(),
                    PlotSpec(mark="box", x="Total_kg", y="TankStatus", title="T"))
        qtbot.wait(20)
        labels = [t.get_text() for t in panel.ax.get_yticklabels()]
        assert any("not reported" in label for label in labels)

    def test_all_null_hue_does_not_crash(self, qtbot):
        # ZeroDivisionError: band = 0.8 / len(hue_levels) with everything dropped.
        panel = _panel(qtbot)
        df = pd.DataFrame({
            "WasteSiteId": ["A", "B"], "TankStatus": [None, None], "Total_kg": [3.0, 4.0],
        })
        render_plot(panel, df, PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId",
                                        hue="TankStatus", title="T"))
        qtbot.wait(20)
        assert sum(p.get_width() for p in panel.ax.patches) == pytest.approx(7.0)


class TestTrimNotesCountTheWholeResult:
    def test_note_denominator_includes_dropped_categories(self, qtbot):
        # 5 categories, 2 of them zero -> the note must say "of 5", not "of 3".
        panel = _panel(qtbot)
        df = pd.DataFrame({
            "WasteType": ["a", "b", "c", "d", "e"],
            "Total_kg": [5.0, 4.0, 3.0, 0.0, 0.0],
        })
        render_plot(panel, df, PlotSpec(mark="bar", x="Total_kg", y="WasteType",
                                        title="T", top_n=2))
        qtbot.wait(20)
        assert "of 5" in panel.ax.get_title()

    def test_non_positive_categories_are_disclosed(self, qtbot):
        panel = _panel(qtbot)
        df = pd.DataFrame({"WasteType": ["a", "b"], "Total_kg": [5.0, 0.0]})
        render_plot(panel, df, PlotSpec(mark="bar", x="Total_kg", y="WasteType", title="T"))
        qtbot.wait(20)
        assert "1 of 2" in panel.ax.get_title() and "not plottable" in panel.ax.get_title()


class TestRatioBasesAreNotStacked:
    """wt% and g/L values each carry their OWN layer denominator. Summing
    them (a stacked bar) yields sum(m/M) rather than sum(m)/sum(M) -- on
    real data that drew a tank at 107.7 "wt% of layer mass"."""

    def test_planner_refuses_a_stacked_ratio_chart(self):
        dataset = _chem_dataset()
        result = run_query(dataset, subject=("basket", BASKET_PRESETS["Salt formers"]),
                           basis="wt%", group_by=["WasteSiteId", "WastePhase"])
        plan = plan_plot(result, ["WasteSiteId", "WastePhase"], "stackedbar",
                         subject_label="Salt formers")
        assert not plan.is_drawable
        assert "denominator" in plan.refusal.lower() or "wt%" in plan.refusal

    def test_mass_bases_still_stack(self):
        dataset = _chem_dataset()
        result = run_query(dataset, subject=("basket", BASKET_PRESETS["Salt formers"]),
                           basis="kg", group_by=["WasteSiteId", "WastePhase"])
        plan = plan_plot(result, ["WasteSiteId", "WastePhase"], "stackedbar",
                         subject_label="Salt formers")
        assert plan.is_drawable

    def test_grouped_ratio_bars_rank_by_peak_not_by_a_meaningless_sum(self, qtbot):
        # Category A: one layer at 40 wt%. Category B: four layers at 15 wt%.
        # Ranking by sum puts B first (60 > 40) although no B layer is
        # anywhere near A's concentration.
        panel = _panel(qtbot)
        df = pd.DataFrame({
            "WasteSiteId": ["A"] + ["B"] * 4,
            "WastePhase": ["p1", "p1", "p2", "p3", "p4"],
            "wt_pct": [40.0, 15.0, 15.0, 15.0, 15.0],
        })
        render_plot(panel, df, PlotSpec(mark="bar", x="wt_pct", y="WasteSiteId",
                                        hue="WastePhase", title="T", top_n=1, intensive=True))
        qtbot.wait(20)
        labels = [t.get_text() for t in panel.ax.get_yticklabels()]
        assert labels == ["A"], f"ranked by sum-of-ratios instead of peak: {labels}"

    def test_planner_marks_ratio_bases_intensive(self):
        dataset = _chem_dataset()
        for basis, expected in (("wt%", True), ("g/L", True), ("kg", False), ("Ci", False)):
            result = run_query(dataset, subject=("element", "Cl"), basis=basis,
                               group_by=["WasteSiteId"])
            plan = plan_plot(result, ["WasteSiteId"], "bar", subject_label="Cl")
            assert plan.spec.intensive is expected, basis


class TestGroupingByUnitsDoesNotCrash:
    def test_units_as_the_category_column_renders(self, qtbot):
        # extras = pdf.set_index("Units") consumed the column, then
        # extras["Units"] raised KeyError straight out of the Qt slot.
        panel = _panel(qtbot)
        df = pd.DataFrame({"Units": ["kg"], "Total_g": [12.0]})
        render_plot(panel, df, PlotSpec(mark="bar", x="Total_g", y="Units", title="T"))
        qtbot.wait(20)
        assert len(panel.ax.patches) == 1


class TestHeatmapBlankVersusZero:
    def test_an_all_nan_cell_reads_as_no_data_not_as_zero(self, qtbot):
        panel = _panel(qtbot)
        df = pd.DataFrame({
            "WasteSiteId": ["T1", "T1", "T2"],
            "WastePhase": ["Liquid", "Solid", "Liquid"],
            "wt_pct": [np.nan, 5.0, 3.0],
        })
        render_plot(panel, df, PlotSpec(mark="heatmap", x="wt_pct", y="WasteSiteId",
                                        hue="WastePhase", title="T"))
        qtbot.wait(20)
        image = np.asarray(panel.ax.get_images()[0].get_array(), dtype=float)
        assert np.isnan(image).sum() == 2  # the blanked cell AND the absent one


class TestFacetTrimIsDisclosed:
    def test_per_facet_trim_reaches_the_figure_footer(self, qtbot):
        panel = _panel(qtbot)
        df = pd.DataFrame({
            "WasteSiteId": [f"T{i}" for i in range(40)] * 2,
            "TankFarm": ["A"] * 40 + ["B"] * 40,
            "Total_kg": [float(i + 1) for i in range(40)] * 2,
        })
        render_plot(panel, df, PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId",
                                        facet="TankFarm", title="F", top_n=5))
        qtbot.wait(20)
        footer = " ".join(t.get_text() for t in panel.figure.texts)
        assert "5" in footer and "40" in footer

    def test_user_top_n_is_honored_per_facet(self, qtbot):
        panel = _panel(qtbot)
        df = pd.DataFrame({
            "WasteSiteId": [f"T{i}" for i in range(40)],
            "TankFarm": ["A"] * 40,
            "Total_kg": [float(i + 1) for i in range(40)],
        })
        render_plot(panel, df, PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId",
                                        facet="TankFarm", title="F", top_n=25))
        qtbot.wait(20)
        drawn = sum(len(ax.patches) for ax in panel.figure.axes)
        assert drawn == 25, f"hardcoded cap overrode top_n: drew {drawn}"


# ----------------------------------------------------------------------
# Page-level state: what the chart claims must match what produced it.
# ----------------------------------------------------------------------
def _chem_dataset() -> dm.HanfordDataset:
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-A-101"] * 5 + ["241-AN-104"],
        "Analyte": ["SO4", "Sulfide", "Cl", "129I", "Cr", "Cl"],
        "WastePhase": ["Supernatant"] * 3 + ["Sludge"] * 3,
        "WasteType": ["T1"] * 3 + ["T2"] * 3,
        "Inventory": [100.0, 5.0, 3.0, 2.0, 4.0, 6.0],
        "Units": ["kg", "kg", "kg", "Ci", "kg", "kg"],
        "Volume": [2.0] * 3 + [1.0] * 3,
        "ComponentDensity": [1.5] * 3 + [1.0] * 3,
    }))
    dataset.report = None
    return dataset


class TestPlotMatchesTheResultItDraws:
    def test_changing_grouping_without_rerunning_does_not_redraw_stale_data(self, qtbot):
        page = _page(qtbot)
        page.on_dataset_changed(_chem_dataset())
        page.kind_combo.setCurrentText("Element")
        page.subject_edit.setText("Cl")
        page.basis_combo.setCurrentText("wt%")
        page.group1_combo.setCurrentText("Tank")
        page.group2_combo.setCurrentText("Waste phase")
        page.run()
        qtbot.wait(20)
        # User narrows the grouping but does NOT re-run, then switches mark.
        page.group2_combo.setCurrentText("—")
        page.mark_combo.setCurrentText("Bar")
        qtbot.wait(20)
        # The plot must still describe the result that was actually run
        # (tank x phase), never collapse a stale ratio table to tank sums.
        drawn = sum(p.get_width() for p in page.plot.ax.patches)
        engine_total = page._result.table["wt_pct"].sum()
        assert drawn == pytest.approx(engine_total, rel=1e-6)

    def test_subtitle_states_the_scope_the_result_was_computed_under(self, qtbot):
        page = _page(qtbot)
        page.on_dataset_changed(_chem_dataset())
        page.kind_combo.setCurrentText("Element")
        page.subject_edit.setText("Cl")
        page.basis_combo.setCurrentText("kg")
        page.run()
        qtbot.wait(20)
        assert "Scope:" not in page.plot.ax.get_title()

        # A scope change whose re-run early-returns (blank subject) must not
        # relabel the standing, unscoped result as scoped.
        page.subject_edit.setText("")
        page.scope.set_filter("WastePhase", ["Sludge"])
        page.on_scope_changed()
        qtbot.wait(20)
        assert "Scope: Sludge" not in page.plot.ax.get_title()


@requires_real_data
class TestChartAgreesWithTheTableOnRealData:
    """The end-to-end statement of every finding above, on the real
    dataset: whatever the Result tab totals, the figure beside it must
    draw -- or say what it left out. TankStatus is null on 31,945 of the
    46,894 real rows, which is what made this the sharpest available
    test case (85.3% of the sulfur mass used to vanish from the chart)."""

    @pytest.fixture
    def loaded_page(self, qtbot):
        dataset = dm.HanfordDataset()
        dataset.load_local_default()
        window = QMainWindow()
        qtbot.addWidget(window)
        page = QueryPage(window)
        qtbot.addWidget(page)
        page.on_dataset_changed(dataset)
        return page

    def _run(self, page, qtbot, *, dim1, dim2, mark):
        page.kind_combo.setCurrentText("Element")
        page.subject_edit.setText("S")
        page.basis_combo.setCurrentText("g")
        page.group1_combo.setCurrentText(dim1)
        page.group2_combo.setCurrentText(dim2)
        page.mark_combo.setCurrentText(mark)
        page.run()
        qtbot.wait(50)

    @pytest.mark.parametrize("mark", ["Bar", "Stacked bar"])
    def test_drawn_mass_matches_the_result_table(self, loaded_page, qtbot, mark):
        self._run(loaded_page, qtbot, dim1="Farm", dim2="Tank status", mark=mark)
        table = loaded_page.result_view.dataframe()
        table_total = float(table[loaded_page._result.value_column].sum())
        drawn = sum(float(p.get_width()) for p in loaded_page.plot.ax.patches)
        assert drawn == pytest.approx(table_total, rel=0.01), (
            f"{mark}: chart lost mass the result table still counts"
        )

    def test_trim_note_counts_the_categories_the_table_holds(self, loaded_page, qtbot):
        self._run(loaded_page, qtbot, dim1="Tank", dim2="Tank status", mark="Bar")
        table = loaded_page.result_view.dataframe()
        n_tanks = table["WasteSiteId"].nunique(dropna=False)
        assert f"of {n_tanks} " in loaded_page.plot.ax.get_title()

    def test_heatmap_keeps_every_farm_the_table_lists(self, loaded_page, qtbot):
        # Six DST farms (AN, AP, AW, AY, AZ, SY) have null TankStatus
        # throughout and used to disappear from the matrix entirely.
        self._run(loaded_page, qtbot, dim1="Farm", dim2="Tank status", mark="Heatmap")
        table = loaded_page.result_view.dataframe()
        drawn_farms = {t.get_text() for t in loaded_page.plot.ax.get_yticklabels()}
        for farm in ("AN", "AP", "AW", "AY", "AZ", "SY"):
            assert farm in drawn_farms, f"{farm} vanished from the heatmap"
        assert drawn_farms >= set(table["TankFarm"].astype(str))


class TestFixedScopeDimsReconcileToo:
    def test_sst_filter_is_dropped_when_the_new_dataset_has_no_sst(self, qtbot):
        from qt_scope import GlobalScope, GlobalScopeBar

        def _ds(tank_types):
            dataset = dm.HanfordDataset()
            names = [f"241-A-10{i}" for i in range(len(tank_types))]
            df = dataset._clean_dataframe(pl.DataFrame({
                "WasteSiteId": names,
                "Analyte": ["Fe"] * len(names),
                "WastePhase": ["Liquid"] * len(names),
                "WasteType": ["T1"] * len(names),
                "Inventory": [10.0] * len(names),
                "Units": ["kg"] * len(names),
            }))
            attrs = dataset._clean_attributes_dataframe(pl.DataFrame({
                "Name": names, "TankType": list(tank_types),
                "TankStatus": [""] * len(names), "TankIntegrity": ["Sound"] * len(names),
                "Capacity": ["1000"] * len(names), "DIL_Gal": ["0"] * len(names),
            }))
            dataset.attrs_df = attrs
            dataset.df = dataset._merge_tank_attributes(df, attrs)
            dataset.report = None
            return dataset

        scope = GlobalScope()
        bar = GlobalScopeBar(scope)
        qtbot.addWidget(bar)
        bar.on_dataset_changed(_ds(["SST-4", "DST"]))
        bar._combos["TankSystem"].setCurrentText("SST")
        assert scope.as_scope() == {"TankSystem": ["SST"]}

        bar.on_dataset_changed(_ds(["DST", "DST"]))  # no SST tanks any more
        assert scope.as_scope() == {}, "an SST filter survived onto an all-DST dataset"
        assert bar._combos["TankSystem"].currentText() == "All"
