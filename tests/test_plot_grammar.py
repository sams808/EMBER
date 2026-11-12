"""
L2 plot grammar tests — the pure spec layer (serialization, validation,
guardrail constants) plus, further down, the renderer against PlotWidget.
"""
import pytest

from plot_grammar import (
    MARKS,
    MAX_BARS,
    MAX_FACETS,
    MAX_HUE_LEVELS,
    PlotSpec,
    default_axis_label,
    spec_replace,
    validate_spec,
)


class TestSpecSerialization:
    def test_round_trips_through_dict(self):
        spec = PlotSpec(
            mark="bar", x="Total_kg", y="WasteSiteId", hue="WastePhase",
            log_x=True, top_n=20, title="T", subtitle="Scope: Farm A",
            annotations=({"kind": "vline", "value": 1.0, "label": "limit"},),
        )
        assert PlotSpec.from_dict(spec.to_dict()) == spec

    def test_to_dict_is_plain_json_types(self):
        import json
        spec = PlotSpec(x="a", y="b", annotations=({"kind": "vline", "value": 2.0},))
        json.dumps(spec.to_dict())  # must not raise

    def test_unknown_field_is_refused_not_silently_dropped(self):
        with pytest.raises(ValueError, match="Unknown PlotSpec field"):
            PlotSpec.from_dict({"x": "a", "y": "b", "colour": "red"})

    def test_spec_is_frozen_and_replace_derives(self):
        spec = PlotSpec(x="a", y="b")
        with pytest.raises(Exception):
            spec.x = "c"  # type: ignore[misc]
        assert spec_replace(spec, x="c").x == "c"
        assert spec.x == "a"


class TestValidation:
    _COLS = ["WasteSiteId", "WastePhase", "Total_kg", "Err"]

    def test_valid_bar_spec_passes(self):
        validate_spec(PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId", hue="WastePhase",
                               errorbars="Err"), self._COLS)

    def test_unknown_mark_refused(self):
        with pytest.raises(ValueError, match="mark"):
            validate_spec(PlotSpec(mark="pie", x="Total_kg", y="WasteSiteId"), self._COLS)

    def test_missing_axis_column_refused(self):
        with pytest.raises(ValueError, match="not a column"):
            validate_spec(PlotSpec(mark="bar", x="Total_g", y="WasteSiteId"), self._COLS)

    def test_empty_axis_role_refused(self):
        with pytest.raises(ValueError, match="required"):
            validate_spec(PlotSpec(mark="bar", x="Total_kg", y=""), self._COLS)

    def test_hue_duplicating_an_axis_refused(self):
        with pytest.raises(ValueError, match="duplicates"):
            validate_spec(PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId",
                                   hue="WasteSiteId"), self._COLS)

    def test_heatmap_requires_hue(self):
        with pytest.raises(ValueError, match="heatmap"):
            validate_spec(PlotSpec(mark="heatmap", x="Total_kg", y="WasteSiteId"), self._COLS)

    def test_heatmap_refuses_facet(self):
        with pytest.raises(ValueError, match="facet"):
            validate_spec(PlotSpec(mark="heatmap", x="Total_kg", y="WasteSiteId",
                                   hue="WastePhase", facet="WastePhase"), self._COLS)

    def test_bad_sort_refused(self):
        with pytest.raises(ValueError, match="sort"):
            validate_spec(PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId", sort="up"),
                          self._COLS)


class TestGuardrails:
    def test_guardrail_constants_are_sane(self):
        # These are UX limits, not physics -- but they must exist and be
        # positive, because the renderer trims against them loudly.
        assert 0 < MAX_HUE_LEVELS <= MAX_BARS
        assert 0 < MAX_FACETS < 100


class TestAxisLabels:
    def test_engine_value_columns_get_unit_honest_labels(self):
        assert "kg" in default_axis_label("Total_kg")
        assert "Ci" in default_axis_label("Total_Ci")
        assert "wt%" in default_axis_label("wt_pct")

    def test_unknown_column_falls_back_to_its_own_name(self):
        assert default_axis_label("Capacity") == "Capacity"


class TestMarkRegistry:
    def test_marks_are_the_single_registry(self):
        assert "bar" in MARKS and "heatmap" in MARKS
        assert len(MARKS) == len(set(MARKS))

    def test_every_mark_has_exactly_one_ui_label(self):
        from plot_grammar import MARK_BY_LABEL, MARK_LABELS
        assert set(MARK_LABELS) == set(MARKS)
        assert len(set(MARK_LABELS.values())) == len(MARKS)  # no two marks share a label
        assert MARK_BY_LABEL == {label: mark for mark, label in MARK_LABELS.items()}

    def test_axis_labels_cover_every_basis_the_engine_can_produce(self):
        # The grammar's unit-honest-label rule has teeth only if a new
        # engine basis can't ship without one.
        from query_engine import VALUE_COLUMN_BY_BASIS
        from plot_grammar import VALUE_AXIS_LABELS
        assert set(VALUE_AXIS_LABELS) == set(VALUE_COLUMN_BY_BASIS.values())


# ======================================================================
# Renderer tests -- real PlotWidget, real matplotlib artists.
# ======================================================================
import pandas as pd

from plot_grammar import render_plot
from qt_widgets import PlotWidget


def _panel(qtbot):
    panel = PlotWidget()
    qtbot.addWidget(panel)
    return panel


def _bar_df():
    return pd.DataFrame({
        "WasteSiteId": ["T1", "T2", "T3"],
        "Total_kg": [30.0, 10.0, 20.0],
    })


def _hue_df():
    return pd.DataFrame({
        "WasteSiteId": ["T1", "T1", "T2", "T2"],
        "WastePhase": ["Liquid", "Solid", "Liquid", "Solid"],
        "Total_kg": [5.0, 15.0, 8.0, 2.0],
    })


class TestRenderBar:
    def test_bars_drawn_largest_at_top_with_unit_honest_xlabel(self, qtbot):
        panel = _panel(qtbot)
        render_plot(panel, _bar_df(), PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId", title="T"))
        qtbot.wait(20)
        assert len(panel.ax.patches) == 3
        labels = [t.get_text() for t in panel.ax.get_yticklabels()]
        assert labels[-1] == "T1"  # largest value reads at the top
        assert "kg" in panel.ax.get_xlabel()
        assert panel.ax.get_title() == "T"

    def test_top_n_trim_is_loud_on_the_figure(self, qtbot):
        panel = _panel(qtbot)
        render_plot(panel, _bar_df(), PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId",
                                               title="T", top_n=2))
        qtbot.wait(20)
        assert len(panel.ax.patches) == 2
        assert "top 2 of 3" in panel.ax.get_title()

    def test_subtitle_moves_title_to_suptitle(self, qtbot):
        panel = _panel(qtbot)
        render_plot(panel, _bar_df(), PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId",
                                               title="Main", subtitle="Scope: Farm A"))
        qtbot.wait(20)
        assert panel.figure.get_suptitle() == "Main"       # suggested_caption stays the real title
        assert "Scope: Farm A" in panel.ax.get_title()

    def test_error_bars_drawn_when_one_row_per_category(self, qtbot):
        df = _bar_df().assign(Err=[3.0, 1.0, 2.0])
        panel = _panel(qtbot)
        render_plot(panel, df, PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId",
                                        title="T", errorbars="Err"))
        qtbot.wait(20)
        assert len(panel.ax.containers) >= 1  # bar container carries the errorbar art
        assert "error bars" not in panel.ax.get_title()

    def test_grouped_bars_one_per_category_hue_pair_with_legend(self, qtbot):
        panel = _panel(qtbot)
        render_plot(panel, _hue_df(), PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId",
                                               hue="WastePhase", title="T"))
        qtbot.wait(20)
        assert len(panel.ax.patches) == 4  # 2 tanks x 2 phases
        assert panel.ax.get_legend() is not None
        # The legend names the dimension as a reader would, not as a column.
        assert panel.ax.get_legend().get_title().get_text() == "Waste phase"

    def test_stacked_bar_widths_sum_to_category_totals(self, qtbot):
        panel = _panel(qtbot)
        render_plot(panel, _hue_df(), PlotSpec(mark="stackedbar", x="Total_kg", y="WasteSiteId",
                                               hue="WastePhase", title="T"))
        qtbot.wait(20)
        widths_by_y = {}
        for patch in panel.ax.patches:
            y = round(patch.get_y() + patch.get_height() / 2, 3)
            widths_by_y[y] = widths_by_y.get(y, 0.0) + patch.get_width()
        assert sorted(widths_by_y.values()) == pytest.approx([10.0, 20.0])


class TestRenderHeatmap:
    def test_matrix_with_colorbar_and_honest_label(self, qtbot):
        panel = _panel(qtbot)
        render_plot(panel, _hue_df(), PlotSpec(mark="heatmap", x="Total_kg", y="WasteSiteId",
                                               hue="WastePhase", title="T"))
        qtbot.wait(20)
        assert len(panel.figure.axes) == 2  # matrix + colorbar
        assert len(panel.ax.get_images()) == 1
        cbar_ax = panel.figure.axes[1]
        assert "kg" in cbar_ax.get_ylabel()

    def test_log_x_transforms_values_and_labels_colorbar(self, qtbot):
        panel = _panel(qtbot)
        render_plot(panel, _hue_df(), PlotSpec(mark="heatmap", x="Total_kg", y="WasteSiteId",
                                               hue="WastePhase", title="T", log_x=True))
        qtbot.wait(20)
        assert "log10" in panel.figure.axes[1].get_ylabel()


class TestRenderScatterAndBox:
    def test_scatter_hue_series_with_legend(self, qtbot):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 1.0, 4.0, 3.0],
            "grp": ["a", "a", "b", "b"],
        })
        panel = _panel(qtbot)
        render_plot(panel, df, PlotSpec(mark="scatter", x="x", y="y", hue="grp", title="T"))
        qtbot.wait(20)
        assert len(panel.ax.collections) == 2
        assert panel.ax.get_legend() is not None

    def test_box_draws_one_box_per_category(self, qtbot):
        df = pd.DataFrame({
            "WastePhase": ["L", "L", "L", "S", "S", "S"],
            "wt_pct": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        panel = _panel(qtbot)
        render_plot(panel, df, PlotSpec(mark="box", x="wt_pct", y="WastePhase", title="T"))
        qtbot.wait(20)
        labels = [t.get_text() for t in panel.ax.get_yticklabels()]
        assert set(labels) == {"L", "S"}
        assert "wt%" in panel.ax.get_xlabel()


class TestRenderFacet:
    def test_one_axes_per_facet_level_plus_suptitle(self, qtbot):
        df = pd.DataFrame({
            "WasteSiteId": ["T1", "T2", "T1", "T2"],
            "TankFarm": ["A", "A", "B", "B"],
            "Total_kg": [1.0, 2.0, 3.0, 4.0],
        })
        panel = _panel(qtbot)
        render_plot(panel, df, PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId",
                                        facet="TankFarm", title="Faceted"))
        qtbot.wait(20)
        visible = [ax for ax in panel.figure.axes if ax.get_visible() and ax.axison]
        assert len(visible) == 2
        assert panel.figure.get_suptitle() == "Faceted"


class TestRendererRefusals:
    def test_empty_data_shows_message(self, qtbot):
        panel = _panel(qtbot)
        render_plot(panel, pd.DataFrame(), PlotSpec(mark="bar", x="a", y="b", title="T"))
        qtbot.wait(20)
        texts = [t.get_text() for t in panel.ax.texts]
        assert any("No data" in t for t in texts)

    def test_invalid_spec_shows_the_reason_not_a_crash(self, qtbot):
        panel = _panel(qtbot)
        render_plot(panel, _bar_df(), PlotSpec(mark="bar", x="NotAColumn", y="WasteSiteId"))
        qtbot.wait(20)
        texts = [t.get_text() for t in panel.ax.texts]
        assert any(t.startswith("Plot spec:") for t in texts)

    def test_hue_guardrail_trims_loudly(self, qtbot):
        n = MAX_HUE_LEVELS + 3
        df = pd.DataFrame({
            "WasteSiteId": ["T1"] * n,
            "WastePhase": [f"P{i}" for i in range(n)],
            "Total_kg": [float(i + 1) for i in range(n)],
        })
        panel = _panel(qtbot)
        render_plot(panel, df, PlotSpec(mark="stackedbar", x="Total_kg", y="WasteSiteId",
                                        hue="WastePhase", title="T"))
        qtbot.wait(20)
        assert len(panel.ax.patches) == MAX_HUE_LEVELS
        assert f"top {MAX_HUE_LEVELS} of {n}" in panel.ax.get_title()


class TestComposability:
    def test_render_is_captured_as_a_recipe(self, qtbot):
        panel = _panel(qtbot)
        render_plot(panel, _bar_df(), PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId", title="T"))
        qtbot.wait(20)
        recipe = panel._last_recipe
        assert recipe is not None
        fn, args, kwargs = recipe
        assert fn.__name__ == "render_plot"
        assert isinstance(args[1], PlotSpec)
