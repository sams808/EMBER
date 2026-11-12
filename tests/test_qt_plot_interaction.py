"""What a chart answers when the pointer is on a mark.

Every test draws a real figure on a real PlotWidget, then drives the
canvas with synthetic matplotlib mouse events at the pixel position of a
known mark. What the tooltip must say and what the signals must carry is
written out from the DATA the test drew, never read back from the layer
under test."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.backend_bases import MouseEvent
from PySide6.QtWidgets import QApplication, QMainWindow

import plot_helpers as ph
from plot_grammar import PlotSpec, render_plot
from qt_widgets import PlotWidget

TANKS = ["241-A-101", "241-AW-106", "241-BX-110"]


class _Host(QMainWindow):
    """A window with the shell's open_tank(), recording what it is asked."""

    def __init__(self):
        super().__init__()
        self.opened = []

    def open_tank(self, tank_id):
        self.opened.append(tank_id)


@pytest.fixture
def host(qtbot):
    """Kept alive by the fixture until the test's pending Qt events have
    run: pytest-qt only holds a weak reference to a widget."""
    window = _Host()
    qtbot.addWidget(window)
    yield window


def _panel(qtbot, host=None):
    panel = PlotWidget()
    if host is not None:
        host.setCentralWidget(panel)
    else:
        qtbot.addWidget(panel)
    return panel


def _ready(panel):
    """Draw, so transforms and tick labels are final."""
    panel.canvas.draw()
    return panel


def _px(ax, x, y):
    return tuple(float(v) for v in ax.transData.transform((x, y)))


def _move(panel, xy):
    panel.canvas.callbacks.process(
        "motion_notify_event", MouseEvent("motion_notify_event", panel.canvas, xy[0], xy[1]))


def _press(panel, xy, button=1, dblclick=False):
    panel.canvas.callbacks.process(
        "button_press_event",
        MouseEvent("button_press_event", panel.canvas, xy[0], xy[1], button=button, dblclick=dblclick))


def _hover(panel, xy):
    seen = []
    panel.markHovered.connect(seen.append)
    try:
        _move(panel, xy)
    finally:
        panel.markHovered.disconnect(seen.append)
    return seen[-1] if seen else None


def _tooltip(panel):
    tip = panel.interaction.tooltip
    return "" if tip.isHidden() else tip.text()


def _bar_centre(ax, patch):
    return _px(ax, patch.get_x() + patch.get_width() / 2.0, patch.get_y() + patch.get_height() / 2.0)


def _tank_bars(panel):
    df = pd.DataFrame({"WasteSiteId": TANKS, "Total_kg": [30.0, 2500.0, 10.0]})
    render_plot(panel, df, PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId", title="T"))
    return _ready(panel)


class TestBarMarks:
    def test_horizontal_bar_names_its_category_value_and_unit(self, qtbot):
        panel = _tank_bars(_panel(qtbot))
        # Bars are drawn bottom to top in ascending value: 10, 30, 2500.
        top = panel.ax.patches[-1]
        mark = _hover(panel, _bar_centre(panel.ax, top))
        assert mark["kind"] == "bar"
        assert mark["label"] == "241-AW-106"
        assert mark["value"] == pytest.approx(2500.0)
        assert mark["unit"] == "kg"
        assert mark["tank_id"] == "241-AW-106"
        assert mark["series"] is None
        tip = _tooltip(panel)
        assert "241-AW-106" in tip and "Total mass (kg)" in tip and "2,500 kg" in tip

    def test_each_bar_answers_for_itself(self, qtbot):
        panel = _tank_bars(_panel(qtbot))
        expected = {"241-BX-110": 10.0, "241-A-101": 30.0, "241-AW-106": 2500.0}
        got = {}
        for patch in panel.ax.patches:
            mark = _hover(panel, _bar_centre(panel.ax, patch))
            got[mark["label"]] = mark["value"]
        assert got == pytest.approx(expected)

    def test_stacked_segment_reports_its_series_and_its_own_length(self, qtbot):
        df = pd.DataFrame({
            "WasteSiteId": ["241-A-101", "241-A-101", "241-AW-106", "241-AW-106"],
            "WastePhase": ["Sludge", "Saltcake", "Sludge", "Saltcake"],
            "Total_kg": [5.0, 15.0, 8.0, 2.0],
        })
        panel = _panel(qtbot)
        render_plot(panel, df, PlotSpec(mark="stackedbar", x="Total_kg", y="WasteSiteId", hue="WastePhase", title="T"))
        _ready(panel)
        seen = {}
        for patch in panel.ax.patches:
            mark = _hover(panel, _bar_centre(panel.ax, patch))
            seen[(mark["label"], mark["series"])] = mark["value"]
        assert seen == pytest.approx({
            ("241-A-101", "Sludge"): 5.0, ("241-A-101", "Saltcake"): 15.0,
            ("241-AW-106", "Sludge"): 8.0, ("241-AW-106", "Saltcake"): 2.0,
        })
        assert "Waste phase" in _tooltip(panel)          # the series is named as a reader would

    def test_grouped_profile_bar_stands_for_the_tank_of_its_series(self, qtbot):
        df = pd.DataFrame({
            "WasteSiteId": ["241-A-101", "241-A-101", "241-AW-106", "241-AW-106"],
            "Element": ["Na", "Fe", "Na", "Fe"],
            "Inventory_sum": [100.0, 10.0, 300.0, 30.0],
            "Units": ["kg"] * 4,
        })
        panel = _panel(qtbot)
        ph.plot_grouped_tank_profile(panel, df, "Inventory_sum", "Tank composition")
        _ready(panel)
        assert panel.ax.get_xlabel() == "Inventory (kg)"     # not the column name "Inventory_sum"
        found = {}
        for container in panel.ax.containers:
            for patch in container.patches:
                mark = _hover(panel, _bar_centre(panel.ax, patch))
                found[(mark["label"], mark["series"])] = (mark["value"], mark["tank_id"])
        assert found == {
            ("Na", "241-A-101"): (pytest.approx(100.0), "241-A-101"),
            ("Na", "241-AW-106"): (pytest.approx(300.0), "241-AW-106"),
            ("Fe", "241-A-101"): (pytest.approx(10.0), "241-A-101"),
            ("Fe", "241-AW-106"): (pytest.approx(30.0), "241-AW-106"),
        }
        # One cluster per element: two categories, not one per (element, tank) row.
        assert [t.get_text() for t in panel.ax.get_yticklabels()] == ["Fe", "Na"]

    def test_vertical_bars_with_no_metadata_fall_back_to_the_tick_labels(self, qtbot):
        panel = _panel(qtbot)
        ax = panel.reset_axes()
        ax.bar(["alpha", "beta"], [3.0, 7.0], label="Counts")
        ax.set_ylabel("Tanks")
        _ready(panel)
        mark = _hover(panel, _bar_centre(ax, ax.patches[1]))
        assert (mark["label"], mark["value"], mark["series"]) == ("beta", pytest.approx(7.0), "Counts")
        assert "Tanks" in _tooltip(panel)

    def test_histogram_bar_reports_its_range(self, qtbot):
        panel = _panel(qtbot)
        ax = panel.reset_axes()
        ax.bar([10.0, 20.0], [4.0, 6.0], width=10.0, align="edge")
        _ready(panel)
        mark = _hover(panel, _bar_centre(ax, ax.patches[1]))
        assert mark["label"] == "20 to 30"
        assert mark["value"] == pytest.approx(6.0)
        assert mark["tank_id"] is None


class TestPointMarks:
    def _pair(self, panel):
        matrix = pd.DataFrame({"WasteSiteId": TANKS, "Cs": [1.0, 5.0, 9.0], "Sr": [2.0, 4.0, 8.0]})
        ph.plot_pair_scatter(panel, matrix, ["Cs", "Sr"], "kg", "inventory")
        return _ready(panel)

    def test_scatter_point_is_named_after_its_tank_with_both_coordinates(self, qtbot):
        panel = self._pair(_panel(qtbot))
        mark = _hover(panel, _px(panel.ax, 5.0, 4.0))
        assert mark["kind"] == "point"
        assert mark["label"] == "241-AW-106" and mark["tank_id"] == "241-AW-106"
        assert (mark["x"], mark["y"]) == pytest.approx((5.0, 4.0))
        tip = _tooltip(panel)
        assert "241-AW-106" in tip
        assert "Cs (inventory, kg)" in tip and "5 kg" in tip
        assert "Sr (inventory, kg)" in tip and "4 kg" in tip

    def test_the_nearest_point_wins_and_empty_space_is_silent(self, qtbot):
        panel = self._pair(_panel(qtbot))
        x, y = _px(panel.ax, 9.0, 8.0)
        assert _hover(panel, (x + 3.0, y - 2.0))["label"] == "241-BX-110"
        _move(panel, _px(panel.ax, 3.0, 7.0))            # nothing drawn here
        assert panel.interaction.current_mark() is None
        assert _tooltip(panel) == ""

    def test_scatter_without_metadata_names_its_series(self, qtbot):
        panel = _panel(qtbot)
        ax = panel.reset_axes()
        ax.scatter([1.0, 2.0], [10.0, 20.0], label="Series A")
        ax.set_xlabel("Volume (kL)")
        ax.set_ylabel("Mass (kg)")
        _ready(panel)
        mark = _hover(panel, _px(ax, 2.0, 20.0))
        assert mark["label"] is None and mark["series"] == "Series A" and mark["tank_id"] is None
        tip = _tooltip(panel)
        assert "Series A" in tip and "2 kL" in tip and "20 kg" in tip

    def test_grammar_scatter_points_carry_the_tank(self, qtbot):
        df = pd.DataFrame({"WasteSiteId": TANKS, "Total_kg": [1.0, 2.0, 3.0], "Total_Ci": [10.0, 20.0, 30.0]})
        panel = _panel(qtbot)
        render_plot(panel, df, PlotSpec(mark="scatter", x="Total_kg", y="Total_Ci", title="T"))
        _ready(panel)
        mark = _hover(panel, _px(panel.ax, 2.0, 20.0))
        assert mark["label"] == "241-AW-106" and mark["tank_id"] == "241-AW-106"
        assert "Total activity (Ci)" in _tooltip(panel)

    def test_pca_scatter_points_are_tanks(self, qtbot):
        if not ph.seaborn_available():
            pytest.skip("seaborn not installed")
        summary = pd.DataFrame({
            "WasteSiteId": TANKS, "PC1": [-1.0, 0.5, 2.0], "PC2": [0.3, -0.7, 1.1],
            "TankFarm": ["A", "AW", "BX"],
        })
        variance = pd.DataFrame({"PC": ["PC1", "PC2"], "ExplainedVarianceRatio": [0.5, 0.2]})
        panel = _panel(qtbot)
        ph.plot_pca_scatter(panel, summary, "TankFarm", variance)
        _ready(panel)
        mark = _hover(panel, _px(panel.ax, 2.0, 1.1))
        assert mark["label"] == "241-BX-110" and mark["tank_id"] == "241-BX-110" and mark["series"] == "BX"
        tip = _tooltip(panel)
        assert "PC1 (50.0%)" in tip and "Farm" in tip
        assert panel.ax.get_legend().get_title().get_text() == "Farm"

    def test_dendrogram_leaf_is_its_tank(self, qtbot):
        if not ph.seaborn_available():
            pytest.skip("seaborn not installed")
        from scipy.cluster.hierarchy import linkage
        points = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]])
        panel = _panel(qtbot)
        ph.plot_dendrogram(panel, linkage(points, method="average"), TANKS)
        _ready(panel)
        order = [t.get_text() for t in panel.ax.get_xticklabels()]
        assert sorted(order) == sorted(TANKS)
        for i, tank in enumerate(order):
            mark = _hover(panel, _px(panel.ax, 5.0 + 10.0 * i, 0.0))
            assert mark["label"] == tank and mark["tank_id"] == tank
        assert "Distance" not in _tooltip(panel)                 # a leaf has a name, not coordinates

    def test_network_node_is_its_element(self, qtbot):
        if not ph.seaborn_available():
            pytest.skip("seaborn not installed")
        nodes = pd.DataFrame({"Element": ["Cs", "Sr"], "x": [0.0, 1.0], "y": [0.0, 1.0],
                              "LogTotalInventory": [2.0, 4.0], "N_edges": [1, 1]})
        edges = pd.DataFrame({"Element_A": ["Cs"], "Element_B": ["Sr"], "AbsCorrelation": [0.9], "Sign": ["positive"]})
        panel = _panel(qtbot)
        ph.plot_element_network(panel, nodes, edges)
        _ready(panel)
        mark = _hover(panel, _px(panel.ax, 1.0, 1.0))
        assert mark["label"] == "Sr" and mark["tank_id"] is None
        tip = _tooltip(panel)
        assert "Links" in tip and "log10 total inventory" in tip and "Layout" not in tip

    def test_line_vertex(self, qtbot):
        panel = _panel(qtbot)
        ax = panel.reset_axes()
        ax.plot([2015, 2016, 2017], [5.0, 7.0, 6.0], marker="o", label="Volume")
        ax.set_ylabel("Volume (kL)")
        _ready(panel)
        mark = _hover(panel, _px(ax, 2016, 7.0))
        assert mark["kind"] == "line" and mark["series"] == "Volume"
        assert (mark["x"], mark["y"]) == pytest.approx((2016.0, 7.0))
        assert "7 kL" in _tooltip(panel)


class TestCellMarks:
    def _matrix(self, panel, **spec):
        df = pd.DataFrame({
            "WasteSiteId": ["241-A-101", "241-A-101", "241-AW-106", "241-AW-106"],
            "WastePhase": ["Sludge", "Saltcake", "Sludge", "Saltcake"],
            "Total_kg": [5.0, 15.0, 80.0, np.nan],
        })
        render_plot(panel, df, PlotSpec(mark="heatmap", x="Total_kg", y="WasteSiteId", hue="WastePhase",
                                        title="T", **spec))
        return _ready(panel)

    def test_imshow_cell_gives_row_column_and_value(self, qtbot):
        panel = self._matrix(_panel(qtbot))
        image = panel.ax.images[0]
        array = np.asarray(image.get_array(), dtype=float)
        rows = [t for t in ("241-AW-106", "241-A-101")]              # ordered by row total, largest first
        cols = sorted(["Saltcake", "Sludge"])
        for r, row in enumerate(rows):
            for c, col in enumerate(cols):
                mark = _hover(panel, _px(panel.ax, c, r))
                assert mark["kind"] == "cell"
                assert (mark["row"], mark["column"]) == (row, col)
                assert mark["tank_id"] == row
                if np.isfinite(array[r, c]):
                    assert mark["value"] == pytest.approx(array[r, c])
                else:
                    assert mark["value"] is None
        tip = _tooltip(panel)
        assert "Tank" in tip and "Waste phase" in tip and "Total mass (kg)" in tip

    def test_values_match_the_data_that_was_drawn(self, qtbot):
        panel = self._matrix(_panel(qtbot))
        assert _hover(panel, _px(panel.ax, 1, 0))["value"] == pytest.approx(80.0)     # AW-106 / Sludge
        assert _hover(panel, _px(panel.ax, 0, 1))["value"] == pytest.approx(15.0)     # A-101 / Saltcake
        assert "80 kg" not in _tooltip(panel) and "15 kg" in _tooltip(panel)

    def test_a_cell_with_no_measurement_says_so(self, qtbot):
        panel = self._matrix(_panel(qtbot))
        mark = _hover(panel, _px(panel.ax, 0, 0))                                      # AW-106 / Saltcake: blank
        assert mark["value"] is None
        assert "not reported" in _tooltip(panel)

    def test_the_colorbar_is_not_a_matrix(self, qtbot):
        panel = self._matrix(_panel(qtbot))
        cax = panel.figure.axes[1]
        box = cax.bbox
        _move(panel, ((box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0))
        assert panel.interaction.current_mark() is None

    def test_seaborn_mesh_cell(self, qtbot):
        if not ph.seaborn_available():
            pytest.skip("seaborn not installed")
        corr = pd.DataFrame({"Element": ["Cs", "Sr", "Mo"], "Cs": [1.0, 0.9, -0.8], "Sr": [0.9, 1.0, -0.7],
                             "Mo": [-0.8, -0.7, 1.0]})
        panel = _panel(qtbot)
        ph.plot_correlation_heatmap(panel, corr, style="Seaborn lower triangle")
        _ready(panel)
        mark = _hover(panel, _px(panel.ax, 0.5, 2.5))            # row Mo, column Cs (cells centred on i + 0.5)
        assert (mark["row"], mark["column"], mark["value"]) == ("Mo", "Cs", pytest.approx(-0.8))
        assert "Correlation r" in _tooltip(panel)
        _move(panel, _px(panel.ax, 2.5, 0.5))                    # the masked upper triangle is not a cell
        assert panel.interaction.current_mark() is None

    def test_association_matrix_cell_and_blank_pair(self, qtbot):
        table = pd.DataFrame({"Component": ["F", "Cl", "Cr"], "F": [1.0, 0.5, np.nan], "Cl": [0.5, 1.0, -0.25],
                              "Cr": [np.nan, -0.25, 1.0]})
        panel = _panel(qtbot)
        ph.plot_association_heatmap(panel, table, title="Cross-correlation")
        _ready(panel)
        # Rows are components 2..n, columns components 1..n-1.
        mark = _hover(panel, _px(panel.ax, 1, 1))
        assert (mark["row"], mark["column"], mark["value"]) == ("Cr", "Cl", pytest.approx(-0.25))
        blank = _hover(panel, _px(panel.ax, 0, 1))
        assert (blank["row"], blank["column"], blank["value"]) == ("Cr", "F", None)
        assert "left blank" in _tooltip(panel)


class TestLegibleLabels:
    def _many(self, panel, n):
        df = pd.DataFrame({"Element": [f"Element {i:03d}" for i in range(n)],
                           "TotalInventory": [float(n - i) for i in range(n)]})
        ph.plot_barh(panel, df, "Element", "TotalInventory", "t", "Total_kg", top_n=n)
        return panel

    def test_category_fit_shrinks_then_thins(self):
        # 100 dpi: a line of 8.5 pt text needs 8.5 * 100/72 * 1.22 = 14.4 px.
        assert ph.category_fit(10, 30.0, 100.0) == (ph.LABEL_FONT_MAX, 1)
        size, step = ph.category_fit(40, 12.0, 100.0)                  # 7 pt needs 11.9 px
        assert step == 1 and size == pytest.approx(7.0)
        size, step = ph.category_fit(80, 5.0, 100.0)                   # 7.5 pt needs 12.7 px: every 3rd
        assert (size, step) == (pytest.approx(7.5), 3)

    def test_too_many_bars_for_the_room_thins_the_labels_not_the_bars(self, qtbot):
        panel = self._many(_panel(qtbot), 80)
        panel.figure.set_size_inches(6.0, 3.5)                         # a short pane
        panel.relayout(draw=False)
        _ready(panel)
        assert len(panel.ax.patches) == 80                             # every bar is still drawn
        labels = [t.get_text() for t in panel.ax.get_yticklabels()]
        assert 2 <= len(labels) < 80
        assert labels[-1] == "Element 000"                             # the top-ranked bar keeps its label
        assert "label shown" in panel.ax.get_xlabel() and "Total (kg)" in panel.ax.get_xlabel()
        # No two labels overlap.
        renderer = panel.canvas.get_renderer()
        boxes = [t.get_window_extent(renderer) for t in panel.ax.get_yticklabels()]
        for lower, upper in zip(boxes, boxes[1:]):
            assert lower.y1 <= upper.y0 + 0.5

    def test_a_thinned_bar_still_answers_with_its_full_name(self, qtbot):
        panel = self._many(_panel(qtbot), 80)
        panel.figure.set_size_inches(6.0, 3.5)
        panel.relayout(draw=False)
        _ready(panel)
        shown = {t.get_text() for t in panel.ax.get_yticklabels()}
        hidden = next(i for i in range(80) if f"Element {i:03d}" not in shown)
        patch = panel.ax.patches[79 - hidden]                          # drawn bottom to top
        mark = _hover(panel, _bar_centre(panel.ax, patch))
        assert mark["label"] == f"Element {hidden:03d}"
        assert mark["value"] == pytest.approx(80.0 - hidden)

    def test_room_coming_back_restores_every_label_and_removes_the_note(self, qtbot):
        panel = self._many(_panel(qtbot), 80)
        panel.figure.set_size_inches(6.0, 3.5)
        panel.relayout(draw=False)
        panel.figure.set_size_inches(6.0, 16.0)
        panel.relayout(draw=False)
        assert len(panel.ax.get_yticklabels()) == 80
        assert panel.ax.get_xlabel() == "Total (kg)"

    def test_long_labels_are_cut_with_the_full_text_in_the_tooltip(self, qtbot):
        long_name = "241-AW-102/Saltcake Interstitial Liquid with a long qualifier"
        df = pd.DataFrame({"Layer": [long_name, "short"], "TotalInventory": [5.0, 3.0]})
        panel = _panel(qtbot)
        ph.plot_barh(panel, df, "Layer", "TotalInventory", "t", "x")
        _ready(panel)
        labels = [t.get_text() for t in panel.ax.get_yticklabels()]
        assert labels[-1].endswith("…") and len(labels[-1]) <= ph.LABEL_MAX_CHARS
        assert long_name.startswith(labels[-1][:-1])
        mark = _hover(panel, _bar_centre(panel.ax, panel.ax.patches[-1]))
        assert mark["label"] == long_name
        assert mark["tank_id"] == "241-AW-102"

    def test_elide_label(self):
        assert ph.elide_label("short") == "short"
        assert ph.elide_label("abcdefghij", 6) == "abcde…"


class TestClickAndMenu:
    def test_click_and_double_click_emit_and_open_the_tank(self, qtbot, host):
        panel = _tank_bars(_panel(qtbot, host))
        clicked, activated = [], []
        panel.markClicked.connect(clicked.append)
        panel.markActivated.connect(activated.append)
        spot = _bar_centre(panel.ax, panel.ax.patches[-1])
        _press(panel, spot)
        assert [m["label"] for m in clicked] == ["241-AW-106"] and not activated and host.opened == []
        _press(panel, spot, dblclick=True)
        assert [m["tank_id"] for m in activated] == ["241-AW-106"]
        assert host.opened == ["241-AW-106"]
        assert set(activated[0]) >= {"kind", "label", "series", "value", "row", "column", "tank_id", "unit"}
        assert not any(key.startswith("_") for key in activated[0])

    def test_double_click_without_a_shell_only_emits(self, qtbot):
        panel = _tank_bars(_panel(qtbot))                     # top-level: its window has no open_tank
        activated = []
        panel.markActivated.connect(activated.append)
        _press(panel, _bar_centre(panel.ax, panel.ax.patches[0]), dblclick=True)
        assert [m["tank_id"] for m in activated] == ["241-BX-110"]

    def test_a_mark_that_is_not_a_tank_opens_nothing(self, qtbot, host):
        panel = _panel(qtbot, host)
        df = pd.DataFrame({"Element": ["Na", "Fe"], "TotalInventory": [5.0, 3.0]})
        ph.plot_barh(panel, df, "Element", "TotalInventory", "t", "x")
        _ready(panel)
        _press(panel, _bar_centre(panel.ax, panel.ax.patches[-1]), dblclick=True)
        assert host.opened == []

    def test_context_menu_copies_value_and_label_and_opens_the_tank(self, qtbot, host):
        panel = _tank_bars(_panel(qtbot, host))
        mark = _hover(panel, _bar_centre(panel.ax, panel.ax.patches[-1]))
        menu = panel.interaction.build_menu(mark)
        actions = {a.text(): a for a in menu.actions() if a.text()}
        assert list(actions) == ["Copy value", "Copy label", "Open tank 241-AW-106"]
        actions["Copy value"].trigger()
        assert float(QApplication.clipboard().text()) == 2500.0          # full precision, no formatting
        actions["Copy label"].trigger()
        assert QApplication.clipboard().text() == "241-AW-106"
        actions["Open tank 241-AW-106"].trigger()
        assert host.opened == ["241-AW-106"]

    def test_no_open_tank_entry_without_a_shell_or_a_tank(self, qtbot):
        panel = _tank_bars(_panel(qtbot))
        mark = _hover(panel, _bar_centre(panel.ax, panel.ax.patches[-1]))
        assert [a.text() for a in panel.interaction.build_menu(mark).actions() if a.text()] == [
            "Copy value", "Copy label"]

    def test_right_click_off_a_mark_does_nothing(self, qtbot):
        panel = _tank_bars(_panel(qtbot))
        box = panel.ax.bbox
        _press(panel, (box.x1 - 2.0, box.y0 + 2.0), button=3)
        assert panel.interaction._menu is None


class TestLifecycle:
    def test_nothing_is_indexed_until_the_pointer_moves(self, qtbot):
        panel = _tank_bars(_panel(qtbot))
        assert panel.interaction.index._axes == {}
        _move(panel, _bar_centre(panel.ax, panel.ax.patches[0]))
        assert len(panel.interaction.index._axes) == 1

    def test_a_redraw_discards_the_index_so_no_stale_mark_can_answer(self, qtbot):
        panel = _tank_bars(_panel(qtbot))
        _move(panel, _bar_centre(panel.ax, panel.ax.patches[-1]))
        assert panel.interaction.current_mark()["label"] == "241-AW-106"
        df = pd.DataFrame({"WasteSiteId": ["241-C-101", "241-C-102"], "Total_kg": [7.0, 9.0]})
        render_plot(panel, df, PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId", title="T"))
        _ready(panel)
        assert panel.interaction.index._axes == {}
        assert panel.interaction.current_mark() is None and _tooltip(panel) == ""
        mark = _hover(panel, _bar_centre(panel.ax, panel.ax.patches[-1]))
        assert (mark["label"], mark["value"]) == ("241-C-102", pytest.approx(9.0))

    def test_inactive_while_a_toolbar_mode_is_engaged(self, qtbot):
        panel = _tank_bars(_panel(qtbot))
        spot = _bar_centre(panel.ax, panel.ax.patches[-1])
        panel.toolbar.zoom()
        clicked = []
        panel.markClicked.connect(clicked.append)
        _move(panel, spot)
        _press(panel, spot)
        assert panel.interaction.current_mark() is None and _tooltip(panel) == "" and clicked == []
        panel.toolbar.zoom()                                   # off again
        _move(panel, spot)
        assert panel.interaction.current_mark()["label"] == "241-AW-106"

    def test_leaving_the_figure_hides_the_tooltip_and_clears_the_readout(self, qtbot):
        panel = _tank_bars(_panel(qtbot))
        _move(panel, _bar_centre(panel.ax, panel.ax.patches[-1]))
        assert _tooltip(panel) and panel.coords_label.text()
        panel.canvas.callbacks.process("figure_leave_event", None)
        assert _tooltip(panel) == "" and panel.coords_label.text() == ""

    def test_readout_names_the_mark_then_the_position(self, qtbot):
        panel = _tank_bars(_panel(qtbot))
        _move(panel, _bar_centre(panel.ax, panel.ax.patches[-1]))
        text = panel.coords_label.text()
        assert "241-AW-106" in text and "Total mass (kg): 2,500 kg" in text
        box = panel.ax.bbox
        _move(panel, (box.x1 - 3.0, _bar_centre(panel.ax, panel.ax.patches[0])[1]))    # right of the short bar
        text = panel.coords_label.text()
        assert "241-BX-110" in text and "Total mass (kg):" in text            # category, then a formatted value
        assert "e+" not in text

    def test_the_outline_is_never_part_of_the_figure(self, qtbot):
        panel = _tank_bars(_panel(qtbot))
        before = (len(panel.ax.patches), len(panel.ax.lines), len(panel.figure.artists))
        _move(panel, _bar_centre(panel.ax, panel.ax.patches[-1]))
        assert (len(panel.ax.patches), len(panel.ax.lines), len(panel.figure.artists)) == before


class TestReplay:
    """A captured plot replays in the Figure Composer exactly as drawn."""

    def _texts(self, ax):
        return {
            "title": ax.get_title(), "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
            "yticks": [t.get_text() for t in ax.get_yticklabels()],
            "bars": [(round(p.get_width(), 9), tuple(round(c, 6) for c in p.get_facecolor())) for p in ax.patches],
        }

    def test_bar_recipe_replays_with_the_same_labels_values_and_colours(self, qtbot):
        from matplotlib.figure import Figure
        from figure_composer import ComposerCellPanel

        panel = _tank_bars(_panel(qtbot))
        render_fn, args, kwargs = panel._last_recipe
        cell = ComposerCellPanel(Figure(figsize=(6.0, 4.5)))
        render_fn(cell, *args, **kwargs)
        assert self._texts(cell.ax) == self._texts(panel.ax)
        assert cell.ax._ember_marks["categories"] == panel.ax._ember_marks["categories"]

    def test_composed_figure_keeps_readable_labels_and_house_colours(self, qtbot):
        from matplotlib.colors import to_rgba
        import figure_composer as fcomp
        from composer_store import store

        store.clear()
        try:
            panel = _panel(qtbot)
            df = pd.DataFrame({"WasteSiteId": TANKS, "User_search_score_proxy": [3.0, 2.0, 1.0]})
            ph.plot_candidate_scores(panel, df)
            panel._send_to_composer()
            out = _panel(qtbot)
            composed = fcomp.compose_figure(out, store.items(), cols=1)
            ax = composed.cells[0].ax
            assert ax.get_xlabel() == "Search score (proxy), dimensionless"
            assert [t.get_text() for t in ax.get_yticklabels()] == ["241-BX-110", "241-AW-106", "241-A-101"]
            assert {tuple(p.get_facecolor()) for p in ax.patches} == {to_rgba("#2a78d6")}
            assert not composed.cells[0].render_failed
        finally:
            store.clear()


class TestReadings:
    def test_a_log_coloured_cell_gives_the_quantity_itself_first(self, qtbot):
        import math
        df = pd.DataFrame({
            "WasteSiteId": ["241-A-101", "241-A-101", "241-AW-106", "241-AW-106"],
            "WastePhase": ["Sludge", "Saltcake", "Sludge", "Saltcake"],
            "Total_kg": [5.0, 15.0, 80.0, 1000.0],
        })
        panel = _panel(qtbot)
        render_plot(panel, df, PlotSpec(mark="heatmap", x="Total_kg", y="WasteSiteId", hue="WastePhase",
                                        title="T", log_x=True))
        _ready(panel)
        mark = _hover(panel, _px(panel.ax, 0, 0))                    # 241-AW-106 / Saltcake = 1000 kg
        assert mark["value"] == pytest.approx(math.log10(1000.0))
        tip = _tooltip(panel)
        assert "Total mass (kg)" in tip and "1,000 kg" in tip and "log10" in tip
        assert "3 kg" not in tip                                     # a logarithm is not given the unit

    def test_a_matrix_readout_names_the_row_and_column(self, qtbot):
        df = pd.DataFrame({
            "WasteSiteId": ["241-A-101", "241-A-101", "241-AW-106", "241-AW-106"],
            "WastePhase": ["Sludge", "Saltcake", "Sludge", "Saltcake"],
            "Total_kg": [5.0, 15.0, 80.0, 1000.0],
        })
        panel = _panel(qtbot)
        render_plot(panel, df, PlotSpec(mark="heatmap", x="Total_kg", y="WasteSiteId", hue="WastePhase", title="T"))
        _ready(panel)
        x, y = _px(panel.ax, 1, 1)
        assert panel.interaction.index.readout(x, y) == "Tank: 241-A-101     Waste phase: Sludge"

    def test_a_tooltip_writes_billions_out(self, qtbot):
        panel = _panel(qtbot)
        df = pd.DataFrame({"Element": ["F", "Na"], "Total_g": [1.238e9, 4.642e10]})
        render_plot(panel, df, PlotSpec(mark="bar", x="Total_g", y="Element", title="T"))
        _ready(panel)
        _hover(panel, _bar_centre(panel.ax, panel.ax.patches[0]))
        assert "1,238,000,000 g" in _tooltip(panel)

    def test_matrix_columns_are_all_labelled_when_they_fit(self, qtbot):
        df = pd.DataFrame({
            "WasteSiteId": ["T1"] * 6, "Element": ["Na", "Al", "Fe", "Cs", "Sr", "U"],
            "Total_kg": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        panel = _panel(qtbot)
        render_plot(panel, df, PlotSpec(mark="heatmap", x="Total_kg", y="WasteSiteId", hue="Element", title="T"))
        assert [t.get_text() for t in panel.ax.get_xticklabels()] == sorted(["Na", "Al", "Fe", "Cs", "Sr", "U"])
        assert [t.get_text() for t in panel.ax.get_yticklabels()] == ["T1"]
