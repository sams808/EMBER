"""PlotWidget: the canvas fills its pane, the export size does not depend
on the window, an empty panel says so, and the toolbar is tidy."""
import pandas as pd
import pytest
from PySide6.QtWidgets import QToolBar

import plot_helpers as ph
from plot_grammar import PlotSpec, render_plot
from qt_widgets import EMPTY_TITLE, PlotWidget


def _shown(qtbot, width=900, height=640):
    panel = PlotWidget()
    qtbot.addWidget(panel)
    panel.resize(width, height)
    panel.show()
    qtbot.waitExposed(panel)
    qtbot.wait(30)
    return panel


def _bars(panel, n=3):
    df = pd.DataFrame({"WasteSiteId": [f"241-A-{101 + i}" for i in range(n)],
                       "Total_kg": [float(n - i) for i in range(n)]})
    render_plot(panel, df, PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId", title="T", top_n=n))
    return panel


def _png_size(path):
    import struct
    with open(path, "rb") as handle:
        head = handle.read(24)
    return struct.unpack(">II", head[16:24])


class TestCanvasFillsItsPane:
    def test_canvas_takes_all_the_room_between_toolbar_and_readout(self, qtbot):
        panel = _shown(qtbot)
        try:
            expected = panel.height() - panel.toolbar.height() - panel.coords_label.height()
            assert abs(panel.canvas.height() - expected) <= 1
            assert panel.canvas.width() == panel.width()
        finally:
            panel.hide()

    def test_a_small_chart_does_not_shrink_the_canvas(self, qtbot):
        panel = _shown(qtbot)
        try:
            before = (panel.canvas.width(), panel.canvas.height())
            _bars(panel, 2)                                   # asks for the default 6.0 x 4.5 in
            qtbot.wait(30)
            assert (panel.canvas.width(), panel.canvas.height()) == before
            ratio = panel.canvas.device_pixel_ratio
            w_px, h_px = panel.canvas.get_width_height()
            assert abs(w_px / ratio - before[0]) <= 1 and abs(h_px / ratio - before[1]) <= 1
        finally:
            panel.hide()

    def test_a_tall_request_does_not_grow_the_canvas_past_the_pane(self, qtbot):
        panel = _shown(qtbot, height=500)
        try:
            before = panel.canvas.height()
            _bars(panel, 60)                                  # 60 bars ask for far more than 500 px
            qtbot.wait(30)
            assert panel.canvas.height() == before
            assert len(panel.ax.patches) == 60
        finally:
            panel.hide()

    def test_figure_follows_a_resize_and_is_laid_out_again(self, qtbot):
        panel = _shown(qtbot)
        try:
            _bars(panel, 3)
            qtbot.wait(30)
            panel.canvas.draw()
            wide = panel.ax.get_position().width * panel.figure.get_size_inches()[0]
            panel.resize(520, 640)
            qtbot.wait(400)                                   # the debounced relayout
            panel.canvas.draw()
            assert panel.figure.get_size_inches()[0] < 6.0
            narrow = panel.ax.get_position().width * panel.figure.get_size_inches()[0]
            assert narrow < wide
            # Margins were recomputed: the tick labels are still inside the figure.
            renderer = panel.canvas.get_renderer()
            for label in panel.ax.get_yticklabels():
                assert label.get_window_extent(renderer).x0 >= 0
        finally:
            panel.hide()

    def test_thinned_labels_are_announced_under_the_axis_label(self, qtbot):
        # Drawn straight into a short, visible pane: the labels are thinned
        # at once, and the sentence saying so must survive the plot
        # function setting its own axis label afterwards.
        panel = _shown(qtbot, height=420)
        try:
            df = pd.DataFrame({"WasteSiteId": [f"241-A-{101 + i}" for i in range(60)],
                               "TotalInventory": [float(60 - i) for i in range(60)]})
            ph.plot_barh(panel, df, "WasteSiteId", "TotalInventory", "Target inventory by tank",
                         "Target inventory: Se", top_n=60)
            qtbot.wait(30)
            lines = panel.ax.get_xlabel().split("\n")
            assert lines[0] == "Target inventory: Se"
            assert len(lines) == 2 and "tank label shown" in lines[1] and "Point at a bar" in lines[1]
            assert len(panel.ax.get_yticklabels()) < 60 and len(panel.ax.patches) == 60
        finally:
            panel.hide()

    def test_minimum_size_is_modest_and_fixed(self, qtbot):
        panel = PlotWidget()
        qtbot.addWidget(panel)
        _bars(panel, 60)
        assert (panel.canvas.minimumWidth(), panel.canvas.minimumHeight()) == PlotWidget.MIN_CANVAS_SIZE

    def test_a_long_readout_cannot_widen_the_widget(self, qtbot):
        panel = PlotWidget()
        qtbot.addWidget(panel)
        before = panel.minimumSizeHint().width()
        panel.coords_label.setText("x" * 2000)
        assert panel.minimumSizeHint().width() == before


class TestTeardown:
    def test_a_redraw_queued_on_a_destroyed_canvas_is_dropped(self, qtbot):
        from PySide6.QtWidgets import QApplication, QWidget
        import shiboken6

        parent = QWidget()
        panel = PlotWidget(parent)
        canvas = panel.canvas
        _bars(panel, 3)                     # leaves a draw_idle queued on a zero-delay timer
        assert canvas._draw_pending
        shiboken6.delete(parent)            # the page is torn down before the timer fires
        assert not shiboken6.isValid(canvas)
        with qtbot.captureExceptions() as raised:
            QApplication.processEvents()
        assert raised == []


class TestExportSize:
    def test_natural_size_is_what_the_plot_asked_for(self, qtbot):
        panel = PlotWidget()
        qtbot.addWidget(panel)
        _bars(panel, 40)
        assert panel.natural_size_inches() == (6.0, pytest.approx(ph.bar_figure_height(40)))
        assert tuple(panel.figure.get_size_inches()) == pytest.approx(panel.natural_size_inches())

    def test_savefig_from_a_shown_panel_uses_the_export_size_not_the_pane(self, qtbot, tmp_path):
        panel = _shown(qtbot, width=900, height=640)
        try:
            _bars(panel, 40)
            qtbot.wait(30)
            live = tuple(panel.figure.get_size_inches())
            export_w, export_h = panel.export_size_inches()
            # The pane is 900 px wide and ~560 px tall; 40 bars ask for more height.
            assert export_h > live[1] and export_w == pytest.approx(live[0])
            path = tmp_path / "plot.png"
            panel.figure.savefig(path, dpi=100)               # exactly what the pages' exports call
            width, height = _png_size(path)
            assert abs(width - export_w * 100) <= 2 and abs(height - export_h * 100) <= 2
            # ...and the on-screen figure is back to the pane's size.
            assert tuple(panel.figure.get_size_inches()) == pytest.approx(live)
        finally:
            panel.hide()

    def test_export_is_the_same_whatever_the_window_height(self, qtbot, tmp_path):
        sizes = []
        for height in (480, 760):
            panel = _shown(qtbot, width=900, height=height)
            try:
                _bars(panel, 12)
                qtbot.wait(30)
                path = tmp_path / f"plot_{height}.png"
                panel.figure.savefig(path, dpi=100)
                sizes.append(_png_size(path))
            finally:
                panel.hide()
        assert sizes[0] == sizes[1]

    def test_unshown_panel_exports_at_the_requested_size(self, qtbot, tmp_path):
        panel = PlotWidget()
        qtbot.addWidget(panel)
        _bars(panel, 40)
        path = tmp_path / "plot.png"
        panel.figure.savefig(path, dpi=100)
        width, height = _png_size(path)
        assert (width, height) == (600, int(round(ph.bar_figure_height(40) * 100)))

    def test_export_at_size_cm_is_exact_and_restores_the_figure(self, qtbot, tmp_path):
        panel = _shown(qtbot)
        try:
            _bars(panel, 5)
            qtbot.wait(30)
            live = tuple(panel.figure.get_size_inches())
            path = tmp_path / "plot.png"
            panel.export_at_size_cm(str(path), 12.7, 7.62, dpi=100)       # 5 x 3 in
            assert _png_size(path) == (500, 300)
            assert tuple(panel.figure.get_size_inches()) == pytest.approx(live)
        finally:
            panel.hide()

    def test_export_figure_writes_vector_and_raster(self, qtbot, tmp_path):
        panel = PlotWidget()
        qtbot.addWidget(panel)
        _bars(panel, 3)
        for suffix in ("png", "svg", "pdf"):
            path = tmp_path / f"chart.{suffix}"
            panel.export_figure(str(path))
            assert path.exists() and path.stat().st_size > 500


class TestEmptyState:
    def test_show_empty_draws_a_sentence_not_axes(self, qtbot):
        panel = PlotWidget()
        qtbot.addWidget(panel)
        panel.show_empty("No tanks selected", "Pick one or more tanks, then Run.")
        assert panel.ax.axison is False
        assert [t.get_text() for t in panel.ax.texts] == ["No tanks selected", "Pick one or more tanks, then Run."]

    def test_clear_goes_to_the_empty_state(self, qtbot):
        panel = PlotWidget()
        qtbot.addWidget(panel)
        _bars(panel, 3)
        panel.clear()
        assert panel.ax.axison is False and not panel.ax.patches
        assert panel.ax.texts[0].get_text() == EMPTY_TITLE
        assert panel._last_recipe is None                      # nothing left to send to the composer

    def test_a_never_drawn_panel_is_not_bare_axes_once_shown(self, qtbot):
        panel = _shown(qtbot)
        try:
            assert panel.ax.axison is False
            assert panel.ax.texts[0].get_text() == EMPTY_TITLE
        finally:
            panel.hide()

    def test_showing_a_panel_never_replaces_what_a_page_already_drew(self, qtbot):
        panel = PlotWidget()
        qtbot.addWidget(panel)
        _bars(panel, 3)
        panel.show()
        qtbot.waitExposed(panel)
        try:
            assert len(panel.ax.patches) == 3
        finally:
            panel.hide()

    def test_a_hint_written_on_the_first_axes_is_left_alone(self, qtbot):
        panel = PlotWidget()
        qtbot.addWidget(panel)
        panel.ax.axis("off")
        panel.ax.text(0.5, 0.5, "Choose components, then Run.")
        panel.show()
        qtbot.waitExposed(panel)
        try:
            assert [t.get_text() for t in panel.ax.texts] == ["Choose components, then Run."]
        finally:
            panel.hide()

    def test_show_message_is_one_centred_text(self, qtbot):
        panel = PlotWidget()
        qtbot.addWidget(panel)
        panel.show_message("No data for these settings.")
        assert panel.ax.axison is False
        assert [t.get_text() for t in panel.ax.texts] == ["No data for these settings."]


class TestToolbar:
    def test_navigation_then_composer_and_export(self, qtbot):
        panel = PlotWidget()
        qtbot.addWidget(panel)
        assert isinstance(panel.toolbar, QToolBar)
        texts = [a.text() for a in panel.toolbar.actions() if a.text()]
        assert texts == ["Home", "Back", "Forward", "Pan", "Zoom", "Send to Figure Composer", "Export…"]

    def test_a_narrow_pane_shortens_the_wording_so_export_stays_in_reach(self, qtbot):
        panel = _shown(qtbot, width=900, height=480)
        try:
            assert panel.composer_action.text() == "Send to Figure Composer"
            wide = panel.toolbar.sizeHint().width()
            panel.resize(430, 480)
            qtbot.wait(30)
            assert panel.composer_action.text() == "To composer"
            assert "Figure Composer" in panel.composer_action.toolTip()      # the full name is one hover away
            assert panel.toolbar.sizeHint().width() < wide                    # the toolbar needs less room
            panel.resize(900, 480)
            qtbot.wait(30)
            assert panel.composer_action.text() == "Send to Figure Composer"
        finally:
            panel.hide()

    def test_send_to_composer_action_sends_the_recipe(self, qtbot):
        from composer_store import store
        store.clear()
        try:
            panel = PlotWidget()
            qtbot.addWidget(panel)
            _bars(panel, 3)
            panel.composer_action.trigger()
            assert len(store.items()) == 1 and store.items()[0].caption == "T"
        finally:
            store.clear()

    def test_export_action_saves_at_300_dpi(self, qtbot, tmp_path, monkeypatch):
        from PySide6.QtWidgets import QFileDialog
        target = tmp_path / "chart.png"
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (str(target), "PNG image (*.png)")))
        panel = PlotWidget()
        qtbot.addWidget(panel)
        _bars(panel, 3)
        panel.export_action.trigger()
        width, _height = _png_size(target)
        assert 1500 <= width <= 1800                             # 6 in at 300 dpi, trimmed to the content

    def test_navigation_icons_follow_the_theme_ink(self, qtbot, monkeypatch):
        import qt_widgets
        panel = PlotWidget()
        qtbot.addWidget(panel)
        home = panel.toolbar._actions["home"]

        def darkest_pixel(action):
            image = action.icon().pixmap(48, 48).toImage()
            values = [image.pixelColor(x, y) for x in range(0, image.width(), 2) for y in range(0, image.height(), 2)]
            opaque = [c for c in values if c.alpha() > 200]
            return min(c.lightness() for c in opaque), max(c.lightness() for c in opaque)

        monkeypatch.setattr(qt_widgets, "_ui_palette", lambda: {"ink": "#20242a"})
        panel._tint_toolbar_icons()
        assert darkest_pixel(home)[1] < 80                       # dark ink on a light toolbar
        monkeypatch.setattr(qt_widgets, "_ui_palette", lambda: {"ink": "#e6e8ea"})
        panel._tint_toolbar_icons()
        assert darkest_pixel(home)[0] > 180                      # light ink on a dark toolbar


class TestChartStyle:
    def test_single_series_bars_are_blue(self, qtbot):
        from matplotlib.colors import to_rgba
        panel = PlotWidget()
        qtbot.addWidget(panel)
        _bars(panel, 3)
        assert {tuple(p.get_facecolor()) for p in panel.ax.patches} == {to_rgba("#2a78d6")}

    def test_highlighted_bar_takes_the_second_categorical_colour(self, qtbot):
        from matplotlib.colors import to_rgba
        panel = PlotWidget()
        qtbot.addWidget(panel)
        df = pd.DataFrame({"Element": ["Na", "Cs", "Fe"], "TotalInventory": [500.0, 300.0, 60.0]})
        ph.plot_barh(panel, df, "Element", "TotalInventory", "t", "x", highlighted_label="Cs")
        colours = {t.get_text(): tuple(p.get_facecolor()) for t, p in zip(panel.ax.get_yticklabels(), panel.ax.patches)}
        assert colours["Cs"] == to_rgba("#eb6834")
        assert colours["Na"] == colours["Fe"] == to_rgba("#2a78d6")

    def test_categorical_series_follow_the_fixed_order(self, qtbot):
        from matplotlib.colors import to_rgba
        df = pd.DataFrame({
            "WasteSiteId": ["T1"] * 3 + ["T2"] * 3,
            "WastePhase": ["Supernatant", "Saltcake", "Sludge"] * 2,
            "Total_kg": [30.0, 20.0, 10.0, 3.0, 2.0, 1.0],
        })
        panel = PlotWidget()
        qtbot.addWidget(panel)
        render_plot(panel, df, PlotSpec(mark="stackedbar", x="Total_kg", y="WasteSiteId", hue="WastePhase", title="T"))
        by_series = {c.get_label(): tuple(c.patches[0].get_facecolor()) for c in panel.ax.containers}
        # Series are ordered by their total (largest first) and coloured in the fixed order.
        assert by_series == {"Supernatant": to_rgba("#2a78d6"), "Saltcake": to_rgba("#eb6834"),
                             "Sludge": to_rgba("#1baf7a")}

    def test_magnitude_uses_the_blue_ramp_and_correlation_the_diverging_map(self, qtbot):
        df = pd.DataFrame({"WasteSiteId": ["T1", "T1", "T2", "T2"], "WastePhase": ["a", "b", "a", "b"],
                           "Total_kg": [5.0, 15.0, 8.0, 2.0]})
        panel = PlotWidget()
        qtbot.addWidget(panel)
        render_plot(panel, df, PlotSpec(mark="heatmap", x="Total_kg", y="WasteSiteId", hue="WastePhase", title="T"))
        assert panel.ax.images[0].get_cmap().name == ph.CMAP_CELLS          # the blue ramp, from its second step
        corr = pd.DataFrame({"Element": ["Cs", "Sr"], "Cs": [1.0, 0.9], "Sr": [0.9, 1.0]})
        ph.plot_correlation_heatmap(panel, corr)
        assert panel.ax.images[0].get_cmap().name == ph.CMAP_DIVERGING
        # Zero is the neutral grey, the poles are blue and red.
        from matplotlib.colors import to_hex, to_rgb
        import matplotlib
        cmap = matplotlib.colormaps[ph.CMAP_DIVERGING]
        middle, grey = cmap(0.5)[:3], to_rgb("#f0efec")
        assert max(abs(a - b) for a, b in zip(middle, grey)) < 0.02        # one step of a 256-entry table
        assert to_hex(cmap(0.0)) == "#184f95" and to_hex(cmap(1.0)) == "#a11d21"

    def test_axes_carry_no_raw_identifier(self, qtbot):
        panel = PlotWidget()
        qtbot.addWidget(panel)
        df = pd.DataFrame({"WasteSiteId": ["241-A-101"], "User_search_score_proxy": [3.0]})
        ph.plot_candidate_scores(panel, df)
        assert "_" not in panel.ax.get_xlabel() and panel.ax.get_xlabel().startswith("Search score (proxy)")
        ph.plot_barh(panel, pd.DataFrame({"Element": ["Na"], "v": [1.0]}), "Element", "v",
                     "Tank count by TankType", "N_PRIMARY_ELEMENTS")
        assert panel.ax.get_xlabel() == "Elements"
        assert panel.ax.get_title() == "Tank count by tank type"

    def test_trim_note_reads_as_a_sentence(self, qtbot):
        panel = PlotWidget()
        qtbot.addWidget(panel)
        df = pd.DataFrame({"WasteSiteId": [f"T{i}" for i in range(5)], "Total_kg": [5.0, 4.0, 3.0, 2.0, 1.0]})
        render_plot(panel, df, PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId", title="T", top_n=2))
        assert "top 2 of 5 tanks shown" in panel.ax.get_title()

    def test_long_subtitle_wraps_instead_of_running_off_the_figure(self, qtbot):
        panel = PlotWidget()
        qtbot.addWidget(panel)
        df = pd.DataFrame({"WasteSiteId": ["T1", "T2"], "Total_kg": [5.0, 4.0]})
        subtitle = " · ".join(f"disclosure number {i} of this result" for i in range(8))
        render_plot(panel, df, PlotSpec(mark="bar", x="Total_kg", y="WasteSiteId", title="Main", subtitle=subtitle))
        panel.canvas.draw()
        renderer = panel.canvas.get_renderer()
        box = panel.ax.title.get_window_extent(renderer)
        assert "\n" in panel.ax.get_title()
        assert box.x0 >= 0 and box.x1 <= panel.figure.bbox.width
        # The title sits clear of the subtitle below it.
        assert panel.figure._suptitle.get_window_extent(renderer).y0 >= box.y1 - 1

    def test_house_style_is_the_process_default(self):
        import matplotlib
        assert matplotlib.rcParams["axes.spines.top"] is False
        assert matplotlib.rcParams["axes.prop_cycle"].by_key()["color"][:3] == ["#2a78d6", "#eb6834", "#1baf7a"]
        assert matplotlib.rcParams["image.cmap"] == ph.CMAP_SEQUENTIAL


class TestComposedCells:
    """A Figure Composer cell is a SubFigure, with no layout engine: its
    text must still stay inside the cell, the heading clear of the
    subtitle and of the panel letter."""

    def _compose(self, qtbot, cols):
        import figure_composer as fcomp
        from composer_store import store

        store.clear()
        source = PlotWidget()
        qtbot.addWidget(source)
        df = pd.DataFrame({"WasteSiteId": [f"241-AW-{101 + i}" for i in range(12)],
                           "Total_g": [float(2e8 - i * 1e7) for i in range(12)]})
        subtitle = " · ".join(["Apportioned (stoichiometric)", "38 of 68 elements omitted (showing the top 30 by g)",
                               "Scope: Sludge"])
        render_plot(source, df, PlotSpec(mark="bar", x="Total_g", y="WasteSiteId", log_x=True,
                                         title="Top tanks by inventory of salt formers", subtitle=subtitle))
        source._send_to_composer()
        source._send_to_composer()
        source._send_to_composer()
        out = PlotWidget()
        qtbot.addWidget(out)
        composed = fcomp.compose_figure(out, store.items(), cols=cols)
        store.clear()
        out.canvas.draw()
        return out, composed

    @pytest.mark.parametrize("cols", [1, 3])
    def test_text_stays_inside_the_cell_and_headings_do_not_collide(self, qtbot, cols):
        out, composed = self._compose(qtbot, cols)
        renderer = out.canvas.get_renderer()
        for i, sub in composed.subfigs.items():
            ax = composed.cells[i].ax
            cell = sub.bbox
            drawn = ax.get_tightbbox(renderer)
            assert drawn.x0 >= cell.x0 - 1 and drawn.x1 <= cell.x1 + 1, "tick labels or subtitle cut at the cell's edge"
            assert drawn.y0 >= cell.y0 - 1
            heading = sub._suptitle.get_window_extent(renderer)
            assert ax.title.get_window_extent(renderer).y1 <= heading.y0 + 1, "the subtitle climbs into the heading"
            assert heading.x0 >= cell.x0 and heading.x1 <= cell.x1
            assert heading.y1 <= cell.y1 + 1

    def test_every_tank_label_survives_the_replay(self, qtbot):
        out, composed = self._compose(qtbot, 1)
        labels = [t.get_text() for t in composed.cells[0].ax.get_yticklabels()]
        assert labels[-1] == "241-AW-101" and len(labels) == 12


class TestSmallPanelText:
    def test_a_statement_longer_than_the_line_is_broken_at_a_space(self):
        text = ph.wrap_subtitle(["38 of 68 elements omitted (showing the top 30 by g)", "Scope: Sludge"], width=30)
        assert all(len(line) <= 30 for line in text.split("\n"))
        assert text.replace("\n", " ").replace("  ", " ") == (
            "38 of 68 elements omitted (showing the top 30 by g) Scope: Sludge")

    def test_crowded_log_tick_labels_are_thinned_until_none_overlap(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(1.6, 2.0), dpi=100)
        canvas = FigureCanvasAgg(figure)
        ax = figure.add_subplot(111)
        ax.barh([0, 1], [5e7, 2.5e8])
        ax.set_xscale("log")
        ax.set_xlim(4e7, 3e8)
        canvas.draw()
        renderer = canvas.get_renderer()

        def boxes():
            labels = ax.xaxis.get_majorticklabels() + ax.xaxis.get_minorticklabels()
            return sorted((t.get_window_extent(renderer) for t in labels if t.get_text() and t.get_visible()),
                          key=lambda b: b.x0)

        before = boxes()
        assert any(a.x1 > b.x0 for a, b in zip(before, before[1:])), "the fixture must start crowded"
        ph.declutter_ticks(ax, renderer)
        canvas.draw()
        after = boxes()
        assert 1 <= len(after) < len(before)
        assert all(a.x1 <= b.x0 for a, b in zip(after, after[1:]))
