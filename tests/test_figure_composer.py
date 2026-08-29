import pytest

import figure_composer as fcomp
from composer_store import ComposerItem
from qt_widgets import PlotWidget


def _fake_plot(panel, value, *, label="default"):
    ax = panel.reset_axes()
    ax.plot([0, 1], [0, value])
    ax.set_title(f"Fake plot ({label})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    panel.ax = ax
    panel.canvas.draw_idle()


def _erroring_plot(panel, *args, **kwargs):
    raise ValueError("boom")


def _item(value=1, label="default", caption="", **overrides) -> ComposerItem:
    return ComposerItem(render_fn=_fake_plot, render_args=(value,), render_kwargs={"label": label}, caption=caption, **overrides)


@pytest.fixture
def panel(qtbot):
    p = PlotWidget()
    qtbot.addWidget(p)
    return p


class TestComposeFigureEmpty:
    def test_no_items_shows_message(self, panel, qtbot):
        fcomp.compose_figure(panel, [])
        qtbot.wait(20)
        # show_message() turns the axes off and writes centered text --
        # confirm no crash and that a message-style axes was drawn.
        assert panel.ax.axison is False


class TestComposeFigureRenders:
    def test_single_panel_creates_one_real_axes(self, panel, qtbot):
        fcomp.compose_figure(panel, [_item()])
        qtbot.wait(20)
        assert len(panel.figure.axes) == 1

    def test_panel_is_a_live_vector_axes_not_a_raster_image(self, panel, qtbot):
        # The whole point of the recipe-based redesign: the panel's axes
        # holds a real Line2D artist from the recipe, not an AxesImage
        # showing a rasterized picture of one.
        fcomp.compose_figure(panel, [_item(value=5)])
        qtbot.wait(20)
        ax = panel.figure.axes[0]
        assert len(ax.lines) == 1
        assert ax.lines[0].get_ydata()[-1] == 5
        assert len(ax.images) == 0

    def test_multiple_panels_create_one_axes_each(self, panel, qtbot):
        items = [_item(value=i) for i in range(4)]
        fcomp.compose_figure(panel, items)
        qtbot.wait(20)
        assert len(panel.figure.axes) == 4

    def test_multiple_panels_use_isolated_subfigures(self, panel, qtbot):
        # Confirms the isolation SubFigure is relied on for: each panel's
        # own reset_axes()/figure.clear() must never touch a sibling's.
        items = [_item(value=1, label="one"), _item(value=2, label="two")]
        fcomp.compose_figure(panel, items)
        qtbot.wait(20)
        titles = sorted(ax.get_title() for ax in panel.figure.axes)
        assert titles == ["Fake plot (one)", "Fake plot (two)"]

    def test_panel_labels_are_drawn_on_the_subfigure(self, panel, qtbot):
        items = [_item(), _item()]
        fcomp.compose_figure(panel, items, label_style="A, B, C")
        qtbot.wait(20)
        all_texts = [t.get_text() for sf in panel.figure.subfigs for t in sf.texts]
        assert "A" in all_texts
        assert "B" in all_texts

    def test_label_style_none_draws_no_text(self, panel, qtbot):
        items = [_item(), _item()]
        fcomp.compose_figure(panel, items, label_style="none")
        qtbot.wait(20)
        all_texts = [t.get_text() for sf in panel.figure.subfigs for t in sf.texts]
        assert all_texts == []

    def test_recipe_supplies_the_title_by_default(self, panel, qtbot):
        fcomp.compose_figure(panel, [_item(label="mine")])
        qtbot.wait(20)
        assert panel.figure.axes[0].get_title() == "Fake plot (mine)"

    def test_title_override_replaces_the_recipes_own_title(self, panel, qtbot):
        fcomp.compose_figure(panel, [_item(title_override="Custom title")])
        qtbot.wait(20)
        assert panel.figure.axes[0].get_title() == "Custom title"

    def test_title_override_empty_string_hides_title(self, panel, qtbot):
        fcomp.compose_figure(panel, [_item(title_override="")])
        qtbot.wait(20)
        assert panel.figure.axes[0].get_title() == ""

    def test_xlabel_and_ylabel_overrides(self, panel, qtbot):
        fcomp.compose_figure(panel, [_item(xlabel_override="Se (kg)", ylabel_override="Tank")])
        qtbot.wait(20)
        ax = panel.figure.axes[0]
        assert ax.get_xlabel() == "Se (kg)"
        assert ax.get_ylabel() == "Tank"

    def test_kwarg_override_changes_what_the_recipe_draws(self, panel, qtbot):
        item = ComposerItem(
            render_fn=_fake_plot, render_args=(1,), render_kwargs={"label": "original"},
            caption="", kwarg_overrides={"label": "overridden"},
        )
        fcomp.compose_figure(panel, [item])
        qtbot.wait(20)
        assert panel.figure.axes[0].get_title() == "Fake plot (overridden)"

    def test_erroring_recipe_shows_a_message_instead_of_crashing(self, panel, qtbot):
        item = ComposerItem(render_fn=_erroring_plot, render_args=(), render_kwargs={}, caption="")
        fcomp.compose_figure(panel, [item])  # must not raise
        qtbot.wait(20)
        assert len(panel.figure.axes) == 1

    def test_explicit_cols_produces_requested_columns(self, panel, qtbot):
        items = [_item() for _ in range(4)]
        fcomp.compose_figure(panel, items, cols=1)
        qtbot.wait(20)
        # 4 panels, 1 column requested -> 4 rows x 1 col grid -> figure
        # taller than it is wide relative to the per-panel aspect.
        w, h = panel.figure.get_size_inches()
        assert h > w

    def test_figure_size_grows_with_more_panels(self, panel, qtbot):
        fcomp.compose_figure(panel, [_item()])
        qtbot.wait(20)
        small = tuple(panel.figure.get_size_inches())

        many = [_item() for _ in range(6)]
        fcomp.compose_figure(panel, many)
        qtbot.wait(20)
        large = tuple(panel.figure.get_size_inches())
        assert large[0] * large[1] > small[0] * small[1]


class TestRenderThumbnailPng:
    def test_returns_valid_png_bytes(self):
        png = fcomp.render_thumbnail_png(_item(value=3))
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_erroring_recipe_still_returns_a_png(self):
        item = ComposerItem(render_fn=_erroring_plot, render_args=(), render_kwargs={}, caption="")
        png = fcomp.render_thumbnail_png(item)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def _counting_plot(counter):
    """A recipe whose invocations are countable -- the unit of cost every
    Stage-0 assertion below is written in."""

    def _plot(panel, value, *, label="default"):
        counter.append(1)
        _fake_plot(panel, value, label=label)

    return _plot


def _sized_plot(panel, height, *, label="default"):
    """A recipe with a content-driven size request, for exercising the
    measured-size-changed -> full-recompose fallback."""
    ax = panel.reset_axes()
    ax.plot([0, 1], [0, 1])
    ax.set_title(f"Sized ({label})")
    panel.set_figure_size_inches(fcomp.PANEL_WIDTH_IN, height)
    panel.ax = ax
    panel.canvas.draw_idle()


class TestMeasurementCache:
    def test_second_measurement_is_a_cache_hit(self):
        calls = []
        item = ComposerItem(render_fn=_counting_plot(calls), render_args=(1,), render_kwargs={}, caption="")
        first = fcomp.measure_preferred_size(item)
        second = fcomp.measure_preferred_size(item)
        assert first == second
        assert len(calls) == 1

    def test_kwarg_override_change_invalidates_the_cache(self):
        from dataclasses import replace
        calls = []
        item = ComposerItem(render_fn=_counting_plot(calls), render_args=(1,), render_kwargs={"label": "a"}, caption="")
        fcomp.measure_preferred_size(item)
        edited = replace(item, kwarg_overrides={"label": "b"})
        fcomp.measure_preferred_size(edited)
        assert len(calls) == 2  # same uid, different overrides -> re-measured

    def test_text_overrides_do_not_invalidate_the_cache(self):
        # Title/label overrides are applied post-render and never reach the
        # recipe -- they must not cost a re-measurement.
        from dataclasses import replace
        calls = []
        item = ComposerItem(render_fn=_counting_plot(calls), render_args=(1,), render_kwargs={}, caption="")
        fcomp.measure_preferred_size(item)
        edited = replace(item, title_override="New title")
        fcomp.measure_preferred_size(edited)
        assert len(calls) == 1

    def test_use_cache_false_forces_a_re_render(self):
        calls = []
        item = ComposerItem(render_fn=_counting_plot(calls), render_args=(1,), render_kwargs={}, caption="")
        fcomp.measure_preferred_size(item)
        fcomp.measure_preferred_size(item, use_cache=False)
        assert len(calls) == 2


class TestComposedPanelsHandles:
    def test_compose_returns_handles_for_every_panel(self, panel, qtbot):
        items = [_item(value=1), _item(value=2), _item(value=3)]
        composed = fcomp.compose_figure(panel, items)
        qtbot.wait(20)
        assert composed is not None
        assert set(composed.subfigs) == {0, 1, 2}
        assert set(composed.cells) == {0, 1, 2}
        assert composed.uids == {i: item.uid for i, item in enumerate(items)}
        assert composed.matches(items)

    def test_compose_with_no_items_returns_none(self, panel, qtbot):
        assert fcomp.compose_figure(panel, []) is None

    def test_matches_is_false_after_reorder(self, panel, qtbot):
        items = [_item(), _item()]
        composed = fcomp.compose_figure(panel, items)
        qtbot.wait(20)
        assert composed.matches(items[::-1]) is False


class TestRerenderPanel:
    def test_rerenders_only_the_target_panel(self, panel, qtbot):
        from dataclasses import replace
        calls = []
        items = [
            ComposerItem(render_fn=_counting_plot(calls), render_args=(1,), render_kwargs={"label": "left"}, caption=""),
            ComposerItem(render_fn=_counting_plot(calls), render_args=(2,), render_kwargs={"label": "right"}, caption=""),
        ]
        composed = fcomp.compose_figure(panel, items)
        qtbot.wait(20)
        untouched_ax = composed.cells[0].ax
        calls.clear()

        edited = replace(items[1], kwarg_overrides={"label": "edited"})
        fcomp.rerender_panel(panel, composed, 1, edited)
        qtbot.wait(20)

        assert len(calls) == 1  # exactly one recipe replay
        assert composed.cells[0].ax is untouched_ax  # sibling untouched
        assert composed.cells[1].ax.get_title() == "Fake plot (edited)"

    def test_rerendered_panel_keeps_its_grid_label(self, panel, qtbot):
        items = [_item(), _item()]
        composed = fcomp.compose_figure(panel, items, label_style="A, B, C")
        qtbot.wait(20)
        fcomp.rerender_panel(panel, composed, 1, items[1])
        qtbot.wait(20)
        labels = [t.get_text() for t in composed.subfigs[1].texts]
        assert "B" in labels


class TestApplyTextOverride:
    def test_string_override_is_applied_in_place(self, panel, qtbot):
        from dataclasses import replace
        items = [_item()]
        composed = fcomp.compose_figure(panel, items)
        qtbot.wait(20)
        edited = replace(items[0], title_override="In-place title")
        assert fcomp.apply_text_override(panel, composed, 0, edited, "title") is True
        assert composed.cells[0].ax.get_title() == "In-place title"

    def test_xlabel_and_ylabel_apply_in_place(self, panel, qtbot):
        from dataclasses import replace
        items = [_item()]
        composed = fcomp.compose_figure(panel, items)
        qtbot.wait(20)
        edited = replace(items[0], xlabel_override="Se (kg)", ylabel_override="Tank")
        assert fcomp.apply_text_override(panel, composed, 0, edited, "xlabel") is True
        assert fcomp.apply_text_override(panel, composed, 0, edited, "ylabel") is True
        assert composed.cells[0].ax.get_xlabel() == "Se (kg)"
        assert composed.cells[0].ax.get_ylabel() == "Tank"

    def test_cleared_override_refuses_in_place_and_demands_re_render(self, panel, qtbot):
        # None means "restore whatever the recipe draws" -- only the recipe
        # knows that text, so the in-place path must decline.
        items = [_item(title_override="Custom")]
        composed = fcomp.compose_figure(panel, items)
        qtbot.wait(20)
        from dataclasses import replace
        cleared = replace(items[0], title_override=None)
        assert fcomp.apply_text_override(panel, composed, 0, cleared, "title") is False

    def test_unknown_panel_index_refuses(self, panel, qtbot):
        items = [_item(title_override="T")]
        composed = fcomp.compose_figure(panel, items)
        qtbot.wait(20)
        assert fcomp.apply_text_override(panel, composed, 5, items[0], "title") is False

    def test_failed_panel_refuses_in_place_text(self, panel, qtbot):
        # Review finding: the full pipeline (_render_item) skips overrides
        # when the recipe raises, so applying one in place onto the error-
        # placeholder axes would make the live preview diverge from what
        # any recompose/export renders. The in-place path must decline.
        from dataclasses import replace
        item = ComposerItem(render_fn=_erroring_plot, render_args=(), render_kwargs={}, caption="")
        composed = fcomp.compose_figure(panel, [item])
        qtbot.wait(20)
        edited = replace(item, title_override="Should not appear")
        assert fcomp.apply_text_override(panel, composed, 0, edited, "title") is False
