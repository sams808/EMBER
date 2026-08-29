import pytest
from PySide6.QtWidgets import QMainWindow

import figure_composer as fcomp
from composer_store import store
from qt_figure_composer import FigureComposerPage, _parse_param_text

# Every _fake_plot invocation (preview panel, measurement pass, thumbnail)
# appends here -- the unit of cost the incremental-rendering tests below
# are written in. Reset by the autouse _clean_store fixture.
RENDER_CALLS = []


def _fake_plot(panel, value, *, label="default"):
    RENDER_CALLS.append(1)
    ax = panel.reset_axes()
    ax.plot([0, 1], [0, value])
    ax.set_title(f"Fake plot ({label})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    panel.ax = ax
    panel.canvas.draw_idle()


def _sized_fake_plot(panel, value, *, extra_height=0.0):
    RENDER_CALLS.append(1)
    ax = panel.reset_axes()
    ax.plot([0, 1], [0, value])
    ax.set_title("Sized fake plot")
    panel.set_figure_size_inches(fcomp.PANEL_WIDTH_IN, fcomp.PANEL_HEIGHT_IN + extra_height)
    panel.ax = ax
    panel.canvas.draw_idle()


def _add(caption="", value=1, label="default", title_override=None, xlabel_override=None, ylabel_override=None):
    store.add_recipe(_fake_plot, (value,), {"label": label}, caption)
    index = len(store.items()) - 1
    if title_override is not None:
        store.set_title_override(index, title_override)
    if xlabel_override is not None:
        store.set_xlabel_override(index, xlabel_override)
    if ylabel_override is not None:
        store.set_ylabel_override(index, ylabel_override)


@pytest.fixture(autouse=True)
def _clean_store():
    # The page reads composer_store.store, a module-level singleton shared
    # with the real app -- reset it around every test so panels captured
    # (or left) by one test can't leak into the next.
    store.clear()
    RENDER_CALLS.clear()
    yield
    store.clear()
    RENDER_CALLS.clear()


@pytest.fixture
def page(qtbot):
    app_window = QMainWindow()  # only used for .statusBar()-style calls, none needed here
    qtbot.addWidget(app_window)
    p = FigureComposerPage(app_window)
    qtbot.addWidget(p)
    # Shown, not just constructed: signal-driven refreshes are visibility-
    # guarded (a hidden page defers work to showEvent), and these tests
    # exercise the live-page behavior. The hidden path has its own tests
    # in TestVisibilityDeferral below.
    p.show()
    qtbot.waitExposed(p)
    qtbot.wait(20)  # let the initial (empty-state) compose_figure()'s draw_idle flush
    yield p
    qtbot.wait(20)  # flush anything the test itself queued, before teardown deletes the widgets


class TestParseParamText:
    def test_parses_python_literals(self):
        assert _parse_param_text("42") == 42
        assert _parse_param_text("3.14") == 3.14
        assert _parse_param_text("True") is True
        assert _parse_param_text("None") is None
        assert _parse_param_text('"quoted"') == "quoted"

    def test_bare_word_falls_back_to_plain_string(self):
        assert _parse_param_text("pearson") == "pearson"
        assert _parse_param_text("Coherent colors") == "Coherent colors"


class TestGallerySync:
    def test_starts_empty(self, page):
        assert page.gallery_list.count() == 0

    def test_gallery_reflects_items_already_in_the_store(self, qtbot):
        _add(caption="Pre-existing panel")
        app_window = QMainWindow()
        qtbot.addWidget(app_window)
        p = FigureComposerPage(app_window)
        qtbot.addWidget(p)
        qtbot.wait(20)
        assert p.gallery_list.count() == 1
        assert p.gallery_list.item(0).text() == "Pre-existing panel"
        qtbot.wait(20)  # flush before this test's own widgets get torn down

    def test_gallery_updates_when_store_changes_after_construction(self, page, qtbot):
        _add(caption="Added later")
        qtbot.wait(20)
        assert page.gallery_list.count() == 1
        assert page.gallery_list.item(0).text() == "Added later"

    def test_untitled_caption_shown_for_empty_caption(self, page, qtbot):
        _add(caption="")
        qtbot.wait(20)
        assert page.gallery_list.item(0).text() == "(untitled)"

    def test_gallery_icon_is_a_real_rendered_thumbnail(self, page, qtbot):
        _add(caption="one")
        qtbot.wait(20)
        icon = page.gallery_list.item(0).icon()
        assert not icon.isNull()


class TestSelectionAndCaptionEditing:
    def test_selecting_item_populates_caption_field(self, page, qtbot):
        _add(caption="First")
        _add(caption="Second")
        qtbot.wait(20)
        page.gallery_list.setCurrentRow(1)
        assert page.caption_edit.text() == "Second"

    def test_caption_field_disabled_with_nothing_selected(self, page):
        assert page.caption_edit.isEnabled() is False

    def test_editing_caption_updates_the_store(self, page, qtbot):
        _add(caption="Original")
        qtbot.wait(20)
        page.gallery_list.setCurrentRow(0)
        page.caption_edit.setText("Edited caption")
        page.caption_edit.editingFinished.emit()
        qtbot.wait(20)
        assert store.items()[0].caption == "Edited caption"


class TestTitleAndAxisOverrides:
    def test_title_field_disabled_with_nothing_selected(self, page):
        assert page.title_edit.isEnabled() is False
        assert page.hide_title_check.isEnabled() is False

    def test_selecting_item_populates_title_and_axis_fields(self, page, qtbot):
        _add(caption="c", title_override="Custom", xlabel_override="Se (kg)", ylabel_override="Tank")
        qtbot.wait(20)
        page.gallery_list.setCurrentRow(0)
        assert page.title_edit.text() == "Custom"
        assert page.xlabel_edit.text() == "Se (kg)"
        assert page.ylabel_edit.text() == "Tank"

    def test_editing_title_updates_the_store(self, page, qtbot):
        _add(caption="c")
        qtbot.wait(20)
        page.gallery_list.setCurrentRow(0)
        page.title_edit.setText("New title")
        page.title_edit.editingFinished.emit()
        qtbot.wait(20)
        assert store.items()[0].title_override == "New title"

    def test_clearing_title_field_reverts_to_original(self, page, qtbot):
        _add(caption="c", title_override="Custom")
        qtbot.wait(20)
        page.gallery_list.setCurrentRow(0)
        page.title_edit.setText("")
        page.title_edit.editingFinished.emit()
        qtbot.wait(20)
        assert store.items()[0].title_override is None

    def test_hide_title_checkbox_forces_empty_override(self, page, qtbot):
        _add(caption="c")
        qtbot.wait(20)
        page.gallery_list.setCurrentRow(0)
        page.hide_title_check.setChecked(True)
        qtbot.wait(20)
        assert store.items()[0].title_override == ""
        assert page.title_edit.isEnabled() is False

    def test_editing_xlabel_updates_the_store(self, page, qtbot):
        _add(caption="c")
        qtbot.wait(20)
        page.gallery_list.setCurrentRow(0)
        page.xlabel_edit.setText("Se (kg)")
        page.xlabel_edit.editingFinished.emit()
        qtbot.wait(20)
        assert store.items()[0].xlabel_override == "Se (kg)"

    def test_editing_ylabel_updates_the_store(self, page, qtbot):
        _add(caption="c")
        qtbot.wait(20)
        page.gallery_list.setCurrentRow(0)
        page.ylabel_edit.setText("Tank")
        page.ylabel_edit.editingFinished.emit()
        qtbot.wait(20)
        assert store.items()[0].ylabel_override == "Tank"


class TestParametersTable:
    def test_populates_with_recipe_kwargs(self, page, qtbot):
        _add(caption="c", label="hello")
        qtbot.wait(20)
        page.gallery_list.setCurrentRow(0)
        assert page.params_table.rowCount() == 1
        assert page.params_table.item(0, 0).text() == "label"
        assert page.params_table.item(0, 1).text() == "hello"

    def test_editing_a_value_sets_a_kwarg_override(self, page, qtbot):
        _add(caption="c", label="hello")
        qtbot.wait(20)
        page.gallery_list.setCurrentRow(0)
        page.params_table.item(0, 1).setText("goodbye")
        qtbot.wait(20)
        assert store.items()[0].kwarg_overrides == {"label": "goodbye"}

    def test_editing_a_numeric_looking_value_parses_as_a_literal(self, page, qtbot):
        store.add_recipe(_fake_plot, (1,), {"value": 5}, "c")
        qtbot.wait(20)
        page.gallery_list.setCurrentRow(0)
        row = next(r for r in range(page.params_table.rowCount()) if page.params_table.item(r, 0).text() == "value")
        page.params_table.item(row, 1).setText("42")
        qtbot.wait(20)
        assert store.items()[0].kwarg_overrides == {"value": 42}

    def test_reset_clears_overrides(self, page, qtbot):
        _add(caption="c", label="hello")
        qtbot.wait(20)
        page.gallery_list.setCurrentRow(0)
        page.params_table.item(0, 1).setText("goodbye")
        qtbot.wait(20)
        page._reset_params()
        qtbot.wait(20)
        assert store.items()[0].kwarg_overrides == {}
        assert page.params_table.item(0, 1).text() == "hello"

    def test_disabled_with_nothing_selected(self, page):
        assert page.params_table.isEnabled() is False


class TestReorderAndRemove:
    def test_move_down_swaps_store_order(self, page, qtbot):
        _add(caption="A")
        _add(caption="B")
        qtbot.wait(20)
        page.gallery_list.setCurrentRow(0)
        page._move(1)
        qtbot.wait(20)
        assert [i.caption for i in store.items()] == ["B", "A"]

    def test_remove_deletes_from_store(self, page, qtbot):
        _add(caption="keep")
        _add(caption="remove me")
        qtbot.wait(20)
        page.gallery_list.setCurrentRow(1)
        page._remove_selected()
        qtbot.wait(20)
        captions = [i.caption for i in store.items()]
        assert captions == ["keep"]

    def test_remove_with_nothing_selected_is_a_noop(self, page, qtbot):
        _add(caption="solo")
        qtbot.wait(20)
        page.gallery_list.setCurrentRow(-1)
        page._remove_selected()
        qtbot.wait(20)
        assert len(store.items()) == 1

    def test_clear_all_empties_the_store(self, page, qtbot):
        # _prevent_blocking_qt_dialogs (conftest, autouse) makes the
        # QMessageBox.question confirmation resolve to Yes automatically.
        _add(caption="one")
        _add(caption="two")
        qtbot.wait(20)
        page._clear_all()
        qtbot.wait(20)
        assert store.items() == []


class TestPreviewRebuildsOnLayoutChange:
    def test_preview_has_no_axes_when_empty(self, page):
        # compose_figure([]) calls panel.show_message(), which clears and
        # turns off the PlotWidget's original axes rather than removing it.
        assert len(page.plot.figure.axes) == 1
        assert page.plot.figure.axes[0].axison is False

    def test_preview_grows_axes_count_with_panels(self, page, qtbot):
        _add(caption="one")
        _add(caption="two")
        _add(caption="three")
        qtbot.wait(20)
        assert len(page.plot.figure.axes) == 3

    def test_changing_label_style_rebuilds_preview_without_raising(self, page, qtbot):
        _add(caption="one")
        qtbot.wait(20)
        page.label_style_combo.setCurrentText("none")
        qtbot.wait(20)
        all_texts = [t.get_text() for sf in page.plot.figure.subfigs for t in sf.texts]
        assert all_texts == []


class TestExport:
    def test_export_with_no_panels_shows_message_not_crash(self, page):
        page.export_figure()  # QMessageBox.information neutralized by conftest; must not raise

    def test_export_writes_a_file(self, page, qtbot, tmp_path, monkeypatch):
        _add(caption="panel")
        qtbot.wait(20)
        out_path = tmp_path / "combined_figure.png"
        monkeypatch.setattr(
            "qt_figure_composer.QFileDialog.getSaveFileName",
            staticmethod(lambda *a, **k: (str(out_path), "")),
        )
        page.export_figure()
        qtbot.wait(20)
        assert out_path.exists()

    def test_cancelled_dialog_writes_nothing(self, page, qtbot, tmp_path, monkeypatch):
        _add(caption="panel")
        qtbot.wait(20)
        monkeypatch.setattr(
            "qt_figure_composer.QFileDialog.getSaveFileName",
            staticmethod(lambda *a, **k: ("", "")),
        )
        page.export_figure()
        qtbot.wait(20)
        assert list(tmp_path.iterdir()) == []


class TestOnDatasetChanged:
    def test_does_not_raise(self, page):
        page.on_dataset_changed(None)  # Figure Composer ignores the dataset; must not raise


@pytest.fixture
def compose_counter(monkeypatch):
    """Counts full compose_figure() grid rebuilds, preserving behavior."""
    calls = []
    real = fcomp.compose_figure

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(fcomp, "compose_figure", counting)
    return calls


class TestIncrementalRendering:
    """The Stage-0 performance contract, asserted in units of recipe
    replays: with 4 panels, the old pipeline charged a title edit 9 full
    renders (4 measurements + 4 panel replays + 1 thumbnail). The new
    routing must charge ~0 immediately and at most a debounced thumbnail
    afterward -- and must never call compose_figure for a cosmetic edit."""

    def _add_four(self, qtbot):
        for i in range(4):
            store.add_recipe(_fake_plot, (i + 1,), {"label": f"p{i}"}, f"panel {i}")
        qtbot.wait(30)

    def test_title_edit_costs_zero_immediate_renders_and_no_recompose(self, page, qtbot, compose_counter):
        self._add_four(qtbot)
        RENDER_CALLS.clear()
        compose_counter.clear()

        store.set_title_override(0, "New title")

        assert compose_counter == []
        assert RENDER_CALLS == []  # in-place: zero recipe replays right now
        # The preview really shows the new title already (pixel-equivalent
        # to a recompose, without one).
        assert page._composed.cells[0].ax.get_title() == "New title"

        qtbot.wait(350)  # debounce settles: at most the one thumbnail render
        assert compose_counter == []
        assert len(RENDER_CALLS) <= 1

    def test_axis_label_edit_is_also_in_place(self, page, qtbot, compose_counter):
        self._add_four(qtbot)
        RENDER_CALLS.clear()
        compose_counter.clear()
        store.set_xlabel_override(2, "Se (kg)")
        assert compose_counter == []
        assert RENDER_CALLS == []
        assert page._composed.cells[2].ax.get_xlabel() == "Se (kg)"

    def test_clearing_a_title_re_renders_exactly_one_panel(self, page, qtbot, compose_counter):
        self._add_four(qtbot)
        store.set_title_override(0, "Custom")
        qtbot.wait(350)
        RENDER_CALLS.clear()
        compose_counter.clear()

        store.set_title_override(0, None)  # restore recipe original: needs that panel's recipe
        qtbot.wait(350)

        assert compose_counter == []  # still no grid rebuild
        # measurement is cache-hit; re-render (1) + thumbnail (1)
        assert len(RENDER_CALLS) <= 2
        assert page._composed.cells[0].ax.get_title() == "Fake plot (p0)"

    def test_param_edit_re_renders_one_panel_not_the_grid(self, page, qtbot, compose_counter):
        self._add_four(qtbot)
        RENDER_CALLS.clear()
        compose_counter.clear()

        store.set_kwarg_override(1, "label", "edited")
        qtbot.wait(350)

        assert compose_counter == []
        # re-measure (1, cache miss on new overrides) + panel replay (1)
        # + thumbnail (1) = 3, vs 9 for the old full pipeline.
        assert len(RENDER_CALLS) == 3
        assert page._composed.cells[1].ax.get_title() == "Fake plot (edited)"
        assert len(page.plot.figure.axes) == 4  # grid intact

    def test_param_edit_that_changes_measured_size_falls_back_to_full_recompose(self, page, qtbot, compose_counter):
        store.add_recipe(_sized_fake_plot, (1,), {"extra_height": 0.0}, "sized")
        store.add_recipe(_fake_plot, (2,), {"label": "plain"}, "plain")
        qtbot.wait(30)
        compose_counter.clear()

        store.set_kwarg_override(0, "extra_height", 3.0)  # taller panel: grid geometry is stale
        qtbot.wait(350)

        assert compose_counter == [1]  # exactly one full rebuild, not per-keystroke storms

    def test_caption_edit_renders_nothing_at_all(self, page, qtbot, compose_counter):
        self._add_four(qtbot)
        RENDER_CALLS.clear()
        compose_counter.clear()

        store.set_caption(2, "renamed")
        qtbot.wait(350)

        assert compose_counter == []
        assert RENDER_CALLS == []
        assert page.gallery_list.item(2).text() == "renamed"

    def test_thumbnail_catches_up_after_debounce(self, page, qtbot):
        self._add_four(qtbot)
        icon_before = page.gallery_list.item(0).icon().cacheKey()
        store.set_kwarg_override(0, "label", "different")
        qtbot.wait(350)
        assert page.gallery_list.item(0).icon().cacheKey() != icon_before


class TestStructuralChangeDuringDebounce:
    """Review finding (confirmed by 3 independent finders): a dirty panel
    index queued by an edit outlived a structural removal inside the 200 ms
    debounce window and indexed past the end of store.items() in the timer
    slot. Focus-out commit followed by a Remove click lands both events in
    one window essentially every time."""

    def test_remove_after_edit_within_debounce_does_not_crash(self, page, qtbot):
        for i in range(4):
            store.add_recipe(_fake_plot, (i + 1,), {"label": f"p{i}"}, f"panel {i}")
        qtbot.wait(30)

        store.set_kwarg_override(3, "label", "edited")  # dirty index 3, timer armed
        store.remove(0)                                  # now only 3 items: index 3 is stale
        qtbot.wait(350)                                  # timer fires -- must not raise

        assert len(store.items()) == 3
        assert len(page.plot.figure.axes) == 3  # preview rebuilt cleanly

    def test_clear_all_after_edit_within_debounce_does_not_crash(self, page, qtbot):
        store.add_recipe(_fake_plot, (1,), {"label": "only"}, "solo")
        qtbot.wait(30)
        store.set_kwarg_override(0, "label", "edited")
        store.clear()
        qtbot.wait(350)  # must not raise
        assert store.items() == []


class TestVisibilityDeferral:
    """Capturing plots from other workspaces used to pay for a full
    recompose of a page nobody was looking at; now a hidden page defers
    everything to showEvent."""

    def test_capture_while_hidden_defers_all_rendering(self, page, qtbot, compose_counter):
        page.hide()
        qtbot.wait(20)
        RENDER_CALLS.clear()
        compose_counter.clear()

        store.add_recipe(_fake_plot, (1,), {"label": "hidden-add"}, "captured while hidden")
        qtbot.wait(50)

        assert compose_counter == []
        assert RENDER_CALLS == []
        assert page.gallery_list.count() == 0  # not refreshed yet
        assert page._stale_full is True

    def test_show_settles_the_deferred_work_once(self, page, qtbot, compose_counter):
        page.hide()
        qtbot.wait(20)
        store.add_recipe(_fake_plot, (1,), {"label": "a"}, "one")
        store.add_recipe(_fake_plot, (2,), {"label": "b"}, "two")
        compose_counter.clear()

        page.show()
        qtbot.waitExposed(page)
        qtbot.wait(50)

        assert compose_counter == [1]  # one rebuild for two captures
        assert page.gallery_list.count() == 2
        assert page._stale_full is False

    def test_edit_while_hidden_defers_to_show(self, page, qtbot, compose_counter):
        store.add_recipe(_fake_plot, (1,), {"label": "a"}, "one")
        qtbot.wait(30)
        page.hide()
        qtbot.wait(20)
        compose_counter.clear()
        RENDER_CALLS.clear()

        store.set_kwarg_override(0, "label", "edited-hidden")
        qtbot.wait(350)
        assert RENDER_CALLS == []  # nothing while hidden

        page.show()
        qtbot.waitExposed(page)
        qtbot.wait(50)
        assert compose_counter == [1]
        assert page._composed.cells[0].ax.get_title() == "Fake plot (edited-hidden)"
