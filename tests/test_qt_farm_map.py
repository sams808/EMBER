"""
FarmMapWidget and RankedTankList: value-to-class mapping (no record and a
reported zero included), legend breaks, hit-testing by synthetic mouse
events at tile centres, selection, activation and the keyboard.
"""
import pandas as pd
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QPixmap
from PySide6.QtWidgets import QApplication

import qt_farm_map as fm
from qt_farm_map import FarmMapWidget, RankedTankList, class_breaks, class_colours, classify


def _hover(widget, pos):
    """A synthetic pointer move, delivered straight to the widget."""
    event = QMouseEvent(QEvent.MouseMove, QPointF(pos), QPointF(widget.mapToGlobal(pos)),
                        Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    QApplication.sendEvent(widget, event)


def _tanks():
    """Twelve single-shell tanks in farm A-like TX, two double-shell in AW,
    with volumes: one retrieved tank holding a heel."""
    ids = [f"241-TX-{100 + i}" for i in range(1, 13)] + ["241-AW-101", "241-AW-102"]
    volumes = [1000.0] * 11 + [8.0] + [4000.0, 3500.0]
    return pd.DataFrame({
        "WasteSiteId": ids,
        "TankFarm": ["TX"] * 12 + ["AW"] * 2,
        "TankSystem": ["SST"] * 12 + ["DST"] * 2,
        "Volume_kL": volumes,
    })


def _values():
    """TX-101..TX-110 = 10..100, TX-111 a reported zero, TX-112 (the
    heel) an outlier, AW-101 = 55, AW-102 has no record."""
    values = {f"241-TX-{100 + i}": 10.0 * i for i in range(1, 11)}
    values["241-TX-111"] = 0.0
    values["241-TX-112"] = 5000.0
    values["241-AW-101"] = 55.0
    return values


@pytest.fixture
def widget(qtbot):
    w = FarmMapWidget()
    qtbot.addWidget(w)
    w.resize(640, 480)
    w.set_tanks(_tanks())
    w.set_values(_values(), label="SO4", unit="mol per mol Na", additive=False)
    w.show()
    qtbot.waitExposed(w)
    return w


class TestClassBreaks:
    def test_quantile_breaks_are_the_quintiles_of_the_positive_values(self):
        values = [float(v) for v in range(1, 101)]
        breaks = class_breaks(values, "quantile")
        assert breaks == pytest.approx([20.8, 40.6, 60.4, 80.2])

    def test_the_reference_subset_sets_the_scale(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 5000.0]
        with_outlier = class_breaks(values, "quantile")
        without = class_breaks(values, "quantile", reference=values[:-1])
        assert max(without) < max(with_outlier)
        assert classify(5000.0, without) == 4          # still drawn, in the top class

    def test_log_breaks_are_equal_ratios(self):
        breaks = class_breaks([1.0, 10.0, 100.0, 1000.0, 100000.0], "log")
        ratios = [b / a for a, b in zip(breaks, breaks[1:])]
        assert ratios == pytest.approx([10.0, 10.0, 10.0])
        assert breaks[0] == pytest.approx(10.0)

    def test_linear_breaks_are_equal_steps(self):
        assert class_breaks([0.0, 50.0, 100.0], "linear") == pytest.approx([20.0, 40.0, 60.0, 80.0])

    def test_a_single_value_has_no_breaks(self):
        assert class_breaks([3.0, 3.0, 3.0], "quantile") == []
        assert class_breaks([], "quantile") == []

    def test_tied_quantiles_collapse_instead_of_repeating(self):
        breaks = class_breaks([1.0] * 50 + [2.0] * 5, "quantile")
        assert breaks == sorted(set(breaks))

    def test_diverging_breaks_are_symmetric_about_zero(self):
        breaks = class_breaks([-4.0, -2.0, -1.0, 1.0, 2.0, 4.0], "quantile", palette="diverging")
        assert len(breaks) == 4
        assert breaks[0] == pytest.approx(-breaks[3]) and breaks[1] == pytest.approx(-breaks[2])
        assert classify(0.0, breaks) == 2

    def test_unknown_scale_is_refused(self):
        with pytest.raises(ValueError):
            class_breaks([1.0, 2.0], "cubic")


class TestClassify:
    def test_no_record_is_not_a_class(self):
        assert classify(None, [1.0, 2.0]) is None
        assert classify(float("nan"), [1.0, 2.0]) is None

    def test_a_reported_zero_is_the_lowest_class(self):
        assert classify(0.0, [1.0, 2.0, 3.0, 4.0]) == 0

    def test_values_fall_in_ascending_classes(self):
        breaks = [10.0, 20.0, 30.0, 40.0]
        assert [classify(v, breaks) for v in (5, 10, 15, 25, 35, 45, 1e9)] == [0, 1, 1, 2, 3, 4, 4]

    def test_colours_run_light_to_dark_on_a_light_surface(self):
        colours = class_colours(5, "sequential", dark=False)
        lightness = [fm.QColor(c).lightness() for c in colours]
        assert lightness == sorted(lightness, reverse=True)
        dark = [fm.QColor(c).lightness() for c in class_colours(5, "sequential", dark=True)]
        assert dark == sorted(dark)

    def test_fewer_classes_still_span_the_ramp(self):
        assert len(class_colours(1)) == 1
        three = class_colours(3)
        five = class_colours(5)
        assert three[0] == five[0] and three[-1] == five[-1]


class TestValueMapping:
    def test_every_tank_is_drawn(self, widget):
        assert sorted(widget.tanks()) == sorted(_tanks()["WasteSiteId"])

    def test_no_record_is_hollow_and_zero_is_the_lowest_class(self, widget):
        assert widget.class_of("241-AW-102") is None
        assert widget.colour_of("241-AW-102") is None
        assert widget.class_of("241-TX-111") == 0
        assert widget.colour_of("241-TX-111") is not None

    def test_breaks_come_from_the_tanks_holding_real_waste(self, widget):
        # The heel in TX-112 (8 kL) reports 5000; it must not set the scale.
        assert max(widget.breaks()) < 100.0
        assert widget.class_of("241-TX-112") == widget.class_count() - 1
        assert widget.class_of("241-TX-101") == 0
        assert widget.class_of("241-TX-110") == widget.class_count() - 1

    def test_classes_are_monotonic_in_the_value(self, widget):
        ordered = sorted(_values().items(), key=lambda kv: kv[1])
        classes = [widget.class_of(t) for t, _v in ordered]
        assert classes == sorted(classes)

    def test_legend_states_the_real_breaks(self, widget):
        labels = widget.legend_labels()
        assert len(labels) == len(widget.breaks()) == 4
        assert [float(text) for text in labels] == pytest.approx(widget.breaks(), rel=5e-3)

    def test_scale_changes_the_breaks(self, widget):
        quantile = widget.breaks()
        widget.set_values(_values(), label="SO4", unit="mol per mol Na", scale="log")
        assert widget.breaks() != quantile
        widget.set_values(_values(), label="SO4", unit="mol per mol Na", scale="linear")
        steps = [b - a for a, b in zip(widget.breaks(), widget.breaks()[1:])]
        assert steps == pytest.approx([steps[0]] * len(steps))

    def test_a_tank_holding_little_waste_is_drawn_smaller(self, widget):
        assert widget.size_fraction("241-TX-112") < 1.0
        assert widget.size_fraction("241-TX-101") == 1.0
        widget.set_sizes({"241-TX-101": 0.25})
        assert widget.size_fraction("241-TX-101") == 0.25
        widget.set_sizes(None)
        assert widget.size_fraction("241-TX-101") == 1.0

    def test_out_of_scope_tanks_are_hollow(self, widget):
        widget.set_scope_tanks(["241-AW-101", "241-AW-102"])
        assert widget.colour_of("241-TX-105") is None
        assert widget.colour_of("241-AW-101") is not None
        assert "Outside the current scope" in widget.tooltip_text("241-TX-105")
        widget.set_scope_tanks(None)
        assert widget.colour_of("241-TX-105") is not None

    def test_categories_colour_by_name(self, widget):
        widget.set_categories({"241-TX-101": "Saltcake"}, {"Saltcake": "#eb6834"}, label="Phase")
        assert widget.colour_of("241-TX-101") == "#eb6834"
        assert widget.colour_of("241-TX-102") is None
        assert "Saltcake" in widget.tooltip_text("241-TX-101")

    def test_bad_arguments_are_refused(self, widget):
        with pytest.raises(ValueError):
            widget.set_values({}, scale="cubic")
        with pytest.raises(ValueError):
            widget.set_values({}, palette="rainbow")


class TestHover:
    def test_hover_at_a_tile_centre_names_the_tank(self, widget, qtbot):
        hovered = []
        widget.tankHovered.connect(hovered.append)
        _hover(widget, widget.tile_center("241-TX-103"))
        assert hovered[-1] == "241-TX-103"
        assert widget.tank_at(widget.tile_center("241-TX-103")) == "241-TX-103"

    def test_tooltip_gives_tank_farm_shell_volume_value_and_rank(self, widget):
        text = widget.tooltip_text("241-TX-108")
        assert "241-TX-108" in text
        assert "TX farm" in text and "single-shell" in text
        assert "1,000 kL" in text and "gallons" in text
        assert "SO4  80 mol per mol Na" in text
        assert "4th of 13 tanks" in text          # 5000, 100, 90, then 80

    def test_tooltip_of_a_double_shell_tank(self, widget):
        assert "double-shell" in widget.tooltip_text("241-AW-101")

    def test_tooltip_says_no_record_rather_than_zero(self, widget):
        text = widget.tooltip_text("241-AW-102")
        assert "No record of SO4" in text
        assert "SO4  0" not in text

    def test_every_tile_centre_hits_its_own_tank(self, widget):
        for tank in widget.tanks():
            assert widget.tank_at(widget.tile_center(tank)) == tank

    def test_farm_name_is_hoverable_with_farm_figures(self, widget, qtbot):
        farms = []
        widget.farmHovered.connect(farms.append)
        _hover(widget, widget.farm_label_rect("TX").center())
        assert farms[-1] == "TX"
        text = widget.farm_tooltip_text("TX")
        assert "TX farm" in text and "12 tanks" in text and "single-shell" in text
        assert "median" in text                       # an intensive quantity is not summed
        widget.set_values(_values(), label="Na", unit="kg", additive=True)
        assert "in total" in widget.farm_tooltip_text("TX")

    def test_leaving_the_tanks_clears_the_hover(self, widget, qtbot):
        hovered = []
        widget.tankHovered.connect(hovered.append)
        _hover(widget, widget.tile_center("241-TX-103"))
        _hover(widget, QPoint(3, 3))
        assert hovered[-1] == ""


class TestSelection:
    def test_click_selects_one_tank(self, widget, qtbot):
        changes, clicks = [], []
        widget.selectionChanged.connect(changes.append)
        widget.tankClicked.connect(clicks.append)
        qtbot.mouseClick(widget, Qt.LeftButton, pos=widget.tile_center("241-TX-104"))
        assert widget.selection() == ["241-TX-104"]
        assert changes[-1] == ["241-TX-104"] and clicks == ["241-TX-104"]

    def test_ctrl_click_extends_and_toggles(self, widget, qtbot):
        qtbot.mouseClick(widget, Qt.LeftButton, pos=widget.tile_center("241-TX-104"))
        qtbot.mouseClick(widget, Qt.LeftButton, Qt.ControlModifier, widget.tile_center("241-AW-101"))
        assert widget.selection() == ["241-TX-104", "241-AW-101"]
        qtbot.mouseClick(widget, Qt.LeftButton, Qt.ControlModifier, widget.tile_center("241-TX-104"))
        assert widget.selection() == ["241-AW-101"]

    def test_shift_click_extends(self, widget, qtbot):
        qtbot.mouseClick(widget, Qt.LeftButton, pos=widget.tile_center("241-TX-101"))
        qtbot.mouseClick(widget, Qt.LeftButton, Qt.ShiftModifier, widget.tile_center("241-TX-102"))
        assert widget.selection() == ["241-TX-101", "241-TX-102"]

    def test_click_on_empty_space_clears(self, widget, qtbot):
        qtbot.mouseClick(widget, Qt.LeftButton, pos=widget.tile_center("241-TX-104"))
        qtbot.mouseClick(widget, Qt.LeftButton, pos=QPoint(3, 3))
        assert widget.selection() == []

    def test_rubber_band_selects_the_tiles_inside(self, widget, qtbot):
        a, b = widget.tile_center("241-TX-101"), widget.tile_center("241-TX-105")
        qtbot.mousePress(widget, Qt.LeftButton, pos=a - QPoint(8, 8))
        qtbot.mouseMove(widget, b + QPoint(8, 8))
        qtbot.mouseRelease(widget, Qt.LeftButton, pos=b + QPoint(8, 8))
        assert set(widget.selection()) == {"241-TX-101", "241-TX-102", "241-TX-104", "241-TX-105"}

    def test_set_selection_from_code_is_silent(self, widget):
        changes = []
        widget.selectionChanged.connect(changes.append)
        widget.set_selection(["241-TX-101", "not a tank"])
        assert widget.selection() == ["241-TX-101"] and changes == []

    def test_double_click_activates_and_keeps_the_selection(self, widget, qtbot):
        activated = []
        widget.tankActivated.connect(activated.append)
        qtbot.mouseDClick(widget, Qt.LeftButton, pos=widget.tile_center("241-TX-107"))
        assert activated == ["241-TX-107"]
        qtbot.mouseRelease(widget, Qt.LeftButton, pos=widget.tile_center("241-TX-107"))
        assert widget.selection() == ["241-TX-107"]

    def test_highlight_does_not_select(self, widget):
        widget.highlight(["241-TX-101"])
        assert widget.selection() == []
        widget.highlight([])


class TestKeyboard:
    def test_arrows_move_between_tanks_and_enter_opens(self, widget, qtbot):
        activated = []
        widget.tankActivated.connect(activated.append)
        widget.setFocus()
        qtbot.mouseClick(widget, Qt.LeftButton, pos=widget.tile_center("241-TX-101"))
        qtbot.keyClick(widget, Qt.Key_Right)
        assert widget.cursor_tank() == "241-TX-102"
        qtbot.keyClick(widget, Qt.Key_Down)
        assert widget.cursor_tank() == "241-TX-105"
        qtbot.keyClick(widget, Qt.Key_Return)
        assert activated == ["241-TX-105"]

    def test_space_selects_and_escape_clears(self, widget, qtbot):
        widget.setFocus()
        qtbot.keyClick(widget, Qt.Key_Right)          # no cursor yet: the first tank
        first = widget.cursor_tank()
        qtbot.keyClick(widget, Qt.Key_Space)
        assert widget.selection() == [first]
        qtbot.keyClick(widget, Qt.Key_Escape)
        assert widget.selection() == []

    def test_accessible_name_and_description(self, widget):
        assert widget.accessibleName() == "Tank farm map"
        description = widget.accessibleDescription()
        assert "not to scale" in description and "SO4" in description


class TestPainting:
    def test_paints_in_both_themes_and_compact(self, widget, qtbot, monkeypatch):
        import qt_theme
        seen = []
        for palette in (qt_theme.PALETTE, qt_theme.DARK_PALETTE):
            monkeypatch.setattr(qt_theme, "current_palette", lambda p=palette: p, raising=False)
            pixmap = QPixmap(widget.size())
            widget.render(pixmap)
            assert not pixmap.isNull()
            seen.append(pixmap.toImage().pixelColor(2, 2).name())
        assert seen[0] != seen[1], "the map did not follow the theme"
        compact = FarmMapWidget(compact=True)
        qtbot.addWidget(compact)
        compact.resize(260, 190)
        compact.set_tanks(_tanks())
        compact.set_categories({"241-AW-101": "This tank"}, {"This tank": fm.SERIES_BLUE})
        pixmap = QPixmap(compact.size())
        compact.render(pixmap)
        assert not pixmap.isNull()

    def test_legend_goes_where_the_tanks_come_out_larger(self, qtbot):
        w = FarmMapWidget()
        qtbot.addWidget(w)
        w.set_tanks(_tanks())
        w.set_values(_values(), label="SO4", unit="mol per mol Na")
        w.resize(900, 330)                    # wide and short: beside the map
        assert w.legend_mode() == "side"
        w.resize(420, 600)                    # narrow and tall: under the map
        assert w.legend_mode() == "bottom"
        for size in ((900, 330), (420, 600)):
            w.resize(*size)
            for tank in w.tanks():            # every tile stays inside the widget
                centre = w.tile_center(tank)
                assert 0 < centre.x() < size[0] and 0 < centre.y() < size[1]
            pixmap = QPixmap(w.size())
            w.render(pixmap)
        w.set_legend_visible(False)
        assert w.legend_mode() == "none"
        assert FarmMapWidget(compact=True).legend_mode() == "none"

    def test_legend_names_each_class_by_its_range(self, widget):
        labels = widget._class_range_labels()
        assert labels == ["under 30", "30 to 50", "50 to 60", "60 to 80", "80 and above"]

    def test_empty_widget_paints(self, qtbot):
        w = FarmMapWidget()
        qtbot.addWidget(w)
        w.resize(300, 200)
        pixmap = QPixmap(w.size())
        w.render(pixmap)
        assert w.tank_at(QPoint(10, 10)) is None

    def test_palette_always_has_the_keys_the_map_reads(self):
        palette = fm.theme_palette()
        for key in ("card", "ink", "muted", "faint", "border", "border_strong", "accent", "selection_bg"):
            assert palette[key].startswith("#")


class TestRankedTankList:
    def _table(self):
        return pd.DataFrame({
            "WasteSiteId": ["241-TX-101", "241-TX-102", "241-AW-101"],
            "Value": [10.0, 30.0, 20.0],
            "TankFarm": ["TX", "TX", "AW"],
        })

    def test_rows_are_ranked_by_value(self, qtbot):
        view = RankedTankList()
        qtbot.addWidget(view)
        view.set_table(self._table(), "Na", "kg")
        assert view.tanks() == ["241-TX-102", "241-AW-101", "241-TX-101"]
        model = view.model()
        assert model.data(model.index(0, 0)) == "1"
        assert model.data(model.index(0, 2)) == "30"
        assert "1st of 3" in model.data(model.index(0, 1), Qt.ToolTipRole)

    def test_selection_from_code_is_silent_and_from_the_user_is_not(self, qtbot):
        view = RankedTankList()
        qtbot.addWidget(view)
        view.set_table(self._table(), "Na", "kg")
        seen = []
        view.selectionChanged_.connect(seen.append)
        view.set_selection(["241-AW-101"])
        assert view.selected_tanks() == ["241-AW-101"] and seen == []
        view.selectRow(0)
        assert seen[-1] == ["241-TX-102"]

    def test_double_click_activates_the_tank(self, qtbot):
        view = RankedTankList()
        qtbot.addWidget(view)
        view.set_table(self._table(), "Na", "kg")
        activated = []
        view.tankActivated.connect(activated.append)
        view.doubleClicked.emit(view.model().index(1, 1))
        assert activated == ["241-AW-101"]

    def test_empty_table(self, qtbot):
        view = RankedTankList()
        qtbot.addWidget(view)
        view.set_table(None)
        assert view.tanks() == []
        view.highlight("241-TX-101")
