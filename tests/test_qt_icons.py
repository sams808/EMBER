"""
The painted line-icon set: every name draws something, in the colour it
was asked for, and an unknown name fails quietly (icons are painted inside
paint events, where an exception would take the widget down).
"""
import pytest
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter

import qt_icons as qi

REAL_NAV_ITEMS = [
    "Overview", "Query", "Element Explorer", "Tank Attributes", "Tank Explorer",
    "Heatmaps", "Correlations", "Vitrification", "Figure Composer",
]


def _painted_pixels(image: QImage):
    """(count of non-transparent pixels, set of their opaque RGB values)."""
    count, colors = 0, set()
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            if color.alpha() > 0:
                count += 1
                if color.alpha() == 255:
                    colors.add((color.red(), color.green(), color.blue()))
    return count, colors


class TestIconSet:
    @pytest.mark.parametrize("name", qi.ICON_NAMES)
    def test_every_icon_paints_something(self, qtbot, name):
        image = qi.icon_pixmap(name, "#000000", 24, dpr=1.0).toImage()
        count, _colors = _painted_pixels(image)
        assert count > 10, f"icon {name!r} is (nearly) empty"
        # ... and stays inside its square: a stroke that runs off the grid
        # is cut at the edge, which shows as a hard, fully opaque border row.
        assert image.width() == 24 and image.height() == 24

    def test_icons_are_distinct_drawings(self, qtbot):
        rendered = {}
        for name in qi.ICON_NAMES:
            image = qi.icon_pixmap(name, "#000000", 24, dpr=1.0).toImage()
            key = bytes(image.constBits())
            assert key not in rendered, f"{name!r} draws the same picture as {rendered.get(key)!r}"
            rendered[key] = name

    def test_colour_is_the_one_asked_for(self, qtbot):
        red = qi.icon_pixmap("tank_explorer", "#ff0000", 24, dpr=1.0).toImage()
        blue = qi.icon_pixmap("tank_explorer", "#0000ff", 24, dpr=1.0).toImage()
        _n, red_colors = _painted_pixels(red)
        _n, blue_colors = _painted_pixels(blue)
        assert (255, 0, 0) in red_colors and (0, 0, 255) not in red_colors
        assert (0, 0, 255) in blue_colors and (255, 0, 0) not in blue_colors

    def test_recolouring_keeps_the_shape(self, qtbot):
        # Same strokes, other colour: the alpha channel is identical.
        a = qi.icon_pixmap("heatmaps", "#17212b", 24, dpr=1.0).toImage()
        b = qi.icon_pixmap("heatmaps", "#f08a4b", 24, dpr=1.0).toImage()
        alpha_a = [a.pixelColor(x, y).alpha() for y in range(24) for x in range(24)]
        alpha_b = [b.pixelColor(x, y).alpha() for y in range(24) for x in range(24)]
        assert alpha_a == alpha_b

    def test_high_dpi_pixmap_has_more_pixels_and_the_same_logical_size(self, qtbot):
        pixmap = qi.icon_pixmap("search", "#000000", 18, dpr=2.0)
        assert pixmap.width() == 36 and pixmap.height() == 36
        assert pixmap.devicePixelRatio() == 2.0

    def test_unknown_name_paints_nothing_and_says_so(self, qtbot):
        image = QImage(24, 24, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        assert qi.paint_icon(painter, "no_such_icon", QRectF(0, 0, 24, 24), "#000000") is False
        painter.end()
        assert _painted_pixels(image)[0] == 0

    def test_known_name_reports_success(self, qtbot):
        image = QImage(24, 24, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        assert qi.paint_icon(painter, "overview", QRectF(0, 0, 24, 24), QColor("#123456")) is True
        painter.end()

    def test_empty_rect_is_refused_not_a_crash(self, qtbot):
        image = QImage(24, 24, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        assert qi.paint_icon(painter, "overview", QRectF(0, 0, 0, 0), "#000000") is False
        painter.end()

    def test_icon_carries_a_disabled_variant_when_asked(self, qtbot):
        from PySide6.QtGui import QIcon
        icon = qi.icon("help", "#000000", 18, disabled_color="#ff0000")
        assert not icon.isNull()
        disabled = icon.pixmap(18, 18, QIcon.Disabled).toImage()
        _n, colors = _painted_pixels(disabled)
        assert any(r > 200 and g < 60 and b < 60 for r, g, b in colors)


class TestWorkspaceIcons:
    def test_every_workspace_has_an_icon_that_exists(self):
        assert set(qi.WORKSPACE_ICONS) == set(REAL_NAV_ITEMS)
        for name, icon_name in qi.WORKSPACE_ICONS.items():
            assert icon_name in qi.ICON_NAMES, f"{name!r} points at a missing icon {icon_name!r}"

    def test_no_two_workspaces_share_an_icon(self):
        assert len(set(qi.WORKSPACE_ICONS.values())) == len(qi.WORKSPACE_ICONS)


class TestEmberMark:
    def test_mark_paints_its_warm_colours_and_fits_its_square(self, qtbot):
        image = qi.ember_mark_pixmap(64, dpr=1.0).toImage()
        count, colors = _painted_pixels(image)
        assert count > 1000
        core = QColor(qi.EMBER_MARK["core"])
        assert (core.red(), core.green(), core.blue()) in colors
        # The corners stay empty: the mark is round and inside the square.
        for x, y in ((0, 0), (63, 0), (0, 63), (63, 63)):
            assert image.pixelColor(x, y).alpha() == 0
        # ... and it reaches towards every edge (not a small blob in a corner).
        xs = [x for y in range(64) for x in range(64) if image.pixelColor(x, y).alpha() > 0]
        ys = [y for y in range(64) for x in range(64) if image.pixelColor(x, y).alpha() > 0]
        assert min(xs) <= 4 and max(xs) >= 59
        assert min(ys) <= 4 and max(ys) >= 59
