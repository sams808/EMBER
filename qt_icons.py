"""
qt_icons.py -- the application's line-icon set, drawn with QPainter.

No icon font, no image files, no downloads: every icon is a few strokes
on a 24-unit grid, painted at whatever size and colour the caller asks
for. That is what lets one icon follow the theme (light / dark) and the
state of the thing it labels (selected, disabled, on the navy header).

    paint_icon(painter, "tank_explorer", rect, "#17212b")   # in a delegate
    icon("search", palette["chrome_muted"])                 # a QIcon
    icon_pixmap("help", "#ffffff", 18)                      # a QPixmap

Strokes are 1.5 px with round caps and joins. `paint_ember_mark` draws the
product mark (the layered ember of assets/ember_logo.png) without its
dark tile, for use on the header band and the splash.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Union

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

GRID = 24.0
STROKE = 1.5

ColorLike = Union[str, QColor]


def _line(p: QPainter, x1: float, y1: float, x2: float, y2: float) -> None:
    p.drawLine(QPointF(x1, y1), QPointF(x2, y2))


def _poly(p: QPainter, *points: tuple, closed: bool = False) -> None:
    path = QPainterPath()
    path.moveTo(*points[0])
    for x, y in points[1:]:
        path.lineTo(x, y)
    if closed:
        path.closeSubpath()
    p.drawPath(path)


def _dot(p: QPainter, x: float, y: float, r: float = 1.3) -> None:
    p.save()
    p.setBrush(p.pen().color())
    p.setPen(Qt.NoPen)
    p.drawEllipse(QPointF(x, y), r, r)
    p.restore()


def _rrect(p: QPainter, x: float, y: float, w: float, h: float, r: float = 1.5) -> None:
    p.drawRoundedRect(QRectF(x, y, w, h), r, r)


def _fill(p: QPainter, x: float, y: float, w: float, h: float) -> None:
    p.save()
    p.setBrush(p.pen().color())
    p.setPen(Qt.NoPen)
    p.drawRect(QRectF(x, y, w, h))
    p.restore()


# ---- workspaces -------------------------------------------------------
def _overview(p: QPainter) -> None:
    _rrect(p, 3.5, 3.5, 7, 8)
    _rrect(p, 13.5, 3.5, 7, 4.5)
    _rrect(p, 13.5, 11, 7, 9.5)
    _rrect(p, 3.5, 14.5, 7, 6)


def _query(p: QPainter) -> None:
    _poly(p, (3.5, 5), (20.5, 5), (14, 12.5), (14, 18.5), (10, 20.5), (10, 12.5), closed=True)


def _explorer(p: QPainter) -> None:
    for angle in (-45.0, 45.0):
        p.save()
        p.translate(12, 12)
        p.rotate(angle)
        p.drawEllipse(QPointF(0, 0), 9.5, 4.0)
        p.restore()
    _dot(p, 12, 12, 1.6)


def _tank_attributes(p: QPainter) -> None:
    for y in (6.0, 12.0, 18.0):
        _dot(p, 4.5, y, 1.2)
        _line(p, 9, y, 20.5, y)


def _tank_explorer(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 6.5), 7.5, 2.8)
    _line(p, 4.5, 6.5, 4.5, 17.5)
    _line(p, 19.5, 6.5, 19.5, 17.5)
    p.drawArc(QRectF(4.5, 14.7, 15, 5.6), 180 * 16, 180 * 16)
    p.drawArc(QRectF(4.5, 9.2, 15, 5.6), 180 * 16, 180 * 16)


def _heatmaps(p: QPainter) -> None:
    _rrect(p, 3.5, 3.5, 17, 17)
    for v in (9.17, 14.83):
        _line(p, v, 3.5, v, 20.5)
        _line(p, 3.5, v, 20.5, v)
    _fill(p, 9.17, 3.5, 5.66, 5.67)
    _fill(p, 3.5, 9.17, 5.67, 5.66)
    _fill(p, 14.83, 14.83, 5.67, 5.67)


def _correlations(p: QPainter) -> None:
    _poly(p, (4, 3.5), (4, 20), (20.5, 20))
    for x, y in ((8, 15.5), (11.5, 12), (14.5, 13.5), (17.5, 7.5)):
        _dot(p, x, y, 1.5)


def _vitrification(p: QPainter) -> None:
    _line(p, 8.5, 3.5, 15.5, 3.5)
    path = QPainterPath()
    path.moveTo(10, 3.5)
    path.lineTo(10, 9.5)
    path.lineTo(4.8, 18.4)
    path.quadTo(3.9, 20.5, 6.2, 20.5)
    path.lineTo(17.8, 20.5)
    path.quadTo(20.1, 20.5, 19.2, 18.4)
    path.lineTo(14, 9.5)
    path.lineTo(14, 3.5)
    p.drawPath(path)
    _line(p, 7.3, 15, 16.7, 15)


def _figure_composer(p: QPainter) -> None:
    _rrect(p, 3.5, 4.5, 17, 15)
    _line(p, 12, 4.5, 12, 19.5)
    _line(p, 12, 12, 20.5, 12)


# ---- actions ----------------------------------------------------------
def _search(p: QPainter) -> None:
    p.drawEllipse(QPointF(10.5, 10.5), 6.5, 6.5)
    _line(p, 15.4, 15.4, 20.5, 20.5)


def _help(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 12), 9, 9)
    path = QPainterPath()
    path.moveTo(9.3, 9.6)
    path.cubicTo(9.3, 5.9, 14.9, 5.9, 14.9, 9.4)
    path.cubicTo(14.9, 11.6, 12, 11.6, 12, 14)
    p.drawPath(path)
    _dot(p, 12, 17.2, 1.1)


def _info(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 12), 9, 9)
    _line(p, 12, 11, 12, 17)
    _dot(p, 12, 7.6, 1.1)


def _sun(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 12), 4, 4)
    for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0), (0.707, 0.707), (-0.707, 0.707), (0.707, -0.707),
                   (-0.707, -0.707)):
        _line(p, 12 + dx * 7, 12 + dy * 7, 12 + dx * 9.5, 12 + dy * 9.5)


def _moon(p: QPainter) -> None:
    full = QPainterPath()
    full.addEllipse(QPointF(11.5, 12.5), 8.5, 8.5)
    bite = QPainterPath()
    bite.addEllipse(QPointF(16.5, 8), 7.5, 7.5)
    p.drawPath(full.subtracted(bite))


def _chevron_left(p: QPainter) -> None:
    _poly(p, (14.5, 5.5), (8, 12), (14.5, 18.5))


def _chevron_right(p: QPainter) -> None:
    _poly(p, (9.5, 5.5), (16, 12), (9.5, 18.5))


def _chevron_down(p: QPainter) -> None:
    _poly(p, (5.5, 9.5), (12, 16), (18.5, 9.5))


def _close(p: QPainter) -> None:
    _line(p, 6, 6, 18, 18)
    _line(p, 18, 6, 6, 18)


def _database(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 6), 7.5, 2.8)
    _line(p, 4.5, 6, 4.5, 18)
    _line(p, 19.5, 6, 19.5, 18)
    p.drawArc(QRectF(4.5, 9.2, 15, 5.6), 180 * 16, 180 * 16)
    p.drawArc(QRectF(4.5, 15.2, 15, 5.6), 180 * 16, 180 * 16)


def _folder(p: QPainter) -> None:
    _poly(p, (3.5, 18.5), (3.5, 5.5), (9.5, 5.5), (11.5, 8), (20.5, 8), (20.5, 18.5), closed=True)


def _export(p: QPainter) -> None:
    _line(p, 12, 3.5, 12, 15)
    _poly(p, (7.5, 8), (12, 3.5), (16.5, 8))
    _poly(p, (4, 14.5), (4, 20), (20, 20), (20, 14.5))


def _panel(p: QPainter) -> None:
    _rrect(p, 3.5, 4.5, 17, 15)
    _line(p, 9.5, 4.5, 9.5, 19.5)


def _menu(p: QPainter) -> None:
    for y in (6.5, 12.0, 17.5):
        _line(p, 4, y, 20, y)


def _play(p: QPainter) -> None:
    _poly(p, (8, 5), (19, 12), (8, 19), closed=True)


_ICONS: Dict[str, Callable[[QPainter], None]] = {
    "overview": _overview,
    "query": _query,
    "explorer": _explorer,
    "tank_attributes": _tank_attributes,
    "tank_explorer": _tank_explorer,
    "heatmaps": _heatmaps,
    "correlations": _correlations,
    "vitrification": _vitrification,
    "figure_composer": _figure_composer,
    "search": _search,
    "help": _help,
    "info": _info,
    "sun": _sun,
    "moon": _moon,
    "chevron_left": _chevron_left,
    "chevron_right": _chevron_right,
    "chevron_down": _chevron_down,
    "close": _close,
    "database": _database,
    "folder": _folder,
    "export": _export,
    "panel": _panel,
    "menu": _menu,
    "play": _play,
}

ICON_NAMES = tuple(_ICONS)

# Workspace (nav) name -> icon name. Plain strings on purpose: this module
# must stay importable from qt_help, which qt_shell itself imports.
WORKSPACE_ICONS: Dict[str, str] = {
    "Overview": "overview",
    "Query": "query",
    "Element Explorer": "explorer",
    "Tank Attributes": "tank_attributes",
    "Tank Explorer": "tank_explorer",
    "Heatmaps": "heatmaps",
    "Correlations": "correlations",
    "Vitrification": "vitrification",
    "Figure Composer": "figure_composer",
}


def paint_icon(painter: QPainter, name: str, rect: QRectF, color: ColorLike,
               stroke: float = STROKE) -> bool:
    """Paint icon `name` into `rect` (logical pixels) in `color`.

    Returns False, having painted nothing, for an unknown name: this runs
    inside paint events, where an exception would take the widget down."""
    draw = _ICONS.get(name)
    if draw is None:
        return False
    side = min(rect.width(), rect.height())
    if side <= 0:
        return False
    scale = side / GRID
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.translate(rect.x() + (rect.width() - side) / 2.0, rect.y() + (rect.height() - side) / 2.0)
    painter.scale(scale, scale)
    pen = QPen(QColor(color))
    pen.setWidthF(stroke / scale)      # 1.5 px on screen whatever the size
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    draw(painter)
    painter.restore()
    return True


def icon_pixmap(name: str, color: ColorLike, size: int = 18, dpr: Optional[float] = None) -> QPixmap:
    """A transparent `size` x `size` (logical px) pixmap of one icon."""
    if dpr is None:
        from PySide6.QtGui import QGuiApplication
        app = QGuiApplication.instance()
        dpr = float(app.devicePixelRatio()) if app is not None else 1.0
    dpr = max(1.0, float(dpr))
    pixmap = QPixmap(int(round(size * dpr)), int(round(size * dpr)))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    paint_icon(painter, name, QRectF(0, 0, size, size), color)
    painter.end()
    return pixmap


def icon(name: str, color: ColorLike, size: int = 18, disabled_color: Optional[ColorLike] = None) -> QIcon:
    """A QIcon of one icon; `disabled_color` adds the disabled-state pixmap."""
    result = QIcon(icon_pixmap(name, color, size))
    if disabled_color is not None:
        result.addPixmap(icon_pixmap(name, disabled_color, size), QIcon.Disabled)
    return result


# ---- the product mark ---------------------------------------------------
EMBER_MARK = {
    "core": "#7a2a17",
    "outer": "#c1502e",
    "inner": "#e0703f",
    "tip": "#ffd27a",
}


def paint_ember_mark(painter: QPainter, rect: QRectF, sparks: bool = True) -> None:
    """The layered ember of the application icon, on a transparent ground.
    Geometry follows assets/ember_logo.png (a 10 x 10 design grid, y up)."""
    side = min(rect.width(), rect.height())
    if side <= 0:
        return
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.translate(rect.x() + (rect.width() - side) / 2.0, rect.y() + (rect.height() - side) / 2.0)
    # The mark occupies x 1.85..8.15 and y 1.45..7.75 of the design grid:
    # a 6.3-unit square, fitted here with a 0.15-unit margin.
    unit = side / 6.6
    painter.scale(unit, unit)
    painter.translate(-1.7, -2.1)

    def disc(x: float, y: float, r: float, color: str, alpha: float = 1.0) -> None:
        c = QColor(color)
        c.setAlphaF(alpha)
        painter.setBrush(c)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(x, 10.0 - y), r, r)

    disc(5.0, 4.6, 3.15, EMBER_MARK["core"])
    disc(5.0, 4.3, 2.55, EMBER_MARK["outer"])
    disc(5.05, 3.85, 1.85, EMBER_MARK["inner"], 0.95)
    disc(5.1, 3.55, 1.05, EMBER_MARK["tip"], 0.95)
    if sparks:
        for x, y, r in ((7.6, 6.6, 0.22), (7.95, 7.55, 0.14), (2.55, 7.0, 0.16)):
            disc(x, y, r, EMBER_MARK["tip"], 0.85)
    painter.restore()


def ember_mark_pixmap(size: int = 28, dpr: Optional[float] = None, sparks: bool = True) -> QPixmap:
    if dpr is None:
        from PySide6.QtGui import QGuiApplication
        app = QGuiApplication.instance()
        dpr = float(app.devicePixelRatio()) if app is not None else 1.0
    dpr = max(1.0, float(dpr))
    pixmap = QPixmap(int(round(size * dpr)), int(round(size * dpr)))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    paint_ember_mark(painter, QRectF(0, 0, size, size), sparks=sparks)
    painter.end()
    return pixmap
