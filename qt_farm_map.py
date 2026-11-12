"""
qt_farm_map.py -- the interactive tank-farm map and its ranked companion.

FarmMapWidget paints the 177 tanks as discs in a schematic grouped by farm
(farm_layout.py: not a geographic map, not to scale). A disc's colour
encodes a value in classes with real break values; a tank with no record
is hollow, never drawn as a zero. Double-shell tanks show a second ring,
and tanks holding little waste are drawn with a smaller disc inside their
outline. The widget answers to the mouse and the keyboard: hover for a
callout, click to select (Ctrl or Shift extends, drag a rubber band),
double-click or Enter to open a tank, arrow keys to move between tanks.

RankedTankList is the same quantity as a ranked table with in-cell bars;
the two are meant to be wired together (linked hover and selection).

Everything is drawn by QPainter from the active theme palette, so the map
is crisp at any size and follows light and dark mode. The callout is
painted inside the widget rather than shown as a QToolTip window.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PySide6.QtCore import (
    QAbstractTableModel, QModelIndex, QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal,
)
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QHeaderView, QSizePolicy, QStyle,
    QStyledItemDelegate, QTableView, QWidget,
)

import farm_layout as fl
from site_summary import (
    format_compact, format_gallons, format_number, format_volume, ordinal,
)

# Data colours of the design system (fixed values; charts and maps share them).
BLUE_RAMP = ["#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES_BLUE = "#2a78d6"
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

SCALES = ("quantile", "log", "linear")
PALETTES = ("sequential", "diverging")
N_CLASSES = 5
MIN_VOLUME_KL = 100.0     # below this a tank holds a heel: smaller disc, not in the class breaks
SMALL_FRACTION = 0.56
CAPTION = "Schematic: tanks grouped by farm, not to scale"

_FALLBACK_PALETTE = {
    "bg": "#f7f7f8", "bg_alt": "#eef0f2", "card": "#ffffff", "ink": "#20242a",
    "muted": "#6b7280", "border": "#d9dce1", "accent": "#c1502e",
    "selection_bg": "#eef0f2",
}


# ---------------------------------------------------------------------------
# theme
# ---------------------------------------------------------------------------
def _blend(a: str, b: str, t: float) -> str:
    ca, cb = QColor(a), QColor(b)
    return QColor(
        round(ca.red() + (cb.red() - ca.red()) * t),
        round(ca.green() + (cb.green() - ca.green()) * t),
        round(ca.blue() + (cb.blue() - ca.blue()) * t),
    ).name()


def is_dark(colour: str) -> bool:
    c = QColor(colour)
    return (0.2126 * c.red() + 0.7152 * c.green() + 0.0722 * c.blue()) < 128


def theme_palette() -> Dict[str, str]:
    """The active theme palette, with every key this module reads.

    qt_theme.current_palette() is used when the theme provides it; an
    older theme module is read through its two palettes and the
    application stylesheet. Keys a palette lacks are derived, so the map
    never fails on a missing colour."""
    palette: Dict[str, str] = {}
    try:
        import qt_theme
        current = getattr(qt_theme, "current_palette", None)
        if callable(current):
            palette = dict(current())
        else:
            app = QApplication.instance()
            sheet = app.styleSheet() if app is not None else ""
            dark = getattr(qt_theme, "DARK_PALETTE", {})
            light = getattr(qt_theme, "PALETTE", {})
            use_dark = bool(sheet) and dark.get("bg", "#") in sheet and light.get("bg", "#") not in sheet
            palette = dict(dark if use_dark else light)
    except Exception:
        palette = {}
    merged = dict(_FALLBACK_PALETTE)
    merged.update({k: v for k, v in palette.items() if isinstance(v, str)})
    merged.setdefault("faint", _blend(merged["muted"], merged["card"], 0.35))
    merged.setdefault("border_strong", _blend(merged["border"], merged["ink"], 0.30))
    merged.setdefault("chrome", "#16324f")
    merged.setdefault("chrome_ink", "#ffffff")
    merged.setdefault("chrome_muted", "#b9c6d6")
    return merged


# ---------------------------------------------------------------------------
# value -> class
# ---------------------------------------------------------------------------
def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    return arr[np.isfinite(arr)]


def class_breaks(values: Iterable[float], scale: str = "quantile", n_classes: int = N_CLASSES,
                 reference: Optional[Iterable[float]] = None,
                 palette: str = "sequential") -> List[float]:
    """Interior class boundaries (at most n_classes - 1, ascending).

    quantile: equal-count classes of the positive reference values;
    log: equal steps of log10 between their smallest and largest;
    linear: equal steps between the smallest and largest reference value.
    `reference` is the subset that sets the scale (the tanks holding a
    real volume of waste); without enough of it, every value is used.
    A diverging palette gets boundaries symmetric about zero.
    """
    if scale not in SCALES:
        raise ValueError(f"Unknown scale {scale!r}; expected one of {SCALES}")
    vals = _finite(values)
    if vals.size == 0 or float(vals.min()) == float(vals.max()):
        return []
    ref = _finite(reference) if reference is not None else vals
    if palette == "diverging":
        magnitude = np.abs(ref if ref.size >= n_classes else vals)
        magnitude = magnitude[magnitude > 0]
        if magnitude.size == 0:
            return []
        if scale == "linear":
            inner, outer = 0.2 * float(magnitude.max()), 0.6 * float(magnitude.max())
        else:
            inner, outer = (float(q) for q in np.quantile(magnitude, [0.2, 0.6]))
        edges = [-outer, -inner, inner, outer]
    elif scale == "linear":
        if ref.size < 2 or float(ref.min()) == float(ref.max()):
            ref = vals
        edges = np.linspace(float(ref.min()), float(ref.max()), n_classes + 1)[1:-1]
    else:
        positive = ref[ref > 0]
        if positive.size < n_classes:
            positive = vals[vals > 0]
        if positive.size == 0 or float(positive.min()) == float(positive.max()):
            return []
        if scale == "log":
            edges = 10 ** np.linspace(math.log10(float(positive.min())),
                                      math.log10(float(positive.max())), n_classes + 1)[1:-1]
        else:
            edges = np.quantile(positive, [i / n_classes for i in range(1, n_classes)])
    out: List[float] = []
    for edge in edges:
        edge = float(edge)
        if math.isfinite(edge) and (not out or edge > out[-1]):
            out.append(edge)
    return out


def classify(value: Optional[float], breaks: Sequence[float]) -> Optional[int]:
    """Class index of a value (0 = lowest), or None for no record.
    A value equal to a boundary belongs to the class above it."""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return int(np.searchsorted(np.asarray(breaks, dtype=float), value, side="right"))


def class_colours(n: int, palette: str = "sequential", dark: bool = False) -> List[str]:
    """n colours for n classes, low to high. On a dark surface the
    sequential ramp runs from deep to pale so the high end still stands
    out; the diverging pair keeps a neutral midpoint."""
    n = max(1, int(n))
    if palette == "diverging":
        mid = "#6b7078" if dark else DIVERGING[1]
        five = [DIVERGING[0], _blend(DIVERGING[0], mid, 0.55), mid,
                _blend(DIVERGING[2], mid, 0.55), DIVERGING[2]]
    elif dark:
        five = [BLUE_RAMP[6], BLUE_RAMP[5], BLUE_RAMP[4], BLUE_RAMP[2], BLUE_RAMP[1]]
    else:
        five = [BLUE_RAMP[1], BLUE_RAMP[2], BLUE_RAMP[4], BLUE_RAMP[5], BLUE_RAMP[7]]
    if n == 1:
        return [five[2]]
    if n >= 5:
        return five
    return [five[round(i * 4 / (n - 1))] for i in range(n)]


# ---------------------------------------------------------------------------
# the map
# ---------------------------------------------------------------------------
class FarmMapWidget(QWidget):
    """Schematic tank-farm map: one disc per tank, coloured by a value."""

    tankHovered = Signal(str)        # "" when the pointer leaves the tanks
    farmHovered = Signal(str)
    tankClicked = Signal(str)
    tankActivated = Signal(str)      # double-click or Enter
    selectionChanged = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None, *, compact: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName("FarmMap")
        self._compact = bool(compact)
        self._legend_visible = not compact
        self._caption_visible = True
        self._layout = fl.FarmLayout()
        self._info: Dict[str, Dict[str, object]] = {}
        self._values: Dict[str, float] = {}
        self._label, self._unit = "", ""
        self._scale, self._palette = "quantile", "sequential"
        self._additive = True
        self._breaks: List[float] = []
        self._classes: Dict[str, Optional[int]] = {}
        self._ranks: Dict[str, int] = {}
        self._categories: Dict[str, str] = {}
        self._category_colours: Dict[str, str] = {}
        self._mode = "values"
        self._sizes: Optional[Dict[str, float]] = None
        self._in_scope: Optional[set] = None
        self._selection: List[str] = []
        self._highlight: set = set()
        self._hover_tank: Optional[str] = None
        self._hover_farm: Optional[str] = None
        self._cursor_tank: Optional[str] = None
        self._press_pos: Optional[QPoint] = None
        self._press_tank: Optional[str] = None
        self._band: Optional[QRect] = None
        self._skip_release = False
        self._key_nav = False
        self._geometry: Optional[dict] = None
        self._version = 0
        self._base: Optional[QPixmap] = None
        self._base_key: object = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAccessibleName("Tank farm map")
        self._update_accessible_description()

    # ---- size ---------------------------------------------------------
    def sizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(260, 190) if self._compact else QSize(720, 540)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(180, 130) if self._compact else QSize(420, 340)

    # ---- data ---------------------------------------------------------
    def set_tanks(self, table: object) -> None:
        """The tanks to draw: a table with WasteSiteId and, when known,
        TankFarm, TankSystem and Volume_kL (or a plain list of ids)."""
        self._layout = fl.farm_layout(table)
        self._info = {}
        if isinstance(table, pd.DataFrame) and "WasteSiteId" in table.columns:
            for record in table.to_dict("records"):
                self._info[str(record["WasteSiteId"])] = record
        known = set(self._layout.tiles)
        self._selection = [t for t in self._selection if t in known]
        self._highlight &= known
        if self._hover_tank not in known:
            self._hover_tank = None
        if self._cursor_tank not in known:
            self._cursor_tank = None
        self._geometry = None
        self._reclassify()
        self._update_accessible_description()
        self._invalidate()

    def tanks(self) -> List[str]:
        return self._layout.tanks()

    def layout_model(self) -> fl.FarmLayout:
        return self._layout

    def set_values(self, values: Mapping[str, float], label: str = "", unit: str = "",
                   scale: str = "quantile", palette: str = "sequential",
                   additive: bool = True) -> None:
        """Colour the tanks by a magnitude. A tank missing from `values`
        (or mapped to None / NaN) has no record and is drawn hollow."""
        if scale not in SCALES:
            raise ValueError(f"Unknown scale {scale!r}; expected one of {SCALES}")
        if palette not in PALETTES:
            raise ValueError(f"Unknown palette {palette!r}; expected one of {PALETTES}")
        self._mode = "values"
        self._values = {}
        for tank, value in (values or {}).items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                self._values[str(tank)] = number
        self._label, self._unit = str(label or ""), str(unit or "")
        self._scale, self._palette, self._additive = scale, palette, bool(additive)
        self._reclassify()
        self._update_accessible_description()
        self._invalidate()

    def set_categories(self, categories: Mapping[str, str], colours: Mapping[str, str],
                       label: str = "") -> None:
        """Colour the tanks by a category. Tanks missing from the mapping
        are hollow; `colours` maps each category to its colour."""
        self._mode = "categories"
        self._categories = {str(t): str(c) for t, c in (categories or {}).items() if c is not None}
        self._category_colours = dict(colours or {})
        self._label, self._unit = str(label or ""), ""
        self._update_accessible_description()
        self._invalidate()

    def set_sizes(self, sizes: Optional[Mapping[str, float]]) -> None:
        """Disc size per tank as a fraction of its outline (0 to 1).
        None restores the default: a smaller disc under 100 kL of waste."""
        self._sizes = None if sizes is None else {str(t): float(v) for t, v in sizes.items()}
        self._invalidate()

    def set_scope_tanks(self, tanks: Optional[Iterable[str]]) -> None:
        """The tanks inside the active scope; the others are drawn faint.
        None means every tank is in scope."""
        self._in_scope = None if tanks is None else {str(t) for t in tanks}
        self._reclassify()
        self._invalidate()

    def set_selection(self, tanks: Iterable[str]) -> None:
        """Select tanks from code (no selectionChanged is emitted)."""
        known = self._layout.tiles
        selection = [str(t) for t in tanks if str(t) in known]
        if selection != self._selection:
            self._selection = selection
            self.update()

    def selection(self) -> List[str]:
        return list(self._selection)

    def highlight(self, tanks: Iterable[str]) -> None:
        """Emphasise tanks without selecting them (linked hover)."""
        highlight = {str(t) for t in tanks if str(t) in self._layout.tiles}
        if highlight != self._highlight:
            self._highlight = highlight
            self.update()

    def set_legend_visible(self, visible: bool) -> None:
        self._legend_visible = bool(visible)
        self._geometry = None
        self._invalidate()

    def set_caption_visible(self, visible: bool) -> None:
        self._caption_visible = bool(visible)
        self._invalidate()

    def _invalidate(self) -> None:
        """Something the cached base picture shows has changed."""
        self._version += 1
        self.update()

    # ---- derived state ------------------------------------------------
    def _volume(self, tank: str) -> Optional[float]:
        value = self._info.get(tank, {}).get("Volume_kL")
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _reference_values(self) -> List[float]:
        """Values of the tanks holding a real volume of waste: the ones
        that set the class breaks, so a heel in a retrieved tank does not."""
        return [v for t, v in self._values.items()
                if (self._volume(t) or 0.0) >= MIN_VOLUME_KL and self._is_in_scope(t)]

    def _is_in_scope(self, tank: str) -> bool:
        return self._in_scope is None or tank in self._in_scope

    def _reclassify(self) -> None:
        shown = {t: v for t, v in self._values.items()
                 if t in self._layout.tiles and self._is_in_scope(t)}
        reference = self._reference_values()
        self._breaks = class_breaks(
            shown.values(), self._scale, N_CLASSES,
            reference=reference if len(reference) >= N_CLASSES else None,
            palette=self._palette,
        )
        self._classes = {t: classify(v, self._breaks) for t, v in shown.items()}
        ordered = sorted(shown.items(), key=lambda item: (-item[1], item[0]))
        self._ranks = {}
        previous, rank = None, 0
        for position, (tank, value) in enumerate(ordered, start=1):
            if value != previous:
                rank, previous = position, value
            self._ranks[tank] = rank

    def breaks(self) -> List[float]:
        return list(self._breaks)

    def class_count(self) -> int:
        return len(self._breaks) + 1

    def class_of(self, tank: str) -> Optional[int]:
        """Class index of a tank (0 = lowest), None when it has no record."""
        return self._classes.get(str(tank))

    def value_of(self, tank: str) -> Optional[float]:
        return self._values.get(str(tank))

    def rank_of(self, tank: str) -> Optional[int]:
        return self._ranks.get(str(tank))

    def legend_labels(self) -> List[str]:
        """The break values as the legend prints them, low to high."""
        return [format_compact(b) for b in self._breaks]

    def size_fraction(self, tank: str) -> float:
        if self._sizes is not None:
            value = self._sizes.get(tank, 1.0)
            return min(1.0, max(0.0, value)) if math.isfinite(value) else 1.0
        volume = self._volume(tank)
        return SMALL_FRACTION if volume is not None and volume < MIN_VOLUME_KL else 1.0

    def colour_of(self, tank: str, dark: Optional[bool] = None) -> Optional[str]:
        """Fill colour of a tank, None when it is hollow."""
        if not self._is_in_scope(tank):
            return None
        if self._mode == "categories":
            category = self._categories.get(tank)
            return self._category_colours.get(category) if category is not None else None
        index = self._classes.get(tank)
        if index is None:
            return None
        if dark is None:
            dark = is_dark(theme_palette()["card"])
        colours = class_colours(self.class_count(), self._palette, dark)
        return colours[min(index, len(colours) - 1)]

    def _update_accessible_description(self) -> None:
        n = len(self._layout.tiles)
        if self._mode == "categories":
            what = f"coloured by {self._label}" if self._label else "coloured by category"
        elif self._values:
            unit = f" in {self._unit}" if self._unit else ""
            what = f"coloured by {self._label}{unit}, {self._scale} classes"
        else:
            what = "no quantity mapped"
        self.setAccessibleDescription(
            f"{CAPTION}. {n} tanks, {what}. Arrow keys move between tanks, "
            f"Enter opens a tank, Space selects it."
        )

    # ---- callout text -------------------------------------------------
    def tooltip_lines(self, tank: str) -> List[Tuple[str, str]]:
        """(text, kind) lines describing a tank; kind is title, body or muted."""
        tile = self._layout.tiles.get(tank)
        if tile is None:
            return []
        lines: List[Tuple[str, str]] = [(tank, "title")]
        shell = "double-shell tank" if tile.double_shell else "single-shell tank"
        lines.append((f"{tile.farm} farm · {shell}", "muted"))
        volume = self._volume(tank)
        if volume is not None:
            lines.append((f"Waste volume  {format_number(volume)} kL ({format_gallons(volume)})", "body"))
        if not self._is_in_scope(tank):
            lines.append(("Outside the current scope", "muted"))
            return lines
        if self._mode == "categories":
            category = self._categories.get(tank)
            if category is not None:
                prefix = f"{self._label}  " if self._label else ""
                lines.append((f"{prefix}{category}", "body"))
            return lines
        if not self._label and not self._values:
            return lines
        value = self._values.get(tank)
        if value is None:
            lines.append((f"No record of {self._label}" if self._label else "No record", "muted"))
            return lines
        unit = f" {self._unit}" if self._unit else ""
        lines.append((f"{self._label}  {format_number(value)}{unit}".strip(), "body"))
        rank = self._ranks.get(tank)
        if rank is not None:
            lines.append((f"{ordinal(rank)} of {len(self._ranks):,} tanks", "muted"))
        return lines

    def tooltip_text(self, tank: str) -> str:
        return "\n".join(text for text, _kind in self.tooltip_lines(tank))

    def farm_tooltip_lines(self, farm: str) -> List[Tuple[str, str]]:
        block = self._layout.farm(farm)
        if block is None:
            return []
        tiles = [self._layout.tiles[t] for t in block.tanks]
        n_double = sum(1 for t in tiles if t.double_shell)
        if n_double == len(tiles):
            kind = "double-shell"
        elif n_double == 0:
            kind = "single-shell"
        else:
            kind = f"{n_double} double-shell"
        lines: List[Tuple[str, str]] = [
            (f"{farm} farm", "title"),
            (f"{len(tiles)} tank{'s' if len(tiles) != 1 else ''} · {kind}", "muted"),
        ]
        volumes = [v for v in (self._volume(t) for t in block.tanks) if v is not None]
        if volumes:
            total = float(sum(volumes))
            lines.append((f"Waste volume  {format_volume(total)} ({format_gallons(total)})", "body"))
        if self._mode == "values" and self._label:
            values = [self._values[t] for t in block.tanks
                      if t in self._values and self._is_in_scope(t)]
            unit = f" {self._unit}" if self._unit else ""
            if not values:
                lines.append((f"No record of {self._label}", "muted"))
            elif self._additive:
                lines.append((f"{self._label}  {format_number(sum(values))}{unit} in total", "body"))
            else:
                lines.append((f"{self._label}  median {format_number(float(np.median(values)))}{unit}", "body"))
                lines.append((f"range {format_number(min(values))} to {format_number(max(values))}", "muted"))
            if values:
                lines.append((f"{len(values)} of {len(tiles)} tanks with a record", "muted"))
        return lines

    def farm_tooltip_text(self, farm: str) -> str:
        return "\n".join(text for text, _kind in self.farm_tooltip_lines(farm))

    # ---- geometry -----------------------------------------------------
    def _font(self, px: float, weight: int = QFont.Normal, spacing: float = 0.0) -> QFont:
        font = QFont("Segoe UI")
        font.setPixelSize(max(7, int(round(px))))
        font.setWeight(QFont.Weight(weight))
        if spacing:
            font.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
        return font

    SIDE_LEGEND_WIDTH = 154
    SIDE_LEGEND_MIN_HEIGHT = 290
    BOTTOM_LEGEND_HEIGHT = 62

    def _ensure_geometry(self) -> dict:
        key = (self.width(), self.height(), id(self._layout), self._legend_visible,
               self._caption_visible, self._compact, self.devicePixelRatioF())
        if self._geometry is not None and self._geometry["key"] == key:
            return self._geometry
        margin = 6 if self._compact else 10
        width, height = float(self.width()), float(self.height())
        lw, lh = max(self._layout.width, 1.0), max(self._layout.height, 1.0)
        caption_h = 16.0 if self._caption_visible else 0.0

        # Candidate areas for the tiles: legend strip under the map, or a
        # legend column at its right. The one that draws larger tanks wins.
        mode = "none"
        area = QRectF(margin, margin, max(10.0, width - 2 * margin),
                      max(10.0, height - 2 * margin - (caption_h if self._compact else 0.0)))
        if self._legend_visible:
            bottom = QRectF(margin, margin, max(10.0, width - 2 * margin),
                            max(10.0, height - 2 * margin - self.BOTTOM_LEGEND_HEIGHT - 14))
            side = QRectF(margin, margin, max(10.0, width - 2 * margin - self.SIDE_LEGEND_WIDTH - 18),
                          max(10.0, height - 2 * margin))
            scale_bottom = min(bottom.width() / lw, bottom.height() / lh)
            scale_side = min(side.width() / lw, side.height() / lh)
            if scale_side > scale_bottom * 1.04 and height - 2 * margin >= self.SIDE_LEGEND_MIN_HEIGHT:
                mode, area = "side", side
            else:
                mode, area = "bottom", bottom
        scale = min(area.width() / lw, area.height() / lh)
        if not self._compact:
            scale = min(scale, 40.0)
        # A whole number of device pixels per tile, and an origin on the
        # pixel grid: every disc then lands on the same sub-pixel phase
        # and the rows read evenly.
        ratio = max(self.devicePixelRatioF(), 1.0)
        if scale * ratio >= 8.0:
            scale = math.floor(scale * ratio) / ratio
        ox = area.left() + (area.width() - lw * scale) / 2.0
        oy = area.top() + (area.height() - lh * scale) / 2.0
        if mode == "side":
            # Keep the map and its legend together, centred as one block.
            block = lw * scale + 18 + self.SIDE_LEGEND_WIDTH
            ox = margin + max(0.0, (width - 2 * margin - block) / 2.0)
        ox = round(ox * ratio) / ratio
        oy = round(oy * ratio) / ratio
        centres = {
            tank: (ox + tile.x * scale, oy + tile.y * scale, max(1.5, tile.radius * scale))
            for tank, tile in self._layout.tiles.items()
        }
        label_px = 11.0 if scale >= 17 else (10.0 if scale >= 12 else max(7.0, scale * 0.8))
        metrics = QFontMetricsF(self._font(label_px, QFont.DemiBold))
        farm_rects = {}
        for block_ in self._layout.farms:
            w = metrics.horizontalAdvance(block_.name) + 10
            h = metrics.height() + 2
            cx, cy = ox + block_.label_x * scale, oy + block_.label_y * scale
            farm_rects[block_.name] = QRectF(cx - w / 2, cy - h / 2, w, h)
        legend_rect = QRectF()
        if mode == "side":
            legend_rect = QRectF(ox + lw * scale + 18, oy + 2, self.SIDE_LEGEND_WIDTH,
                                 height - margin - oy - 2)
        elif mode == "bottom":
            legend_rect = QRectF(margin, height - margin - self.BOTTOM_LEGEND_HEIGHT,
                                 width - 2 * margin, self.BOTTOM_LEGEND_HEIGHT)
        self._geometry = {
            "key": key, "scale": scale, "ox": ox, "oy": oy, "centres": centres,
            "farm_rects": farm_rects, "label_px": label_px, "margin": margin,
            "legend_mode": mode, "legend_rect": legend_rect,
            "legend_top": legend_rect.top(), "map_bottom": oy + lh * scale,
        }
        return self._geometry

    def legend_mode(self) -> str:
        """Where the legend is drawn at the current size: side, bottom or none."""
        return self._ensure_geometry()["legend_mode"]

    def tile_center(self, tank: str) -> QPoint:
        """Widget coordinates of a tank's centre (for tests and links)."""
        cx, cy, _r = self._ensure_geometry()["centres"][str(tank)]
        return QPoint(int(round(cx)), int(round(cy)))

    def tank_at(self, pos: QPoint) -> Optional[str]:
        geometry = self._ensure_geometry()
        best, best_d = None, math.inf
        for tank, (cx, cy, r) in geometry["centres"].items():
            d = math.hypot(pos.x() - cx, pos.y() - cy)
            if d <= r + 2.5 and d < best_d:
                best, best_d = tank, d
        return best

    def farm_label_rect(self, farm: str) -> QRect:
        rect = self._ensure_geometry()["farm_rects"].get(farm)
        return rect.toRect() if rect is not None else QRect()

    def farm_at(self, pos: QPoint) -> Optional[str]:
        for farm, rect in self._ensure_geometry()["farm_rects"].items():
            if rect.adjusted(-2, -2, 2, 2).contains(QPointF(pos)):
                return farm
        return None

    # ---- events -------------------------------------------------------
    def resizeEvent(self, event) -> None:  # noqa: N802
        self._geometry = None
        super().resizeEvent(event)

    def _set_hover(self, tank: Optional[str], farm: Optional[str]) -> None:
        if tank != self._hover_tank:
            self._hover_tank = tank
            self.tankHovered.emit(tank or "")
            self.update()
        if farm != self._hover_farm:
            self._hover_farm = farm
            self.farmHovered.emit(farm or "")
            self.update()
        self.setCursor(Qt.PointingHandCursor if (tank or farm) else Qt.ArrowCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position().toPoint()
        if self._key_nav:
            self._key_nav = False       # the pointer takes over from the keyboard
            self.update()
        if self._press_pos is not None and (event.buttons() & Qt.LeftButton):
            if self._band is not None or (pos - self._press_pos).manhattanLength() > 6:
                self._band = QRect(self._press_pos, pos).normalized()
                self._set_hover(None, None)
                self.update()
                return
        tank = self.tank_at(pos)
        self._set_hover(tank, None if tank else self.farm_at(pos))

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._set_hover(None, None)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        self.setFocus(Qt.MouseFocusReason)
        self._press_pos = event.position().toPoint()
        self._press_tank = self.tank_at(self._press_pos)
        self._band = None

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            return super().mouseReleaseEvent(event)
        press_tank, band, press_pos = self._press_tank, self._band, self._press_pos
        self._press_pos, self._press_tank, self._band = None, None, None
        if self._skip_release:
            self._skip_release = False
            self.update()
            return
        release_pos = event.position().toPoint()
        if band is None and press_pos is not None and (release_pos - press_pos).manhattanLength() > 6:
            # A drag whose intermediate moves were not delivered is still a drag.
            band = QRect(press_pos, release_pos).normalized()
        extend = bool(event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier))
        toggle = bool(event.modifiers() & Qt.ControlModifier)
        if band is not None:
            inside = [
                tank for tank, (cx, cy, _r) in self._ensure_geometry()["centres"].items()
                if band.contains(QPoint(int(cx), int(cy)))
            ]
            selection = list(self._selection) if extend else []
            selection += [t for t in inside if t not in selection]
            self._apply_user_selection(selection)
            self.update()
            return
        tank = self.tank_at(event.position().toPoint())
        if tank is None or tank != press_tank:
            if tank is None and not extend:
                self._apply_user_selection([])
            return
        self._cursor_tank = tank
        if toggle and tank in self._selection:
            selection = [t for t in self._selection if t != tank]
        elif extend:
            selection = self._selection + ([tank] if tank not in self._selection else [])
        elif self._selection == [tank]:
            selection = []
        else:
            selection = [tank]
        self._apply_user_selection(selection)
        self.tankClicked.emit(tank)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            return super().mouseDoubleClickEvent(event)
        tank = self.tank_at(event.position().toPoint())
        if tank is None:
            return
        # The first click of the pair already selected the tank; the
        # release that follows this event must not toggle it off again.
        self._skip_release = True
        if tank not in self._selection:
            self._apply_user_selection([tank])
        self._cursor_tank = tank
        self.tankActivated.emit(tank)

    def _apply_user_selection(self, selection: List[str]) -> None:
        if selection != self._selection:
            self._selection = selection
            self.selectionChanged.emit(list(selection))
        self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        moves = {Qt.Key_Left: (-1, 0), Qt.Key_Right: (1, 0), Qt.Key_Up: (0, -1), Qt.Key_Down: (0, 1)}
        if key in moves and self._layout.tiles:
            current = self._cursor_tank or (self._selection[0] if self._selection else None)
            if current is None:
                target = next(iter(self._layout.tiles))
            else:
                target = self._layout.neighbour(current, *moves[key]) or current
            self._cursor_tank = target
            self._key_nav = True
            self.tankHovered.emit(target)
            self.update()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter) and self._cursor_tank:
            self.tankActivated.emit(self._cursor_tank)
            return
        if key == Qt.Key_Space and self._cursor_tank:
            tank = self._cursor_tank
            if tank in self._selection:
                selection = [t for t in self._selection if t != tank]
            elif event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier):
                selection = self._selection + [tank]
            else:
                selection = [tank]
            self._apply_user_selection(selection)
            return
        if key == Qt.Key_Escape and (self._selection or self._cursor_tank):
            self._cursor_tank = None
            self._apply_user_selection([])
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802
        self.update()
        super().focusOutEvent(event)

    def cursor_tank(self) -> Optional[str]:
        return self._cursor_tank

    def _keyboard_cursor(self) -> bool:
        """The keyboard cursor is shown (ring and callout) while the keys
        are driving and the map has the focus, not after a mouse click."""
        return self._key_nav and self.hasFocus()

    # ---- painting -----------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        palette = theme_palette()
        dark = is_dark(palette["card"])
        painter = QPainter(self)
        if not self._layout.tiles:
            painter.fillRect(self.rect(), QColor(palette["card"]))
            painter.end()
            return
        geometry = self._ensure_geometry()
        painter.drawPixmap(0, 0, self._base_pixmap(geometry, palette, dark))
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        self._paint_overlays(painter, geometry, palette, dark)
        if self._band is not None:
            accent = QColor(palette["accent"])
            painter.setPen(QPen(accent, 1))
            fill = QColor(accent)
            fill.setAlpha(36)
            painter.setBrush(fill)
            painter.drawRect(self._band)
        self._paint_callout(painter, geometry, palette)
        painter.end()

    def _base_pixmap(self, geometry: dict, palette: dict, dark: bool) -> QPixmap:
        """Everything that does not depend on the pointer, painted once
        and reused, so a hover repaint costs a blit and a few rings."""
        ratio = self.devicePixelRatioF()
        key = (geometry["key"], self._version, ratio, tuple(sorted(palette.items())))
        if self._base is not None and self._base_key == key:
            return self._base
        pixmap = QPixmap(int(math.ceil(self.width() * ratio)), int(math.ceil(self.height() * ratio)))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(QColor(palette["card"]))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        self._paint_frame(painter, geometry, palette)
        colours = class_colours(self.class_count(), self._palette, dark)
        for tank, (cx, cy, r) in geometry["centres"].items():
            self._draw_tile(painter, tank, cx, cy, r, palette, colours)
        if self._legend_visible:
            self._paint_legend(painter, geometry, palette, dark)
        elif self._caption_visible and self._compact:
            painter.setFont(self._font(9))
            painter.setPen(QColor(palette["faint"]))
            painter.drawText(QRectF(0, self.height() - 16, self.width() - 6, 14),
                             Qt.AlignRight | Qt.AlignVCenter, "Schematic, not to scale")
        painter.end()
        self._base, self._base_key = pixmap, key
        return pixmap

    def _paint_frame(self, painter: QPainter, geometry: dict, palette: dict) -> None:
        scale, ox, oy = geometry["scale"], geometry["ox"], geometry["oy"]
        title_px = 11.0 if scale >= 15 else max(7.0, scale * 0.72)
        painter.setFont(self._font(title_px, QFont.DemiBold, spacing=0.6 if scale >= 15 else 0.0))
        for area in self._layout.areas:
            x0, x1 = ox + area.x0 * scale, ox + area.x1 * scale
            rule_y = oy + area.rule_y * scale
            painter.setPen(QColor(palette["muted"]))
            painter.drawText(QRectF(x0, rule_y - title_px - 7, max(x1 - x0, 10), title_px + 5),
                             Qt.AlignLeft | Qt.AlignBottom | Qt.TextDontClip, area.title.upper())
            painter.setPen(QPen(QColor(palette["border"]), 1))
            painter.drawLine(QPointF(x0, round(rule_y) + 0.5), QPointF(x1, round(rule_y) + 0.5))
        painter.setFont(self._font(geometry["label_px"], QFont.DemiBold))
        painter.setPen(QColor(palette["ink"]))
        for block in self._layout.farms:
            painter.drawText(geometry["farm_rects"][block.name], Qt.AlignCenter | Qt.TextDontClip, block.name)

    def _fill_of(self, tank: str, colours: Sequence[str]) -> Optional[str]:
        if not self._is_in_scope(tank):
            return None
        if self._mode == "categories":
            category = self._categories.get(tank)
            return self._category_colours.get(category) if category is not None else None
        index = self._classes.get(tank)
        return colours[min(index, len(colours) - 1)] if index is not None else None

    def _draw_tile(self, painter: QPainter, tank: str, cx: float, cy: float, radius: float,
                   palette: dict, colours: Sequence[str]) -> None:
        tile = self._layout.tiles[tank]
        fill = self._fill_of(tank, colours)
        centre = QPointF(cx, cy)
        edge = palette["border_strong"] if self._is_in_scope(tank) else palette["border"]
        painter.setBrush(QColor(palette["card"]))
        painter.setPen(QPen(QColor(edge), 1.0))
        painter.drawEllipse(centre, radius, radius)
        inner = radius
        if tile.double_shell:
            # The second ring: a gap between the outline and the disc.
            inner = max(1.0, radius - max(2.0, min(3.2, radius * 0.24)))
            if fill is None:
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(centre, inner, inner)
        if fill is not None:
            disc = max(1.2, inner * self.size_fraction(tank) - (0.0 if tile.double_shell else 0.5))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(fill))
            painter.drawEllipse(centre, disc, disc)

    def _paint_overlays(self, painter: QPainter, geometry: dict, palette: dict, dark: bool) -> None:
        centres = geometry["centres"]
        accent, ink = QColor(palette["accent"]), QColor(palette["ink"])
        scale, ox, oy = geometry["scale"], geometry["ox"], geometry["oy"]
        if self._hover_farm is not None:
            block = self._layout.farm(self._hover_farm)
            if block is not None:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(palette["border_strong"]), 1.0))
                painter.drawRoundedRect(
                    QRectF(ox + block.x0 * scale - 3, oy + (block.y0 - fl.LABEL_HEIGHT) * scale + 2,
                           (block.x1 - block.x0) * scale + 6,
                           (block.y1 - block.y0 + fl.LABEL_HEIGHT) * scale + 1), 4, 4)
                painter.setFont(self._font(geometry["label_px"], QFont.DemiBold))
                rect = geometry["farm_rects"][block.name]
                painter.fillRect(rect, QColor(palette["card"]))
                painter.setPen(accent)
                painter.drawText(rect, Qt.AlignCenter | Qt.TextDontClip, block.name)
        active = self._hover_tank or (self._cursor_tank if self._keyboard_cursor() else None)
        selected = set(self._selection)
        painter.setBrush(Qt.NoBrush)
        for tank in self._highlight - selected - {active}:
            cx, cy, r = centres[tank]
            painter.setPen(QPen(ink, 1.6))
            painter.drawEllipse(QPointF(cx, cy), r + 1.4, r + 1.4)
        ring = 3.5 if self._compact else 1.4
        for tank in self._selection:
            if tank == active or tank not in centres:
                continue
            cx, cy, r = centres[tank]
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(accent, 2.0))
            painter.drawEllipse(QPointF(cx, cy), r + ring, r + ring)
        if active is not None and active in centres:
            # The disc under the pointer lifts: redrawn larger, accent outline.
            cx, cy, r = centres[active]
            colours = class_colours(self.class_count(), self._palette, dark)
            self._draw_tile(painter, active, cx, cy, r + 2.0, palette, colours)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(accent, 2.0))
            painter.drawEllipse(QPointF(cx, cy), r + 3.0, r + 3.0)

    def _legend_disc(self, painter: QPainter, x: float, y: float, r: float, palette: dict,
                     fill: Optional[str], double: bool = False, fraction: float = 1.0,
                     faint: bool = False) -> None:
        centre = QPointF(x, y)
        painter.setBrush(QColor(palette["card"]))
        painter.setPen(QPen(QColor(palette["border"] if faint else palette["border_strong"]), 1.0))
        painter.drawEllipse(centre, r, r)
        inner = r
        if double:
            inner = r - 2.4
            if fill is None:
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(centre, inner, inner)
        if fill is not None:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(fill))
            disc = inner * fraction - (0.0 if double else 0.5)
            painter.drawEllipse(centre, disc, disc)

    def _class_range_labels(self) -> List[str]:
        """One label per class, low to high: "under a", "a to b", "b and above"."""
        breaks = [format_compact(b) for b in self._breaks]
        if not breaks:
            shown = [v for t, v in self._values.items() if t in self._classes]
            if not shown:
                return []
            low, high = format_compact(min(shown)), format_compact(max(shown))
            return [low if low == high else f"{low} to {high}"]
        labels = [f"under {breaks[0]}"]
        labels += [f"{a} to {b}" for a, b in zip(breaks, breaks[1:])]
        labels.append(f"{breaks[-1]} and above")
        return labels

    def _paint_legend_side(self, painter: QPainter, geometry: dict, palette: dict, dark: bool) -> None:
        rect = geometry["legend_rect"]
        ink, muted, faint = QColor(palette["ink"]), QColor(palette["muted"]), QColor(palette["faint"])
        text_font, bold_font = self._font(11), self._font(12, QFont.DemiBold)
        metrics = QFontMetricsF(text_font)
        x, y, width = rect.left(), rect.top(), rect.width()
        row = 20.0

        def line(text: str, font: QFont, colour: QColor, height: float = 17.0) -> None:
            nonlocal y
            painter.setFont(font)
            painter.setPen(colour)
            shown = QFontMetricsF(font).elidedText(text, Qt.ElideRight, width)
            painter.drawText(QRectF(x, y, width, height), Qt.AlignLeft | Qt.AlignVCenter, shown)
            y += height

        if self._mode == "categories":
            line(self._label or "Category", bold_font, ink, 19.0)
            y += 4
            for category, colour in self._category_colours.items():
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(colour))
                painter.drawRoundedRect(QRectF(x, y + 4, 12, 12), 2, 2)
                painter.setFont(text_font)
                painter.setPen(muted)
                painter.drawText(QRectF(x + 20, y, width - 20, row), Qt.AlignLeft | Qt.AlignVCenter,
                                 metrics.elidedText(str(category), Qt.ElideRight, width - 20))
                y += row
        else:
            line(self._label or "No quantity mapped", bold_font, ink, 19.0)
            if self._unit:
                line(self._unit, text_font, muted)
            if self._classes:
                line(f"{self._scale} classes", text_font, faint)
                y += 6
                colours = class_colours(self.class_count(), self._palette, dark)
                labels = self._class_range_labels()
                for colour, label in reversed(list(zip(colours, labels))):   # highest first
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(colour))
                    painter.drawRect(QRectF(x, y + 3, 14, 14))
                    painter.setFont(text_font)
                    painter.setPen(muted)
                    painter.drawText(QRectF(x + 22, y, width - 22, row), Qt.AlignLeft | Qt.AlignVCenter, label)
                    y += row
        y += 10
        painter.setPen(QPen(QColor(palette["border"]), 1))
        painter.drawLine(QPointF(x, round(y) + 0.5), QPointF(x + width - 8, round(y) + 0.5))
        y += 10
        sample = class_colours(self.class_count(), self._palette, dark)
        sample_fill = sample[min(len(sample) - 1, 3)] if self._mode == "values" else SERIES_BLUE

        def key(text: str, **disc) -> None:
            nonlocal y
            if y + row > rect.bottom():
                return
            self._legend_disc(painter, x + 7, y + row / 2, 6.5, palette, **disc)
            painter.setFont(text_font)
            painter.setPen(muted)
            painter.drawText(QRectF(x + 22, y, width - 22, row), Qt.AlignLeft | Qt.AlignVCenter, text)
            y += row

        key("No record", fill=None)
        if self._in_scope is not None:
            key("Outside scope", fill=None, faint=True)
        if any(t.double_shell for t in self._layout.tiles.values()):
            key("Double-shell tank", fill=sample_fill, double=True)
            key("Single-shell tank", fill=sample_fill)
        if self._sizes is None and any(self.size_fraction(t) < 1.0 for t in self._layout.tiles):
            key(f"Under {MIN_VOLUME_KL:.0f} kL of waste", fill=sample_fill, fraction=SMALL_FRACTION)
        if self._caption_visible and rect.bottom() - y >= 34:
            painter.setFont(self._font(10))
            painter.setPen(faint)
            painter.drawText(QRectF(x, rect.bottom() - 30, width, 30),
                             Qt.AlignLeft | Qt.AlignBottom | Qt.TextWordWrap, CAPTION)

    def _paint_legend(self, painter: QPainter, geometry: dict, palette: dict, dark: bool) -> None:
        if geometry["legend_mode"] == "side":
            self._paint_legend_side(painter, geometry, palette, dark)
            return
        margin = geometry["margin"]
        top = geometry["legend_top"]
        left, right = margin + 2.0, self.width() - margin - 2.0
        ink, muted, faint = QColor(palette["ink"]), QColor(palette["muted"]), QColor(palette["faint"])
        text_font = self._font(11)
        bold_font = self._font(11, QFont.DemiBold)
        small_font = self._font(10)
        metrics = QFontMetricsF(text_font)
        row_a, row_b, row_c = top + 2.0, top + 22.0, top + 40.0

        painter.setPen(QPen(QColor(palette["border"]), 1))
        painter.drawLine(QPointF(margin, top - 7.5), QPointF(self.width() - margin, top - 7.5))

        x = left
        ramp_right = left
        if self._mode == "categories":
            painter.setFont(bold_font)
            painter.setPen(ink)
            painter.drawText(QRectF(x, row_a, 400, 16), Qt.AlignLeft | Qt.AlignVCenter,
                             self._label or "Category")
            painter.setFont(text_font)
            cx = x
            for category, colour in self._category_colours.items():
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(colour))
                painter.drawRoundedRect(QRectF(cx, row_b + 1, 12, 12), 2, 2)
                painter.setPen(muted)
                width = metrics.horizontalAdvance(str(category))
                painter.drawText(QRectF(cx + 17, row_b - 1, width + 4, 16),
                                 Qt.AlignLeft | Qt.AlignVCenter, str(category))
                cx += 17 + width + 18
            ramp_right = cx
        else:
            title = self._label or "No quantity mapped"
            painter.setFont(bold_font)
            painter.setPen(ink)
            title_w = QFontMetricsF(bold_font).horizontalAdvance(title)
            painter.drawText(QRectF(x, row_a, title_w + 4, 16), Qt.AlignLeft | Qt.AlignVCenter, title)
            painter.setFont(text_font)
            painter.setPen(muted)
            detail = ", ".join(part for part in (self._unit, f"{self._scale} classes" if self._values else "") if part)
            painter.drawText(QRectF(x + title_w + 8, row_a, 320, 16), Qt.AlignLeft | Qt.AlignVCenter, detail)
            ramp_right = max(ramp_right, x + title_w + 8 + metrics.horizontalAdvance(detail))
            if self._classes:
                colours = class_colours(self.class_count(), self._palette, dark)
                swatch_w = 54.0 if len(colours) >= 4 else 72.0
                for i, colour in enumerate(colours):
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(colour))
                    painter.drawRect(QRectF(x + i * swatch_w, row_b, swatch_w - 1.5, 11))
                painter.setFont(small_font)
                small_metrics = QFontMetricsF(small_font)
                shown = [v for t, v in self._values.items() if t in self._classes]
                labels = [(x + (i + 1) * swatch_w - 0.75, format_compact(b), Qt.AlignHCenter)
                          for i, b in enumerate(self._breaks)]
                if shown:
                    labels.append((x, format_compact(min(shown)), Qt.AlignLeft))
                    labels.append((x + len(colours) * swatch_w - 1.5, format_compact(max(shown)), Qt.AlignRight))
                occupied: List[Tuple[float, float]] = []
                for anchor, text, align in labels:
                    width = small_metrics.horizontalAdvance(text)
                    x0 = anchor - width / 2 if align == Qt.AlignHCenter else (anchor if align == Qt.AlignLeft else anchor - width)
                    if any(x0 < b + 5 and x0 + width > a - 5 for a, b in occupied):
                        continue
                    occupied.append((x0, x0 + width))
                    painter.setPen(muted if align == Qt.AlignHCenter else faint)
                    painter.drawText(QRectF(x0, row_c - 3, width + 2, 14),
                                     Qt.AlignLeft | Qt.AlignVCenter, text)
                ramp_right = max(ramp_right, x + len(colours) * swatch_w)

        # keys: no record, scope, size, shell
        painter.setFont(text_font)
        keys_x = ramp_right + 30.0
        sample = class_colours(self.class_count(), self._palette, dark)
        sample_fill = sample[min(len(sample) - 1, 3)] if self._mode == "values" else SERIES_BLUE

        def key(xpos: float, ypos: float, text: str, **disc) -> float:
            width = metrics.horizontalAdvance(text)
            if xpos + 20 + width > right:
                return xpos
            self._legend_disc(painter, xpos + 7, ypos + 8, 6.5, palette, **disc)
            painter.setPen(muted)
            painter.drawText(QRectF(xpos + 19, ypos, width + 4, 16), Qt.AlignLeft | Qt.AlignVCenter, text)
            return xpos + 19 + width + 18

        kx = keys_x
        kx = key(kx, row_a, "No record", fill=None)
        if self._in_scope is not None:
            kx = key(kx, row_a, "Outside scope", fill=None, faint=True)
        has_small = any(self.size_fraction(t) < 1.0 for t in self._layout.tiles)
        kx = keys_x
        if any(t.double_shell for t in self._layout.tiles.values()):
            kx = key(kx, row_b - 2, "Double-shell", fill=sample_fill, double=True)
            kx = key(kx, row_b - 2, "Single-shell", fill=sample_fill)
        if has_small and self._sizes is None:
            kx = key(kx, row_b - 2, f"Under {MIN_VOLUME_KL:.0f} kL of waste", fill=sample_fill,
                     fraction=SMALL_FRACTION)
        if self._caption_visible:
            painter.setFont(small_font)
            painter.setPen(faint)
            painter.drawText(QRectF(left, row_c - 3, right - left, 14),
                             Qt.AlignRight | Qt.AlignVCenter, CAPTION)

    def _paint_callout(self, painter: QPainter, geometry: dict, palette: dict) -> None:
        tank = self._hover_tank or (self._cursor_tank if self._keyboard_cursor() else None)
        if tank is not None and tank in geometry["centres"]:
            lines = self.tooltip_lines(tank)
            cx, cy, r = geometry["centres"][tank]
            anchor = QRectF(cx - r, cy - r, 2 * r, 2 * r)
        elif self._hover_farm is not None and self._hover_farm in geometry["farm_rects"]:
            lines = self.farm_tooltip_lines(self._hover_farm)
            anchor = geometry["farm_rects"][self._hover_farm]
        else:
            return
        if not lines:
            return
        fonts = {"title": self._font(12, QFont.DemiBold), "body": self._font(12), "muted": self._font(11)}
        pad_x, pad_y, gap = 10.0, 8.0, 3.0
        heights = [QFontMetricsF(fonts[kind]).height() for _text, kind in lines]
        width = max(QFontMetricsF(fonts[kind]).horizontalAdvance(text) for text, kind in lines) + 2 * pad_x + 2
        height = sum(heights) + gap * (len(lines) - 1) + 2 * pad_y
        x = anchor.right() + 10
        if x + width > self.width() - 4:
            x = anchor.left() - 10 - width
        x = max(4.0, min(x, self.width() - width - 4))
        y = anchor.center().y() - height / 2
        y = max(4.0, min(y, self.height() - height - 4))
        box = QRectF(x, y, width, height)
        background = palette.get("tooltip_bg", palette["chrome"])
        text_colour = palette.get("tooltip_ink", palette["chrome_ink"])
        quiet = _blend(text_colour, background, 0.32)
        painter.setPen(QPen(QColor(palette.get("tooltip_border", background)), 1))
        painter.setBrush(QColor(background))
        painter.drawRoundedRect(box, 4, 4)
        ty = y + pad_y
        for (text, kind), line_height in zip(lines, heights):
            painter.setFont(fonts[kind])
            painter.setPen(QColor(quiet if kind == "muted" else text_colour))
            painter.drawText(QRectF(x + pad_x, ty, width - 2 * pad_x, line_height),
                             Qt.AlignLeft | Qt.AlignVCenter, text)
            ty += line_height + gap


# ---------------------------------------------------------------------------
# the ranked list
# ---------------------------------------------------------------------------
class _RankModel(QAbstractTableModel):
    HEADERS = ["#", "Tank", "Value"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._tanks: List[str] = []
        self._values: List[float] = []
        self._ranks: List[int] = []
        self._farms: List[str] = []
        self.unit = ""
        self.label = ""
        self.maximum = 0.0

    def set_rows(self, table: Optional[pd.DataFrame], label: str, unit: str) -> None:
        self.beginResetModel()
        self.label, self.unit = label, unit
        if table is None or table.empty:
            self._tanks, self._values, self._ranks, self._farms = [], [], [], []
        else:
            ordered = table.sort_values(["Value", "WasteSiteId"], ascending=[False, True], kind="mergesort")
            self._tanks = [str(t) for t in ordered["WasteSiteId"]]
            self._values = [float(v) for v in ordered["Value"]]
            self._farms = ([str(f) if f is not None else "" for f in ordered["TankFarm"]]
                           if "TankFarm" in ordered.columns else [""] * len(self._tanks))
            self._ranks, previous, rank = [], None, 0
            for position, value in enumerate(self._values, start=1):
                if value != previous:
                    rank, previous = position, value
                self._ranks.append(rank)
        self.maximum = max(self._values) if self._values else 0.0
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._tanks)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else 3

    def tank(self, row: int) -> str:
        return self._tanks[row]

    def value(self, row: int) -> float:
        return self._values[row]

    def row_of(self, tank: str) -> int:
        try:
            return self._tanks.index(tank)
        except ValueError:
            return -1

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        row, column = index.row(), index.column()
        if role == Qt.DisplayRole:
            if column == 0:
                return str(self._ranks[row])
            if column == 1:
                return self._tanks[row]
            return format_number(self._values[row])
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignVCenter | (Qt.AlignLeft if column == 1 else Qt.AlignRight))
        if role == Qt.ToolTipRole:
            unit = f" {self.unit}" if self.unit else ""
            farm = f" ({self._farms[row]} farm)" if self._farms[row] else ""
            return (f"{self._tanks[row]}{farm}\n{self.label}  {format_number(self._values[row])}{unit}\n"
                    f"{ordinal(self._ranks[row])} of {len(self._tanks):,} tanks")
        if role == Qt.UserRole:
            return self._values[row]
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # noqa: N802
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            # The quantity is named above the list: a theme that upper-cases
            # table headers would turn "mol per mol Na" into "NA".
            return self.HEADERS[section]
        if role == Qt.TextAlignmentRole and orientation == Qt.Horizontal:
            return int(Qt.AlignVCenter | (Qt.AlignLeft if section == 1 else Qt.AlignRight))
        return None


class _BarDelegate(QStyledItemDelegate):
    """Paints the value as a bar from the left with the number at the right."""

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        view = self.parent()
        palette = theme_palette()
        model = index.model()
        row = index.row()
        selected = bool(option.state & QStyle.State_Selected)
        hovered = getattr(view, "_hover_row", -1) == row or getattr(view, "_linked_row", -1) == row
        painter.save()
        if selected or hovered:
            painter.fillRect(option.rect, QColor(palette["selection_bg"]))
        if selected and index.column() == 0:
            painter.fillRect(QRect(option.rect.left(), option.rect.top(), 3, option.rect.height()),
                             QColor(palette["accent"]))
        rect = option.rect.adjusted(8, 0, -8, 0)
        font = QFont(option.font)
        if index.column() == 2:
            value = float(index.data(Qt.UserRole) or 0.0)
            maximum = float(getattr(model, "maximum", 0.0) or 0.0)
            text = str(index.data(Qt.DisplayRole))
            metrics = QFontMetricsF(font)
            text_w = max(metrics.horizontalAdvance(text) + 8, 62.0)
            bar_area = rect.width() - text_w
            if maximum > 0 and value > 0 and bar_area > 6:
                width = max(2.0, bar_area * min(1.0, value / maximum))
                painter.setRenderHint(QPainter.Antialiasing, False)
                painter.fillRect(QRectF(rect.left(), rect.center().y() - 4.5, width, 10.0),
                                 QColor(SERIES_BLUE))
            painter.setPen(QColor(palette["ink"]))
            painter.setFont(font)
            painter.drawText(rect, int(Qt.AlignRight | Qt.AlignVCenter), text)
        else:
            if index.column() == 1 and selected:
                font.setWeight(QFont.DemiBold)
            painter.setFont(font)
            painter.setPen(QColor(palette["muted"] if index.column() == 0 else palette["ink"]))
            painter.drawText(rect, int(index.data(Qt.TextAlignmentRole) or Qt.AlignVCenter),
                             str(index.data(Qt.DisplayRole)))
        painter.restore()

    def sizeHint(self, option, index: QModelIndex) -> QSize:  # noqa: N802
        return QSize(60, 24)


class RankedTankList(QTableView):
    """Tanks ranked by the mapped value, with a bar in each row.

    Signals mirror FarmMapWidget so the two can be linked: hovering a row
    emits tankHovered, selecting rows emits selectionChanged (user changes
    only), a double-click emits tankActivated."""

    tankHovered = Signal(str)
    tankActivated = Signal(str)
    selectionChanged_ = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("RankedTankList")
        self._model = _RankModel(self)
        self.setModel(self._model)
        self.setItemDelegate(_BarDelegate(self))
        self._hover_row = -1
        self._linked_row = -1
        self._syncing = False
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setShowGrid(False)
        self.setAlternatingRowColors(False)
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(24)
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setHighlightSections(False)
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setColumnWidth(0, 44)
        self.setColumnWidth(1, 96)
        self.setAccessibleName("Tanks ranked by the mapped quantity")
        self.entered.connect(self._on_entered)
        self.doubleClicked.connect(lambda index: self.tankActivated.emit(self._model.tank(index.row())))
        self.selectionModel().selectionChanged.connect(self._on_selection_changed)

    def set_table(self, table: Optional[pd.DataFrame], label: str = "", unit: str = "") -> None:
        """table: WasteSiteId, Value (and TankFarm), one row per tank with a record."""
        self._syncing = True
        try:
            self._hover_row = self._linked_row = -1
            self._model.set_rows(table, label, unit)
        finally:
            self._syncing = False

    def tanks(self) -> List[str]:
        return [self._model.tank(r) for r in range(self._model.rowCount())]

    def selected_tanks(self) -> List[str]:
        rows = sorted({i.row() for i in self.selectionModel().selectedRows()})
        return [self._model.tank(r) for r in rows]

    def set_selection(self, tanks: Iterable[str], scroll: bool = True) -> None:
        """Select rows from code (selectionChanged_ is not emitted)."""
        from PySide6.QtCore import QItemSelection, QItemSelectionModel
        rows = [r for r in (self._model.row_of(str(t)) for t in tanks) if r >= 0]
        selection = QItemSelection()
        for row in rows:
            selection.select(self._model.index(row, 0), self._model.index(row, 2))
        self._syncing = True
        try:
            self.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect)
        finally:
            self._syncing = False
        if rows and scroll:
            self.scrollTo(self._model.index(min(rows), 0), QAbstractItemView.EnsureVisible)
        self.viewport().update()

    def highlight(self, tank: str) -> None:
        """Mark the row of a tank hovered elsewhere (no scrolling)."""
        row = self._model.row_of(tank) if tank else -1
        if row != self._linked_row:
            self._linked_row = row
            self.viewport().update()

    def _on_entered(self, index: QModelIndex) -> None:
        if index.row() != self._hover_row:
            self._hover_row = index.row()
            self.viewport().update()
            self.tankHovered.emit(self._model.tank(index.row()))

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self._hover_row != -1:
            self._hover_row = -1
            self.viewport().update()
            self.tankHovered.emit("")
        super().leaveEvent(event)

    def _on_selection_changed(self, *_args) -> None:
        self.viewport().update()
        if not self._syncing:
            self.selectionChanged_.emit(self.selected_tanks())
