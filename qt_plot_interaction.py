"""
qt_plot_interaction.py -- what a chart says when the pointer is on it.

Two layers:

- `MarkIndex` (no Qt): given a matplotlib figure and a pixel position,
  names the mark under the pointer -- a bar, a scatter point, a line
  vertex, a heat-map cell -- with its label, series, value and unit, and
  the tank it stands for when it stands for one. It works on any axes:
  categories come from the tick labels, values from the artists. A plot
  function can sharpen the answer by leaving a small dictionary on the
  axes (`ax._ember_marks`, written through plot_helpers.mark_categories /
  mark_points / mark_cells): the full label of every category when the
  axis shows only some of them, the tank behind each point of a scatter.
- `PlotInteraction` (Qt): connects that index to a canvas. Hovering shows
  a tooltip, outlines the mark and writes a formatted readout; a click
  emits `mark_clicked`, a double click `mark_activated`, a right click
  opens a menu (copy value, copy label, open tank).

Nothing is computed while the pointer is still: the index is built on the
first move after a draw and discarded at the next draw, so it can never
describe artists that are no longer on screen. While a toolbar mode (zoom
or pan) is engaged the layer stays out of the way.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from qt_table_format import format_number, humanize_text, tank_id_of, unit_from_label, with_unit

MARKS_ATTR = "_ember_marks"
POINT_RADIUS_PX = 7.0
MIN_BAR_PX = 4.0
PRIVATE_KEYS = ("_rect", "_xy", "_radius", "_axes")


# ---------------------------------------------------------------------------
# Reading an axes
# ---------------------------------------------------------------------------

def _marks(ax) -> Dict[str, Any]:
    marks = getattr(ax, MARKS_ATTR, None)
    return marks if isinstance(marks, dict) else {}


def _clean_axis_label(text: str) -> str:
    """An axis label without its scale note ('Total mass (g) — log scale')."""
    return humanize_text(str(text or "").split(" — ")[0].replace("\n", " ").strip())


def _is_colorbar_axes(ax) -> bool:
    if ax.get_label() == "<colorbar>" or hasattr(ax, "_colorbar"):
        return True
    figure = ax.get_figure(root=True) if _supports_root(ax) else ax.figure
    for other in getattr(figure, "axes", []):
        for mappable in list(other.images) + list(other.collections):
            bar = getattr(mappable, "colorbar", None)
            if bar is not None and getattr(bar, "ax", None) is ax:
                return True
    return False


def _supports_root(ax) -> bool:
    try:
        ax.get_figure(root=True)
        return True
    except TypeError:
        return False


def _tick_lookup(axis) -> Dict[int, str]:
    """Position -> label for a categorical axis, from the axis itself."""
    mapping = getattr(getattr(axis, "units", None), "_mapping", None)
    if mapping:
        return {int(pos): str(label) for label, pos in mapping.items()}
    out: Dict[int, str] = {}
    if not _has_fixed_labels(axis):
        return out                                # a numeric axis: its tick text is a number, not a name
    try:
        locs = axis.get_majorticklocs()
        labels = _full_tick_labels(axis)
    except Exception:
        return out
    for loc, label in zip(locs, labels):
        if label and abs(loc - round(loc)) < 1e-6:
            out[int(round(loc))] = label
    return out


def _full_tick_labels(axis) -> List[str]:
    """The major tick labels in full. On a narrow pane the layout cuts long
    names with an ellipsis (plot_helpers.contain_region) by wrapping the
    axis' formatter; the wrapped one still gives the whole name."""
    inner = getattr(axis.get_major_formatter(), "ember_inner", None)
    if inner is not None:
        try:
            return [str(v) for v in inner.format_ticks(axis.get_majorticklocs())]
        except Exception:
            pass
    return [t.get_text() for t in axis.get_majorticklabels()]


def _has_fixed_labels(axis) -> bool:
    """True when the tick labels were SET (set_ticklabels: names), not
    computed from the tick values by a numeric formatter."""
    try:
        from matplotlib.ticker import FixedFormatter
        formatter = axis.get_major_formatter()
        return isinstance(getattr(formatter, "ember_inner", None) or formatter, FixedFormatter)
    except Exception:
        return False


def _half_tick_lookup(axis) -> Dict[int, str]:
    """Same for a seaborn heat map, whose cells are centred on i + 0.5."""
    out: Dict[int, str] = {}
    if not _has_fixed_labels(axis):
        return out
    try:
        locs = axis.get_majorticklocs()
        labels = _full_tick_labels(axis)
    except Exception:
        return out
    for loc, label in zip(locs, labels):
        if label:
            out[int(np.floor(loc))] = label
    return out


def _user_label(artist) -> Optional[str]:
    label = artist.get_label() if hasattr(artist, "get_label") else None
    if label is None:
        return None
    label = str(label)
    return None if (not label or label.startswith("_")) else label


def _finite(a: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(np.nan_to_num(a, nan=low, posinf=high, neginf=low), low - 1.0, high + 1.0)


class _AxesIndex:
    """Everything hit-testable on one axes, in display pixels."""

    def __init__(self, ax) -> None:
        self.ax = ax
        self.marks = _marks(ax)
        self.bars: List[Tuple[Any, Any]] = []            # (container, patch)
        self.bar_boxes = np.zeros((0, 4))
        self.point_xy = np.zeros((0, 2))
        self.point_ref: List[Tuple[Any, int]] = []       # (collection, index)
        self.point_radius = np.zeros(0)
        self.line_xy = np.zeros((0, 2))
        self.line_ref: List[Tuple[Any, int]] = []
        self.cells: List[Any] = []                       # images and meshes
        self._build()

    # -- construction -----------------------------------------------------
    def _build(self) -> None:
        ax = self.ax
        box = ax.bbox
        boxes = []
        for container in getattr(ax, "containers", []):
            patches = getattr(container, "patches", None)
            if patches is None:
                continue
            for patch in patches:
                try:
                    x0, y0 = patch.get_x(), patch.get_y()
                    x1, y1 = x0 + patch.get_width(), y0 + patch.get_height()
                    # The DATA transform: a Rectangle's own get_transform()
                    # starts from the unit square, not from data coordinates.
                    corners = patch.get_data_transform().transform(np.array([[x0, y0], [x1, y1]], dtype=float))
                except Exception:
                    continue
                xs = _finite(corners[:, 0], box.x0, box.x1)
                ys = _finite(corners[:, 1], box.y0, box.y1)
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
                self.bars.append((container, patch))
        if boxes:
            arr = np.asarray(boxes, dtype=float)
            # A sliver is still a bar: widen it to something a pointer can hold.
            for lo, hi in ((0, 2), (1, 3)):
                thin = (arr[:, hi] - arr[:, lo]) < MIN_BAR_PX
                mid = (arr[:, hi] + arr[:, lo]) / 2.0
                arr[thin, lo] = mid[thin] - MIN_BAR_PX / 2.0
                arr[thin, hi] = mid[thin] + MIN_BAR_PX / 2.0
            self.bar_boxes = arr

        from matplotlib.collections import PathCollection, QuadMesh

        xy_parts, radii = [], []
        dpi_scale = ax.figure.dpi / 72.0
        for coll in ax.collections:
            if not coll.get_visible():
                continue
            if isinstance(coll, QuadMesh):
                self.cells.append(coll)
                continue
            if not isinstance(coll, PathCollection):
                continue
            offsets = np.ma.filled(np.ma.asarray(coll.get_offsets(), dtype=float), np.nan)
            if offsets.ndim != 2 or not len(offsets):
                continue
            try:
                xy = coll.get_offset_transform().transform(offsets)
            except Exception:
                continue
            sizes = np.asarray(coll.get_sizes(), dtype=float)
            if sizes.size == 0:
                sizes = np.array([36.0])
            radius = np.sqrt(np.resize(sizes, len(xy))) / 2.0 * dpi_scale
            xy_parts.append(xy)
            radii.append(np.maximum(radius + 2.0, POINT_RADIUS_PX))
            self.point_ref.extend((coll, i) for i in range(len(xy)))
        if xy_parts:
            self.point_xy = np.vstack(xy_parts)
            self.point_radius = np.concatenate(radii)

        line_parts = []
        for line in ax.lines:
            if not line.get_visible() or line.get_transform() is not ax.transData:
                continue
            has_marker = str(line.get_marker()) not in ("", "None", "none", " ")
            if _user_label(line) is None and not has_marker:
                continue
            data = np.column_stack([np.asarray(line.get_xdata(orig=False), dtype=float),
                                    np.asarray(line.get_ydata(orig=False), dtype=float)])
            if not len(data):
                continue
            try:
                xy = ax.transData.transform(data)
            except Exception:
                continue
            line_parts.append(xy)
            self.line_ref.extend((line, i) for i in range(len(xy)))
        if line_parts:
            self.line_xy = np.vstack(line_parts)

        self.cells.extend(im for im in ax.images if im.get_visible())

    # -- lookup -------------------------------------------------------------
    def lookup(self, x: float, y: float) -> Optional[Dict[str, Any]]:
        for finder in (self._point_at, self._bar_at, self._line_at, self._cell_at):
            mark = finder(x, y)
            if mark is not None:
                mark["_axes"] = self.ax
                return mark
        return None

    def _nearest(self, xy: np.ndarray, radius, x: float, y: float) -> Optional[int]:
        if not len(xy):
            return None
        d2 = (xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2
        d2 = np.where(np.isfinite(d2), d2, np.inf)
        i = int(np.argmin(d2))
        limit = radius[i] if isinstance(radius, np.ndarray) else radius
        return i if d2[i] <= float(limit) ** 2 else None

    def _point_at(self, x: float, y: float) -> Optional[Dict[str, Any]]:
        i = self._nearest(self.point_xy, self.point_radius, x, y)
        if i is None:
            return None
        coll, k = self.point_ref[i]
        ax = self.ax
        px, py = (float(v) for v in np.ma.filled(np.ma.asarray(coll.get_offsets(), dtype=float), np.nan)[k])
        meta = self._point_meta(coll)
        label = _pick(meta.get("labels"), k)
        tank = _pick(meta.get("tank_ids"), k)
        if tank is None and label is not None:
            tank = tank_id_of(label)
        series = _user_label(coll)
        x_label = meta.get("x_label") or _clean_axis_label(ax.get_xlabel()) or "x"
        y_label = meta.get("y_label") or _clean_axis_label(ax.get_ylabel()) or "y"
        rows = []
        if series and label is not None:
            rows.append((meta.get("series_name") or "Series", series))
        if meta.get("show_position", True):
            rows.append((x_label, with_unit(px, unit_from_label(x_label))))
            rows.append((y_label, with_unit(py, unit_from_label(y_label))))
        array = coll.get_array()
        if array is not None and np.size(array) > k and getattr(coll, "colorbar", None) is not None:
            c_label = _clean_axis_label(coll.colorbar.ax.get_ylabel()) or "Colour value"
            rows.append((c_label, format_number(float(np.ravel(array)[k]))))
        for name, values in meta.get("extra", ()) or ():
            value = _pick(values, k)
            if value is not None:
                rows.append((str(name), value if isinstance(value, str) else format_number(value)))
        title = str(label) if label is not None else (series or meta.get("name") or "Point")
        return _mark(
            "point", title=title, rows=rows, label=None if label is None else str(label), series=series,
            value=py, x=px, y=py, tank_id=tank, unit=unit_from_label(y_label),
            _xy=tuple(self.point_xy[i]), _radius=float(self.point_radius[i]),
        )

    def _point_meta(self, coll) -> Dict[str, Any]:
        for entry in self.marks.get("points", ()) or ():
            if entry.get("artist") is coll:
                return entry
        return {}

    def _bar_at(self, x: float, y: float) -> Optional[Dict[str, Any]]:
        b = self.bar_boxes
        if not len(b):
            return None
        hit = np.nonzero((b[:, 0] <= x) & (x <= b[:, 2]) & (b[:, 1] <= y) & (y <= b[:, 3]))[0]
        if not len(hit):
            return None
        i = int(hit[-1])                       # the last drawn is on top
        container, patch = self.bars[i]
        ax = self.ax
        horizontal = getattr(container, "orientation", None) == "horizontal"
        if getattr(container, "orientation", None) is None:
            horizontal = patch.get_height() < patch.get_width() and bool(_tick_lookup(ax.yaxis))
        if horizontal:
            value = float(patch.get_width())
            centre = patch.get_y() + patch.get_height() / 2.0
            span = (patch.get_y(), patch.get_y() + patch.get_height())
            cat_axis, value_axis_label = ax.yaxis, ax.get_xlabel()
        else:
            value = float(patch.get_height())
            centre = patch.get_x() + patch.get_width() / 2.0
            span = (patch.get_x(), patch.get_x() + patch.get_width())
            cat_axis, value_axis_label = ax.xaxis, ax.get_ylabel()
        marks = self.marks
        position = int(round(centre))
        label = _pick(marks.get("categories"), position) if abs(centre - position) <= 0.5 else None
        if label is None:
            label = _tick_lookup(cat_axis).get(position) if abs(centre - position) <= 0.5 else None
        numeric_bin = label is None
        if numeric_bin:
            label = f"{format_number(span[0])} to {format_number(span[1])}"
        tank = _pick(marks.get("category_tank_ids"), position) if not numeric_bin else None
        if tank is None and not numeric_bin:
            tank = tank_id_of(label)
        series = _user_label(container)
        if tank is None and series is not None:
            tank = tank_id_of(series)          # one series per tank (a tank profile)
        value_label = marks.get("value_label") or _clean_axis_label(value_axis_label) or "Value"
        unit = marks.get("unit") or unit_from_label(value_label)
        rows = []
        if series:
            rows.append((marks.get("series_name") or ("Tank" if tank_id_of(series) else "Series"), series))
        rows.append((value_label, with_unit(value, unit)))
        for name, values in marks.get("category_extra", ()) or ():
            extra = _pick(values, position)
            if extra is not None:
                rows.append((str(name), extra if isinstance(extra, str) else format_number(extra)))
        return _mark(
            "bar", title=str(label), rows=rows, label=str(label), series=series, value=value,
            tank_id=tank, unit=unit, category_name=marks.get("category_name") or "",
            _rect=tuple(self.bar_boxes[i]),
        )

    def _line_at(self, x: float, y: float) -> Optional[Dict[str, Any]]:
        i = self._nearest(self.line_xy, POINT_RADIUS_PX, x, y)
        if i is None:
            return None
        line, k = self.line_ref[i]
        ax = self.ax
        px = float(np.asarray(line.get_xdata(orig=False), dtype=float)[k])
        py = float(np.asarray(line.get_ydata(orig=False), dtype=float)[k])
        series = _user_label(line)
        x_label = _clean_axis_label(ax.get_xlabel()) or "x"
        y_label = _clean_axis_label(ax.get_ylabel()) or "y"
        rows = [(x_label, with_unit(px, unit_from_label(x_label))),
                (y_label, with_unit(py, unit_from_label(y_label)))]
        return _mark("line", title=series or "Line", rows=rows, label=series, series=series, value=py,
                     x=px, y=py, unit=unit_from_label(y_label), _xy=tuple(self.line_xy[i]), _radius=5.0)

    def _cell_at(self, x: float, y: float) -> Optional[Dict[str, Any]]:
        if not self.cells:
            return None
        ax = self.ax
        try:
            dx, dy = ax.transData.inverted().transform((x, y))
        except Exception:
            return None
        for artist in reversed(self.cells):
            located = _locate_cell(artist, dx, dy)
            if located is None:
                continue
            row, col, value, edges = located
            cells = self.marks.get("cells") or {}
            if cells.get("triangle") == "lower" and col > row:
                continue                          # the unused half of a symmetric matrix
            half = hasattr(artist, "get_coordinates")
            row_label = _pick(cells.get("rows"), row)
            col_label = _pick(cells.get("cols"), col)
            if row_label is None:
                row_label = (_half_tick_lookup(ax.yaxis) if half else _tick_lookup(ax.yaxis)).get(row)
            if col_label is None:
                col_label = (_half_tick_lookup(ax.xaxis) if half else _tick_lookup(ax.xaxis)).get(col)
            row_label = f"row {row + 1}" if row_label is None else str(row_label)
            col_label = f"column {col + 1}" if col_label is None else str(col_label)
            row_name = cells.get("row_name") or _clean_axis_label(ax.get_ylabel()) or "Row"
            col_name = cells.get("col_name") or _clean_axis_label(ax.get_xlabel()) or "Column"
            bar = getattr(artist, "colorbar", None)
            value_label = cells.get("value_label") or (
                _clean_axis_label(bar.ax.get_ylabel()) if bar is not None else "") or "Value"
            unit = cells["unit"] if "unit" in cells else unit_from_label(value_label)
            # The title already names the row and the column; the body
            # says what they are only when they are different things.
            rows = [] if row_name == col_name else [(row_name, row_label), (col_name, col_label)]
            if value is None:
                rows.append((value_label, cells.get("blank_text") or "no value"))
            elif value_label.lower().startswith("log10 "):
                # A logarithm is how the cell is coloured, not what a
                # reader wants to read: give the quantity itself first.
                base = value_label[6:].strip()
                base = base[:1].upper() + base[1:]
                rows.append((base, with_unit(10.0 ** value, unit_from_label(base))))
                rows.append(("log10", format_number(value)))
            else:
                rows.append((value_label, with_unit(value, unit)))
            tank = None
            if cells.get("rows_are_tanks", True):
                tank = tank_id_of(row_label)
            if tank is None and cells.get("cols_are_tanks", True):
                tank = tank_id_of(col_label)
            corners = ax.transData.transform(np.array([[edges[0], edges[2]], [edges[1], edges[3]]]))
            rect = (min(corners[:, 0]), min(corners[:, 1]), max(corners[:, 0]), max(corners[:, 1]))
            return _mark(
                "cell", title=f"{row_label} · {col_label}", rows=rows, label=f"{row_label} · {col_label}",
                row=row_label, column=col_label, value=value, tank_id=tank, unit=unit, _rect=rect,
            )
        return None


def _locate_cell(artist, dx: float, dy: float):
    """(row, column, value or None, (x0, x1, y0, y1)) of the cell at a data
    position, or None when the position is outside the artist. A masked
    or non-finite cell has the value None."""
    if hasattr(artist, "get_coordinates"):                      # QuadMesh
        coords = np.asarray(artist.get_coordinates())
        x_edges, y_edges = coords[0, :, 0], coords[:, 0, 1]
        n_rows, n_cols = len(y_edges) - 1, len(x_edges) - 1
        col = _bin(x_edges, dx)
        row = _bin(y_edges, dy)
        if row is None or col is None:
            return None
        array = np.ma.asarray(artist.get_array())
        if array.size != n_rows * n_cols:
            return None
        array = array.reshape(n_rows, n_cols)
        edges = (x_edges[col], x_edges[col + 1], y_edges[row], y_edges[row + 1])
    else:                                                       # AxesImage
        array = np.ma.asarray(artist.get_array())
        if array.ndim != 2:
            return None
        n_rows, n_cols = array.shape
        x0, x1, y0, y1 = artist.get_extent()
        if x1 == x0 or y1 == y0:
            return None
        fx, fy = (dx - x0) / (x1 - x0), (dy - y0) / (y1 - y0)
        if not (0.0 <= fx < 1.0 and 0.0 <= fy < 1.0):
            return None
        col = int(fx * n_cols)
        # get_extent() is (left, right, bottom, top). The first array row
        # is drawn at the top for origin "upper", at the bottom for "lower".
        from_bottom = int(fy * n_rows)
        row = (n_rows - 1 - from_bottom) if getattr(artist, "origin", "upper") == "upper" else from_bottom
        cw, ch = (x1 - x0) / n_cols, (y1 - y0) / n_rows
        edges = (x0 + col * cw, x0 + (col + 1) * cw, y0 + from_bottom * ch, y0 + (from_bottom + 1) * ch)
    value = array[row, col]
    if np.ma.is_masked(value):
        return row, col, None, edges              # drawn as "no value": still a cell
    value = float(value)
    return row, col, (None if not np.isfinite(value) else value), edges


def _bin(edges: np.ndarray, value: float) -> Optional[int]:
    if len(edges) < 2:
        return None
    ascending = edges[-1] >= edges[0]
    ordered = edges if ascending else edges[::-1]
    i = int(np.searchsorted(ordered, value, side="right")) - 1
    if i < 0 or i >= len(edges) - 1:
        return None
    return i if ascending else len(edges) - 2 - i


def _pick(values, index: int):
    if values is None:
        return None
    try:
        if isinstance(values, dict):
            return values.get(index)
        if 0 <= index < len(values):
            value = values[index]
            return None if value is None else value
    except Exception:
        return None
    return None


def _mark(kind: str, *, title: str, rows: Sequence[Tuple[str, str]], **fields) -> Dict[str, Any]:
    mark: Dict[str, Any] = {
        "kind": kind, "title": title, "rows": [(str(k), str(v)) for k, v in rows],
        "label": None, "series": None, "value": None, "row": None, "column": None,
        "tank_id": None, "unit": "",
    }
    mark.update(fields)
    mark["text"] = "\n".join([title] + [f"{k}: {v}" for k, v in mark["rows"]])
    return mark


def public_mark(mark: Dict[str, Any]) -> Dict[str, Any]:
    """The mark as emitted to listeners: no drawing internals."""
    return {k: v for k, v in mark.items() if k not in PRIVATE_KEYS}


class MarkIndex:
    """Lazy, per-draw index of every hit-testable mark of a figure."""

    def __init__(self, figure) -> None:
        self.figure = figure
        self._axes: Dict[int, _AxesIndex] = {}

    def invalidate(self) -> None:
        self._axes.clear()

    def axes_at(self, x: float, y: float, pad: float = 0.0):
        """The axes under a pixel. `pad` widens every axes by that many
        pixels: a sliver of a bar, or a point on the frame, is drawn half
        outside the axes and must still be reachable."""
        axes = [ax for ax in reversed(list(self.figure.axes)) if ax.get_visible()]
        for ax in axes:
            if ax.bbox.contains(x, y):
                return ax
        if pad > 0:
            for ax in axes:
                box = ax.bbox
                if box.x0 - pad <= x <= box.x1 + pad and box.y0 - pad <= y <= box.y1 + pad:
                    return ax
        return None

    def lookup(self, x: float, y: float) -> Optional[Dict[str, Any]]:
        ax = self.axes_at(x, y, pad=POINT_RADIUS_PX)
        if ax is None or _is_colorbar_axes(ax):
            return None
        index = self._axes.get(id(ax))
        if index is None or index.ax is not ax:
            index = self._axes[id(ax)] = _AxesIndex(ax)
        return index.lookup(x, y)

    def readout(self, x: float, y: float) -> str:
        """The formatted position under the pointer, for the line below the
        chart: a category name on a categorical axis, a formatted number
        with the axis label otherwise."""
        ax = self.axes_at(x, y)
        if ax is None or _is_colorbar_axes(ax) or not ax.axison:
            return ""
        try:
            dx, dy = ax.transData.inverted().transform((x, y))
        except Exception:
            return ""
        parts = []
        marks = _marks(ax)
        cells = marks.get("cells") or {}
        if cells.get("rows") is not None and cells.get("cols") is not None and np.isfinite(dx) and np.isfinite(dy):
            # A matrix: say which row and column, not a pair of coordinates.
            offset = 0.0 if ax.images else -0.5          # image cells are centred on i, mesh cells on i + 0.5
            row = _pick(cells["rows"], int(round(dy + offset)))
            col = _pick(cells["cols"], int(round(dx + offset)))
            if row is not None and col is not None:
                row_name = cells.get("row_name") or "Row"
                col_name = cells.get("col_name") or "Column"
                return f"{row_name}: {row}     {col_name}: {col}"
        for axis, value, label_text, name in ((ax.xaxis, dx, ax.get_xlabel(), "x"), (ax.yaxis, dy, ax.get_ylabel(), "y")):
            label = _clean_axis_label(label_text)
            categories = None
            if marks.get("category_axis", "y") == name and marks.get("categories") is not None:
                categories = marks.get("categories")
            position = int(round(value)) if np.isfinite(value) else None
            category = _pick(categories, position) if position is not None else None
            if category is None and position is not None:
                lookup = _tick_lookup(axis)
                if lookup and len(lookup) >= 2 and getattr(getattr(axis, "units", None), "_mapping", None):
                    category = lookup.get(position)
            if category is not None:
                parts.append(f"{label}: {category}" if label else str(category))
            elif np.isfinite(value):
                parts.append(f"{label or name}: {format_number(float(value))}")
        return "     ".join(parts)


# ---------------------------------------------------------------------------
# Qt layer
# ---------------------------------------------------------------------------

from PySide6.QtCore import QObject, QPoint, Qt, Signal  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtWidgets import QLabel, QMenu  # noqa: E402


def _palette() -> Dict[str, str]:
    """The active UI palette, read at use so a theme switch is followed."""
    try:
        import qt_theme
    except Exception:
        return {}
    current = getattr(qt_theme, "current_palette", None)
    if callable(current):
        try:
            return dict(current())
        except Exception:
            pass
    return dict(getattr(qt_theme, "PALETTE", {}))


def _escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class PlotTooltip(QLabel):
    """The tooltip of a chart mark: a title, then one line per quantity.

    A child of the canvas rather than a QToolTip, so it follows the theme,
    stays inside the chart, and is part of what `grab()` captures."""

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setObjectName("PlotTooltip")
        self.setTextFormat(Qt.RichText)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setWordWrap(False)
        self.hide()

    def show_mark(self, mark: Dict[str, Any], x: float, y: float, *, can_open_tank: bool = False) -> None:
        palette = _palette()
        background = palette.get("chrome", "#1d2733")
        ink = palette.get("chrome_ink", "#ffffff")
        muted = palette.get("chrome_muted", "#b7c0cc")
        self.setStyleSheet(
            f"QLabel#PlotTooltip {{ background: {background}; color: {ink}; border: 1px solid {background};"
            f" border-radius: 4px; padding: 6px 9px; font-size: 12px; }}"
        )
        rows = "".join(
            f"<tr><td style='color:{muted}; padding-right:10px;'>{_escape(key)}</td>"
            f"<td align='right' style='color:{ink};'>{_escape(value)}</td></tr>"
            for key, value in mark.get("rows", [])
        )
        hint = ""
        if mark.get("tank_id") and can_open_tank:
            hint = f"<div style='color:{muted}; font-size:11px; margin-top:3px;'>Double-click to open this tank</div>"
        self.setText(
            f"<div style='font-weight:600; color:{ink}; margin-bottom:2px;'>{_escape(mark.get('title', ''))}</div>"
            f"<table cellspacing='0' cellpadding='0'>{rows}</table>{hint}"
        )
        self.adjustSize()
        self.move(self._position(x, y))
        self.show()
        self.raise_()

    def _position(self, x: float, y: float) -> QPoint:
        parent = self.parentWidget()
        width, height = self.width(), self.height()
        px, py = int(x) + 14, int(y) + 16
        if parent is not None:
            if px + width > parent.width() - 4:
                px = int(x) - width - 12
            if py + height > parent.height() - 4:
                py = int(y) - height - 10
            px = max(4, min(px, max(4, parent.width() - width - 4)))
            py = max(4, min(py, max(4, parent.height() - height - 4)))
        return QPoint(px, py)


class PlotInteraction(QObject):
    """Hover, click and context menu on the marks of one canvas."""

    mark_hovered = Signal(dict)       # {} when the pointer leaves every mark
    mark_clicked = Signal(dict)
    mark_activated = Signal(dict)
    readout_changed = Signal(str)

    def __init__(self, canvas, figure, *, toolbar=None, parent=None) -> None:
        super().__init__(parent if parent is not None else canvas)
        self.canvas = canvas
        self.figure = figure
        self.toolbar = toolbar
        self.index = MarkIndex(figure)
        self.tooltip = PlotTooltip(canvas)
        self.enabled = True
        self._current: Optional[Dict[str, Any]] = None
        self._background = None
        self._overlay_drawn = False
        self._menu: Optional[QMenu] = None
        self._open_tank = None
        self._open_tank_available = None
        self._cids = [
            canvas.mpl_connect("draw_event", self._on_draw),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("figure_leave_event", self._on_leave),
            canvas.mpl_connect("axes_leave_event", self._on_leave),
            canvas.mpl_connect("button_press_event", self._on_press),
        ]

    # -- configuration ------------------------------------------------------
    def set_open_tank_handler(self, handler, available=None) -> None:
        """`handler(tank_id)` is called when a mark that stands for a tank
        is double-clicked or 'Open tank' is chosen. `available()` says
        whether there is anywhere to open a tank right now (it decides
        whether the menu entry and the tooltip's hint are shown)."""
        self._open_tank = handler
        self._open_tank_available = available

    def can_open_tank(self) -> bool:
        if self._open_tank is None:
            return False
        if self._open_tank_available is None:
            return True
        try:
            return bool(self._open_tank_available())
        except Exception:
            return False

    def current_mark(self) -> Optional[Dict[str, Any]]:
        return None if self._current is None else public_mark(self._current)

    def disconnect(self) -> None:
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        self._cids = []

    def _toolbar_busy(self) -> bool:
        mode = getattr(self.toolbar, "mode", "")
        return bool(str(getattr(mode, "value", mode) or ""))

    # -- events ---------------------------------------------------------------
    def _on_draw(self, _event=None) -> None:
        self.index.invalidate()
        self._background = None
        self._overlay_drawn = False
        if self._current is not None:
            self._current = None
            self.tooltip.hide()

    def _on_leave(self, _event=None) -> None:
        self._set_current(None, None, None)
        self.readout_changed.emit("")

    def _on_motion(self, event) -> None:
        if not self.enabled or self._toolbar_busy() or event.x is None:
            if self._current is not None:
                self._set_current(None, None, None)
            if self._toolbar_busy() and event.inaxes is not None and event.xdata is not None:
                self.readout_changed.emit(self.index.readout(event.x, event.y))
            return
        mark = self.index.lookup(event.x, event.y)
        self._set_current(mark, event.x, event.y)
        if mark is not None:
            self.readout_changed.emit("     ".join([mark["title"]] + [f"{k}: {v}" for k, v in mark["rows"]]))
        else:
            self.readout_changed.emit(self.index.readout(event.x, event.y))

    def _on_press(self, event) -> None:
        if not self.enabled or self._toolbar_busy() or event.x is None:
            return
        mark = self.index.lookup(event.x, event.y)
        if mark is None:
            return
        public = public_mark(mark)
        button = getattr(event.button, "value", event.button)
        if button == 3:
            self._show_menu(public)
        elif button == 1:
            if event.dblclick:
                self.mark_activated.emit(public)
                if public.get("tank_id") and self.can_open_tank():
                    self._open_tank(public["tank_id"])
            else:
                self.mark_clicked.emit(public)

    # -- hover state ------------------------------------------------------------
    def _set_current(self, mark, x, y) -> None:
        previous = self._current
        same = (
            mark is not None and previous is not None
            and mark.get("_rect") == previous.get("_rect") and mark.get("_xy") == previous.get("_xy")
            and mark.get("title") == previous.get("title")
        )
        self._current = mark
        if mark is None:
            if previous is not None:
                self.tooltip.hide()
                self._draw_overlay(None)
                self.mark_hovered.emit({})
            return
        ratio = self._pixel_ratio()
        self.tooltip.show_mark(mark, x / ratio, (self.figure.bbox.height - y) / ratio,
                               can_open_tank=self.can_open_tank())
        if not same:
            self._draw_overlay(mark)
            self.mark_hovered.emit(public_mark(mark))

    def _pixel_ratio(self) -> float:
        return float(getattr(self.canvas, "device_pixel_ratio", 1.0) or 1.0)

    def _draw_overlay(self, mark) -> None:
        """Outline the hovered mark without redrawing the chart: restore the
        clean frame, draw the outline on top, blit."""
        canvas = self.canvas
        if not (hasattr(canvas, "copy_from_bbox") and hasattr(canvas, "get_renderer")):
            return
        try:
            if self._background is None:
                if mark is None:
                    return
                self._background = canvas.copy_from_bbox(self.figure.bbox)
            canvas.restore_region(self._background)
            if mark is not None:
                renderer = canvas.get_renderer()
                for artist in self._overlay_artists(mark):
                    artist.set_figure(self.figure)
                    artist.draw(renderer)
            canvas.blit(self.figure.bbox)
            self._overlay_drawn = mark is not None
        except Exception:
            # Highlighting is a nicety; a backend that cannot blit must not
            # take the tooltip down with it.
            self._background = None

    def _overlay_artists(self, mark) -> List[Any]:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import IdentityTransform

        scale = self.figure.dpi / 100.0
        ink, halo = "#0b0b0b", "#ffffff"
        artists: List[Any] = []
        rect = mark.get("_rect")
        if rect is not None:
            x0, y0, x1, y1 = rect
            for color, width in ((halo, 3.2 * scale), (ink, 1.3 * scale)):
                artists.append(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color,
                                         linewidth=width, transform=IdentityTransform(), clip_on=False))
        xy = mark.get("_xy")
        if xy is not None:
            radius_pt = (mark.get("_radius", POINT_RADIUS_PX) + 2.0) * 72.0 / self.figure.dpi * 2.0
            for color, width in ((halo, 3.2 * scale), (ink, 1.3 * scale)):
                artists.append(Line2D([xy[0]], [xy[1]], marker="o", markersize=radius_pt,
                                      markerfacecolor="none", markeredgecolor=color, markeredgewidth=width,
                                      linestyle="none", transform=IdentityTransform(), clip_on=False))
            ax = mark.get("_axes")
            if ax is not None:
                box = ax.bbox
                for xs, ys in (([box.x0, box.x1], [xy[1], xy[1]]), ([xy[0], xy[0]], [box.y0, box.y1])):
                    artists.insert(0, Line2D(xs, ys, color="#52514e", linewidth=0.7 * scale, alpha=0.55,
                                             linestyle=(0, (3, 3)), transform=IdentityTransform(), clip_on=False))
        return artists

    # -- context menu -------------------------------------------------------------
    def build_menu(self, mark: Dict[str, Any]) -> QMenu:
        menu = QMenu(self.canvas)
        value = mark.get("value")
        value_text = "" if value is None else repr(float(value))
        label_text = str(mark.get("label") or mark.get("title") or "")
        copy_value = menu.addAction("Copy value")
        copy_value.setEnabled(bool(value_text))
        copy_value.triggered.connect(lambda _checked=False, t=value_text: QGuiApplication.clipboard().setText(t))
        copy_label = menu.addAction("Copy label")
        copy_label.setEnabled(bool(label_text))
        copy_label.triggered.connect(lambda _checked=False, t=label_text: QGuiApplication.clipboard().setText(t))
        tank = mark.get("tank_id")
        if tank and self.can_open_tank():
            menu.addSeparator()
            open_action = menu.addAction(f"Open tank {tank}")
            open_action.triggered.connect(lambda _checked=False, t=tank: self._open_tank(t))
        return menu

    def _show_menu(self, mark: Dict[str, Any]) -> None:
        from PySide6.QtGui import QCursor

        self.tooltip.hide()
        self._menu = self.build_menu(mark)
        self._menu.popup(QCursor.pos())
