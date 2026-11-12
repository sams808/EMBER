"""
qt_search.py -- the global search box of the application header.

Type a fragment of a tank id ("AW-106", "B107"), an element or an analyte
("Cs", "137Cs", "SO4") or a workspace name; the matches appear in a list
under the box; Enter (or a click) opens the chosen one. The shell decides
what "open" means for each kind of match (see EmberMainWindow.open_hit).

Two pieces, kept apart so the matching can be tested without a window:
- SearchIndex: plain Python. Holds the tanks, elements, analytes and
  workspace names; search(text) returns ranked SearchHit rows.
- GlobalSearch: the QLineEdit (object name SearchBox) and its result list.
  The list is a child of the main window rather than a separate popup
  window: it never takes the keyboard focus away from the box, it cannot
  be left behind on screen, and it is part of a window grab.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QFrame, QLabel, QLineEdit, QListWidget, QListWidgetItem, QStyle, QStyledItemDelegate,
    QVBoxLayout, QWidget,
)

KIND_TANK = "tank"
KIND_ELEMENT = "element"
KIND_ANALYTE = "analyte"
KIND_WORKSPACE = "workspace"

KIND_LABELS = {
    KIND_TANK: "Tank",
    KIND_ELEMENT: "Element",
    KIND_ANALYTE: "Analyte",
    KIND_WORKSPACE: "Workspace",
}
KIND_ICONS = {
    KIND_TANK: "tank_explorer",
    KIND_ELEMENT: "explorer",
    KIND_ANALYTE: "explorer",
    KIND_WORKSPACE: "panel",
}

_HIT_ROLE = Qt.UserRole + 1


@dataclass(frozen=True)
class SearchHit:
    kind: str      # one of the KIND_* constants
    key: str       # tank id, element symbol, analyte name or workspace name
    label: str     # what the row shows
    detail: str    # the quiet text at the right of the row
    score: int     # lower is a better match


def _compact(text: str) -> str:
    """Upper-case letters and digits only: "241-AW-106" and "aw106" meet."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def _natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


class SearchIndex:
    """What the search box can find. Refill with set_dataset() after every load."""

    def __init__(self, workspaces: Optional[Dict[str, str]] = None) -> None:
        self._workspaces: Dict[str, str] = dict(workspaces or {})
        self._tanks: List[str] = []
        self._tank_farm: Dict[str, str] = {}
        self._elements: List[str] = []
        self._analytes: List[str] = []

    # -- filling ---------------------------------------------------------
    def set_entries(self, tanks: Iterable[str] = (), elements: Iterable[str] = (),
                    analytes: Iterable[str] = (), tank_farms: Optional[Dict[str, str]] = None) -> None:
        self._tanks = sorted({str(t) for t in tanks if t}, key=_natural_key)
        self._tank_farm = dict(tank_farms or {})
        self._elements = sorted({str(e) for e in elements if e})
        self._analytes = sorted({str(a) for a in analytes if a}, key=_natural_key)

    def set_dataset(self, dataset) -> None:
        """Read the tanks, elements and analytes of a loaded HanfordDataset
        (anything else empties the data part of the index)."""
        if dataset is None or not dataset.is_loaded():
            self.set_entries()
            return
        df = dataset.require_df()
        columns = set(df.columns)

        def unique(column: str) -> List[str]:
            if column not in columns:
                return []
            return [str(v) for v in df.get_column(column).drop_nulls().unique().to_list()]

        farms: Dict[str, str] = {}
        if {"WasteSiteId", "TankFarm"} <= columns:
            pairs = df.select(["WasteSiteId", "TankFarm"]).unique().drop_nulls()
            farms = {str(t): str(f) for t, f in pairs.iter_rows()}
        self.set_entries(unique("WasteSiteId"), unique("Element"), unique("Analyte"), farms)

    def counts(self) -> Dict[str, int]:
        return {
            KIND_TANK: len(self._tanks), KIND_ELEMENT: len(self._elements),
            KIND_ANALYTE: len(self._analytes), KIND_WORKSPACE: len(self._workspaces),
        }

    # -- matching ----------------------------------------------------------
    def search(self, text: str, limit: int = 10) -> List[SearchHit]:
        query = str(text or "").strip()
        if not query:
            return []
        low = query.lower()
        compact = _compact(query)
        hits: List[SearchHit] = []

        # Elements: the symbol itself, a longer symbol it begins, or a name
        # the element normaliser knows ("iodine").
        element_keys = set()
        alias: Optional[str] = None
        try:
            from elements import normalize_element_symbol
            alias = normalize_element_symbol(query)
        except Exception:
            alias = None
        for symbol in self._elements:
            sym_low = symbol.lower()
            if sym_low == low:
                score = 0
            elif alias is not None and symbol == alias:
                score = 1
            elif len(low) == 1 and sym_low.startswith(low):
                score = 20
            else:
                continue
            element_keys.add(symbol)
            hits.append(SearchHit(KIND_ELEMENT, symbol, symbol, "Element", score))

        # Analytes: exact, then the isotopes of a symbol ("Cs" -> 137Cs),
        # then names that begin with or contain the text.
        for analyte in self._analytes:
            if analyte in element_keys:
                continue          # the element row already stands for it
            a_low = analyte.lower()
            if a_low == low:
                score = 2
            elif a_low.endswith(low) and a_low[: -len(low)].rstrip("m/").replace("/", "").isdigit():
                score = 12
            elif a_low.startswith(low):
                score = 24
            elif len(low) >= 2 and low in a_low:
                score = 40
            else:
                continue
            hits.append(SearchHit(KIND_ANALYTE, analyte, analyte, "Analyte", score))

        # Tanks: compare without punctuation, and without the leading area
        # number, so "B107", "b-107" and "241-B-107" all find the same tank.
        if compact:
            for tank in self._tanks:
                full = _compact(tank)
                short = full[3:] if full.startswith("241") and len(full) > 3 else full
                if compact in (full, short):
                    score = 3
                elif short.startswith(compact) or full.startswith(compact):
                    score = 14
                elif len(compact) >= 2 and compact in full:
                    score = 30
                else:
                    continue
                farm = self._tank_farm.get(tank)
                detail = f"Tank · farm {farm}" if farm else "Tank"
                hits.append(SearchHit(KIND_TANK, tank, tank, detail, score))

        for name in self._workspaces:
            n_low = name.lower()
            if n_low == low:
                score = 1
            elif n_low.startswith(low) or any(word.startswith(low) for word in n_low.split()):
                score = 10
            elif len(low) >= 2 and low in n_low:
                score = 35
            else:
                continue
            hits.append(SearchHit(KIND_WORKSPACE, name, name, "Workspace", score))

        order = {KIND_ELEMENT: 0, KIND_ANALYTE: 1, KIND_TANK: 2, KIND_WORKSPACE: 3}
        hits.sort(key=lambda h: (h.score, order.get(h.kind, 9), _natural_key(h.label)))
        return hits[: max(0, int(limit))]


class _HitDelegate(QStyledItemDelegate):
    """One result row: icon, label, and the kind at the right."""

    ROW_HEIGHT = 34

    def sizeHint(self, option, index) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(200, self.ROW_HEIGHT)

    def paint(self, painter: QPainter, option, index) -> None:
        from qt_icons import paint_icon
        from qt_theme import current_palette

        palette = current_palette()
        hit = index.data(_HIT_ROLE)
        rect = option.rect.adjusted(2, 1, -2, -1)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        selected = bool(option.state & QStyle.State_Selected)
        hovered = bool(option.state & QStyle.State_MouseOver)
        if hit is not None and (selected or hovered):
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(palette["selection_bg"] if selected else palette["hover"]))
            painter.drawRoundedRect(QRectF(rect), 3, 3)
        font = QFont(option.font)
        font.setPixelSize(13)
        if hit is None:
            painter.setFont(font)
            painter.setPen(QColor(palette["muted"]))
            painter.drawText(rect.adjusted(10, 0, -10, 0), Qt.AlignVCenter | Qt.AlignLeft,
                             str(index.data(Qt.DisplayRole) or ""))
            painter.restore()
            return
        icon_rect = QRectF(rect.left() + 10, rect.center().y() - 7.5, 16, 16)
        paint_icon(painter, KIND_ICONS.get(hit.kind, "search"), icon_rect,
                   palette["accent"] if selected else palette["muted"])
        detail_font = QFont(font)
        detail_font.setPixelSize(11)
        detail_width = QFontMetrics(detail_font).horizontalAdvance(hit.detail) + 4
        painter.setFont(detail_font)
        painter.setPen(QColor(palette["muted"]))
        painter.drawText(QRect(rect.right() - detail_width - 10, rect.top(), detail_width, rect.height()),
                         Qt.AlignVCenter | Qt.AlignRight, hit.detail)
        font.setWeight(QFont.DemiBold)
        painter.setFont(font)
        painter.setPen(QColor(palette["ink"]))
        label_rect = QRect(rect.left() + 36, rect.top(), rect.width() - 36 - detail_width - 24, rect.height())
        label = QFontMetrics(font).elidedText(hit.label, Qt.ElideRight, label_rect.width())
        painter.drawText(label_rect, Qt.AlignVCenter | Qt.AlignLeft, label)
        painter.restore()


class SearchPopup(QFrame):
    """The result list. A plain child widget, raised above the page."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("SearchPopup")
        self.setFocusPolicy(Qt.NoFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        self.list = QListWidget()
        self.list.setObjectName("SearchResults")
        self.list.setFocusPolicy(Qt.NoFocus)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setMouseTracking(True)
        self.list.setItemDelegate(_HitDelegate(self.list))
        self.list.setUniformItemSizes(True)
        layout.addWidget(self.list)
        self.hint = QLabel("Enter to open  ·  Up / Down to choose  ·  Esc to close")
        self.hint.setObjectName("SearchHint")
        layout.addWidget(self.hint)
        self.hide()

    def set_hits(self, hits: Sequence[SearchHit], query: str) -> None:
        self.list.clear()
        if not hits:
            item = QListWidgetItem(f"No tank, element, analyte or workspace matches “{query}”.")
            item.setFlags(Qt.NoItemFlags)
            self.list.addItem(item)
        for hit in hits:
            item = QListWidgetItem(hit.label)
            item.setData(_HIT_ROLE, hit)
            item.setToolTip(f"{KIND_LABELS.get(hit.kind, hit.kind)}: {hit.key}")
            self.list.addItem(item)
        if hits:
            self.list.setCurrentRow(0)

    def content_height(self) -> int:
        rows = max(1, self.list.count())
        return rows * _HitDelegate.ROW_HEIGHT + 8 + self.hint.sizeHint().height() + 2

    def current_hit(self) -> Optional[SearchHit]:
        item = self.list.currentItem()
        return item.data(_HIT_ROLE) if item is not None else None


class GlobalSearch(QLineEdit):
    """The search box. Emits `activated(SearchHit)` when a match is chosen."""

    activated = Signal(object)

    def __init__(self, index: Optional[SearchIndex] = None, parent: Optional[QWidget] = None,
                 limit: int = 10) -> None:
        super().__init__(parent)
        self.setObjectName("SearchBox")
        self.index = index if index is not None else SearchIndex()
        self.limit = limit
        self.setPlaceholderText("Search tanks, elements, workspaces")
        self.setToolTip("Find a tank, an element, an analyte or a workspace (Ctrl+K).\n"
                        "Examples: AW-106, B107, Cs, 137Cs, SO4, heatmaps.")
        self.setClearButtonEnabled(False)
        self.popup: Optional[SearchPopup] = None
        self._icon_action = None
        self.refresh_icon()
        self.textEdited.connect(self._on_text_edited)

    # -- appearance ----------------------------------------------------------
    def refresh_icon(self) -> None:
        """(Re)draw the magnifier in the header's secondary colour."""
        from qt_icons import icon
        from qt_theme import current_palette

        if self._icon_action is not None:
            self.removeAction(self._icon_action)
        self._icon_action = self.addAction(icon("search", current_palette()["chrome_muted"], 16),
                                           QLineEdit.LeadingPosition)

    # -- popup ------------------------------------------------------------------
    def _ensure_popup(self) -> SearchPopup:
        host = self.window()
        if self.popup is None or self.popup.parent() is not host:
            if self.popup is not None:
                self.popup.deleteLater()
            self.popup = SearchPopup(host)
            self.popup.list.itemClicked.connect(self._on_item_clicked)
        return self.popup

    def _place_popup(self) -> None:
        popup = self._ensure_popup()
        host = popup.parentWidget()
        if host is None:
            return
        origin = self.mapTo(host, QPoint(0, self.height() + 4))
        width = max(self.width(), 380)
        x = min(origin.x(), max(0, host.width() - width - 8))
        height = min(popup.content_height(), max(120, host.height() - origin.y() - 12))
        popup.setGeometry(x, origin.y(), width, height)
        popup.raise_()

    def show_results(self) -> List[SearchHit]:
        """Search for the current text and show the list; returns the hits."""
        query = self.text().strip()
        popup = self._ensure_popup()
        if not query:
            popup.hide()
            return []
        hits = self.index.search(query, self.limit)
        popup.set_hits(hits, query)
        self._place_popup()
        popup.show()
        return hits

    def hide_results(self) -> None:
        if self.popup is not None:
            self.popup.hide()

    def results_visible(self) -> bool:
        return self.popup is not None and not self.popup.isHidden()

    def _on_text_edited(self, _text: str) -> None:
        self.show_results()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        hit = item.data(_HIT_ROLE)
        if hit is not None:
            self._activate(hit)

    # -- choosing -----------------------------------------------------------------
    def _activate(self, hit: SearchHit) -> None:
        self.hide_results()
        self.clear()
        self.activated.emit(hit)

    def activate_current(self) -> Optional[SearchHit]:
        """Open the highlighted match (the best one by default)."""
        if not self.results_visible():
            self.show_results()
        hit = self.popup.current_hit() if self.popup is not None else None
        if hit is not None:
            self._activate(hit)
        return hit

    def _move(self, step: int) -> None:
        if not self.results_visible():
            self.show_results()
            return
        lst = self.popup.list
        rows = [r for r in range(lst.count()) if lst.item(r).data(_HIT_ROLE) is not None]
        if not rows:
            return
        current = lst.currentRow()
        position = rows.index(current) if current in rows else -1
        lst.setCurrentRow(rows[(position + step) % len(rows)])

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        key = event.key()
        if key == Qt.Key_Down:
            self._move(1)
            event.accept()
            return
        if key == Qt.Key_Up:
            self._move(-1)
            event.accept()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.activate_current()
            event.accept()
            return
        if key == Qt.Key_Escape:
            if self.results_visible():
                self.hide_results()
            else:
                self.clear()
                self.clearFocus()
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self.hide_results()
        super().focusOutEvent(event)

    def focusInEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().focusInEvent(event)
        if self.text().strip():
            self.show_results()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        if self.results_visible():
            self._place_popup()
