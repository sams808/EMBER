"""
qt_theme.py -- the design system: palettes, data colours, the application
QPalette and one stylesheet for the whole app.

Character: an engineering data product. Navy chrome, a white working
surface, one warm accent reserved for the primary action and the current
selection. Flat, 1 px hairlines, 4 px radius, no gradients, no shadows.

What lives here
- PALETTE / DARK_PALETTE: the interface colours, by role. Pages read them
  through current_palette() and never hard-code a hex value.
- DATA: the colours of charts and maps. They are the same in both themes
  because chart surfaces stay white in both.
- build_qpalette(): the QPalette that matches a palette dict. The
  stylesheet alone is not enough: anything Qt paints from the palette
  (link colour in rich text, item-view text, icons that a library recolours
  from the widget palette) otherwise follows the operating system, and a
  light theme on a dark-mode Windows session gets white marks on white.
- build_stylesheet(): every widget-class rule and every object name of the
  design system (see the brief: AppHeader, PageTitle, StatTile, ...).
- apply_theme(): sets style, palette and stylesheet together and
  announces the change through theme_notifier().changed.

Type scale (Segoe UI): page title 20/600, section title 14/600, body 13,
secondary 12, caption and table header 11/600 upper-case, stat value
22/600. Spacing sits on a 4 px grid.
"""
from __future__ import annotations

import os
import tempfile
from typing import Dict, Optional

PALETTE = {
    # surfaces
    "bg": "#f4f6f8",            # page plane
    "bg_alt": "#e9edf1",        # recessed strips
    "card": "#ffffff",          # working surfaces
    "sidebar": "#fbfcfd",
    "hover": "#eef1f5",
    "row_alt": "#fafbfc",
    # text
    "ink": "#17212b",
    "muted": "#586474",
    "faint": "#848e9b",         # placeholders, tertiary marks
    # lines
    "border": "#dfe3e8",        # hairline
    "border_strong": "#818b98",  # control outlines (3:1 on the surfaces)
    "scroll_handle": "#818b98",
    # accent: the primary action and the current selection, nothing else
    "accent": "#b84a18",
    "accent_hover": "#9c3d12",
    "accent_ink": "#ffffff",
    "selection_bg": "#fdeee4",
    # chrome: the header band
    "chrome": "#13263b",
    "chrome_deep": "#0e1d2e",
    "chrome_ink": "#ffffff",
    "chrome_muted": "#a9b8c9",
    "chrome_field": "#1f3650",
    "chrome_border": "#5d7896",
    # semantic
    "link": "#1d5fb0",
    "focus": "#2a78d6",
    "good": "#1a7f4b",
    "warn": "#8f5f00",
    "critical": "#b3261e",
    "critical_bg": "#fbeae8",
    "note_bg": "#eef3f9",
    "note_ink": "#2d4762",
    "note_border": "#d3deea",
    "tooltip_bg": "#13263b",
    "tooltip_ink": "#ffffff",
    "tooltip_border": "#13263b",
}

# The interface only: chart surfaces deliberately stay white in both
# themes, so what is on screen always matches the exported figure.
DARK_PALETTE = {
    "bg": "#11161c",
    "bg_alt": "#0c1015",
    "card": "#19202a",
    "sidebar": "#0e1319",
    "hover": "#222b36",
    "row_alt": "#1c242e",
    "ink": "#e7ebf0",
    "muted": "#a5afbb",
    "faint": "#7d8895",
    "border": "#2a333e",
    "border_strong": "#6b7786",
    "scroll_handle": "#6b7786",
    "accent": "#f08a4b",
    "accent_hover": "#ff9f66",
    "accent_ink": "#1c0f07",
    "selection_bg": "#3b2a1f",
    "chrome": "#0b1a2b",
    "chrome_deep": "#071321",
    "chrome_ink": "#ffffff",
    "chrome_muted": "#9db0c5",
    "chrome_field": "#152a41",
    "chrome_border": "#527092",
    "link": "#7fb5f5",
    "focus": "#6aa9ef",
    "good": "#4cc38a",
    "warn": "#e2b254",
    "critical": "#f08379",
    "critical_bg": "#3d211d",
    "note_bg": "#172536",
    "note_ink": "#c9d8e8",
    "note_border": "#29415b",
    "tooltip_bg": "#2a3441",
    "tooltip_ink": "#f2f5f8",
    "tooltip_border": "#44505e",
}

# Charts and maps. One fixed order for categories, one hue for magnitude,
# blue / grey / red for a signed quantity. A single-series bar chart is
# DATA["single"], never the interface accent.
_BLUE_RAMP = ["#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DATA = {
    "categorical": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "phase": {"Supernatant": "#2a78d6", "Saltcake": "#eb6834", "Sludge": "#1baf7a"},
    "sequential": list(_BLUE_RAMP),
    "ramp": list(_BLUE_RAMP),            # alias of "sequential"
    "blue_ramp": list(_BLUE_RAMP),       # alias of "sequential"
    "diverging": ["#2a78d6", "#f0efec", "#e34948"],   # low, middle, high
    "diverging_low": "#2a78d6",
    "diverging_mid": "#f0efec",
    "diverging_high": "#e34948",
    "single": "#2a78d6",
    "bar": "#2a78d6",                    # alias of "single"
    "ink": "#0b0b0b",
    "secondary": "#52514e",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#ffffff",
}

_FONT_FAMILY = '"Segoe UI", "Inter", -apple-system, "Helvetica Neue", Arial, sans-serif'
_MONO_FAMILY = 'Consolas, "Cascadia Mono", "DejaVu Sans Mono", monospace'

_ACTIVE: Dict[str, str] = PALETTE
_NOTIFIER = None


def current_palette() -> Dict[str, str]:
    """The palette of the theme that is applied right now (PALETTE until
    apply_theme() has run). Custom-painted widgets read their colours from
    this at paint time, so a theme change needs no bookkeeping in them."""
    return _ACTIVE


def is_dark() -> bool:
    return _ACTIVE is DARK_PALETTE


def theme_notifier():
    """A QObject with one signal, `changed`, emitted after every
    apply_theme(). Widgets repaint by themselves when the stylesheet
    changes; connect here only to rebuild something cached per theme (an
    icon, a pixmap)."""
    global _NOTIFIER
    if _NOTIFIER is None:
        from PySide6.QtCore import QObject, Signal

        class _ThemeNotifier(QObject):
            changed = Signal()

        _NOTIFIER = _ThemeNotifier()
    return _NOTIFIER


# ----------------------------------------------------------------------
# Small glyphs the stylesheet needs as image files (drop-down and step
# arrows, the check mark). A stylesheet can only name a file, so they are
# drawn once per colour into a cache folder and reused.
# ----------------------------------------------------------------------
_GLYPH_DIR: Optional[str] = None
_GLYPH_CACHE: Dict[tuple, Optional[str]] = {}


def _glyph_dir() -> Optional[str]:
    global _GLYPH_DIR
    if _GLYPH_DIR is None:
        try:
            path = os.path.join(tempfile.gettempdir(), "ember_ui_glyphs")
            os.makedirs(path, exist_ok=True)
            _GLYPH_DIR = path
        except OSError:
            _GLYPH_DIR = ""
    return _GLYPH_DIR or None


def _draw_glyph(kind: str, color: str, size: int, scale: int):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen

    px = size * scale
    image = QImage(px, px, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(QColor(color))
    pen.setWidthF(1.5 * scale)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    s = float(px)

    def poly(points) -> None:
        path = QPainterPath()
        path.moveTo(QPointF(points[0][0] * s, points[0][1] * s))
        for x, y in points[1:]:
            path.lineTo(QPointF(x * s, y * s))
        painter.drawPath(path)

    if kind == "chevron_down":
        poly([(0.2, 0.36), (0.5, 0.66), (0.8, 0.36)])
    elif kind == "chevron_up":
        poly([(0.2, 0.64), (0.5, 0.34), (0.8, 0.64)])
    elif kind == "chevron_right":
        poly([(0.36, 0.2), (0.66, 0.5), (0.36, 0.8)])
    elif kind == "chevron_left":
        poly([(0.64, 0.2), (0.34, 0.5), (0.64, 0.8)])
    elif kind == "check":
        poly([(0.2, 0.53), (0.42, 0.74), (0.8, 0.29)])
    elif kind == "dash":
        poly([(0.24, 0.5), (0.76, 0.5)])
    painter.end()
    return image


def _glyph(kind: str, color: str, size: int = 10) -> Optional[str]:
    """Path (forward slashes) of a glyph PNG, or None if the cache folder
    cannot be written; the rule that wanted it is then left out and Qt
    draws its own arrow."""
    key = (kind, color, size)
    if key in _GLYPH_CACHE:
        return _GLYPH_CACHE[key]
    result: Optional[str] = None
    folder = _glyph_dir()
    if folder:
        base = os.path.join(folder, f"{kind}_{size}_{color.lstrip('#')}")
        try:
            for scale, suffix in ((1, ".png"), (2, "@2x.png")):
                target = base + suffix
                if not os.path.isfile(target):
                    if not _draw_glyph(kind, color, size, scale).save(target, "PNG"):
                        raise OSError(target)
            result = (base + ".png").replace("\\", "/")
        except Exception:
            result = None
    _GLYPH_CACHE[key] = result
    return result


def _image(kind: str, color: str, size: int = 10) -> str:
    """An `image: url(...)` declaration, or nothing."""
    path = _glyph(kind, color, size)
    return f'image: url("{path}");' if path else ""


# ----------------------------------------------------------------------
# QPalette
# ----------------------------------------------------------------------
def build_qpalette(palette: dict = PALETTE):
    """The QPalette equivalent of a palette dict, disabled group included."""
    from PySide6.QtGui import QColor, QPalette

    p = palette
    qp = QPalette()
    roles = {
        QPalette.Window: p["bg"],
        QPalette.WindowText: p["ink"],
        QPalette.Base: p["card"],
        QPalette.AlternateBase: p["row_alt"],
        QPalette.Text: p["ink"],
        QPalette.Button: p["card"],
        QPalette.ButtonText: p["ink"],
        QPalette.BrightText: p["accent_ink"],
        QPalette.Highlight: p["selection_bg"],
        QPalette.HighlightedText: p["ink"],
        QPalette.ToolTipBase: p["tooltip_bg"],
        QPalette.ToolTipText: p["tooltip_ink"],
        QPalette.Link: p["link"],
        QPalette.LinkVisited: p["link"],
        QPalette.PlaceholderText: p["faint"],
        QPalette.Light: p["card"],
        QPalette.Midlight: p["hover"],
        QPalette.Mid: p["border"],
        QPalette.Dark: p["border_strong"],
        QPalette.Shadow: p["border_strong"],
    }
    for role, value in roles.items():
        color = QColor(value)
        for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
            qp.setColor(group, role, color)
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText, QPalette.HighlightedText):
        qp.setColor(QPalette.Disabled, role, QColor(p["faint"]))
    qp.setColor(QPalette.Disabled, QPalette.Base, QColor(p["bg_alt"]))
    qp.setColor(QPalette.Disabled, QPalette.Button, QColor(p["bg_alt"]))
    return qp


# ----------------------------------------------------------------------
# Stylesheet
# ----------------------------------------------------------------------
def build_stylesheet(palette: dict = PALETTE) -> str:
    p = palette
    caption = "font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;"
    # Captions that carry data wording (a unit, an element symbol, a formula)
    # keep their case: upper-casing turns kg into KG, Na into NA and Co into CO.
    data_caption = "font-size: 11px; font-weight: 600; letter-spacing: 0.3px;"
    return f"""
    /* ---------- base ---------- */
    QWidget {{
        font-family: {_FONT_FAMILY};
        font-size: 13px;
        color: {p['ink']};
    }}
    QWidget:disabled {{
        color: {p['faint']};
    }}
    QMainWindow, QDialog {{
        background: {p['bg']};
    }}
    QToolTip {{
        background: {p['tooltip_bg']};
        color: {p['tooltip_ink']};
        border: 1px solid {p['tooltip_border']};
        padding: 5px 8px;
        font-size: 12px;
    }}

    /* ---------- application header (navy chrome) ---------- */
    QMenuBar {{
        background: {p['chrome_deep']};
        color: {p['chrome_ink']};
        border: none;
        padding: 1px 8px;
        font-size: 12px;
    }}
    QMenuBar::item {{
        background: transparent;
        color: {p['chrome_muted']};
        padding: 5px 10px;
        border-radius: 3px;
    }}
    QMenuBar::item:selected, QMenuBar::item:pressed {{
        background: {p['chrome_field']};
        color: {p['chrome_ink']};
    }}
    QFrame#AppHeader {{
        background: {p['chrome']};
        border: none;
    }}
    QLabel#AppTitle {{
        color: {p['chrome_ink']};
        font-size: 17px;
        font-weight: 700;
        letter-spacing: 2px;
    }}
    QLabel#AppSubtitle {{
        color: {p['chrome_muted']};
        font-size: 11px;
    }}
    QLabel#DatasetChip {{
        background: {p['chrome_field']};
        color: {p['chrome_ink']};
        border: 1px solid {p['chrome_border']};
        border-radius: 4px;
        padding: 5px 10px;
        font-size: 12px;
    }}
    /* SearchBox is any search / filter field. On a page it is an ordinary
       input (see "inputs" below); inside the header it takes the chrome. */
    QFrame#AppHeader QLineEdit#SearchBox {{
        background: {p['chrome_field']};
        color: {p['chrome_ink']};
        border: 1px solid {p['chrome_border']};
        border-radius: 4px;
        padding: 5px 10px;
        font-size: 13px;
        selection-background-color: {p['accent']};
        selection-color: {p['accent_ink']};
    }}
    QFrame#AppHeader QLineEdit#SearchBox:hover {{
        border: 1px solid {p['chrome_muted']};
    }}
    QFrame#AppHeader QLineEdit#SearchBox:focus {{
        border: 1px solid {p['chrome_ink']};
    }}
    QToolButton#HeaderButton {{
        background: transparent;
        color: {p['chrome_ink']};
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 5px 9px;
        font-size: 12px;
    }}
    QToolButton#HeaderButton:hover, QToolButton#HeaderButton:pressed {{
        background: {p['chrome_field']};
        border: 1px solid {p['chrome_border']};
    }}
    QToolButton#HeaderButton:focus {{
        border: 1px solid {p['chrome_ink']};
    }}
    QToolButton#HeaderButton::menu-indicator {{
        image: none;
        width: 0px;
    }}
    QFrame#SearchPopup {{
        background: {p['card']};
        border: 1px solid {p['border_strong']};
        border-radius: 4px;
    }}
    QListWidget#SearchResults {{
        background: transparent;
        border: none;
        outline: none;
        padding: 4px;
    }}
    QLabel#SearchHint {{
        color: {p['muted']};
        font-size: 11px;
        padding: 5px 12px 6px 12px;
        border-top: 1px solid {p['border']};
    }}

    /* ---------- sidebar ---------- */
    QWidget#Sidebar {{
        background: {p['sidebar']};
        border-right: 1px solid {p['border']};
    }}
    QListWidget#NavList {{
        background: transparent;
        border: none;
        outline: none;
        padding: 0px;
    }}
    QLabel#NavGroup {{
        color: {p['muted']};
        {caption}
    }}
    QToolButton#SidebarToggle {{
        background: transparent;
        border: none;
        border-top: 1px solid {p['border']};
        border-radius: 0px;
        padding: 8px;
        color: {p['muted']};
        font-size: 12px;
    }}
    QToolButton#SidebarToggle:hover {{
        background: {p['hover']};
        color: {p['ink']};
    }}

    /* ---------- scope strip ---------- */
    QFrame#ScopeStrip {{
        background: {p['card']};
        border: none;
        border-bottom: 1px solid {p['border']};
    }}
    QWidget#ScopeBar {{
        background: transparent;
    }}
    QLabel#ScopeTitle {{
        color: {p['ink']};
        {caption}
    }}
    QLabel#ScopeStamp {{
        font-size: 12px;
        font-weight: 600;
        color: {p['accent']};
    }}
    QLabel#ScopeIdle {{
        font-size: 12px;
        color: {p['muted']};
    }}

    /* ---------- page furniture ---------- */
    QWidget#PageHeader, QWidget#EmptyState {{
        background: transparent;
    }}
    QLabel#PageTitle {{
        font-size: 20px;
        font-weight: 600;
        color: {p['ink']};
    }}
    QLabel#PageSubtitle {{
        font-size: 13px;
        color: {p['muted']};
    }}
    QLabel#SectionTitle {{
        font-size: 14px;
        font-weight: 600;
        color: {p['ink']};
    }}
    QLabel#SectionNote {{
        font-size: 12px;
        color: {p['muted']};
    }}
    QLabel#FieldLabel {{
        font-size: 12px;
        color: {p['muted']};
    }}
    QLabel#WarningNote {{
        font-size: 12px;
        color: {p['note_ink']};
        background: {p['note_bg']};
        border: 1px solid {p['note_border']};
        border-radius: 4px;
        padding: 6px 10px;
    }}
    QWidget#Card, QFrame#Card {{
        background: {p['card']};
        border: 1px solid {p['border']};
        border-radius: 4px;
    }}
    QWidget#Toolbar, QFrame#Toolbar {{
        background: transparent;
        border: none;
    }}
    QFrame#StatTile {{
        background: {p['card']};
        border: 1px solid {p['border']};
        border-radius: 4px;
    }}
    QLabel#StatLabel {{
        color: {p['muted']};
        {data_caption}
    }}
    QLabel#StatValue {{
        font-size: 22px;
        font-weight: 600;
        color: {p['ink']};
    }}
    QLabel#StatHint {{
        font-size: 12px;
        color: {p['muted']};
    }}
    QLabel#EmptyTitle {{
        font-size: 14px;
        font-weight: 600;
        color: {p['ink']};
    }}
    QLabel#EmptyHint {{
        font-size: 12px;
        color: {p['muted']};
    }}
    QFrame#NotesBanner {{
        background: {p['note_bg']};
        border: 1px solid {p['note_border']};
        border-radius: 4px;
    }}
    QLabel#NotesSummary, QLabel#NotesDetail {{
        font-size: 12px;
        color: {p['note_ink']};
    }}
    QToolButton#NotesToggle {{
        background: transparent;
        border: none;
        padding: 0px 2px;
        font-size: 12px;
        font-weight: 600;
        color: {p['link']};
    }}
    QToolButton#NotesToggle:hover {{
        text-decoration: underline;
    }}
    QLabel#Pill {{
        background: {p['bg_alt']};
        color: {p['ink']};
        border: 1px solid {p['border']};
        border-radius: 9px;
        padding: 1px 8px;
        font-size: 11px;
        font-weight: 600;
    }}

    /* ---------- buttons ---------- */
    QPushButton {{
        background: {p['card']};
        border: 1px solid {p['border_strong']};
        border-radius: 4px;
        padding: 6px 14px;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background: {p['hover']};
    }}
    QPushButton:pressed {{
        background: {p['bg_alt']};
    }}
    QPushButton:focus {{
        border: 1px solid {p['focus']};
    }}
    QPushButton:disabled {{
        color: {p['faint']};
        background: {p['bg_alt']};
        border: 1px solid {p['border']};
    }}
    QPushButton#Primary {{
        background: {p['accent']};
        color: {p['accent_ink']};
        border: 1px solid {p['accent']};
        font-weight: 600;
    }}
    QPushButton#Primary:hover, QPushButton#Primary:pressed {{
        background: {p['accent_hover']};
        border: 1px solid {p['accent_hover']};
    }}
    QPushButton#Primary:focus {{
        border: 1px solid {p['ink']};
    }}
    QPushButton#Primary:disabled {{
        color: {p['faint']};
        background: {p['bg_alt']};
        border: 1px solid {p['border']};
    }}
    QPushButton#Secondary {{
        background: transparent;
        border: 1px solid transparent;
        color: {p['link']};
        padding: 6px 10px;
    }}
    QPushButton#Secondary:hover {{
        background: {p['hover']};
    }}
    QPushButton#Secondary:focus {{
        border: 1px solid {p['focus']};
    }}
    QPushButton#Secondary:disabled {{
        color: {p['faint']};
        background: transparent;
        border: 1px solid transparent;
    }}
    QPushButton#HelpButton {{
        border: 1px solid {p['border_strong']};
        border-radius: 12px;
        padding: 0px;
        min-width: 22px;
        max-width: 22px;
        min-height: 22px;
        max-height: 22px;
        font-size: 12px;
        font-weight: 600;
        color: {p['muted']};
    }}
    QPushButton#HelpButton:hover {{
        color: {p['ink']};
        background: {p['hover']};
    }}
    QToolButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 4px 8px;
    }}
    QToolButton:hover {{
        background: {p['hover']};
    }}
    QToolButton:checked {{
        background: {p['selection_bg']};
    }}
    QToolButton:focus {{
        border: 1px solid {p['focus']};
    }}
    QToolButton#CollapsibleHeader {{
        background: transparent;
        border: none;
        padding: 4px 2px;
        font-size: 12px;
        font-weight: 600;
        color: {p['muted']};
        text-align: left;
    }}
    QToolButton#CollapsibleHeader:hover {{
        color: {p['ink']};
        background: transparent;
    }}
    QToolButton#CollapsibleHeader:checked {{
        color: {p['ink']};
        background: transparent;
    }}

    /* ---------- inputs ---------- */
    QLineEdit, QComboBox, QAbstractSpinBox {{
        background: {p['card']};
        border: 1px solid {p['border_strong']};
        border-radius: 4px;
        padding: 5px 8px;
        font-size: 13px;
        selection-background-color: {p['focus']};
        selection-color: #ffffff;
    }}
    QLineEdit:hover, QComboBox:hover, QAbstractSpinBox:hover {{
        border: 1px solid {p['muted']};
    }}
    QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus {{
        border: 1px solid {p['focus']};
    }}
    QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled {{
        background: {p['bg_alt']};
        color: {p['faint']};
        border: 1px solid {p['border']};
    }}
    QComboBox {{
        padding-right: 24px;
        combobox-popup: 0;
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: center right;
        width: 24px;
        border: none;
        background: transparent;
    }}
    QComboBox::down-arrow {{
        {_image('chevron_down', p['muted'])}
        width: 10px;
        height: 10px;
    }}
    QComboBox::down-arrow:disabled {{
        {_image('chevron_down', p['faint'])}
    }}
    QComboBox QAbstractItemView {{
        background: {p['card']};
        border: 1px solid {p['border_strong']};
        border-radius: 0px;
        padding: 2px;
        outline: none;
        selection-background-color: {p['selection_bg']};
        selection-color: {p['ink']};
    }}
    QComboBox QAbstractItemView::item {{
        min-height: 24px;
        padding: 2px 8px;
    }}
    QAbstractSpinBox {{
        padding-right: 22px;
    }}
    QAbstractSpinBox::up-button {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 18px;
        border: none;
        border-left: 1px solid {p['border']};
        border-top-right-radius: 3px;
        background: transparent;
        margin: 1px 1px 0px 0px;
    }}
    QAbstractSpinBox::down-button {{
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 18px;
        border: none;
        border-left: 1px solid {p['border']};
        border-bottom-right-radius: 3px;
        background: transparent;
        margin: 0px 1px 1px 0px;
    }}
    QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
        background: {p['hover']};
    }}
    QAbstractSpinBox::up-arrow {{
        {_image('chevron_up', p['muted'], 8)}
        width: 8px;
        height: 8px;
    }}
    QAbstractSpinBox::down-arrow {{
        {_image('chevron_down', p['muted'], 8)}
        width: 8px;
        height: 8px;
    }}
    QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{
        {_image('chevron_up', p['faint'], 8)}
    }}
    QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{
        {_image('chevron_down', p['faint'], 8)}
    }}
    QPlainTextEdit {{
        background: {p['card']};
        border: 1px solid {p['border']};
        border-radius: 4px;
        padding: 6px;
        font-size: 12px;
        font-family: {_MONO_FAMILY};
        selection-background-color: {p['focus']};
        selection-color: #ffffff;
    }}
    QTextEdit, QTextBrowser {{
        background: {p['card']};
        border: 1px solid {p['border']};
        border-radius: 4px;
        padding: 8px;
        selection-background-color: {p['focus']};
        selection-color: #ffffff;
    }}

    /* ---------- check boxes and radio buttons ---------- */
    QCheckBox, QRadioButton {{
        spacing: 8px;
        background: transparent;
    }}
    QCheckBox::indicator {{
        width: 14px;
        height: 14px;
        border: 1px solid {p['border_strong']};
        border-radius: 3px;
        background: {p['card']};
    }}
    QCheckBox::indicator:hover {{
        border: 1px solid {p['muted']};
    }}
    QCheckBox::indicator:checked {{
        background: {p['accent']};
        border: 1px solid {p['accent']};
        {_image('check', p['accent_ink'], 14)}
    }}
    QCheckBox::indicator:indeterminate {{
        background: {p['accent']};
        border: 1px solid {p['accent']};
        {_image('dash', p['accent_ink'], 14)}
    }}
    QCheckBox::indicator:disabled {{
        background: {p['bg_alt']};
        border: 1px solid {p['border']};
    }}
    QRadioButton::indicator {{
        width: 14px;
        height: 14px;
        border: 1px solid {p['border_strong']};
        border-radius: 8px;
        background: {p['card']};
    }}
    QRadioButton::indicator:hover {{
        border: 1px solid {p['muted']};
    }}
    QRadioButton::indicator:checked {{
        width: 6px;
        height: 6px;
        border: 5px solid {p['accent']};
        border-radius: 8px;
        background: {p['accent_ink']};
    }}
    QRadioButton::indicator:disabled {{
        background: {p['bg_alt']};
        border: 1px solid {p['border']};
    }}

    /* ---------- tables, trees, lists ---------- */
    QTableView, QTreeView, QTableWidget {{
        background: {p['card']};
        alternate-background-color: {p['row_alt']};
        border: 1px solid {p['border']};
        gridline-color: {p['border']};
        selection-background-color: {p['selection_bg']};
        selection-color: {p['ink']};
        font-size: 13px;
        outline: none;
    }}
    QListView, QListWidget {{
        background: {p['card']};
        alternate-background-color: {p['row_alt']};
        border: 1px solid {p['border']};
        selection-background-color: {p['selection_bg']};
        selection-color: {p['ink']};
        outline: none;
    }}
    QTableView::item, QTableWidget::item {{
        padding: 3px 8px;
    }}
    QListView::item, QListWidget::item, QTreeView::item {{
        padding: 4px 8px;
    }}
    QTableView::item:hover, QTreeView::item:hover, QTableWidget::item:hover,
    QListView::item:hover, QListWidget::item:hover {{
        background: {p['hover']};
    }}
    QTableView::item:selected, QTreeView::item:selected, QTableWidget::item:selected,
    QListView::item:selected, QListWidget::item:selected {{
        background: {p['selection_bg']};
        color: {p['ink']};
    }}
    QHeaderView {{
        background: {p['card']};
        border: none;
    }}
    QHeaderView::section {{
        background: {p['card']};
        color: {p['muted']};
        border: none;
        border-bottom: 1px solid {p['border_strong']};
        padding: 7px 8px;
        {data_caption}
    }}
    QHeaderView::section:hover {{
        color: {p['ink']};
    }}
    QHeaderView::section:vertical {{
        border-bottom: none;
        border-right: 1px solid {p['border']};
        padding: 3px 8px;
        font-weight: 400;
        color: {p['faint']};
    }}
    QHeaderView::up-arrow {{
        {_image('chevron_up', p['ink'], 8)}
        width: 8px;
        height: 8px;
        subcontrol-origin: padding;
        subcontrol-position: center right;
        right: 6px;
    }}
    QHeaderView::down-arrow {{
        {_image('chevron_down', p['ink'], 8)}
        width: 8px;
        height: 8px;
        subcontrol-origin: padding;
        subcontrol-position: center right;
        right: 6px;
    }}
    QTableCornerButton::section {{
        background: {p['card']};
        border: none;
        border-bottom: 1px solid {p['border_strong']};
        border-right: 1px solid {p['border']};
    }}

    /* ---------- tabs: underline, no boxes ---------- */
    QTabWidget::pane {{
        border: none;
        border-top: 1px solid {p['border']};
        background: transparent;
        top: -1px;
    }}
    QTabBar {{
        background: transparent;
        qproperty-drawBase: 0;
    }}
    QTabBar::tab {{
        background: transparent;
        color: {p['muted']};
        border: none;
        border-bottom: 2px solid transparent;
        padding: 8px 14px;
        margin: 0px 2px 0px 0px;
        font-size: 13px;
    }}
    QTabBar::tab:hover {{
        color: {p['ink']};
        border-bottom: 2px solid {p['border_strong']};
    }}
    QTabBar::tab:selected {{
        color: {p['ink']};
        border-bottom: 2px solid {p['accent']};
    }}
    QTabBar::tab:disabled {{
        color: {p['faint']};
    }}
    QTabBar QToolButton {{
        background: {p['bg']};
        border: 1px solid {p['border']};
        border-radius: 3px;
        padding: 2px;
    }}

    /* ---------- menus ---------- */
    QMenu {{
        background: {p['card']};
        border: 1px solid {p['border_strong']};
        padding: 4px;
    }}
    QMenu::item {{
        background: transparent;
        padding: 6px 28px 6px 26px;
        border-radius: 3px;
        font-size: 13px;
    }}
    QMenu::item:selected {{
        background: {p['selection_bg']};
        color: {p['ink']};
    }}
    QMenu::item:disabled {{
        color: {p['faint']};
        background: transparent;
    }}
    QMenu::separator {{
        height: 1px;
        background: {p['border']};
        margin: 4px 6px;
    }}
    QMenu::indicator {{
        width: 14px;
        height: 14px;
        left: 6px;
    }}
    QMenu::indicator:checked {{
        {_image('check', p['accent'], 14)}
    }}
    QMenu::right-arrow {{
        {_image('chevron_right', p['muted'])}
        width: 10px;
        height: 10px;
        right: 6px;
    }}

    /* ---------- scroll bars: thin and quiet ---------- */
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 0px;
        border: none;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 0px;
        border: none;
    }}
    QScrollBar::handle:vertical {{
        background: {p['scroll_handle']};
        border-radius: 3px;
        min-height: 32px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal {{
        background: {p['scroll_handle']};
        border-radius: 3px;
        min-width: 32px;
        margin: 2px;
    }}
    QScrollBar::handle:hover, QScrollBar::handle:pressed {{
        background: {p['muted']};
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{
        width: 0px;
        height: 0px;
        border: none;
        background: transparent;
    }}
    QScrollBar::add-page, QScrollBar::sub-page {{
        background: transparent;
    }}
    QAbstractScrollArea::corner {{
        background: transparent;
        border: none;
    }}
    QScrollArea {{
        background: transparent;
        border: none;
    }}

    /* ---------- splitters, group boxes, progress ---------- */
    QSplitter::handle {{
        background: transparent;
    }}
    QSplitter::handle:horizontal {{
        width: 8px;
    }}
    QSplitter::handle:vertical {{
        height: 8px;
    }}
    QSplitter::handle:hover, QSplitter::handle:pressed {{
        background: {p['border']};
    }}
    QGroupBox {{
        background: transparent;
        border: 1px solid {p['border']};
        border-radius: 4px;
        margin-top: 20px;
        padding: 12px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 0px;
        padding: 0px 0px 6px 0px;
        color: {p['muted']};
        {caption}
    }}
    QProgressBar {{
        background: {p['bg_alt']};
        border: none;
        border-radius: 3px;
        min-height: 6px;
        max-height: 6px;
        text-align: center;
        color: transparent;
    }}
    QProgressBar::chunk {{
        background: {p['focus']};
        border-radius: 3px;
    }}
    QSlider::groove:horizontal {{
        height: 4px;
        background: {p['border']};
        border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{
        background: {p['focus']};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        width: 14px;
        height: 14px;
        margin: -6px 0px;
        border-radius: 8px;
        background: {p['card']};
        border: 1px solid {p['border_strong']};
    }}
    QSlider::handle:horizontal:hover {{
        border: 1px solid {p['focus']};
    }}

    /* ---------- status bar and generic tool bars ---------- */
    QStatusBar {{
        background: {p['card']};
        border-top: 1px solid {p['border']};
        color: {p['muted']};
        font-size: 12px;
    }}
    QStatusBar::item {{
        border: none;
    }}
    QStatusBar QLabel {{
        color: {p['muted']};
        font-size: 12px;
        padding: 0px 6px;
    }}
    QToolBar {{
        background: transparent;
        border: none;
        spacing: 4px;
        padding: 2px;
    }}
    QToolBar::separator {{
        background: {p['border']};
        width: 1px;
        margin: 4px 6px;
    }}

    /* ---------- chart tools ----------
       The strip above a chart follows the theme (the chart surface under it
       stays white). Its icons are matplotlib's: black or white is chosen
       from the widget palette when the strip is built, which is why
       apply_theme() sets a QPalette; qt_widgets.PlotWidget re-tints them
       in the theme's ink on every theme change. */
    NavigationToolbar2QT {{
        background: {p['card']};
        border: none;
        border-bottom: 1px solid {p['border']};
        spacing: 2px;
        padding: 2px 6px;
    }}
    NavigationToolbar2QT QToolButton {{
        background: transparent;
        color: {p['ink']};
        border: 1px solid transparent;
        border-radius: 3px;
        padding: 3px 6px;
        font-size: 12px;
    }}
    NavigationToolbar2QT QToolButton:hover {{
        background: {p['hover']};
    }}
    NavigationToolbar2QT QToolButton:checked, NavigationToolbar2QT QToolButton:pressed {{
        background: {p['selection_bg']};
        border: 1px solid {p['accent']};
    }}
    NavigationToolbar2QT QToolButton:disabled {{
        color: {p['faint']};
    }}
    NavigationToolbar2QT QLabel {{
        background: transparent;
        color: {p['muted']};
        font-size: 11px;
    }}
    NavigationToolbar2QT::separator {{
        background: {p['border']};
        width: 1px;
        margin: 4px 6px;
    }}

    /* ---------- dialogs of the shell ---------- */
    QFrame#DialogBand {{
        background: {p['chrome']};
        border: none;
    }}
    QLabel#DialogTitle {{
        color: {p['chrome_ink']};
        font-size: 17px;
        font-weight: 600;
    }}
    QLabel#DialogSubtitle {{
        color: {p['chrome_muted']};
        font-size: 12px;
    }}
    QFrame#OptionRow {{
        background: {p['card']};
        border: 1px solid {p['border']};
        border-radius: 4px;
    }}
    QFrame#OptionRow:hover {{
        border: 1px solid {p['border_strong']};
        background: {p['hover']};
    }}
    QPushButton#OptionTitle {{
        background: transparent;
        border: none;
        padding: 0px;
        text-align: left;
        font-size: 13px;
        font-weight: 600;
        color: {p['ink']};
    }}
    QPushButton#OptionTitle:hover {{
        color: {p['link']};
        background: transparent;
    }}
    QPushButton#OptionTitle:focus {{
        color: {p['link']};
        border: none;
    }}
    QLabel#OptionNote {{
        font-size: 12px;
        color: {p['muted']};
        background: transparent;
    }}
    """


def apply_theme(app, palette: dict = PALETTE, *, dark: bool = False) -> None:
    """Style, QPalette and stylesheet, together. The palette comes first so
    widgets built afterwards (and libraries that read it once, at
    construction) see the right colours."""
    global _ACTIVE
    active = DARK_PALETTE if dark else palette
    _ACTIVE = active
    try:
        style = app.style()
        if style is None or style.objectName().lower() != "fusion":
            # One predictable base under the stylesheet on every platform,
            # and one that paints from the QPalette set below.
            app.setStyle("Fusion")
        app.setPalette(build_qpalette(active))
    except Exception:  # pragma: no cover - a palette problem must not stop the app
        pass
    app.setStyleSheet(build_stylesheet(active))
    if _NOTIFIER is not None:
        _NOTIFIER.changed.emit()
