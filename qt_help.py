"""
qt_help.py — in-app Help: About dialog, the full detailed Notice (rendered
from NOTICE.md via QTextBrowser.setMarkdown -- Qt has native Markdown
support, replacing the old app's ~230-line hand-rolled renderer), and
shared app identity constants.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget,
)

APP_NAME = "Ember"                 # settings key, window title, running text
APP_DISPLAY_NAME = "EMBER"         # the wordmark of the header and the splash
APP_VERSION = "2.0.0"
APP_SUBTITLE = "Hanford tank waste explorer"
APP_TAGLINE = "Hanford tank composition explorer & vitrification screening"


def asset_path(name: str) -> str:
    """Path to a bundled asset (assets/ next to the code, or the PyInstaller
    bundle dir when frozen)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "assets", name)


def notice_path() -> str:
    """Path to NOTICE.md (bundled at the PyInstaller bundle root via
    `--add-data "NOTICE.md;."`, next to the code otherwise)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "NOTICE.md")


def load_notice_markdown() -> str:
    try:
        with open(notice_path(), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return "# Notice\n\nNOTICE.md could not be found next to the application."


def load_notice_section(section_number: int) -> str:
    """Extracts just one numbered "## N. Title" section (through the next
    "## " heading, or end of file) out of the full NOTICE.md -- lets a
    workspace's "?" help button open only its own relevant manual section
    instead of the entire document every time."""
    text = load_notice_markdown()
    start_match = re.search(rf"^## {section_number}\. .*$", text, re.MULTILINE)
    if start_match is None:
        return f"# Section {section_number}\n\nNot found in NOTICE.md."
    next_match = re.search(r"^## \d+\. ", text[start_match.end():], re.MULTILINE)
    end = start_match.end() + next_match.start() if next_match else len(text)
    return text[start_match.start():end].strip()


def show_workspace_help(parent: Optional[QWidget], section_number: int, workspace_title: str) -> None:
    """Opens the per-workspace "?" help dialog, scoped to that workspace's
    own NOTICE.md section -- mirrors PHOENIX's per-application contextual
    help icon (hover/click an app tile for just its own description)
    rather than always dropping the user into the full manual."""
    markdown = load_notice_section(section_number)
    HelpDialog(parent, markdown=markdown, title=f"{workspace_title} — Help").exec()


def make_help_button(parent: QWidget, section_number: int, workspace_title: str) -> QPushButton:
    """The per-workspace "?" button, built in ONE place -- every workspace
    header adds one of these, and a copy-pasted five-line block at each
    site is exactly how a tooltip rewording or a signature change drifts
    silently across 13 call sites (review finding)."""
    button = QPushButton("?", parent)
    button.setObjectName("HelpButton")
    button.setToolTip("What is this workspace for?")
    button.clicked.connect(lambda: show_workspace_help(parent, section_number, workspace_title))
    return button


ABOUT_HTML = f"""
<p class="lead">{APP_TAGLINE}.</p>

<p>Developed by <b>Sami Soudani</b>, in the <b>McCloy</b> group at
Washington State University, with support from the
U.S.&nbsp;Department&nbsp;of&nbsp;Energy&nbsp;(DOE).</p>
<p><a href="https://github.com/sams808/EMBER">github.com/sams808/EMBER</a>
&nbsp;&middot;&nbsp; MIT&nbsp;License &nbsp;&middot;&nbsp; version {APP_VERSION}</p>

<p>Explore Hanford tank composition data by element and analyte, build
tank&times;element heatmaps and correlation/association maps, screen
tank compositions for vitrification (glass immobilization), including
element&rarr;oxide conversion and network-structure estimates, and combine
any of the resulting plots into one labeled multi-panel figure with Figure
Composer. See Help &rarr; User guide for the full manual, including every
formula behind the vitrification screening tools.</p>

<h3>Data source</h3>
<p>The bundled composition data (<code>Hanford.csv</code>, <code>Tank_attributes.csv</code>) comes from
PNNL's <a href="https://phoenix.pnnl.gov">PHOENIX</a> (Hanford Online Information Exchange), the access
mechanism for Tri-Party Agreement tank waste databases:</p>
<p class="small">
Brulotte, P.J., and Christensen, K.C.. "Tri-Party Agreement databases, access mechanism and
procedures". United States. doi:10.2172/10112540. https://www.osti.gov/servlets/purl/10112540<br><br>
"PNNL Hanford Online Information Exchange (PHOENIX)", Pacific Northwest National Laboratory,
Richland WA, U.S. Department of Energy. https://phoenix.pnnl.gov
</p>
<p>{APP_NAME} (this application) is an independent project, not produced by, affiliated with, or
endorsed by PNNL or the U.S. Department of Energy.</p>

<h3>Built on</h3>
<ul>
  <li><b>polars</b> &mdash; DataFrame engine, <a href="https://github.com/pola-rs/polars">pola-rs/polars</a></li>
  <li><b>pandas</b> &mdash; McKinney, <i>Proc.&nbsp;9th&nbsp;Python in Science Conf.</i>, 56&ndash;61 (2010)</li>
  <li><b>NumPy</b> &mdash; Harris et&nbsp;al., <i>Nature</i> <b>585</b>, 357 (2020)</li>
  <li><b>Matplotlib</b> &mdash; Hunter, <i>Comput.&nbsp;Sci.&nbsp;Eng.</i> <b>9</b>, 90 (2007)</li>
  <li><b>seaborn</b> &mdash; Waskom, <i>J.&nbsp;Open Source Softw.</i> <b>6</b>(60), 3021 (2021)</li>
  <li><b>SciPy</b> &mdash; Virtanen et&nbsp;al., <i>Nature Methods</i> <b>17</b>, 261 (2020)</li>
  <li><b>scikit-learn</b> &mdash; Pedregosa et&nbsp;al., <i>J.&nbsp;Mach.&nbsp;Learn.&nbsp;Res.</i> <b>12</b>, 2825&ndash;2830 (2011)</li>
  <li><b>NetworkX</b> &mdash; Hagberg, Schult &amp; Swart, <i>Proc.&nbsp;7th&nbsp;Python in Science Conf.</i> (SciPy2008), 11&ndash;15</li>
  <li><b>Plotly</b> &mdash; interactive HTML export, Plotly Technologies Inc.</li>
  <li><b>xraydb</b> &mdash; Newville, X-ray reference data in Python, <a href="https://github.com/xraypy/XrayDB">xraypy/XrayDB</a></li>
  <li><b>PySide6 / Qt&nbsp;for&nbsp;Python</b> (The Qt Company) &mdash; the interface</li>
</ul>
"""


def document_css(palette: Optional[dict] = None) -> str:
    """The style of rich text shown in a help dialog, from the active
    palette, so About reads the same in the light and the dark theme (the
    HTML itself carries no colour)."""
    if palette is None:
        from qt_theme import current_palette
        palette = current_palette()
    p = palette
    return f"""
    body {{ color: {p['ink']}; font-size: 13px; }}
    p {{ margin-top: 0px; margin-bottom: 10px; }}
    p.lead {{ font-size: 14px; color: {p['muted']}; }}
    p.small {{ font-size: 12px; color: {p['muted']}; }}
    h1 {{ font-size: 20px; font-weight: 600; color: {p['ink']}; }}
    h2 {{ font-size: 16px; font-weight: 600; color: {p['ink']}; }}
    h3 {{ font-size: 14px; font-weight: 600; color: {p['ink']}; margin-top: 16px; margin-bottom: 4px; }}
    a {{ color: {p['link']}; text-decoration: none; }}
    code {{ font-family: Consolas, monospace; color: {p['ink']}; }}
    li {{ margin-bottom: 3px; }}
    """


class _DialogBand(QFrame):
    """The navy band at the top of the shell's dialogs: the product mark,
    a title and an optional second line."""

    def __init__(self, title: str, subtitle: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("DialogBand")
        row = QHBoxLayout(self)
        row.setContentsMargins(20, 14, 20, 14)
        row.setSpacing(12)
        self.mark = QLabel()
        try:
            from qt_icons import ember_mark_pixmap
            self.mark.setPixmap(ember_mark_pixmap(34))
        except Exception:  # the band is still a band without its mark
            pass
        self.mark.setFixedSize(34, 34)
        row.addWidget(self.mark, 0, Qt.AlignVCenter)
        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(1)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("DialogTitle")
        text.addWidget(self.title_label)
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("DialogSubtitle")
        self.subtitle_label.setVisible(bool(subtitle))
        text.addWidget(self.subtitle_label)
        row.addLayout(text, 1)


class HelpDialog(QDialog):
    def __init__(
        self, parent: Optional[QWidget] = None, *, html: str = ABOUT_HTML, markdown: Optional[str] = None,
        title: str = "About Ember", width: int = 640, height: int = 520, subtitle: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(width, height)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.band = _DialogBand(title, subtitle, self)
        layout.addWidget(self.band)

        body = QVBoxLayout()
        body.setContentsMargins(20, 16, 20, 16)
        body.setSpacing(12)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.document().setDocumentMargin(10)
        if markdown is not None:
            self.browser.setMarkdown(markdown)
        else:
            self.browser.document().setDefaultStyleSheet(document_css())
            self.browser.setHtml(html)
        body.addWidget(self.browser, 1)
        footer = QHBoxLayout()
        footer.addStretch(1)
        self.close_btn = QPushButton("Close")
        self.close_btn.setDefault(True)
        self.close_btn.clicked.connect(self.accept)
        footer.addWidget(self.close_btn)
        body.addLayout(footer)
        layout.addLayout(body, 1)


# (button label, one-line plain-language explanation, target workspace --
# must match one of qt_shell.NAV_ITEMS' string values exactly; kept as
# plain strings rather than importing the NAV_* constants to avoid a
# circular import, since qt_shell.py is what imports THIS module).
GETTING_STARTED_OPTIONS = [
    ("Ask a combined question",
     "Any element, analyte or basket, on any unit basis, grouped the way the question needs.", "Query"),
    ("Explore a specific tank",
     "The full composition of one tank, or of several tanks side by side.", "Tank Explorer"),
    ("Find which elements move together",
     "Correlations and associations across the whole inventory.", "Correlations"),
    ("Screen tanks for vitrification",
     "Oxide chemistry, candidate search, blend partners and glass loading rules.", "Vitrification"),
    ("Build a publication figure",
     "Combine plots from any workspace into one labeled multi-panel figure.", "Figure Composer"),
    ("Just show me everything",
     "The site at a glance: headline figures, the tank-farm map and the dataset audit tables.", "Overview"),
]


class _OptionRow(QFrame):
    """One choice of the Getting Started dialog: an icon, the choice (a
    real button, so it works from the keyboard) and one line about it.
    A click anywhere on the row counts as a click on the button."""

    def __init__(self, label: str, explanation: str, target: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("OptionRow")
        self.setCursor(Qt.PointingHandCursor)
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 9, 14, 9)
        row.setSpacing(12)
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(22, 22)
        self._icon_name = ""
        try:
            from qt_icons import WORKSPACE_ICONS
            self._icon_name = WORKSPACE_ICONS.get(target, "")
        except Exception:
            pass
        row.addWidget(self.icon_label, 0, Qt.AlignVCenter)
        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(1)
        self.button = QPushButton(label)
        self.button.setObjectName("OptionTitle")
        self.button.setToolTip(explanation)
        self.button.setCursor(Qt.PointingHandCursor)
        text.addWidget(self.button)
        self.note = QLabel(explanation)
        self.note.setObjectName("OptionNote")
        self.note.setWordWrap(True)
        text.addWidget(self.note)
        row.addLayout(text, 1)
        self.chevron = QLabel()
        self.chevron.setFixedSize(16, 16)
        row.addWidget(self.chevron, 0, Qt.AlignVCenter)
        self._paint_icons()

    def _paint_icons(self) -> None:
        try:
            from qt_icons import icon_pixmap
            from qt_theme import current_palette
            palette = current_palette()
            if self._icon_name:
                self.icon_label.setPixmap(icon_pixmap(self._icon_name, palette["accent"], 22))
            self.chevron.setPixmap(icon_pixmap("chevron_right", palette["faint"], 16))
        except Exception:
            pass

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() == Qt.LeftButton and self.rect().contains(event.position().toPoint()):
            self.button.click()
        super().mouseReleaseEvent(event)


class GettingStartedDialog(QDialog):
    """A "what would you like to do?" picker shown on first launch
    (persisted via QSettings so it doesn't repeat every session -- see
    qt_shell.py's _maybe_show_getting_started) and reachable any time from
    the Help menu: one step, one row per starting point."""

    workspace_chosen = Signal(str)  # emits a GETTING_STARTED_OPTIONS target string

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Getting Started")
        self.resize(540, 620)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(_DialogBand(f"Welcome to {APP_DISPLAY_NAME}", APP_SUBTITLE, self))

        body = QVBoxLayout()
        body.setContentsMargins(20, 16, 20, 16)
        body.setSpacing(8)
        title = QLabel("What would you like to do?")
        title.setObjectName("SectionTitle")
        body.addWidget(title)
        note = QLabel("Choose a starting point. Every workspace stays one click away in the sidebar.")
        note.setWordWrap(True)
        note.setObjectName("SectionNote")
        body.addWidget(note)
        body.addSpacing(4)

        for option_label, explanation, target in GETTING_STARTED_OPTIONS:
            option = _OptionRow(option_label, explanation, target, self)
            option.button.clicked.connect(lambda checked=False, t=target: self._choose(t))
            body.addWidget(option)

        body.addStretch(1)
        footer = QHBoxLayout()
        hint = QLabel("Help → Getting Started reopens this window.")
        hint.setObjectName("SectionNote")
        footer.addWidget(hint)
        footer.addStretch(1)
        skip_btn = QPushButton("Skip for now")
        skip_btn.setObjectName("Secondary")
        skip_btn.clicked.connect(self.reject)
        footer.addWidget(skip_btn)
        body.addLayout(footer)
        layout.addLayout(body, 1)

    def _choose(self, target: str) -> None:
        self.workspace_chosen.emit(target)
        self.accept()
