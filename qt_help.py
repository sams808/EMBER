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

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget

APP_NAME = "Ember"
APP_VERSION = "0.1.0"
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
<h2 style="margin-bottom:2px;">{APP_NAME} <span style="font-size:14px; color:#8a9099; font-weight:normal;">v{APP_VERSION}</span></h2>
<p style="color:#4a4f58; margin-top:0;"><i>{APP_TAGLINE}</i></p>

<p>Developed by <b>Sam Soudani</b>, in the <b>McCloy</b> group at
Washington State University, with support from the
U.S.&nbsp;Department&nbsp;of&nbsp;Energy&nbsp;(DOE).</p>
<p><a href="https://github.com/sams808/EMBER">github.com/sams808/EMBER</a>
&nbsp;&middot;&nbsp; MIT&nbsp;License</p>

<p>Explore Hanford tank composition data by element and analyte, build
tank&times;element heatmaps and correlation/association maps, screen
tank compositions for vitrification (glass immobilization) -- including
element&rarr;oxide conversion and network-structure estimates -- and combine
any of the resulting plots into one labeled multi-panel figure with Figure
Composer. See Help &rarr; Notice for the full manual, including every
formula behind the vitrification screening tools.</p>

<h3 style="margin-bottom:2px;">Data source</h3>
<p style="margin-top:2px;">The bundled composition data (<code>Hanford.csv</code>, <code>Tank_attributes.csv</code>) comes from
PNNL's <a href="https://phoenix.pnnl.gov">PHOENIX</a> (Hanford Online Information Exchange), the access
mechanism for Tri-Party Agreement tank waste databases:</p>
<p style="font-size:10pt; color:#666">
Brulotte, P.J., and Christensen, K.C.. "Tri-Party Agreement databases, access mechanism and
procedures". United States. doi:10.2172/10112540. https://www.osti.gov/servlets/purl/10112540<br><br>
"PNNL Hanford Online Information Exchange (PHOENIX)", Pacific Northwest National Laboratory,
Richland WA, U.S. Department of Energy. https://phoenix.pnnl.gov
</p>
<p>{APP_NAME} (this application) is an independent project, not produced by, affiliated with, or
endorsed by PNNL or the U.S. Department of Energy.</p>

<h3 style="margin-bottom:2px;">Built on</h3>
<ul style="margin-top:2px;">
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


class HelpDialog(QDialog):
    def __init__(
        self, parent: Optional[QWidget] = None, *, html: str = ABOUT_HTML, markdown: Optional[str] = None,
        title: str = "About Ember", width: int = 640, height: int = 520,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(width, height)
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        if markdown is not None:
            self.browser.setMarkdown(markdown)
        else:
            self.browser.setHtml(html)
        layout.addWidget(self.browser)


# (button label, one-line plain-language explanation, target workspace --
# must match one of qt_shell.NAV_ITEMS' string values exactly; kept as
# plain strings rather than importing the NAV_* constants to avoid a
# circular import, since qt_shell.py is what imports THIS module).
GETTING_STARTED_OPTIONS = [
    ("Ask a combined question", "Any element/analyte/basket, any unit basis, grouped however you want -- one query surface.", "Query"),
    ("Explore a specific tank", "See a tank's full composition -- one tank, or several side by side.", "Tank Explorer"),
    ("Find which elements move together", "Statistical correlations and associations across the whole dataset.", "Correlations"),
    ("Screen tanks for vitrification", "Oxide chemistry, candidate search, blend partners.", "Vitrification"),
    ("Build a publication figure", "Combine plots from any workspace into one labeled multi-panel figure.", "Figure Composer"),
    ("Just show me everything", "Dataset audit: units, top elements, waste-phase/type/farm breakdowns.", "Overview"),
]


class GettingStartedDialog(QDialog):
    """A "what are you trying to do?" picker shown on first launch
    (persisted via QSettings so it doesn't repeat every session -- see
    qt_shell.py's _restore_settings) and reachable any time from the Help
    menu. Ports PHOENIX's "Which Application Should I Use?" wizard concept,
    scaled down to Ember's much smaller surface (9 workspaces, not dozens
    of separate applications) as a single picker step instead of PHOENIX's
    4-step Topic/Feature/Task/Application funnel."""

    workspace_chosen = Signal(str)  # emits a GETTING_STARTED_OPTIONS target string

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Getting Started")
        self.resize(460, 380)
        layout = QVBoxLayout(self)

        title = QLabel("What are you trying to do?")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        note = QLabel("Pick one to jump straight to the right workspace -- you can always switch from the nav rail afterward.")
        note.setWordWrap(True)
        note.setObjectName("SectionNote")
        layout.addWidget(note)
        layout.addSpacing(8)

        for option_label, explanation, target in GETTING_STARTED_OPTIONS:
            btn = QPushButton(option_label)
            btn.setToolTip(explanation)
            btn.clicked.connect(lambda checked=False, t=target: self._choose(t))
            layout.addWidget(btn)
            sub = QLabel(explanation)
            sub.setWordWrap(True)
            sub.setObjectName("SectionNote")
            layout.addWidget(sub)

        layout.addStretch(1)
        skip_btn = QPushButton("Skip -- I'll explore on my own")
        skip_btn.clicked.connect(self.reject)
        layout.addWidget(skip_btn)

    def _choose(self, target: str) -> None:
        self.workspace_chosen.emit(target)
        self.accept()
