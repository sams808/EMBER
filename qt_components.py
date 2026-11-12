"""
qt_components.py -- the small presentational building blocks every
workspace shares, so pages read as one product: a page header, a stat
tile, an empty state, a calm notes banner and a tag.

These widgets carry no data logic. Their look comes from qt_theme's
stylesheet through the object names set here (PageHeader, PageTitle,
PageSubtitle, StatTile, StatValue, StatLabel, StatHint, EmptyState,
EmptyTitle, EmptyHint, NotesBanner, NotesSummary, NotesDetail, NotesToggle,
Pill), never from inline styles, so light and dark mode stay in one place.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QCompleter, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy, QToolButton,
    QVBoxLayout, QWidget,
)


def wrapped_label(text: str = "", object_name: str = "SectionNote") -> QLabel:
    """A word-wrapping label that never widens its window.

    A plain QLabel reports the full length of its text as its minimum
    width, so one long sentence can push the main window past the screen.
    Wrapping plus an Ignored horizontal policy lets the layout decide.
    """
    label = QLabel(text)
    label.setObjectName(object_name)
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def field_label(text: str) -> QLabel:
    """The caption in front of a control in a page's control row."""
    label = QLabel(text)
    label.setObjectName("FieldLabel")
    return label


def control_row(spacing: int = 8) -> tuple:
    """A page's control row: (widget styled as `Toolbar`, its horizontal layout)."""
    row = QWidget()
    row.setObjectName("Toolbar")
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    return row, layout


def secondary_button(text: str, tooltip: str = "") -> QPushButton:
    """A quiet button: an export, a reset, anything that is not the page's one primary action."""
    button = QPushButton(text)
    button.setObjectName("Secondary")
    if tooltip:
        button.setToolTip(tooltip)
    return button


def set_completions(edit: QLineEdit, items: Iterable[str]) -> None:
    """Offer `items` as the field is typed in (case-insensitive, matching anywhere).

    Replaces the previous list: call it again when the dataset changes.
    """
    completer = QCompleter(sorted({str(item) for item in items if str(item)}, key=str.lower), edit)
    completer.setCaseSensitivity(Qt.CaseInsensitive)
    completer.setFilterMode(Qt.MatchContains)
    completer.setCompletionMode(QCompleter.PopupCompletion)
    edit.setCompleter(completer)


def dataset_values(dataset, column: str) -> list:
    """The distinct non-null values of one column of a loaded dataset, as text (empty when unavailable)."""
    try:
        if dataset is None or not dataset.is_loaded():
            return []
        df = dataset.require_df()
        if column not in df.columns:
            return []
        return [str(value) for value in df.get_column(column).drop_nulls().unique().to_list()]
    except Exception:
        return []


class ElidedLabel(QLabel):
    """A one-line label that shortens with an ellipsis instead of widening its parent.

    text() always returns the full text; only the painting is shortened,
    and the full text is then offered as the tooltip.
    """

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(0, super().minimumSizeHint().height())

    def is_elided(self) -> bool:
        return self.fontMetrics().horizontalAdvance(self.text()) > self.contentsRect().width()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        rect = self.contentsRect()
        full = self.text()
        shown = self.fontMetrics().elidedText(full, Qt.ElideRight, rect.width())
        self.setToolTip(full if shown != full else "")
        self.style().drawItemText(
            painter, rect, int(self.alignment()), self.palette(), self.isEnabled(), shown, self.foregroundRole(),
        )


class PageHeader(QWidget):
    """Title, one-line description and a right-aligned action area."""

    def __init__(self, title: str, subtitle: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("PageHeader")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(2)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("PageTitle")
        self.subtitle_label = wrapped_label(subtitle, "PageSubtitle")
        text.addWidget(self.title_label)
        text.addWidget(self.subtitle_label)
        row.addLayout(text, 1)
        # Only once the label has a parent: showing a parentless widget
        # creates a top-level window, which costs 0.1 s and can flash.
        self.subtitle_label.setVisible(bool(subtitle))
        self._actions = QHBoxLayout()
        self._actions.setContentsMargins(0, 0, 0, 0)
        self._actions.setSpacing(8)
        row.addLayout(self._actions, 0)

    def set_title(self, title: str) -> None:
        self.title_label.setText(title)

    def set_subtitle(self, subtitle: str) -> None:
        self.subtitle_label.setText(subtitle)
        self.subtitle_label.setVisible(bool(subtitle))

    def add_action_widget(self, widget: QWidget) -> QWidget:
        """Put a button (export, help...) at the right of the header."""
        self._actions.addWidget(widget, 0, Qt.AlignVCenter)
        return widget


class StatTile(QFrame):
    """One headline number with its label, and an optional second line."""

    clicked = Signal()

    def __init__(self, label: str, value: str = "—", hint: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatTile")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(1)
        self.label_label = ElidedLabel(label)
        self.label_label.setObjectName("StatLabel")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("StatValue")
        self.value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.hint_label = ElidedLabel(hint)
        self.hint_label.setObjectName("StatHint")
        layout.addWidget(self.label_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.hint_label)
        self.hint_label.setVisible(bool(hint))        # after it has a parent (see PageHeader)

    def set_value(self, value: str, hint: Optional[str] = None) -> None:
        self.value_label.setText(value)
        if hint is not None:
            self.hint_label.setText(hint)
            self.hint_label.setVisible(bool(hint))

    def value(self) -> str:
        return self.value_label.text()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class EmptyState(QWidget):
    """What a pane shows before it has a result: what will appear, and how to get it."""

    def __init__(self, title: str, hint: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("EmptyState")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(6)
        layout.addStretch(1)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("EmptyTitle")
        self.title_label.setAlignment(Qt.AlignHCenter)
        self.hint_label = wrapped_label(hint, "EmptyHint")
        self.hint_label.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self.title_label)
        layout.addWidget(self.hint_label)
        self.hint_label.setVisible(bool(hint))        # after it has a parent (see PageHeader)
        layout.addStretch(1)

    def set_text(self, title: str, hint: str = "") -> None:
        self.title_label.setText(title)
        self.hint_label.setText(hint)
        self.hint_label.setVisible(bool(hint))


def notes_headline(count: int, subject: str = "this result") -> Optional[str]:
    """The one line shown for several notes (None for one note: it is shown in full)."""
    if count <= 1:
        return None
    return f"{count} notes qualify {subject}: what was set aside, converted or left out, and why."


class NotesBanner(QFrame):
    """Notes that qualify a result, in one calm line with the detail one click away.

    A result can carry a dozen disclosures. Shown in full above the result
    they read as an alarm and push the content down; hidden they are a
    silent caveat. The banner states how many apply and names the first,
    and expands in place to the full list. With no notes it hides itself.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("NotesBanner")
        self._notes: list[str] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(4)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)
        self.summary_label = wrapped_label("", "NotesSummary")
        self.toggle = QToolButton()
        self.toggle.setObjectName("NotesToggle")
        self.toggle.setCheckable(True)
        self.toggle.setText("Show all")
        self.toggle.toggled.connect(self._on_toggled)
        top.addWidget(self.summary_label, 1)
        top.addWidget(self.toggle, 0, Qt.AlignTop)
        outer.addLayout(top)
        self.detail_label = wrapped_label("", "NotesDetail")
        self.detail_label.setVisible(False)
        outer.addWidget(self.detail_label)
        self.setVisible(False)

    def set_notes(self, notes: Iterable[str], headline: Optional[str] = None) -> None:
        """notes: full sentences. headline: the one line to show; by default the first note."""
        self._notes = [str(n).strip() for n in notes if str(n).strip()]
        count = len(self._notes)
        if not count:
            self.setVisible(False)
            return
        first = headline or self._notes[0]
        more = "" if count == 1 or headline else f"   (+{count - 1} more)"
        self.summary_label.setText(f"{first}{more}")
        self.detail_label.setText("\n".join(f"• {note}" for note in self._notes))
        self.toggle.setVisible(count > 1 or bool(headline))
        self.toggle.setChecked(False)
        self.setToolTip("\n\n".join(self._notes))
        self.setVisible(True)

    def set_result_notes(self, notes: Iterable[str], subject: str = "this result") -> None:
        """The house wording for notes that qualify a computed result: one
        note is shown in full; several are announced by their count, with
        the full list one click away."""
        notes = [str(n).strip() for n in notes if str(n).strip()]
        self.set_notes(notes, headline=notes_headline(len(notes), subject))

    def notes(self) -> Sequence[str]:
        return tuple(self._notes)

    def text(self) -> str:
        """Every note, one per line (what an export or a test reads)."""
        return "\n".join(self._notes)

    def _on_toggled(self, checked: bool) -> None:
        self.detail_label.setVisible(checked)
        self.toggle.setText("Hide" if checked else "Show all")


class Pill(QLabel):
    """A small rounded tag: a dataset stamp, a unit, a count."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("Pill")
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
