"""
composer_store.py — shared state for the Figure Composer workspace: a
single in-memory list of captured plot "recipes" (the plot_helpers render
function plus the exact arguments used to call it), mutated by the "->
Figure Composer" button on every PlotWidget's toolbar and read by the
Figure Composer page itself.

Recipes, not snapshots: each captured item remembers *how* to redraw its
plot rather than a frozen raster image of it. figure_composer.compose_figure
replays that call into its own subfigure at compose time, so every panel in
the combined figure stays a real, live, vector matplotlib Axes -- editable
(title, axis labels, any of the recipe's own keyword parameters) and sharp
at any export size -- instead of a fixed-resolution picture of what the
plot looked like at capture time.

Kept as its own tiny module (rather than living in qt_widgets.py or
qt_figure_composer.py) specifically to avoid a circular import: PlotWidget
(in qt_widgets.py) needs to push into this store, and the Figure Composer
page (in qt_figure_composer.py) needs to both read it AND import
PlotWidget for its own preview pane -- putting the store in either of
those two files would make the other import it back.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Signal

_uid_counter = itertools.count()

# The item_edited field vocabulary for text overrides -- one definition
# shared by the setters below and the qt page's edit routing, so a future
# override field is added in one place.
TEXT_OVERRIDE_FIELDS = ("title", "xlabel", "ylabel")


@dataclass(frozen=True)
class ComposerItem:
    render_fn: Callable
    render_args: Tuple[Any, ...]
    render_kwargs: Dict[str, Any]
    caption: str
    source: str = ""
    added_at: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    # Stable across replace()-based edits below (those only ever override
    # OTHER fields), unlike object identity -- lets a thumbnail cache tell
    # "same panel, cosmetic edit" apart from "genuinely different content"
    # without needing render_args/render_kwargs (which may hold DataFrames,
    # not usable as a plain dict key) to be hashable.
    uid: int = field(default_factory=lambda: next(_uid_counter))
    # User overrides layered on top of the recipe at render time. None means
    # "use whatever the recipe itself produces"; "" (title/axis labels
    # only) means "force blank". kwarg_overrides replaces individual keys
    # of render_kwargs without losing the originally-captured values, so
    # "reset to original" is just clearing the dict back to {}.
    title_override: Optional[str] = None
    xlabel_override: Optional[str] = None
    ylabel_override: Optional[str] = None
    kwarg_overrides: Dict[str, Any] = field(default_factory=dict)

    def effective_kwargs(self) -> Dict[str, Any]:
        return {**self.render_kwargs, **self.kwarg_overrides}


class ComposerStore(QObject):
    """Two signals with deliberately different costs downstream:

    - `changed` fires only on STRUCTURAL mutations (add/remove/move/clear)
      -- the composed grid's shape is stale, the page must rebuild it.
    - `item_edited(index, field)` fires on per-item edits (caption, text
      overrides, kwarg overrides) -- the grid's shape is still valid, only
      the one panel (or, for a caption, only its gallery row) needs
      touching. This split is what lets a title edit cost ~zero re-renders
      instead of replaying every panel (the "9 renders per keystroke"
      pipeline this replaced routed EVERY mutation through one
      undifferentiated `changed`).

    Every setter also early-returns without emitting when the new value
    equals the old -- editingFinished-style Qt signals re-fire on focus
    changes with unchanged text, and those must stay free."""

    changed = Signal()
    item_edited = Signal(int, str)  # (index, field): caption|title|xlabel|ylabel|kwargs

    def __init__(self) -> None:
        super().__init__()
        self._items: List[ComposerItem] = []

    def items(self) -> List[ComposerItem]:
        return list(self._items)

    def add_recipe(
        self, render_fn: Callable, render_args: Tuple[Any, ...], render_kwargs: Dict[str, Any],
        caption: str, source: str = "",
    ) -> None:
        # Deep-copy so a later reload/rerun elsewhere in the app that
        # reassigns the source DataFrames can't change what an
        # already-captured panel replays.
        item = ComposerItem(
            render_fn=render_fn, render_args=copy.deepcopy(render_args),
            render_kwargs=copy.deepcopy(render_kwargs), caption=caption, source=source,
        )
        self._items.append(item)
        self.changed.emit()

    def remove(self, index: int) -> None:
        if 0 <= index < len(self._items):
            del self._items[index]
            self.changed.emit()

    def move(self, index: int, delta: int) -> None:
        new_index = index + delta
        if 0 <= index < len(self._items) and 0 <= new_index < len(self._items):
            self._items[index], self._items[new_index] = self._items[new_index], self._items[index]
            self.changed.emit()

    def _edit_field(self, index: int, attr: str, value: Any, field: str) -> None:
        """Shared guard/replace/emit body for every per-item field setter --
        one place for the bounds check, the no-op suppression, and the
        item_edited emission, so the four public setters can't drift."""
        if 0 <= index < len(self._items) and getattr(self._items[index], attr) != value:
            self._items[index] = replace(self._items[index], **{attr: value})
            self.item_edited.emit(index, field)

    def set_caption(self, index: int, caption: str) -> None:
        self._edit_field(index, "caption", caption, "caption")

    def set_title_override(self, index: int, title: Optional[str]) -> None:
        self._edit_field(index, "title_override", title, "title")

    def set_xlabel_override(self, index: int, xlabel: Optional[str]) -> None:
        self._edit_field(index, "xlabel_override", xlabel, "xlabel")

    def set_ylabel_override(self, index: int, ylabel: Optional[str]) -> None:
        self._edit_field(index, "ylabel_override", ylabel, "ylabel")

    def set_kwarg_override(self, index: int, key: str, value: Any) -> None:
        if not (0 <= index < len(self._items)):
            return
        current = self._items[index].kwarg_overrides
        if key in current and current[key] == value:
            return
        overrides = dict(current)
        overrides[key] = value
        self._items[index] = replace(self._items[index], kwarg_overrides=overrides)
        self.item_edited.emit(index, "kwargs")

    def clear_kwarg_overrides(self, index: int) -> None:
        if 0 <= index < len(self._items) and self._items[index].kwarg_overrides:
            self._items[index] = replace(self._items[index], kwarg_overrides={})
            self.item_edited.emit(index, "kwargs")

    def clear(self) -> None:
        if self._items:
            self._items.clear()
            self.changed.emit()


# Module-level singleton: every PlotWidget's "-> Figure Composer" button and
# the Figure Composer page itself share this one instance.
store = ComposerStore()
