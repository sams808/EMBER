"""
query_plots.py — the L1→L2 bridge.

Turning a QueryResult into a picture involves real policy: which of the
two returned tables backs the chart (the grouped `table`, or the
per-member split), which grouping dimension becomes the bar axis versus
the color series, and which (mark, grouping) combinations simply cannot
be drawn honestly. That policy belongs in ONE place — not inside the
Query workspace, where every workspace migrating onto the engine would
have to copy it and each copy could disagree about, say, which table a
stacked bar draws.

It also has to be reconstructable: exports record the PlotSpec JSON, and
a spec with `y="Member"` is only replayable if the replayer knows the
same table-selection rule. `plan_plot()` is that rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd

from plot_grammar import PlotSpec
from query_engine import QueryResult

# The table a plan draws from, recorded so a saved plan can be replayed
# against a re-run query (Stage 3) without re-deriving the rule.
SOURCE_TABLE = "table"
SOURCE_MEMBER_TABLE = "member_table"


@dataclass(frozen=True)
class PlotPlan:
    """Either something to draw (`spec` + `data`), or a `refusal` saying
    in plain language why this mark can't answer this shape of question.
    A refusal is a normal outcome shown on the panel, not an error."""

    data: Optional[pd.DataFrame] = None
    spec: Optional[PlotSpec] = None
    refusal: str = ""
    source: str = SOURCE_TABLE

    @property
    def is_drawable(self) -> bool:
        return self.data is not None and self.spec is not None


def plan_plot(
    result: QueryResult,
    group_by: Sequence[str],
    mark: str,
    *,
    subject_label: str,
    subtitle: str = "",
) -> PlotPlan:
    """Map (result × grouping × mark) onto a table and a PlotSpec.

    - No grouping: the informative picture is the per-member split, so
      bars run over `Member` (from member_table). Other marks have no
      second axis to use and are refused.
    - One dimension: bars run over it (from table). Stacked bars and
      heatmaps use the per-member split *within* each group, so basket
      members become the series/columns (from member_table).
    - Two dimensions: dimension 1 is the bar axis and dimension 2 the
      color series (bars/heatmap); box plots the distribution of the
      dimension-1 values within each dimension-2 level.
    """
    dims: List[str] = list(group_by)
    common = dict(
        x=result.value_column,
        title=f"{subject_label} — {result.basis}",
        subtitle=subtitle,
        # Mass/activity totals span orders of magnitude across tanks;
        # concentrations don't.
        log_x=result.basis in ("g", "kg", "Ci"),
    )

    if not dims:
        if mark != "bar":
            return PlotPlan(refusal="Heatmap, stacked bars, and box need a grouping dimension.")
        return PlotPlan(result.member_table, PlotSpec(mark="bar", y="Member", **common),
                        source=SOURCE_MEMBER_TABLE)

    if len(dims) == 1:
        dim = dims[0]
        if mark == "bar":
            return PlotPlan(result.table, PlotSpec(mark="bar", y=dim, **common))
        if mark in ("stackedbar", "heatmap"):
            return PlotPlan(result.member_table, PlotSpec(mark=mark, y=dim, hue="Member", **common),
                            source=SOURCE_MEMBER_TABLE)
        if mark == "box":
            return PlotPlan(refusal="Box needs two grouping dimensions (distribution of dim 1 within dim 2).")
        return PlotPlan(refusal=f"{mark!r} needs two grouping dimensions.")

    dim1, dim2 = dims[0], dims[1]
    if mark == "box":
        return PlotPlan(result.table, PlotSpec(mark="box", y=dim2, **common))
    if mark == "scatter":
        return PlotPlan(refusal="Scatter needs two value columns; this result has one.")
    return PlotPlan(result.table, PlotSpec(mark=mark, y=dim1, hue=dim2, **common))
