"""
explorer_query.py — engine-backed answers for the Element Explorer.

Element Explorer's search is its own thing and stays: five match modes,
co-occurrence tables, raw rows. What it should NOT keep is its
arithmetic. Its element mode matches every analyte whose parsed formula
contains the symbol and then sums those analytes' FULL masses, so a
search for "S" books all 96.06 g/mol of every sulfate row as sulfur --
and a search for "O" books the same sulfate mass again as oxygen. Two
searches, each claiming the whole of the same material.

This module answers the quantitative half through the query engine, so
Explorer agrees with Heatmaps (§9) and Overview (§5) about what the waste
contains. The exploratory half (co-elements, co-analytes, composition
profiles, raw rows) stays on element_science, which counts reported rows
as they are -- a different and legitimate question, as long as the page
says which one it is answering.

Two of Explorer's match modes have no engine equivalent: "analyte
contains" and "analyte regex" select a SET of analytes by pattern, and
the engine's subjects are one element, one analyte, or one basket. Rather
than quietly ignoring the pattern, `resolve_subject` expands it into a
basket of the analytes it actually matched -- which is both honest and
exactly what the user meant.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import polars as pl

from data_model import HanfordDataset
from elements import normalize_element_symbol
from query_engine import Basket, QueryResult, run_query

# Explorer's grouping views, as (table name, engine dimensions).
TARGET_VIEWS: Dict[str, List[str]] = {
    "Target by tank": ["WasteSiteId"],
    "Target by phase": ["WastePhase"],
    "Target by type": ["WasteType"],
}

_MAX_PATTERN_MEMBERS = 200


@dataclass
class ResolvedSubject:
    """What a search box plus a mode means to the engine."""

    kind: str = ""                       # "element" | "analyte" | "basket"
    subject: object = None               # symbol, label, or Basket
    label: str = ""                      # what to call it on a chart
    note: str = ""                       # how it was resolved, for disclosure
    error: str = ""                      # why it could not be resolved

    @property
    def ok(self) -> bool:
        return bool(self.kind) and not self.error


def matching_analytes(dataset: HanfordDataset, query: str, mode: str) -> List[str]:
    """The analyte labels a pattern mode selects, in dataset order."""
    df = dataset.require_df()
    labels = df.get_column("Analyte").drop_nulls().unique().sort().to_list()
    if mode == "analyte_contains":
        needle = query.lower()
        return [a for a in labels if needle in str(a).lower()]
    if mode == "regex":
        try:
            pattern = re.compile(query)
        except re.error:
            return []
        return [a for a in labels if pattern.search(str(a))]
    return []


def resolve_subject(dataset: HanfordDataset, query: str, mode: str) -> ResolvedSubject:
    """Map Explorer's (query, mode) onto an engine subject.

    "auto" resolves to an element when the text names one and to an exact
    analyte otherwise, matching Explorer's own convention. Pattern modes
    become a basket of the analytes they matched, so the engine answers
    the question the pattern asked instead of the pattern being dropped.
    """
    query = (query or "").strip()
    if not query:
        return ResolvedSubject(error="Enter something to search for.")

    if mode in ("analyte_contains", "regex"):
        matched = matching_analytes(dataset, query, mode)
        if not matched:
            kind_text = "matches" if mode == "regex" else "contains"
            return ResolvedSubject(
                error=f"No analyte {kind_text} {query!r}."
                      + (" (Invalid regular expression.)" if mode == "regex"
                         and not _valid_regex(query) else "")
            )
        trimmed = matched[:_MAX_PATTERN_MEMBERS]
        note = f"{len(matched)} analyte(s) matched: " + ", ".join(trimmed[:8])
        if len(trimmed) > 8:
            note += f", … (+{len(trimmed) - 8} more)"
        if len(matched) > len(trimmed):
            note += (f". Only the first {_MAX_PATTERN_MEMBERS} are included; "
                     f"narrow the pattern to cover the rest.")
        return ResolvedSubject(
            kind="basket", label=query, note=note,
            subject=Basket(query, tuple(trimmed), f"analytes matching {query!r}"),
        )

    if mode == "analyte_exact":
        return ResolvedSubject(kind="analyte", subject=query, label=query,
                               note="exact analyte label")

    symbol = normalize_element_symbol(query)
    if mode == "element":
        if symbol is None:
            return ResolvedSubject(error=f"{query!r} is not an element symbol.")
        return ResolvedSubject(kind="element", subject=symbol, label=symbol,
                               note="element (apportioned across every species reporting it)")

    # auto
    if symbol is not None:
        return ResolvedSubject(kind="element", subject=symbol, label=symbol,
                               note="read as an element symbol")
    return ResolvedSubject(kind="analyte", subject=query, label=query,
                           note="read as an exact analyte label")


def _valid_regex(pattern: str) -> bool:
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


@dataclass
class TargetTables:
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    disclosures: List[str] = field(default_factory=list)
    value_column: str = ""
    basis: str = ""
    label: str = ""
    member_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def is_empty(self) -> bool:
        return all(t.empty for t in self.tables.values()) if self.tables else True


def target_tables(
    dataset: HanfordDataset,
    resolved: ResolvedSubject,
    *,
    basis: str = "kg",
    scope: Optional[Dict[str, Sequence[str]]] = None,
) -> TargetTables:
    """The three grouped views Explorer shows, all from one engine
    subject so they cannot disagree with each other."""
    if not resolved.ok:
        return TargetTables(basis=basis, label=resolved.label,
                            warnings=[resolved.error] if resolved.error else [])

    subject = (resolved.kind, resolved.subject)
    out = TargetTables(basis=basis, label=resolved.label)
    seen_warnings: List[str] = []
    for name, dims in TARGET_VIEWS.items():
        result: QueryResult = run_query(dataset, subject=subject, basis=basis,
                                        group_by=dims, scope=scope)
        out.tables[name] = result.table
        out.value_column = result.value_column
        for warning in result.warnings:
            if warning not in seen_warnings:
                seen_warnings.append(warning)
        if name == "Target by tank":
            out.member_table = result.member_table
            out.disclosures = list(result.disclosures)
    out.warnings = seen_warnings
    if resolved.note:
        out.disclosures.insert(0, f"{resolved.label}: {resolved.note}.")
    return out


def raw_target_rows(
    dataset: HanfordDataset, resolved: ResolvedSubject, *, limit: int = 500,
    scope: Optional[Dict[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """The dataset rows behind the answer -- the provenance a search
    workspace owes its user. Row-level, so no apportionment applies: these
    are the records as reported."""
    if not resolved.ok:
        return pd.DataFrame()
    from query_engine import apply_scope

    df = apply_scope(dataset.require_df(), scope)
    if resolved.kind == "element":
        expr = pl.col("ElementListPadded").str.contains(f";{resolved.subject};", literal=True)
    elif resolved.kind == "analyte":
        expr = pl.col("Analyte") == str(resolved.subject)
    else:
        expr = pl.col("Analyte").is_in(list(resolved.subject.members))
    columns = [c for c in ["WasteSiteId", "TankFarm", "Analyte", "Inventory", "Units",
                           "WastePhase", "WasteType", "Element", "Isotope", "AnalyteClass",
                           "Inventory_RSD", "DecayDate"] if c in df.columns]
    return df.filter(expr).select(columns).head(limit).to_pandas()
