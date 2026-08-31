"""
matrix_query.py — engine-backed tank x element matrices (L1 -> matrix).

The Heatmaps workspace asks one question: how much of each element sits
in each tank? The legacy builder (matrix_science.matrix_long_wide)
answers it by booking every analyte's FULL mass to its first parsed
element, so all of SO4 counts as sulfur and oxygen -- which is never a
first element -- barely registers. Measured on the shipped dataset, that
overstates nitrogen 4.1x and understates oxygen 15.6x.

This module answers the same question through the query engine, which
apportions each species stoichiometrically. It is a composition of
queries, not new science: one basket query over every element the dataset
can speak about, then a pivot of the per-member table.

Both attributions stay available (ATTRIBUTIONS) because they answer
subtly different questions -- "mass reported under analytes whose primary
element is X" versus "mass of X" -- and a user reproducing an older
figure needs the first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import polars as pl

import matrix_science as msci
from data_model import HanfordDataset
from reference_data import SPECIES_ELEMENT_MASS_FRACTIONS
from query_engine import Basket, apply_scope, run_query

APPORTIONED = "Apportioned (stoichiometric)"
AS_REPORTED = "As reported (primary element)"
ATTRIBUTIONS = [APPORTIONED, AS_REPORTED]


@dataclass
class MatrixResult:
    long_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    wide_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    elements: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    disclosures: List[str] = field(default_factory=list)
    value_column: str = ""


@dataclass
class ElementTotals:
    """Ranked element totals for a whole dataset (the Overview headline)."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: List[str] = field(default_factory=list)
    disclosures: List[str] = field(default_factory=list)
    value_column: str = ""
    basis: str = ""


def element_totals(
    dataset: HanfordDataset,
    *,
    basis: str = "kg",
    top_n: int = 40,
    scope: Optional[Dict[str, Sequence[str]]] = None,
) -> ElementTotals:
    """Every element ranked by apportioned total, with how many tanks and
    how many distinct analytes contribute to each.

    Two queries: one ungrouped (totals and tank counts, which cannot be
    summed out of a grouped result) and one grouped by Analyte (which
    species carry each element -- more informative under apportionment
    than it was before, since sulfur genuinely arrives via SO4, sulfide
    and thiosulfate)."""
    elements = candidate_elements(dataset)
    if not elements:
        return ElementTotals(basis=basis)

    basket = Basket("All elements", tuple(elements), "every element in the dataset")
    totals = run_query(dataset, subject=("basket", basket), basis=basis, scope=scope)
    by_species = run_query(dataset, subject=("basket", basket), basis=basis,
                           group_by=["Analyte"], scope=scope)

    members = totals.member_table
    if members.empty:
        return ElementTotals(warnings=list(totals.warnings),
                             disclosures=list(totals.disclosures), basis=basis)

    value_col = totals.value_column
    table = members.rename(columns={"Member": "Element"})[
        ["Element", value_col, "N_tanks"]
    ].copy()

    if not by_species.member_table.empty:
        counts = (
            by_species.member_table.groupby("Member", dropna=False)["Analyte"]
            .nunique().rename("N_analytes")
        )
        table["N_analytes"] = table["Element"].map(counts).fillna(0).astype(int)

    table = table[table[value_col] > 0].sort_values(value_col, ascending=False)
    warnings = list(totals.warnings)
    if len(table) > top_n:
        warnings.append(f"{len(table) - top_n} of {len(table)} elements omitted "
                        f"(showing the top {top_n} by {basis}).")
        table = table.head(top_n)
    warnings.extend(_unattributable_warning(dataset, basis, scope, members))

    return ElementTotals(
        table=table.reset_index(drop=True), warnings=warnings,
        disclosures=list(totals.disclosures), value_column=value_col, basis=basis,
    )


def candidate_elements(dataset: HanfordDataset) -> List[str]:
    """Every element this dataset can speak about: those tagged on a row,
    plus every constituent of a registered species (oxygen only ever
    reaches a matrix through the second kind)."""
    df = dataset.require_df()
    tagged = set(df.get_column("Element").drop_nulls().unique().to_list())
    from_species = {el for fractions in SPECIES_ELEMENT_MASS_FRACTIONS.values() for el in fractions}
    return sorted({str(e) for e in tagged} | from_species)


def engine_matrix(
    dataset: HanfordDataset,
    *,
    basis: str = "kg",
    top_n_elements: int = 35,
    scope: Optional[Dict[str, Sequence[str]]] = None,
    tank_subset: Optional[Sequence[str]] = None,
) -> MatrixResult:
    """Tank x element matrix with stoichiometric apportionment.

    One basket query over every candidate element gives both the ranking
    (by apportioned total) and the per-tank values, so the top-N columns
    are chosen on the same footing they are drawn on.
    """
    elements = candidate_elements(dataset)
    if not elements:
        return MatrixResult()

    query_scope = dict(scope or {})
    if tank_subset is not None:
        query_scope["WasteSiteId"] = list(tank_subset)

    basket = Basket("Matrix", tuple(elements), "every element in the dataset")
    result = run_query(dataset, subject=("basket", basket), basis=basis,
                       group_by=["WasteSiteId"], scope=query_scope or None)
    members = result.member_table
    if members.empty:
        return MatrixResult(warnings=list(result.warnings), disclosures=list(result.disclosures))

    value_col = result.value_column
    ranked = (
        members.groupby("Member", dropna=False)[value_col].sum()
        .sort_values(ascending=False)
    )
    ranked = ranked[ranked > 0]
    kept = [str(e) for e in ranked.index[:max(1, int(top_n_elements))]]

    warnings = list(result.warnings)
    if len(ranked) > len(kept):
        warnings.append(
            f"{len(ranked) - len(kept)} of {len(ranked)} elements omitted "
            f"(showing the top {len(kept)} by {basis})."
        )

    warnings.extend(_unattributable_warning(dataset, basis, query_scope, members))

    long_df = members[members["Member"].astype(str).isin(kept)].copy()
    long_df = long_df.rename(columns={"Member": "Element"})
    # NO fillna: a (tank, element) pair with no measurement must stay
    # blank, not become a reported 0 kg. The figure paints those cells
    # "no data"; the table and the exported CSV have to say the same
    # thing (review finding -- 1,207 of 6,195 real cells asserted a
    # measurement that does not exist).
    wide_df = (
        long_df.pivot_table(index="WasteSiteId", columns="Element", values=value_col,
                            aggfunc="sum", dropna=False)
        .reindex(columns=kept)
        .reset_index()
    )
    # Order rows by total so the densest tanks read first, matching the
    # legacy builder's presentation.
    totals = wide_df[kept].sum(axis=1, skipna=True)
    wide_df = wide_df.assign(_total=totals).sort_values("_total", ascending=False).drop(columns="_total")

    return MatrixResult(
        long_df=long_df.reset_index(drop=True), wide_df=wide_df.reset_index(drop=True),
        elements=kept, warnings=warnings, disclosures=list(result.disclosures),
        value_column=value_col,
    )


def _unattributable_warning(
    dataset: HanfordDataset, basis: str, scope: Optional[Dict[str, Sequence[str]]],
    members: pd.DataFrame,
) -> List[str]:
    """Mass in scope that reached NO element. The engine's own
    unregistered-species warning only sees rows it can parse as compounds,
    so analytes it cannot resolve at all (Aroclors, ketones, phthalates --
    515 kg on the shipped dataset, including the entire PCB inventory)
    were excluded in total silence. Comparing totals catches every cause,
    including ones nobody has thought of yet."""
    if basis not in ("kg", "Ci"):
        return []
    unit = "kg" if basis == "kg" else "Ci"
    scoped = apply_scope(dataset.require_df(), scope)
    reported = float(
        scoped.filter(pl.col("Units") == unit).get_column("Inventory").sum() or 0.0
    )
    value_cols = [c for c in members.columns if c.startswith("Total_")]
    if not value_cols:
        return []
    attributed = float(members[value_cols[0]].sum())
    missing = reported - attributed
    # Tolerance guards float noise on sums of ~1e8; anything real is
    # orders of magnitude above it.
    if reported <= 0 or missing <= abs(reported) * 1e-12:
        return []
    pct = 100.0 * missing / reported
    return [
        f"{missing:,.4g} {unit} ({pct:.2g}% of the reported total) is not attributable to any "
        f"element — analytes whose formula the parser cannot resolve. Those rows are absent "
        f"from this matrix."
    ]


def legacy_matrix(
    dataset: HanfordDataset,
    *,
    unit: str = "kg",
    top_n_elements: int = 35,
    min_inventory: float = 0.0,
    scope: Optional[Dict[str, Sequence[str]]] = None,
    tank_subset: Optional[Sequence[str]] = None,
) -> MatrixResult:
    """The first-element-takes-all matrix, kept for comparison and for
    reproducing older figures. Scoped through the SAME filter the engine
    uses, so "global scope" means one thing everywhere."""
    scoped = _scoped_copy(dataset, scope)
    long_df, wide_df = msci.matrix_long_wide(
        scoped, unit=unit, top_n_elements=top_n_elements, min_inventory=min_inventory,
        value_mode="inventory", tank_subset=tank_subset,
    )
    elements = [c for c in wide_df.columns if c != "WasteSiteId"] if not wide_df.empty else []

    # This path trims and filters exactly like the engine path does, so it
    # owes the same disclosures -- it used to emit none at all, leaving
    # the warnings banner, the Notes tab and the export silent together
    # (review finding).
    warnings: List[str] = []
    available = _available_element_count(scoped, unit, min_inventory)
    if available > len(elements):
        warnings.append(
            f"{available - len(elements)} of {available} elements omitted "
            f"(showing the top {len(elements)} by {unit})."
        )
    if min_inventory > 0:
        warnings.append(
            f"min inventory {min_inventory:g} {unit}: rows at or below it were dropped "
            f"before summing, so tank totals exclude them."
        )

    return MatrixResult(
        long_df=long_df, wide_df=wide_df, elements=elements,
        value_column=f"Inventory_{unit}", warnings=warnings,
        disclosures=[
            "Each analyte's full mass is booked to its first parsed element "
            "(no stoichiometric split): species constituents other than the first "
            "are not credited, so e.g. SO4 counts entirely as S and not at all as O."
        ],
    )


def _available_element_count(dataset: HanfordDataset, unit: str, min_inventory: float) -> int:
    df = dataset.require_df()
    return (
        df.filter(
            (pl.col("Units") == unit)
            & pl.col("Element").is_not_null()
            & (pl.col("Inventory") > float(min_inventory))
        )
        .get_column("Element").n_unique()
    )


def _scoped_copy(dataset: HanfordDataset, scope: Optional[Dict[str, Sequence[str]]]) -> HanfordDataset:
    """A shallow dataset view with the scope applied, for science
    functions that take a dataset rather than a scope dict."""
    if not scope:
        return dataset
    import copy

    view = copy.copy(dataset)
    view.df = apply_scope(dataset.require_df(), scope)
    return view


def attribution_delta(engine: MatrixResult, legacy: MatrixResult) -> pd.DataFrame:
    """Per-element totals under both attributions, so the change a
    migration makes is inspectable rather than asserted."""
    if engine.wide_df.empty or legacy.wide_df.empty:
        return pd.DataFrame()

    def _totals(matrix: MatrixResult) -> pd.Series:
        cols = [c for c in matrix.wide_df.columns if c != "WasteSiteId"]
        return matrix.wide_df[cols].sum()

    eng, leg = _totals(engine), _totals(legacy)
    out = pd.DataFrame({"Element": sorted(set(eng.index) | set(leg.index))})
    out["As_reported"] = out["Element"].map(leg).fillna(0.0)
    out["Apportioned"] = out["Element"].map(eng).fillna(0.0)
    out["Ratio"] = out.apply(
        lambda r: (r["Apportioned"] / r["As_reported"]) if r["As_reported"] else float("nan"),
        axis=1,
    )
    return out.sort_values("As_reported", ascending=False).reset_index(drop=True)
