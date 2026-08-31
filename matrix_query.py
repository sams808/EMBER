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

    long_df = members[members["Member"].astype(str).isin(kept)].copy()
    long_df = long_df.rename(columns={"Member": "Element"})
    wide_df = (
        long_df.pivot_table(index="WasteSiteId", columns="Element", values=value_col,
                            aggfunc="sum", dropna=False)
        .reindex(columns=kept)
        .fillna(0.0)
        .reset_index()
    )
    # Order rows by total so the densest tanks read first, matching the
    # legacy builder's presentation.
    totals = wide_df[kept].sum(axis=1)
    wide_df = wide_df.assign(_total=totals).sort_values("_total", ascending=False).drop(columns="_total")

    return MatrixResult(
        long_df=long_df.reset_index(drop=True), wide_df=wide_df.reset_index(drop=True),
        elements=kept, warnings=warnings, disclosures=list(result.disclosures),
        value_column=value_col,
    )


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
    return MatrixResult(
        long_df=long_df, wide_df=wide_df, elements=elements,
        value_column=f"Inventory_{unit}",
        disclosures=[
            "Each analyte's full mass is booked to its first parsed element "
            "(no stoichiometric split): species constituents other than the first "
            "are not credited, so e.g. SO4 counts entirely as S and not at all as O."
        ],
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
