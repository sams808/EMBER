"""
association_query.py — engine-backed cross-correlation of waste components
(framework-agnostic, L1 -> association).

The question this answers: across the tank farm, which components occur
TOGETHER? The legacy correlation workbench (correlation_science) could not
answer it honestly for three reasons, each of which changes the result:

1. It correlated raw INVENTORIES. Every component scales with how much
   waste a tank holds, so every pair correlates (Spearman 0.4-0.9 on the
   shipped dataset) and the matrix describes tank size, not chemistry.
   Here the default basis is a CONCENTRATION -- mol per mol Na, wt% or
   g/L -- taken from the query engine's layer-correct denominators.
2. It worked in one unit at a time. Iodine exists in the BBI only as
   129I and selenium (site-wide) only as 79Se, both in curies, so a kg
   matrix could never put them beside sulfate. Members here go through
   run_query, which bridges curies to grams (ICRP-107), apportions
   species stoichiometrically and suppresses restatements.
3. It treated every tank as an independent observation. Many BBI layers
   are one waste-type TEMPLATE composition multiplied by a volume: the
   R1 sludge in eight S/SX tanks carries the same SO4/Na to three
   figures. Counting each copy inflates N and manufactures significance.
   `replicate_groups` finds layers whose whole member vector is the same
   composition and `correlate(distinct_only=True)` counts each once.

Nothing here adjudicates the data: values are the engine's, and every
caveat the engine raises travels with the matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl

from data_model import HanfordDataset
from elements import normalize_element_symbol, parse_isotope
from query_engine import Basket, apply_scope, run_query
from reference_data import ATOMIC_MASSES, SPECIES_FORMULAS

# The unit of observation. A LAYER is one (tank, phase, waste type) entry
# of the BBI -- the level at which a composition is actually stated. A
# TANK sums its layers: the right unit for "which tanks", the wrong one
# for "which components go together", because a tank holding saltcake
# over sludge averages two unrelated chemistries.
LEVEL_KEYS: Dict[str, List[str]] = {
    "layer": ["WasteSiteId", "WastePhase", "WasteType"],
    "tank": ["WasteSiteId"],
}
LEVEL_LABELS: Dict[str, str] = {"layer": "Waste layer", "tank": "Tank"}

MOL_PER_MOL_NA = "mol/mol Na"
# basis -> (axis label, engine basis it is computed from)
BASIS_LABELS: Dict[str, str] = {
    MOL_PER_MOL_NA: "mol per mol Na",
    "wt%": "wt% of layer mass",
    "g/L": "g per liter",
    "kg": "inventory (kg, Ci bridged)",
}
BASES = list(BASIS_LABELS)
METHODS = ["spearman", "pearson (log10)", "kendall"]

NULL_LABEL = "(not reported)"
PHASE_FAMILIES = ["Supernatant", "Saltcake", "Sludge"]

# Two layers are the same composition when every member agrees to this
# relative tolerance. BBI inventories carry three significant figures, so
# a ratio of two of them is reproducible to about 1%; 1.5% separates
# rounding from chemistry.
REPLICATE_REL_TOL = 0.015

# Inventory_RSD conventions seen in the BBI export: exactly 0 means no
# uncertainty was assigned; a value sitting at 1.00 is a default of 100%
# rather than a computed statistic (every bromine row lies in 1.000-1.014).
RSD_DEFAULT_BAND = (0.995, 1.02)


def phase_family(phase: object) -> str:
    """Supernatant / Saltcake / Sludge from a BBI WastePhase label."""
    text = str(phase or "")
    for family in PHASE_FAMILIES:
        if text.startswith(family):
            return family
    return NULL_LABEL


def member_molar_mass(member: str) -> Optional[float]:
    """g/mol of what a member's grams are grams OF: the species for a
    registered species, the nuclide for an isotope label, the element for
    a symbol. None when it cannot be stated (a combined-pair label)."""
    if member in SPECIES_FORMULAS:
        return float(sum(ATOMIC_MASSES[el] * n for el, n in SPECIES_FORMULAS[member].items()))
    mass_number, _meta, is_isotope = parse_isotope(member)
    if is_isotope:
        return float(mass_number) if mass_number else None
    symbol = normalize_element_symbol(member)
    if symbol and symbol in ATOMIC_MASSES:
        return float(ATOMIC_MASSES[symbol])
    return None


@dataclass
class AssociationMatrix:
    """One row per unit of observation, one column per member."""

    values: pd.DataFrame = field(default_factory=pd.DataFrame)
    members: List[str] = field(default_factory=list)
    id_columns: List[str] = field(default_factory=list)
    basis: str = MOL_PER_MOL_NA
    level: str = "layer"
    warnings: List[str] = field(default_factory=list)
    disclosures: List[str] = field(default_factory=list)

    @property
    def value_label(self) -> str:
        return BASIS_LABELS.get(self.basis, self.basis)


@dataclass
class AssociationResult:
    correlation: pd.DataFrame = field(default_factory=pd.DataFrame)   # square, members x members
    n_used: pd.DataFrame = field(default_factory=pd.DataFrame)        # square, observations per pair
    pairs: pd.DataFrame = field(default_factory=pd.DataFrame)         # long, one row per pair
    method: str = "spearman"
    n_rows: int = 0            # observations offered
    n_distinct: int = 0        # of which distinct compositions
    notes: List[str] = field(default_factory=list)


def _member_name(member: str) -> str:
    """The name run_query gives a basket member (a plain element symbol
    is normalised; everything else is kept verbatim)."""
    symbol = normalize_element_symbol(member)
    if symbol is not None and member not in SPECIES_FORMULAS and not parse_isotope(member)[2]:
        return symbol
    return member


def _pivot(member_table: pd.DataFrame, keys: List[str], value_col: str,
           names: List[str]) -> pd.DataFrame:
    if member_table.empty:
        return pd.DataFrame(columns=keys + names)
    table = member_table.copy()
    for key in keys:
        # A null key would be dropped by the pivot, and with it the layer:
        # WasteType is null on nine real layers.
        table[key] = table[key].where(table[key].notna(), NULL_LABEL).astype(str)
    wide = (
        table.pivot_table(index=keys, columns="Member", values=value_col,
                          aggfunc="sum", dropna=False)
        .reindex(columns=names)
        .reset_index()
    )
    wide.columns.name = None
    # pivot_table(dropna=False) builds the full key product; keep the
    # units that exist.
    present = table[keys].drop_duplicates()
    return present.merge(wide, on=keys, how="left")


def association_matrix(
    dataset: HanfordDataset,
    members: Sequence[str],
    *,
    basis: str = MOL_PER_MOL_NA,
    level: str = "layer",
    scope: Optional[Dict[str, Sequence[str]]] = None,
) -> AssociationMatrix:
    """Unit x member matrix on one basis, through the query engine.

    A member a unit does not report stays blank (NaN): it is not a
    measured zero. A reported zero stays zero.
    """
    if basis not in BASIS_LABELS:
        raise ValueError(f"Unknown basis {basis!r}; expected one of {BASES}")
    if level not in LEVEL_KEYS:
        raise ValueError(f"Unknown level {level!r}; expected one of {list(LEVEL_KEYS)}")
    names: List[str] = []
    for member in members:
        name = _member_name(str(member).strip())
        if name and name not in names:
            names.append(name)
    if len(names) < 2:
        raise ValueError("Cross-correlation needs at least two components.")

    keys = LEVEL_KEYS[level]
    basket = Basket("Association", tuple(names), "cross-correlation members")
    engine_basis = {"kg": "g", MOL_PER_MOL_NA: "g"}.get(basis, basis)
    result = run_query(dataset, subject=("basket", basket), basis=engine_basis,
                       group_by=keys, scope=scope)
    warnings = list(result.warnings)
    disclosures = list(result.disclosures)
    wide = _pivot(result.member_table, keys, result.value_column, names)

    if basis == "kg":
        wide[names] = wide[names] / 1000.0
    elif basis == MOL_PER_MOL_NA:
        sodium = run_query(dataset, subject=("element", "Na"), basis="g",
                           group_by=keys, scope=scope)
        na = sodium.table.copy()
        for key in keys:
            na[key] = na[key].where(na[key].notna(), NULL_LABEL).astype(str)
        na["_mol_na"] = na[sodium.value_column] / ATOMIC_MASSES["Na"]
        wide = wide.merge(na[keys + ["_mol_na"]], on=keys, how="left")
        no_na = ~(wide["_mol_na"] > 0)
        if no_na.any():
            warnings.append(
                f"{int(no_na.sum())} {LEVEL_LABELS[level].lower()}(s) report no sodium — "
                f"{MOL_PER_MOL_NA} left blank there."
            )
        unstated: List[str] = []
        for name in names:
            molar = member_molar_mass(name)
            if molar is None:
                unstated.append(name)
                wide[name] = np.nan
                continue
            wide[name] = (wide[name] / molar / wide["_mol_na"]).where(~no_na)
        if unstated:
            warnings.append(
                f"{', '.join(unstated)}: no single molar mass (a combined-pair label, or a "
                f"component the app does not recognise) — "
                f"blank on the {MOL_PER_MOL_NA} basis."
            )
        wide = wide.drop(columns="_mol_na")

    wide = _attach_descriptors(dataset, wide, keys, level, scope)
    id_columns = [c for c in wide.columns if c not in names]
    return AssociationMatrix(
        values=wide.reset_index(drop=True), members=names, id_columns=id_columns,
        basis=basis, level=level, warnings=warnings, disclosures=disclosures,
    )


def _attach_descriptors(dataset: HanfordDataset, wide: pd.DataFrame, keys: List[str],
                        level: str, scope: Optional[Dict[str, Sequence[str]]]) -> pd.DataFrame:
    """Farm, SST/DST, phase family and the waste volume behind each unit."""
    df = apply_scope(dataset.require_df(), scope)
    cohort = [c for c in LEVEL_KEYS["layer"] + ["Volume", "ComponentDensity"] if c in df.columns]
    layers = df.select(cohort).unique().with_columns([
        pl.col(k).fill_null(NULL_LABEL) for k in LEVEL_KEYS["layer"] if k in cohort
    ])
    if {"Volume", "ComponentDensity"} <= set(layers.columns):
        size = (
            layers.with_columns((pl.col("Volume") * pl.col("ComponentDensity") * 1000.0).alias("_kg"))
            .group_by(keys).agg([
                pl.col("Volume").sum().alias("Volume_kL"),
                pl.col("_kg").sum().alias("WasteMass_kg"),
            ]).to_pandas()
        )
        wide = wide.merge(size, on=keys, how="left")
    per_tank = [c for c in ("TankFarm", "TankSystem") if c in df.columns]
    if per_tank:
        tanks = df.select(["WasteSiteId"] + per_tank).unique(subset=["WasteSiteId"]).to_pandas()
        wide = wide.merge(tanks, on="WasteSiteId", how="left")
    if level == "layer":
        wide["PhaseFamily"] = wide["WastePhase"].map(phase_family)
    return wide


def replicate_groups(values: pd.DataFrame, members: Sequence[str],
                     rel_tol: float = REPLICATE_REL_TOL) -> pd.Series:
    """Group id per row: rows share an id when EVERY member agrees within
    `rel_tol` (blank matches blank, zero matches zero). A group of several
    rows is one stated composition applied to several volumes -- one
    observation, however many tanks it was written into.

    Meaningful on an intensive basis only; on inventories every row is
    its own group by construction.
    """
    data = values[list(members)].to_numpy(dtype=float)
    n = len(data)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n:
        blank = np.isnan(data)
        filled = np.where(blank, 0.0, data)
        for i in range(n - 1):
            a, b = filled[i], filled[i + 1:]
            scale = np.maximum(np.abs(a), np.abs(b))
            close = np.abs(b - a) <= rel_tol * scale          # 0 vs 0 -> True
            same_blank = blank[i] == blank[i + 1:]
            match = np.all(close & same_blank, axis=1)
            for j in np.nonzero(match)[0]:
                ri, rj = find(i), find(i + 1 + int(j))
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    roots = [find(i) for i in range(n)]
    return pd.Series(roots, index=values.index, name="CompositionGroup")


def correlate(
    matrix: AssociationMatrix,
    *,
    method: str = "spearman",
    distinct_only: bool = True,
    family: Optional[str] = None,
    min_n: int = 8,
) -> AssociationResult:
    """Pairwise association between the matrix's members.

    Each pair uses the units that report BOTH members. "pearson (log10)"
    additionally needs both values positive (a logarithm of a reported
    zero does not exist; Spearman and Kendall keep zeros as the ties they
    are). Pairs with fewer than `min_n` observations are left blank.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
    from scipy import stats  # lazy: keeps scipy out of application start-up

    values = matrix.values
    members = matrix.members
    notes: List[str] = []
    if family is not None:
        if "PhaseFamily" not in values.columns:
            raise ValueError("A phase family can only be selected at the layer level.")
        values = values[values["PhaseFamily"] == family]
    n_rows = len(values)
    n_distinct = n_rows
    if distinct_only and matrix.basis != "kg" and n_rows:
        groups = replicate_groups(values, members)
        n_distinct = int(groups.nunique())
        if n_distinct < n_rows:
            notes.append(
                f"{n_rows - n_distinct} of {n_rows} {LEVEL_LABELS[matrix.level].lower()}s repeat a "
                f"composition already counted (every component within "
                f"{100 * REPLICATE_REL_TOL:.1f}%); each distinct composition is counted once."
            )
        values = values[~groups.duplicated()]
    elif distinct_only and matrix.basis == "kg":
        notes.append("Replicate compositions cannot be recognised on an inventory basis; "
                     "every row is counted.")

    square = pd.DataFrame(np.nan, index=members, columns=members, dtype=float)
    counts = pd.DataFrame(0, index=members, columns=members, dtype=int)
    rows = []
    for name in members:
        counts.loc[name, name] = int(values[name].notna().sum())
        if counts.loc[name, name]:
            square.loc[name, name] = 1.0
    for a, b in combinations(members, 2):
        pair = values[[a, b]].dropna()
        n_both_positive = int(((pair[a] > 0) & (pair[b] > 0)).sum())
        if method == "pearson (log10)":
            pair = np.log10(pair[(pair[a] > 0) & (pair[b] > 0)])
        n = len(pair)
        r = p = np.nan
        if n >= max(int(min_n), 3) and pair[a].nunique() > 1 and pair[b].nunique() > 1:
            if method == "spearman":
                r, p = stats.spearmanr(pair[a], pair[b])
            elif method == "kendall":
                r, p = stats.kendalltau(pair[a], pair[b])
            else:
                r, p = stats.pearsonr(pair[a], pair[b])
        square.loc[a, b] = square.loc[b, a] = r
        counts.loc[a, b] = counts.loc[b, a] = n
        rows.append({
            "A": a, "B": b, "r": float(r) if pd.notna(r) else np.nan,
            "abs_r": abs(float(r)) if pd.notna(r) else np.nan,
            "p_value": float(p) if pd.notna(p) else np.nan,
            "N": n, "N_both_positive": n_both_positive,
        })
    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        pairs = pairs.sort_values("abs_r", ascending=False, na_position="last").reset_index(drop=True)
        short = pairs[pairs["r"].isna()]
        if len(short):
            notes.append(f"{len(short)} pair(s) have fewer than {min_n} shared observations "
                         f"(or a constant member) and are left blank.")
    return AssociationResult(
        correlation=square, n_used=counts, pairs=pairs, method=method,
        n_rows=n_rows, n_distinct=n_distinct, notes=notes,
    )


def member_quality(
    dataset: HanfordDataset,
    members: Sequence[str],
    *,
    scope: Optional[Dict[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """What each member rests on: which analytes, how many tanks, and how
    much of it carries no uncertainty or a default one.

    This is the table to read BEFORE the correlation matrix. A member
    reported in 11 tanks at a default 100% uncertainty (bromine) cannot
    carry a site-wide statement, whatever its coefficient says.
    """
    names = []
    for member in members:
        name = _member_name(str(member).strip())
        if name and name not in names:
            names.append(name)
    df = apply_scope(dataset.require_df(), scope)
    n_tanks_total = df.get_column("WasteSiteId").n_unique()
    rows = []
    for name in names:
        basket = Basket(name, (name,))
        by_analyte = run_query(dataset, subject=("basket", basket), basis="g",
                               group_by=["Analyte", "Units"], scope=scope)
        table = by_analyte.member_table
        analytes = sorted(set(table["Analyte"].astype(str))) if not table.empty else []
        source = df.filter(pl.col("Analyte").is_in(analytes))
        n = source.height
        positive = source.filter(pl.col("Inventory") > 0)
        rsd = source.get_column("Inventory_RSD") if "Inventory_RSD" in source.columns and n else None
        lo, hi = RSD_DEFAULT_BAND
        total_g = float(run_query(dataset, subject=("basket", basket), basis="g",
                                  scope=scope).table["Total_g"].sum()) if n else 0.0
        rows.append({
            "Member": name,
            "Reported_as": ", ".join(
                f"{a} ({u})" for a, u in sorted(set(zip(table["Analyte"].astype(str),
                                                       table["Units"].astype(str))))
            ) if not table.empty else "not reported",
            "Total_kg": total_g / 1000.0,
            "N_tanks": int(source.get_column("WasteSiteId").n_unique()) if n else 0,
            "N_tanks_positive": int(positive.get_column("WasteSiteId").n_unique()) if n else 0,
            "N_tanks_in_scope": int(n_tanks_total),
            "N_rows": int(n),
            "Unassigned_uncertainty_pct": 100.0 * float((rsd == 0).sum()) / n if rsd is not None else np.nan,
            "Default_100pct_uncertainty_pct": (
                100.0 * float(((rsd >= lo) & (rsd <= hi)).sum()) / n if rsd is not None else np.nan
            ),
        })
    return pd.DataFrame(rows)


def co_occurrence(matrix: AssociationMatrix, *, quantile: float = 0.75,
                  members: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Per unit: which members sit above their own `quantile` across the
    matrix, and how many at once. The units at the top are where the
    components coincide -- the question a correlation coefficient only
    answers on average."""
    names = [m for m in (members or matrix.members) if m in matrix.values.columns]
    values = matrix.values
    thresholds = values[names].quantile(quantile)
    high = values[names].gt(thresholds, axis=1)
    out = values[matrix.id_columns].copy()
    out["N_high"] = high.sum(axis=1).astype(int)
    out["High_members"] = [", ".join(n for n in names if row[n]) for _, row in high.iterrows()]
    for name in names:
        out[name] = values[name]
    return out.sort_values(["N_high"] + names[:1], ascending=False).reset_index(drop=True)
