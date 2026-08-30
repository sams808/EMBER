"""
query_engine.py — the L1 universal query engine (framework-agnostic).

One query shape replaces the per-workspace hardcoded aggregations:

    run_query(dataset,
              subject=("element", "S") | ("analyte", "SO4") | ("basket", basket),
              basis="g" | "kg" | "Ci" | "wt%" | "g/L",
              group_by=["WasteSiteId", "WastePhase"],   # any DIMENSIONS keys
              scope={"WastePhase": ["Supernatant"], ...})

and returns a QueryResult carrying the aggregated table, a per-member
breakdown (for baskets), machine-checked WARNINGS (mixed decay cohorts,
unassigned uncertainty, exclusions), and per-member DISCLOSURES stating
exactly what each member resolved to in this dataset -- e.g.
"I = 129I only (Ci->g bridged); stable 127I is not reported: lower bound."

Correctness rules this engine owns centrally (so no view can silently
reproduce the legacy traps):

- Stoichiometric apportionment: an element subject receives the element's
  MASS FRACTION of every species reporting it (S gets 32.06/96.06 of SO4
  kg), never the species' full mass. The legacy first-element-takes-all
  attribution booked ALL SO4 mass as S -- any cited sulfur number was
  inflated by ~3x.
- Unit awareness: kg and Ci are never summed together. The "g" basis puts
  both on one axis by bridging Ci rows to grams via ICRP-107 specific
  activities (reference_data); combined-pair labels (239/240Pu, ...)
  carry no single specific activity and are EXCLUDED from mass bases,
  disclosed rather than silently dropped.
- Layer-correct concentrations: Volume and ComponentDensity are verified
  constant per (tank, phase, waste-type) layer, so wt% and g/L divide by
  the sum of DISTINCT layer masses/volumes in each group -- across all
  in-scope layers of the group, whether or not the subject is present in
  them.
- Mixed DecayDate cohorts in any Ci-involving aggregation raise a warning
  (the dataset carries 2015-07-01 and 2022-01-01 baselines).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import polars as pl

from data_model import HanfordDataset
from elements import normalize_element_symbol, parse_isotope
from reference_data import (
    NUCLIDE_HALF_LIFE_YEARS,
    SPECIES_ELEMENT_MASS_FRACTIONS,
    SPECIES_FORMULAS,
    nuclide_grams_per_ci,
)

# Group-by dimensions the engine accepts, with display names. Tank-attribute
# dimensions are per-tank and therefore layer-compatible for wt%/g-per-L.
DIMENSIONS: Dict[str, str] = {
    "WasteSiteId": "Tank",
    "TankFarm": "Farm",
    "WastePhase": "Waste phase",
    "WasteType": "Waste type",
    "TankSystem": "SST/DST",
    "TankType": "Tank type",
    "TankIntegrity": "Tank integrity",
    "TankStatus": "Tank status",
    "Ventilation": "Ventilation",
    "Element": "Element",
    "Isotope": "Isotope",
    "Analyte": "Analyte (species)",
    "AnalyteClass": "Analyte class",
    "Units": "Units",
    "DecayDate": "Decay baseline",
}

# Dimensions along which layer mass/volume denominators are well-defined
# (constant per layer or per tank). Grouping a concentration basis by
# Element/Isotope/Analyte/etc. would double-count denominators and is
# refused. TankType is per-tank, hence layer-compatible.
LAYER_COMPATIBLE_DIMS = {
    "WasteSiteId", "TankFarm", "WastePhase", "WasteType",
    "TankSystem", "TankType", "TankIntegrity", "TankStatus", "Ventilation",
}

# The value column each basis produces, and the unit-honest axis label for
# that column. ONE registry: run_query names its output column from here
# and the plot grammar labels axes from here, so a new basis cannot ship
# with an unlabeled (or, worse, mislabeled) axis.
VALUE_COLUMN_BY_BASIS: Dict[str, str] = {
    "g": "Total_g",
    "kg": "Total_kg",
    "Ci": "Total_Ci",
    "wt%": "wt_pct",
    "g/L": "g_per_L",
}
VALUE_AXIS_LABELS: Dict[str, str] = {
    "Total_g": "Total mass (g)",
    "Total_kg": "Total mass (kg)",
    "Total_Ci": "Total activity (Ci)",
    "wt_pct": "wt% of layer mass",
    "g_per_L": "g per liter",
}
BASES = list(VALUE_COLUMN_BY_BASIS)
_LAYER_KEYS = ["WasteSiteId", "WastePhase", "WasteType"]


@dataclass(frozen=True)
class Basket:
    """A named, reusable group of subjects that behaves as one queryable
    species. Members are element symbols or exact analyte labels; the
    engine resolves each and reports per-member what it found."""

    name: str
    members: Tuple[str, ...]
    description: str = ""


# Shipped starter baskets (editable copies in the UI). Members are element
# symbols unless they name an exact analyte.
BASKET_PRESETS: Dict[str, Basket] = {
    "Salt formers": Basket(
        "Salt formers", ("F", "Cl", "Br", "I", "S", "Cr"),
        "Halides + sulfate-sulfur + chromium: the molten-salt / gall formation set. "
        "S resolves via SO4/sulfide apportionment; I via 129I (Ci-bridged).",
    ),
    "Halides": Basket("Halides", ("F", "Cl", "Br", "I"), "F, Cl, Br, I."),
    "Glass formers": Basket(
        "Glass formers", ("B", "Si", "P", "Al", "Zr", "Ti"),
        "Glass-former / intermediate elements.",
    ),
    "RCRA metals": Basket(
        "RCRA metals", ("As", "Ba", "Cd", "Cr", "Pb", "Se", "Ag", "Hg"),
        "The eight RCRA 40 CFR 261 toxicity-characteristic metals.",
    ),
    "Mobile long-lived": Basket(
        "Mobile long-lived", ("99Tc", "129I", "NO3"),
        "Low-Kd, long-lived groundwater drivers (99Tc, 129I, nitrate).",
    ),
}


@dataclass
class QueryResult:
    table: pd.DataFrame
    member_table: pd.DataFrame
    warnings: List[str] = field(default_factory=list)
    disclosures: List[str] = field(default_factory=list)
    value_column: str = ""
    basis: str = "g"


def _apply_scope(df: pl.DataFrame, scope: Optional[Dict[str, Sequence[str]]]) -> pl.DataFrame:
    if not scope:
        return df
    missing = [col for col, allowed in scope.items() if allowed and col not in df.columns]
    if missing:
        # Review finding: silently skipping a scope column presented
        # UNFILTERED totals as filtered -- refuse loudly instead.
        raise ValueError(
            f"Scope filter(s) on {missing} can't be applied: column(s) not in this dataset "
            f"(tank-attribute filters require Tank_attributes.csv)."
        )
    for col, allowed in scope.items():
        if allowed:
            df = df.filter(pl.col(col).is_in(list(allowed)))
    return df


def _grams_per_ci_expr() -> pl.Expr:
    """Maps the Isotope label to grams-per-curie (null when not bridgeable).
    Keyed on the CANONICAL "###m?Sym" form rebuilt from the parsed
    MassNumber/IsMetastable/Element columns, so every label form the
    isotope parser recognizes bridges identically ("Cs-137" == "137Cs").
    Combined-pair labels (a "/" in the raw label) keep their raw key and
    hit their explicit not-bridgeable entries."""
    mapping = {n: nuclide_grams_per_ci(n) for n in NUCLIDE_HALF_LIFE_YEARS}
    canonical = (
        pl.when(pl.col("Isotope").str.contains("/", literal=True) | pl.col("MassNumber").is_null())
        .then(pl.col("Isotope"))
        .otherwise(
            pl.col("MassNumber").cast(pl.Utf8)
            + pl.when(pl.col("IsMetastable")).then(pl.lit("m")).otherwise(pl.lit(""))
            + pl.col("Element")
        )
    )
    return canonical.replace_strict(mapping, default=None, return_dtype=pl.Float64)


def _element_contributions(df: pl.DataFrame, element: str) -> pl.DataFrame:
    """All rows contributing to one ELEMENT subject, with per-row Mass_g
    (null where no mass exists: Ci rows whose label is not bridgeable) and
    Ci_value (null for kg rows), plus SourceKind for disclosures."""
    pieces: List[pl.DataFrame] = []

    # kg species rows: apportion the element's mass fraction of the species.
    for analyte, fractions in SPECIES_ELEMENT_MASS_FRACTIONS.items():
        frac = fractions.get(element, 0.0)
        if frac <= 0:
            continue
        rows = df.filter((pl.col("Analyte") == analyte) & (pl.col("Units") == "kg"))
        if rows.is_empty():
            continue
        pieces.append(rows.with_columns([
            (pl.col("Inventory") * frac * 1000.0).alias("Mass_g"),
            pl.lit(None, dtype=pl.Float64).alias("Ci_value"),
            pl.lit(f"kg-species:{analyte}").alias("SourceKind"),
        ]))

    # kg elemental rows (class "element", not a registered species label --
    # UTOTAL is registered as a species so it never lands in both paths;
    # Isotope must be null or a label like "Cs-137", which classifies as
    # "element" AND parses as an isotope, would be counted here and by the
    # kg-nuclide branch below -- exactly 2x the mass; review finding).
    elemental = df.filter(
        (pl.col("Units") == "kg")
        & (pl.col("AnalyteClass") == "element")
        & (pl.col("Element") == element)
        & pl.col("Isotope").is_null()
        & ~pl.col("Analyte").is_in(list(SPECIES_FORMULAS))
    )
    if not elemental.is_empty():
        pieces.append(elemental.with_columns([
            (pl.col("Inventory") * 1000.0).alias("Mass_g"),
            pl.lit(None, dtype=pl.Float64).alias("Ci_value"),
            pl.lit("kg-element").alias("SourceKind"),
        ]))

    # kg rows carrying an ISOTOPE label (mass of a specific nuclide reported
    # in kg) contribute their full mass to the element. The current BBI
    # export reports no analyte in both units, but the engine must not lose
    # mass if a future export (or a user CSV) does.
    kg_nuclides = df.filter(
        (pl.col("Units") == "kg")
        & (pl.col("Element") == element)
        & pl.col("Isotope").is_not_null()
    )
    if not kg_nuclides.is_empty():
        pieces.append(kg_nuclides.with_columns([
            (pl.col("Inventory") * 1000.0).alias("Mass_g"),
            pl.lit(None, dtype=pl.Float64).alias("Ci_value"),
            (pl.lit("kg-nuclide:") + pl.col("Analyte")).alias("SourceKind"),
        ]))

    # Ci nuclide rows for this element: activity always available; mass only
    # where the label carries a single specific activity.
    nuclides = df.filter((pl.col("Units") == "Ci") & (pl.col("Element") == element))
    if not nuclides.is_empty():
        pieces.append(nuclides.with_columns([
            (pl.col("Inventory") * _grams_per_ci_expr()).alias("Mass_g"),
            pl.col("Inventory").alias("Ci_value"),
            (pl.lit("Ci:") + pl.col("Analyte")).alias("SourceKind"),
        ]))

    if not pieces:
        return df.head(0).with_columns([
            pl.lit(None, dtype=pl.Float64).alias("Mass_g"),
            pl.lit(None, dtype=pl.Float64).alias("Ci_value"),
            pl.lit(None, dtype=pl.Utf8).alias("SourceKind"),
        ])
    return pl.concat(pieces, how="vertical_relaxed")


def _analyte_contributions(df: pl.DataFrame, analyte: str) -> pl.DataFrame:
    rows = df.filter(pl.col("Analyte") == analyte)
    if rows.is_empty():
        return rows.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("Mass_g"),
            pl.lit(None, dtype=pl.Float64).alias("Ci_value"),
            pl.lit(None, dtype=pl.Utf8).alias("SourceKind"),
        ])
    return rows.with_columns([
        pl.when(pl.col("Units") == "kg")
          .then(pl.col("Inventory") * 1000.0)
          .otherwise(pl.col("Inventory") * _grams_per_ci_expr())
          .alias("Mass_g"),
        pl.when(pl.col("Units") == "Ci").then(pl.col("Inventory")).otherwise(None).alias("Ci_value"),
        (pl.lit("analyte:") + pl.col("Units")).alias("SourceKind"),
    ])


def _resolve_member(df: pl.DataFrame, member: str) -> Tuple[str, pl.DataFrame]:
    """A plain element symbol resolves as an ELEMENT (apportioned species +
    elemental + nuclide rows), exactly like the Element subject kind --
    review finding: exact-analyte-first meant a bare "S" analyte row
    silently dropped ALL apportioned SO4/Sulfide mass for member "S".
    Everything else (registered species, isotope labels like "129I" --
    which must stay the exact nuclide, not widen to element I -- and other
    exact analyte labels) resolves as an analyte."""
    symbol = normalize_element_symbol(member)
    is_plain_symbol = (
        symbol is not None
        and member not in SPECIES_FORMULAS
        and not parse_isotope(member)[2]
    )
    if is_plain_symbol:
        return symbol, _element_contributions(df, symbol)
    return member, _analyte_contributions(df, member)  # empty when absent; disclosed as not found


def _member_disclosure(name: str, contrib: pl.DataFrame) -> str:
    if contrib.is_empty():
        return f"{name}: not reported in this dataset in any form."
    kinds = contrib.group_by("SourceKind").agg(pl.col("Inventory").sum().alias("inv"))
    parts: List[str] = []
    bridged: List[str] = []
    excluded: List[str] = []
    has_direct_mass = False  # any resolution beyond the Ci->g bridge
    for kind in sorted(kinds.get_column("SourceKind").to_list()):
        if kind.startswith("kg-species:"):
            parts.append(f"apportioned from {kind.split(':', 1)[1]} (kg)")
            has_direct_mass = True
        elif kind == "kg-element":
            parts.append("elemental (kg)")
            has_direct_mass = True
        elif kind.startswith("kg-nuclide:"):
            # Review finding: no branch here -> "Cd: ." (broken, empty) AND
            # a false "mass is a lower bound" claim below despite kg mass.
            parts.append(f"{kind.split(':', 1)[1]} (kg nuclide rows)")
            has_direct_mass = True
        elif kind.startswith("Ci:"):
            label = kind.split(":", 1)[1]
            if nuclide_grams_per_ci(label) is not None:
                bridged.append(label)
            else:
                excluded.append(label)
        elif kind.startswith("analyte:"):
            parts.append(f"analyte rows ({kind.split(':', 1)[1]})")
            has_direct_mass = True
    if bridged:
        parts.append(f"{'/'.join(bridged)} (Ci→g bridged, ICRP-107)")
    if excluded:
        parts.append(f"{'/'.join(excluded)} EXCLUDED from mass bases (combined pair, no single specific activity)")
    text = f"{name}: " + "; ".join(parts) + "."
    if bridged and not has_direct_mass:
        text += " No stable-isotope reporting exists for this member — mass is a lower bound."
    return text


def _layer_denominators(scoped: pl.DataFrame, layer_dims: List[str]) -> pl.DataFrame:
    """Sum of distinct-layer masses (g) and volumes (L) per group of
    layer-compatible dims, over ALL in-scope layers of the group."""
    layers = scoped.select(
        [c for c in dict.fromkeys(_LAYER_KEYS + layer_dims + ["Volume", "ComponentDensity"]) if c in scoped.columns]
    ).unique(subset=_LAYER_KEYS)
    layers = layers.with_columns([
        (pl.col("Volume") * 1000.0).alias("_layer_volume_L"),          # kL -> L
        (pl.col("Volume") * 1.0e6 * pl.col("ComponentDensity")).alias("_layer_mass_g"),  # kL -> mL x g/mL
    ])
    if layer_dims:
        return layers.group_by(layer_dims).agg([
            pl.col("_layer_mass_g").sum().alias("_group_mass_g"),
            pl.col("_layer_volume_L").sum().alias("_group_volume_L"),
        ])
    return layers.select([
        pl.col("_layer_mass_g").sum().alias("_group_mass_g"),
        pl.col("_layer_volume_L").sum().alias("_group_volume_L"),
    ])


def run_query(
    dataset: HanfordDataset,
    *,
    subject: Tuple[str, Union[str, Basket]],
    basis: str = "g",
    group_by: Optional[Sequence[str]] = None,
    scope: Optional[Dict[str, Sequence[str]]] = None,
) -> QueryResult:
    kind, target = subject
    group_by = list(group_by or [])
    # Review findings: a typo'd dimension silently returned UNGROUPED
    # totals, and an attribute-backed dimension with no Tank_attributes.csv
    # loaded escaped as a polars ColumnNotFoundError the UI can't catch.
    # Both are ValueError refusals like every other engine refusal.
    unknown = [g for g in group_by if g not in DIMENSIONS]
    if unknown:
        raise ValueError(f"Unknown group-by dimension(s) {unknown}; expected keys of DIMENSIONS.")
    base_df = dataset.require_df()
    missing_dims = [g for g in group_by if g not in base_df.columns]
    if missing_dims:
        raise ValueError(
            f"Grouping by {missing_dims} needs column(s) this dataset does not carry — "
            f"tank-attribute dimensions require Tank_attributes.csv to be loaded."
        )
    if basis not in BASES:
        raise ValueError(f"Unknown basis {basis!r}; expected one of {BASES}")
    if basis in ("wt%", "g/L"):
        bad = [g for g in group_by if g not in LAYER_COMPATIBLE_DIMS]
        if bad:
            raise ValueError(
                f"Basis {basis!r} needs layer-compatible grouping; {bad} would double-count "
                f"layer denominators. Group by tanks/farms/phases/types instead."
            )
        # Review finding: Volume is converted as kL and ComponentDensity as
        # g/mL unconditionally -- a CSV carrying other units would be
        # silently wrong by the conversion factor. No honest conversion is
        # implemented, so refuse rather than compute a knowingly-wrong number.
        for col, expected in (("VolumeUnits", "kL"), ("DensityUnits", "g/mL")):
            if col in base_df.columns:
                odd = [
                    u for u in base_df.get_column(col).drop_nulls().unique().to_list()
                    if u != expected
                ]
                if odd:
                    raise ValueError(
                        f"{col} carries {odd}; layer denominators assume {expected!r}. "
                        f"Convert the CSV to {expected!r} before using the {basis} basis."
                    )

    df = _apply_scope(base_df, scope)
    warnings: List[str] = []
    disclosures: List[str] = []

    # ---- resolve subject to per-row contributions, tagged by member ----
    if kind == "basket":
        basket: Basket = target  # type: ignore[assignment]
        member_frames = []
        for member in basket.members:
            name, contrib = _resolve_member(df, member)
            disclosures.append(_member_disclosure(name, contrib))
            if not contrib.is_empty():
                member_frames.append(contrib.with_columns(pl.lit(name).alias("Member")))
        contrib = (
            pl.concat(member_frames, how="vertical_relaxed") if member_frames
            else df.head(0).with_columns([
                pl.lit(None, dtype=pl.Float64).alias("Mass_g"),
                pl.lit(None, dtype=pl.Float64).alias("Ci_value"),
                pl.lit(None, dtype=pl.Utf8).alias("SourceKind"),
                pl.lit(None, dtype=pl.Utf8).alias("Member"),
            ])
        )
        subject_label = basket.name
    elif kind == "element":
        symbol = normalize_element_symbol(target) or str(target)
        contrib = _element_contributions(df, symbol).with_columns(pl.lit(symbol).alias("Member"))
        disclosures.append(_member_disclosure(symbol, contrib.drop("Member")))
        subject_label = symbol
    elif kind == "analyte":
        contrib = _analyte_contributions(df, str(target)).with_columns(pl.lit(str(target)).alias("Member"))
        disclosures.append(_member_disclosure(str(target), contrib.drop("Member")))
        subject_label = str(target)
    else:
        raise ValueError(f"Unknown subject kind {kind!r}")

    # ---- basis: choose the value column and inclusion rule ----
    if basis == "Ci":
        included = contrib.filter(pl.col("Ci_value").is_not_null())
        value_expr = pl.col("Ci_value")
        value_col = "Total_Ci"
        n_kg_dropped = contrib.filter(pl.col("Ci_value").is_null()).height
        if n_kg_dropped:
            warnings.append(f"{n_kg_dropped} kg-basis row(s) excluded: activity basis carries Ci rows only.")
    elif basis == "kg":
        included = contrib.filter(pl.col("Units") == "kg")
        value_expr = pl.col("Mass_g") / 1000.0
        value_col = "Total_kg"
        n_ci_dropped = contrib.filter(pl.col("Units") == "Ci").height
        if n_ci_dropped:
            warnings.append(
                f"{n_ci_dropped} Ci-basis row(s) excluded under the kg basis; use basis 'g' to bridge them."
            )
    else:  # g, wt%, g/L -- mass-based
        included = contrib.filter(pl.col("Mass_g").is_not_null())
        value_expr = pl.col("Mass_g")
        value_col = {"g": "Total_g", "wt%": "wt_pct", "g/L": "g_per_L"}[basis]
        dropped = contrib.filter(pl.col("Mass_g").is_null() & (pl.col("Units") == "Ci"))
        if dropped.height:
            labels = sorted(set(dropped.get_column("Analyte").to_list()))
            warnings.append(
                f"{dropped.height} Ci row(s) not mass-bridgeable and excluded: {', '.join(labels)} "
                f"(no single ICRP-107 specific activity — combined-pair label, or a nuclide "
                f"missing from reference_data.NUCLIDE_HALF_LIFE_YEARS)."
            )

    # ---- central data-integrity warnings ----
    # A registered-species miss must never be silent: a kg compound row
    # whose formula the registry lacks, but whose parsed constituents
    # include a requested element, is EXCLUDED (no honest apportionment
    # exists without a formula) and warned about.
    element_members = [
        m for m in (target.members if kind == "basket" else [target])
        if normalize_element_symbol(m) is not None and str(m) not in SPECIES_FORMULAS
    ]
    if element_members:
        needles = [f";{normalize_element_symbol(m)};" for m in element_members]
        unregistered = df.filter(
            (pl.col("Units") == "kg")
            & (pl.col("AnalyteClass") == "compound/formula")
            & ~pl.col("Analyte").is_in(list(SPECIES_FORMULAS))
            & pl.any_horizontal([pl.col("ElementListPadded").str.contains(n, literal=True) for n in needles])
        )
        if unregistered.height:
            labels = sorted(set(unregistered.get_column("Analyte").to_list()))
            warnings.append(
                f"{unregistered.height} kg row(s) of unregistered species excluded from apportionment: "
                f"{', '.join(labels)} — add their formulas to reference_data.SPECIES_FORMULAS to include them."
            )

    ci_rows = included.filter(pl.col("Ci_value").is_not_null()) if basis != "kg" else included.head(0)
    if ci_rows.height:
        cohorts = sorted(set(ci_rows.get_column("DecayDate").drop_nulls().to_list()))
        if len(cohorts) > 1:
            warnings.append(
                "Ci values mix decay baselines "
                + " and ".join(c.split(" ")[0] for c in cohorts)
                + " — summed activities are not decay-consistent (harmonizer: Stage 4)."
            )
    if included.height and "Inventory_RSD" in included.columns:
        n_zero = included.filter(pl.col("Inventory_RSD") == 0.0).height
        if n_zero:
            warnings.append(
                f"{n_zero} of {included.height} contributing row(s) have no assigned uncertainty "
                f"(Inventory_RSD = 0 means unassigned, not exact)."
            )

    # ---- aggregate ----
    def _aggregate(frame: pl.DataFrame, keys: List[str]) -> pl.DataFrame:
        frame = frame.with_columns(value_expr.alias("_value"))
        aggs = [
            pl.col("_value").sum().alias(value_col),
            pl.col("WasteSiteId").n_unique().alias("N_tanks"),
            pl.len().alias("N_rows"),
        ]
        out = frame.group_by(keys).agg(aggs) if keys else frame.select(aggs)
        if basis in ("wt%", "g/L"):
            # Denominators are a property of the LAYER group only -- the
            # per-member split shares its group's denominator ("Member" is
            # an output dimension, not a layer dimension).
            layer_keys = [k for k in keys if k in LAYER_COMPATIBLE_DIMS]
            denom = _layer_denominators(df, layer_keys)
            if layer_keys:
                out = out.join(denom, on=layer_keys, how="left")
            else:
                out = out.join(denom, how="cross")
            # Review finding: a null/zero denominator (Volume or
            # ComponentDensity missing) divided to inf/NaN with no warning
            # -- the one integrity failure the warning system didn't cover.
            # Blank the value and say so instead.
            denom_col = "_group_mass_g" if basis == "wt%" else "_group_volume_L"
            n_no_denom = out.filter(pl.col(denom_col).is_null() | (pl.col(denom_col) <= 0)).height
            if n_no_denom:
                msg = (
                    f"{n_no_denom} group(s) lack layer volume/density data "
                    f"(Volume/ComponentDensity missing or zero) — {basis} left blank there."
                )
                if msg not in warnings:  # _aggregate runs twice (table + member table)
                    warnings.append(msg)
            ratio = (
                pl.col(value_col) / pl.col("_group_mass_g") * 100.0 if basis == "wt%"
                else pl.col(value_col) / pl.col("_group_volume_L")
            )
            out = out.with_columns(
                pl.when(pl.col(denom_col) > 0).then(ratio).otherwise(None).alias(value_col)
            )
            out = out.drop(["_group_mass_g", "_group_volume_L"])
        return out.sort(value_col, descending=True, nulls_last=True)

    table = _aggregate(included, group_by).to_pandas()
    member_table = _aggregate(included, list(dict.fromkeys(["Member"] + group_by))).to_pandas()

    return QueryResult(
        table=table, member_table=member_table, warnings=warnings,
        disclosures=disclosures, value_column=value_col, basis=basis,
    )
