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
from elements import normalize_element_symbol, parse_isotope, primary_element_from_analyte
from reference_data import (
    COMBINED_PAIR_MEMBERS,
    SPECIES_TOTAL_MEMBERS,
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
    # How much the restatement rules set aside, in the basis's own unit
    # (kg, g or Ci). A consumer reconciling this table against the raw
    # reported total must subtract it, or it will attribute deliberately
    # suppressed material to some other cause -- which is exactly how the
    # matrix warning came to blame the parser for 430 t of carbon that
    # the engine had correctly declined to count twice.
    restated_value: float = 0.0


def apply_scope(df: pl.DataFrame, scope: Optional[Dict[str, Sequence[str]]]) -> pl.DataFrame:
    """Public: apply a scope dict to any composition frame. Views that do
    NOT go through run_query (legacy science paths kept for comparison)
    must filter through this same function, or "global scope" would mean
    different things in different workspaces."""
    return _apply_scope(df, scope)


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
            pl.lit(True).alias("MassCounts"),
            pl.lit(True).alias("ActivityCounts"),
            pl.lit(element).alias("SubjectElement"),
        ])
    return (
        pl.concat(pieces, how="vertical_relaxed")
        .with_columns([
            pl.lit(True).alias("MassCounts"),
            pl.lit(True).alias("ActivityCounts"),
            pl.lit(element).alias("SubjectElement"),
        ])
    )


def _suppress_restatements(contrib: pl.DataFrame) -> pl.DataFrame:
    """Flag rows that restate material another row already accounts for.

    The BBI describes the same inventory more than once, and adding any
    of those descriptions to itself inflates a published number. Three
    distinct cases, all resolved PER LAYER so a layer reporting only one
    form keeps it:

    MASS. A mass reported for an element includes its radionuclides: the
    BBI reports UTOTAL (all the uranium there is) and, separately, the
    activity of 232-238U -- the same metal, stated radiologically. Adding
    the bridged isotope grams to UTOTAL put 1,354 t of uranium on the
    front page against the 677 t reported (per-tank ratio 0.9926-1.0050
    in all 177 tanks; 235U/UTOTAL = 0.706 wt%, natural uranium). So on a
    mass basis a layer's Ci rows count only where that layer reports no
    mass for the SAME SUBJECT ELEMENT -- which is what makes this work for
    a basket whose members are "U" and "238U", where per-member handling
    saw no overlap at all.

    ACTIVITY. A combined-pair label is the sum of its members, and the
    BBI often reports both: every one of the 181 layers carrying
    239/240Pu also carries 239Pu and 240Pu. Counting all three inflated
    plutonium activity by 21.5% and curium by 66%. Where a layer reports
    a pair AND its members, exactly one representation is kept -- the
    itemised members when they are all present (more informative), the
    pair otherwise (dropping it would lose the member that is missing).

    MASS TOTALS. A reported total already contains the itemisation beside
    it: TOC is total organic carbon and holds the oxalate, formate,
    acetate and glycolate the BBI reports separately. Summing them
    counted 429.8 t of carbon twice (13.6% of all carbon, 35.8% of the
    organic fraction). Here the keep-rule inverts -- the total is a
    proper superset, so the TOTAL is kept and the members set aside, for
    the element the total describes only.
    """
    if contrib.is_empty() or "SubjectElement" not in contrib.columns:
        return contrib
    layer = [c for c in _LAYER_KEYS if c in contrib.columns]
    if not layer:
        return contrib

    # ---- mass: a reported mass already includes the nuclides ----
    key = layer + ["SubjectElement"]
    mass_here = (
        contrib.filter(
            (pl.col("Units") != "Ci")
            & pl.col("Mass_g").is_not_null()
            & (pl.col("Mass_g") > 0)  # a reported ZERO is not a mass that includes anything
        )
        .select(key).unique()
        .with_columns(pl.lit(True).alias("_mass_here"))
    )
    out = (
        # nulls_equal: WasteType is optional and the shipped CSV writes it
        # as "NA" on 869 rows, so 9 layers carry a null key. Without this a
        # layer does not match its own marker and suppression never fires
        # there -- uranium came back at 2.0004x its reported mass.
        contrib.join(mass_here, on=key, how="left", nulls_equal=True)
        .with_columns(
            (~((pl.col("Units") == "Ci") & pl.col("_mass_here").fill_null(False)))
            .alias("MassCounts")
        )
        .drop("_mass_here")
    )

    # ---- mass: a reported TOTAL already contains its itemisation ----
    # The third restatement shape, and the one where the keep-rule
    # inverts. A combined pair EQUALS its members, so the itemisation is
    # the better representation. A total is a proper SUPERSET -- TOC also
    # holds organics the BBI never itemises -- so the TOTAL is kept and
    # the members are set aside. Scoped to the element the total
    # describes: tributyl phosphate stops counting toward carbon, but its
    # phosphorus is still booked in full.
    subjects = set(out.get_column("SubjectElement").unique().to_list())
    for total, (element, members) in SPECIES_TOTAL_MEMBERS.items():
        if element not in subjects:
            continue
        total_here = (
            out.filter(
                (pl.col("Analyte") == total)
                & (pl.col("Units") != "Ci")
                & pl.col("Mass_g").is_not_null()
                & (pl.col("Mass_g") > 0)
            )
            .select(layer).unique()
            .with_columns(pl.lit(True).alias("_total_here"))
        )
        if total_here.is_empty():
            continue
        out = (
            out.join(total_here, on=layer, how="left", nulls_equal=True)
            .with_columns(
                (
                    pl.col("MassCounts")
                    & ~(
                        pl.col("_total_here").fill_null(False)
                        & pl.col("Analyte").is_in(list(members))
                        & (pl.col("SubjectElement") == element)
                    )
                ).alias("MassCounts")
            )
            .drop("_total_here")
        )

    # ---- activity: a combined pair restates its members ----
    keys = pair_restatement_keys(out)
    if keys.is_empty():
        return out
    flags = keys.with_columns(pl.lit(True).alias("_restates"))
    return (
        out.join(flags, on=list(keys.columns), how="left", nulls_equal=True)
        .with_columns((~pl.col("_restates").fill_null(False)).alias("ActivityCounts"))
        .drop("_restates")
    )


def pair_restatement_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Public: the (layer, Analyte) rows whose activity another row in the
    SAME layer already accounts for.

    Decided PER LAYER: one layer may itemise where the next gives only
    the pair, so a dataset-wide verdict would drop real activity. Where a
    layer holds both, exactly one representation survives -- the itemised
    members when all are present (more informative), the pair otherwise
    (dropping it would lose the member that is missing).

    Returns an empty frame carrying the right schema when nothing
    restates, so callers can join or anti-join unconditionally.
    """
    layer = [c for c in _LAYER_KEYS if c in df.columns]
    if not layer or "Analyte" not in df.columns:
        return pl.DataFrame(schema={"Analyte": pl.Utf8})
    # Schema pinned, not inferred: a layer key that is null on every
    # redundant row infers dtype Null, and the join then raises
    # SchemaError -- not a ValueError, so the workspaces' guards miss it.
    schema = {k: df.schema[k] for k in layer + ["Analyte"]}
    rows = df.filter(pl.col("Units") == "Ci") if "Units" in df.columns else df
    labels = {a for pair, members in COMBINED_PAIR_MEMBERS.items()
              for a in (pair,) + members}
    relevant = rows.filter(pl.col("Analyte").is_in(sorted(labels)))
    if relevant.is_empty():
        return pl.DataFrame(schema=schema)

    redundant: List[Dict[str, object]] = []
    for row in relevant.select(layer + ["Analyte"]).unique().group_by(layer).agg(
            pl.col("Analyte").alias("_analytes")).iter_rows(named=True):
        here = set(row["_analytes"])
        for pair, members in COMBINED_PAIR_MEMBERS.items():
            if pair not in here:
                continue
            itemised = [m for m in members if m in here]
            if not itemised:
                continue                     # only the pair: it is the representation
            for analyte in ([pair] if len(itemised) == len(members) else itemised):
                redundant.append({**{k: row[k] for k in layer}, "Analyte": analyte})
    if not redundant:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(redundant, schema=schema)


def drop_pair_restatements(df: pl.DataFrame) -> Tuple[pl.DataFrame, Optional[str]]:
    """Public: `df` without the activity rows that restate another row in
    the same layer, plus a sentence naming what was set aside.

    Every path that sums Ci rows must go through this. Four science
    modules re-implemented the raw group-and-sum instead, so Tank
    Explorer, Heatmaps, Correlations and Vitrification each published a
    combined pair on top of its own members and contradicted the Query
    workspace about the same tank, element and unit (review finding).
    """
    keys = pair_restatement_keys(df)
    if keys.is_empty():
        return df, None
    out = df.join(keys, on=list(keys.columns), how="anti", nulls_equal=True)
    labels = sorted(set(keys.get_column("Analyte").to_list()))
    note = (
        f"{df.height - out.height} Ci row(s) set aside as restatements "
        f"({', '.join(labels)}): a combined-pair label is the sum of its members, "
        f"and this dataset reports both — counting them together would add the "
        f"same activity twice."
    )
    return out, note


def _mass_restatement_warnings(contrib: pl.DataFrame) -> List[str]:
    """Name the rows set aside on a mass basis, and why. The two shapes
    have opposite causes -- one is an activity row a reported mass
    already covers, the other a mass row a reported TOTAL already
    contains -- so they get separate sentences rather than one that is
    only half true of each."""
    out: List[str] = []
    restated = contrib.filter(~pl.col("MassCounts"))
    if restated.is_empty():
        return out
    bridged = restated.filter(pl.col("Units") == "Ci")
    itemised = restated.filter(pl.col("Units") != "Ci")
    if bridged.height:
        labels = sorted(set(bridged.get_column("Analyte").to_list()))
        shown = ", ".join(labels[:6]) + (" …" if len(labels) > 6 else "")
        out.append(
            f"{bridged.height} Ci row(s) set aside on the mass basis ({shown}): a reported "
            f"elemental mass already includes its radionuclides, so adding the bridged grams "
            f"would count the same material twice."
        )
    if itemised.height:
        labels = sorted(set(itemised.get_column("Analyte").to_list()))
        totals = sorted({
            total for total, (_element, members) in SPECIES_TOTAL_MEMBERS.items()
            if any(label in members for label in labels)
        })
        out.append(
            f"{itemised.height} kg row(s) set aside on the mass basis "
            f"({', '.join(labels)}): {' and '.join(totals)} already contains them, and the "
            f"total is the more complete figure — it also holds species the BBI never itemises."
        )
    return out


def _analyte_contributions(df: pl.DataFrame, analyte: str) -> pl.DataFrame:
    rows = df.filter(pl.col("Analyte") == analyte)
    if rows.is_empty():
        return rows.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("Mass_g"),
            pl.lit(None, dtype=pl.Float64).alias("Ci_value"),
            pl.lit(None, dtype=pl.Utf8).alias("SourceKind"),
            pl.lit(True).alias("MassCounts"),
            pl.lit(True).alias("ActivityCounts"),
            pl.lit(None, dtype=pl.Utf8).alias("SubjectElement"),
        ])
    return rows.with_columns([
        pl.when(pl.col("Units") == "kg")
          .then(pl.col("Inventory") * 1000.0)
          .otherwise(pl.col("Inventory") * _grams_per_ci_expr())
          .alias("Mass_g"),
        pl.when(pl.col("Units") == "Ci").then(pl.col("Inventory")).otherwise(None).alias("Ci_value"),
        (pl.lit("analyte:") + pl.col("Units")).alias("SourceKind"),
        pl.lit(True).alias("MassCounts"),
        pl.lit(True).alias("ActivityCounts"),
        # Which element this analyte describes, so a basket mixing "U"
        # and "238U" is recognised as two views of one inventory.
        pl.lit(primary_element_from_analyte(analyte)).alias("SubjectElement"),
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
            unit = kind.split(":", 1)[1]
            if unit == "Ci":
                # Name the bridge here too. The ELEMENT path over these same
                # rows says "Ci→g bridged, ICRP-107" and the analyte path
                # did not, so a template promising "129I (bridged)" produced
                # notes that never mentioned ICRP at all (review finding).
                # NOT routed through the "Ci:" branch above: that also
                # asserts mass is a lower bound, which is false when the
                # subject IS the nuclide.
                label = f"analyte rows (Ci→g bridged, ICRP-107)"
                if nuclide_grams_per_ci(name) is None:
                    label = ("analyte rows (Ci) — no single ICRP-107 specific activity, "
                             "so this subject carries no mass")
                parts.append(label)
            else:
                parts.append(f"analyte rows ({unit})")
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
    if "Element" in group_by and kind in ("element", "basket"):
        # The value is the SUBJECT's apportioned mass; the Element column
        # names the SOURCE ROW's primary element. Grouping oxygen by
        # Element produced a row labelled "N" holding oxygen's share of
        # the nitrate -- 3.1x the real nitrogen total (review finding).
        raise ValueError(
            "Grouping by Element labels a subject's apportioned mass with the source row's "
            "own element, which is a different quantity. Group by Analyte to see which "
            "species carry the subject, or query a single analyte instead."
        )
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
                pl.lit(True).alias("MassCounts"),
                pl.lit(True).alias("ActivityCounts"),
                pl.lit(None, dtype=pl.Utf8).alias("SubjectElement"),
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

    # Restatement is resolved ONCE, over the assembled frame: a basket
    # whose members are "U" and "238U" overlaps ACROSS members, which
    # per-member handling could never see (review finding).
    contrib = _suppress_restatements(contrib)

    # ---- basis: choose the value column and inclusion rule ----
    if basis == "Ci":
        included = contrib.filter(pl.col("Ci_value").is_not_null() & pl.col("ActivityCounts"))
        value_expr = pl.col("Ci_value")
        value_col = "Total_Ci"
        n_kg_dropped = contrib.filter(pl.col("Ci_value").is_null()).height
        if n_kg_dropped:
            warnings.append(f"{n_kg_dropped} kg-basis row(s) excluded: activity basis carries Ci rows only.")
        restated = contrib.filter(pl.col("Ci_value").is_not_null() & ~pl.col("ActivityCounts"))
        if restated.height:
            labels = sorted(set(restated.get_column("Analyte").to_list()))
            warnings.append(
                f"{restated.height} Ci row(s) set aside as restatements ({', '.join(labels)}): "
                f"a combined-pair label is the sum of its members, and this dataset reports "
                f"both — counting them together would add the same activity twice."
            )
    elif basis == "kg":
        # MassCounts gates this basis too: a kg TOTAL restating a kg
        # itemisation (TOC over the organic anions) is a kg-vs-kg overlap
        # that the unit filter alone cannot see.
        included = contrib.filter((pl.col("Units") == "kg") & pl.col("MassCounts"))
        value_expr = pl.col("Mass_g") / 1000.0
        value_col = "Total_kg"
        n_ci_dropped = contrib.filter(pl.col("Units") == "Ci").height
        if n_ci_dropped:
            warnings.append(
                f"{n_ci_dropped} Ci-basis row(s) excluded under the kg basis; use basis 'g' to bridge them."
            )
        warnings.extend(_mass_restatement_warnings(contrib.filter(pl.col("Units") == "kg")))
    else:  # g, wt%, g/L -- mass-based
        included = contrib.filter(pl.col("Mass_g").is_not_null() & pl.col("MassCounts"))
        value_expr = pl.col("Mass_g")
        value_col = {"g": "Total_g", "wt%": "wt_pct", "g/L": "g_per_L"}[basis]
        warnings.extend(_mass_restatement_warnings(contrib))
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
            # State the MASS, not just the row count: on the shipped
            # dataset these 26 rows carry 0.0024 kg out of 149 million,
            # and a warning that sounds alarming about nothing teaches
            # people to stop reading warnings.
            excluded_kg = float(unregistered.get_column("Inventory").sum() or 0.0)
            warnings.append(
                f"{unregistered.height} kg row(s) of unregistered species excluded from apportionment "
                f"({excluded_kg:,.4g} kg total): {', '.join(labels)} — add their formulas to "
                f"reference_data.SPECIES_FORMULAS to include them."
            )
        # The warning above only sees analytes the parser resolved into a
        # formula. Ones it could not resolve AT ALL (Aroclors, ketones,
        # phthalates) reach no element either, and used to be reported by
        # nothing -- so the one figure a reader saw understated the
        # unbooked mass by four orders of magnitude (review finding).
        unresolvable = df.filter(
            (pl.col("Units") == "kg")
            & (pl.col("AnalyteClass") == "unparsed")
            & (pl.col("Inventory") > 0)
        )
        if unresolvable.height:
            unresolved_kg = float(unresolvable.get_column("Inventory").sum() or 0.0)
            n_labels = unresolvable.get_column("Analyte").n_unique()
            warnings.append(
                f"{unresolvable.height} kg row(s) across {n_labels} analyte(s) the parser "
                f"cannot resolve to elements ({unresolved_kg:,.4g} kg total) reach no element "
                f"and are absent from every elemental figure."
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
            # CONTRIBUTIONS, not dataset rows: one kg row of a species
            # feeds every element it contains, so a basket spanning S and
            # O sees a single SO4 row twice. Calling that "N_rows" told
            # the reader a row count that is not one (review finding).
            pl.len().alias("N_contributions"),
        ]
        out = frame.group_by(keys).agg(aggs) if keys else frame.select(aggs)
        if basis in ("wt%", "g/L"):
            # Denominators are a property of the LAYER group only -- the
            # per-member split shares its group's denominator ("Member" is
            # an output dimension, not a layer dimension).
            layer_keys = [k for k in keys if k in LAYER_COMPATIBLE_DIMS]
            denom = _layer_denominators(df, layer_keys)
            if layer_keys:
                # nulls_equal: group_by keeps null keys, so a null-keyed
                # group HAS a denominator row -- without this it never
                # matched its own, and the blanked value was reported
                # under a warning blaming absent volume/density data.
                out = out.join(denom, on=layer_keys, how="left", nulls_equal=True)
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

    # The same value expression over the rows the restatement rules set
    # aside, so "reported - counted" is explainable rather than residual.
    if basis == "Ci":
        set_aside = contrib.filter(pl.col("Ci_value").is_not_null() & ~pl.col("ActivityCounts"))
    elif basis == "kg":
        set_aside = contrib.filter((pl.col("Units") == "kg") & ~pl.col("MassCounts"))
    else:
        set_aside = contrib.filter(pl.col("Mass_g").is_not_null() & ~pl.col("MassCounts"))
    restated_value = (
        float(set_aside.select(value_expr.alias("_v")).get_column("_v").sum() or 0.0)
        if set_aside.height else 0.0
    )

    return QueryResult(
        table=table, member_table=member_table, warnings=warnings,
        disclosures=disclosures, value_column=value_col, basis=basis,
        restated_value=restated_value,
    )
