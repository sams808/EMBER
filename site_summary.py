"""
site_summary.py -- the per-tank and site-wide figures behind the Site
overview dashboard and the tank profile (framework-agnostic: no Qt).

Nothing here does inventory arithmetic of its own. Component values come
from query_engine.run_query and association_query.association_matrix, so
a number on the map is the number the Query workspace gives for the same
question, with the same restatement rules, unit bridging and disclosures.
The only quantity assembled locally is the waste volume of a tank, which
is the sum over its distinct layer cohorts: the rule the engine uses for
its own g/L and wt% denominators (checked against it in the tests).

Unreported is not zero: a tank the engine returns no row for is absent
from `TankValues.values`; a reported zero stays a zero.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl

import association_query as aq
from data_model import HanfordDataset
from query_engine import Basket, apply_scope, run_query

GALLONS_PER_KL = 264.172052
WASTE_VOLUME = "Waste volume"
MOL_PER_MOL_NA = aq.MOL_PER_MOL_NA


@dataclass(frozen=True)
class MapBasis:
    key: str
    label: str        # what the basis selector shows
    unit: str         # what follows a value
    additive: bool    # True: tank values add up to a farm total


MAP_BASES: Dict[str, MapBasis] = {
    "kg": MapBasis("kg", "Inventory (kg)", "kg", True),
    "Ci": MapBasis("Ci", "Activity (Ci)", "Ci", True),
    MOL_PER_MOL_NA: MapBasis(MOL_PER_MOL_NA, "mol per mol Na", "mol per mol Na", False),
    "wt%": MapBasis("wt%", "wt% of waste", "wt% of waste", False),
    "g/L": MapBasis("g/L", "g/L", "g/L", False),
}

# Offered first in the component selector, when the dataset reports them.
COMMON_COMPONENTS = [
    "Na", "Al", "NO3", "NO2", "SO4", "PO4", "F", "Cl", "Cr", "K", "Fe", "U",
    "137Cs", "90Sr", "99Tc", "129I",
]

_ALIASES = {
    "sulfate": "SO4", "sulphate": "SO4", "nitrate": "NO3", "nitrite": "NO2",
    "phosphate": "PO4", "carbonate": "TIC as CO3", "hydroxide": "Free OH",
    "sodium": "Na", "aluminum": "Al", "aluminium": "Al", "chromium": "Cr",
    "cesium": "Cs", "caesium": "Cs", "strontium": "Sr", "technetium": "Tc",
    "uranium": "U", "plutonium": "Pu", "iodine": "I", "fluoride": "F",
    "chloride": "Cl", "potassium": "K", "iron": "Fe", "volume": WASTE_VOLUME,
}
_ISOTOPE_DASH = re.compile(r"^([A-Za-z]{1,2})[-\s]?(\d{1,3}m?)$")


@dataclass
class TankValues:
    """One mapped quantity: a value per tank, with what qualifies it."""

    component: str = ""
    basis: str = "kg"
    label: str = ""                 # e.g. "SO4"
    unit: str = ""                  # e.g. "mol per mol Na"
    additive: bool = True
    values: Dict[str, float] = field(default_factory=dict)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)   # ranked, one row per tank with a value
    warnings: List[str] = field(default_factory=list)
    disclosures: List[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        if self.component == WASTE_VOLUME:
            return "Waste volume (kL)"
        return f"{self.label} · {MAP_BASES[self.basis].label}" if self.basis in MAP_BASES else self.label


@dataclass
class SiteStats:
    n_tanks: int = 0
    n_tanks_total: int = 0
    n_single_shell: Optional[int] = None
    n_double_shell: Optional[int] = None
    volume_kL: Optional[float] = None
    n_layers: int = 0
    n_records: int = 0
    n_records_total: int = 0
    n_analytes: int = 0
    n_elements: int = 0
    sodium_kg: Optional[float] = None
    sodium_wt_pct: Optional[float] = None
    cs137_ci: Optional[float] = None
    sr90_ci: Optional[float] = None


# ---------------------------------------------------------------------------
# tanks, layers, volumes
# ---------------------------------------------------------------------------
_TANK_COLUMNS = [
    "TankFarm", "TankSystem", "TankType", "Capacity", "CapacityUnits",
    "TankStatus", "TankIntegrity", "InServiceDate", "Ventilation",
]
_LAYER_KEYS = ["WasteSiteId", "WastePhase", "WasteType"]


def _layer_cohorts(df: pl.DataFrame) -> pl.DataFrame:
    """Distinct (layer, Volume, ComponentDensity) rows: one per stated
    waste volume. A layer label can carry more than one cohort, and its
    volume is their sum (the engine's own rule for denominators)."""
    cols = [c for c in _LAYER_KEYS + ["Volume", "ComponentDensity"] if c in df.columns]
    return df.select(cols).unique()


def tank_table(dataset: HanfordDataset,
               scope: Optional[Dict[str, Sequence[str]]] = None) -> pd.DataFrame:
    """One row per tank in scope: farm, system, type, and the waste it
    holds (Volume_kL, WasteMass_kg, N_layers). Volume_kL is blank where
    the dataset states no volume, never zero."""
    df = apply_scope(dataset.require_df(), scope)
    keep = ["WasteSiteId"] + [c for c in _TANK_COLUMNS if c in df.columns]
    tanks = df.select(keep).unique(subset=["WasteSiteId"], keep="first")
    if {"Volume", "ComponentDensity", "WastePhase", "WasteType"} <= set(df.columns):
        cohorts = _layer_cohorts(df).with_columns(
            (pl.col("Volume") * pl.col("ComponentDensity") * 1000.0).alias("_kg"))
        size = cohorts.group_by("WasteSiteId").agg([
            pl.when(pl.col("Volume").count() > 0).then(pl.col("Volume").sum())
              .otherwise(None).alias("Volume_kL"),
            pl.when(pl.col("_kg").count() > 0).then(pl.col("_kg").sum())
              .otherwise(None).alias("WasteMass_kg"),
            pl.struct(["WastePhase", "WasteType"]).n_unique().alias("N_layers"),
        ])
        tanks = tanks.join(size, on="WasteSiteId", how="left")
    return tanks.sort("WasteSiteId").to_pandas()


def tank_layers(dataset: HanfordDataset, tank_id: str) -> pd.DataFrame:
    """The waste layers of one tank: phase, waste type, volume, density."""
    df = dataset.require_df().filter(pl.col("WasteSiteId") == tank_id)
    cohorts = _layer_cohorts(df)
    if cohorts.is_empty():
        return pd.DataFrame(columns=["Waste phase", "Waste type", "Volume (kL)", "Density (g/mL)"])
    rename = {"WastePhase": "Waste phase", "WasteType": "Waste type",
              "Volume": "Volume (kL)", "ComponentDensity": "Density (g/mL)"}
    out = cohorts.drop("WasteSiteId").rename({k: v for k, v in rename.items() if k in cohorts.columns})
    if "Volume (kL)" in out.columns:
        out = out.sort("Volume (kL)", descending=True, nulls_last=True)
    return out.to_pandas()


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------
def component_choices(dataset: HanfordDataset) -> List[str]:
    """Completion list for the component selector: the waste volume, the
    components people ask for first, then every element and analyte."""
    df = dataset.require_df()
    analytes = sorted({str(a) for a in df.get_column("Analyte").drop_nulls().unique().to_list()})
    elements = dataset.available_elements()
    known = set(analytes) | set(elements)
    out = [WASTE_VOLUME] + [c for c in COMMON_COMPONENTS if c in known]
    for name in elements + analytes:
        if name not in out:
            out.append(name)
    return out


def resolve_component(dataset: HanfordDataset, text: str) -> str:
    """The dataset's own spelling of what was typed: an exact analyte or
    element first, then a case-insensitive match, then a few plain names
    ("sulfate", "Cs-137"). Unknown text is returned as typed; the engine
    then reports it as not found."""
    text = str(text or "").strip()
    if not text:
        return ""
    if text.lower() == WASTE_VOLUME.lower():
        return WASTE_VOLUME
    df = dataset.require_df()
    analytes = {str(a) for a in df.get_column("Analyte").drop_nulls().unique().to_list()}
    elements = set(dataset.available_elements())
    if text in analytes or text in elements:
        return text
    alias = _ALIASES.get(text.lower())
    if alias:
        return alias
    lowered = {name.lower(): name for name in sorted(analytes | elements)}
    if text.lower() in lowered:
        return lowered[text.lower()]
    match = _ISOTOPE_DASH.match(text)
    if match:
        candidate = f"{match.group(2)}{match.group(1).capitalize()}"
        if candidate in analytes:
            return candidate
    return text


def tank_values(dataset: HanfordDataset, component: str, basis: str = "kg",
                scope: Optional[Dict[str, Sequence[str]]] = None) -> TankValues:
    """Value of one component in every tank in scope, on one basis.

    basis: "kg" (inventory, activities bridged to mass), "Ci" (activity),
    "mol/mol Na", "wt%" (of waste mass) or "g/L" (of waste volume).
    The component "Waste volume" ignores the basis and returns kL.
    """
    component = resolve_component(dataset, component)
    tanks = tank_table(dataset, scope)
    if component == WASTE_VOLUME:
        series = (tanks.set_index("WasteSiteId")["Volume_kL"]
                  if "Volume_kL" in tanks.columns else pd.Series(dtype=float))
        out = TankValues(component=WASTE_VOLUME, basis="kg", label=WASTE_VOLUME,
                         unit="kL", additive=True)
        if series.dropna().empty:
            out.warnings.append("This dataset states no waste volumes.")
    else:
        if basis not in MAP_BASES:
            raise ValueError(f"Unknown basis {basis!r}; expected one of {list(MAP_BASES)}")
        if not component:
            raise ValueError("Enter a component: an element (Na), an analyte (SO4) or a nuclide (137Cs).")
        spec = MAP_BASES[basis]
        if basis == MOL_PER_MOL_NA:
            # The matrix needs two members; sodium is the natural second
            # one (it is the denominator). Its own notes are kept out.
            partner = "Al" if component == "Na" else "Na"
            matrix = aq.association_matrix(dataset, [component, partner], basis=basis,
                                           level="tank", scope=scope)
            name = matrix.members[0]
            series = matrix.values.set_index("WasteSiteId")[name]
            warnings = list(matrix.warnings)
            disclosures = [d for d in matrix.disclosures if not d.startswith(f"{partner}:")]
        else:
            engine_basis = "g" if basis == "kg" else basis
            result = run_query(dataset, subject=("basket", Basket(component, (component,))),
                               basis=engine_basis, group_by=["WasteSiteId"], scope=scope)
            name = (str(result.member_table["Member"].iloc[0])
                    if not result.member_table.empty else component)
            series = (result.table.set_index("WasteSiteId")[result.value_column]
                      if not result.table.empty else pd.Series(dtype=float))
            if basis == "kg":
                series = series / 1000.0
            warnings, disclosures = list(result.warnings), list(result.disclosures)
        out = TankValues(component=component, basis=basis, label=name, unit=spec.unit,
                         additive=spec.additive, warnings=warnings, disclosures=disclosures)

    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[np.isfinite(series.to_numpy(dtype=float))]
    out.values = {str(tank): float(value) for tank, value in series.items()}
    ranked = series.sort_values(ascending=False, kind="mergesort").rename("Value").reset_index()
    ranked.columns = ["WasteSiteId", "Value"]
    extra = [c for c in ("TankFarm", "TankSystem", "Volume_kL") if c in tanks.columns]
    out.table = ranked.merge(tanks[["WasteSiteId"] + extra], on="WasteSiteId", how="left")
    return out


# ---------------------------------------------------------------------------
# site and tank headline figures
# ---------------------------------------------------------------------------
def engine_total(dataset: HanfordDataset, subject, basis: str,
                 scope: Optional[Dict[str, Sequence[str]]] = None) -> Optional[float]:
    """One ungrouped engine figure (a total, or a site-wide wt% or g/L),
    or None when nothing in scope reports the subject."""
    result = run_query(dataset, subject=subject, basis=basis, scope=scope)
    if result.table.empty or int(result.table["N_contributions"].iloc[0]) == 0:
        return None
    return float(result.table[result.value_column].iloc[0])


def site_stats(dataset: HanfordDataset,
               scope: Optional[Dict[str, Sequence[str]]] = None) -> SiteStats:
    """The headline numbers of the dataset under a scope."""
    full = dataset.require_df()
    df = apply_scope(full, scope)
    tanks = tank_table(dataset, scope)
    stats = SiteStats(
        n_tanks=len(tanks),
        n_tanks_total=full.get_column("WasteSiteId").n_unique(),
        n_records=df.height,
        n_records_total=full.height,
        n_analytes=df.get_column("Analyte").n_unique() if df.height else 0,
        n_elements=(df.filter(pl.col("Element").is_not_null()).get_column("Element").n_unique()
                    if df.height else 0),
    )
    if "TankSystem" in tanks.columns:
        stats.n_single_shell = int((tanks["TankSystem"] == "SST").sum())
        stats.n_double_shell = int((tanks["TankSystem"] == "DST").sum())
    if "Volume_kL" in tanks.columns and tanks["Volume_kL"].notna().any():
        stats.volume_kL = float(tanks["Volume_kL"].sum(skipna=True))
    if "N_layers" in tanks.columns:
        stats.n_layers = int(tanks["N_layers"].fillna(0).sum())
    if df.height:
        stats.sodium_kg = engine_total(dataset, ("element", "Na"), "kg", scope)
        if stats.sodium_kg is not None and stats.volume_kL:
            stats.sodium_wt_pct = engine_total(dataset, ("element", "Na"), "wt%", scope)
        stats.cs137_ci = engine_total(dataset, ("analyte", "137Cs"), "Ci", scope)
        stats.sr90_ci = engine_total(dataset, ("analyte", "90Sr"), "Ci", scope)
    return stats


def tank_headlines(dataset: HanfordDataset, tank_ids: Sequence[str]) -> pd.DataFrame:
    """One row per requested tank: attributes, waste volume, layers,
    sodium (kg and wt% of the tank's waste) and 137Cs and 90Sr activity
    (Ci), through the engine."""
    tank_ids = [str(t) for t in tank_ids]
    scope = {"WasteSiteId": tank_ids}
    table = tank_table(dataset, scope)
    has_mass = "WasteMass_kg" in table.columns and table["WasteMass_kg"].notna().any()
    for column, subject, basis in (
        ("Na_kg", ("element", "Na"), "kg"),
        ("Na_wt_pct", ("element", "Na"), "wt%"),
        ("Cs137_Ci", ("analyte", "137Cs"), "Ci"),
        ("Sr90_Ci", ("analyte", "90Sr"), "Ci"),
    ):
        if basis == "wt%" and not has_mass:
            table[column] = float("nan")      # no stated waste mass: no concentration
            continue
        result = run_query(dataset, subject=subject, basis=basis,
                           group_by=["WasteSiteId"], scope=scope)
        values = (result.table.set_index("WasteSiteId")[result.value_column]
                  if not result.table.empty else pd.Series(dtype=float))
        table[column] = table["WasteSiteId"].map(values)
    order = {tank: i for i, tank in enumerate(tank_ids)}
    return table.sort_values("WasteSiteId", key=lambda s: s.map(order)).reset_index(drop=True)


def scope_sentence(scope: Optional[Dict[str, Sequence[str]]], n_tanks: int, n_total: int) -> str:
    """The active scope in words, for a page subtitle."""
    names = {"TankFarm": "farm", "WastePhase": "waste phase", "WasteType": "waste type",
             "TankSystem": "tank system"}
    systems = {"SST": "single-shell", "DST": "double-shell"}
    parts = []
    for column, values in (scope or {}).items():
        if not values:
            continue
        shown = [systems.get(v, v) if column == "TankSystem" else str(v) for v in values]
        parts.append(f"{names.get(column, column)} {', '.join(shown)}")
    if not parts:
        return f"All {n_total:,} tanks, every waste phase and waste type."
    return f"Scope: {'; '.join(parts)}. {n_tanks:,} of {n_total:,} tanks."


# ---------------------------------------------------------------------------
# display formatting (no Qt)
# ---------------------------------------------------------------------------
_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
BLANK = "—"


def _is_blank(value: object) -> bool:
    return value is None or (isinstance(value, float) and not math.isfinite(value))


def format_number(value: object, sig: int = 3) -> str:
    """A number at `sig` significant figures with thousands separators;
    powers of ten for the very small. Blank for a missing value."""
    if _is_blank(value):
        return BLANK
    value = float(value)
    if value == 0:
        return "0"
    magnitude = abs(value)
    exponent = math.floor(math.log10(magnitude))
    if exponent < -4:
        mantissa = value / 10 ** exponent
        return f"{mantissa:.{max(sig - 1, 0)}f}×10{str(exponent).translate(_SUPERSCRIPT)}"
    rounded = round(value, sig - 1 - exponent)
    if magnitude >= 1000:
        return f"{rounded:,.0f}"
    decimals = max(sig - 1 - exponent, 0)
    text = f"{rounded:.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_compact(value: object, sig: int = 3) -> str:
    """A short form for legends: 12.3k, 3.89M."""
    if _is_blank(value):
        return BLANK
    value = float(value)
    magnitude = abs(value)
    for threshold, suffix in ((1e9, "G"), (1e6, "M"), (1e4, "k")):
        if magnitude >= threshold:
            return f"{format_number(value / (threshold if suffix != 'k' else 1e3), sig)}{suffix}"
    return format_number(value, sig)


def format_volume(kl: object) -> str:
    if _is_blank(kl):
        return BLANK
    kl = float(kl)
    return f"{format_number(kl / 1000.0)} ML" if kl >= 10_000 else f"{format_number(kl)} kL"


def format_gallons(kl: object) -> str:
    if _is_blank(kl):
        return ""
    gallons = float(kl) * GALLONS_PER_KL
    if gallons >= 1e6:
        return f"{format_number(gallons / 1e6)} million gallons"
    if gallons >= 1e4:
        return f"{format_number(gallons / 1e3)} thousand gallons"
    return f"{format_number(gallons)} gallons"


def format_mass(kg: object) -> str:
    if _is_blank(kg):
        return BLANK
    kg = float(kg)
    return f"{format_number(kg / 1000.0)} t" if abs(kg) >= 1000 else f"{format_number(kg)} kg"


def format_activity(ci: object) -> str:
    if _is_blank(ci):
        return BLANK
    ci = float(ci)
    if abs(ci) >= 1e6:
        return f"{format_number(ci / 1e6)} MCi"
    if abs(ci) >= 1e3:
        return f"{format_number(ci / 1e3)} kCi"
    return f"{format_number(ci)} Ci"


def format_activity_split(parts: Sequence[Tuple[str, Optional[float]]]) -> Tuple[str, str, str]:
    """A few nuclide activities as (total, split, sentence): the total with
    its unit, the parts in that same unit without repeating it (a stat
    tile is narrow), and the full statement for a tooltip. Parts that are
    not reported (None) are left out, not counted as zero."""
    known = [(name, float(value)) for name, value in parts if not _is_blank(value)]
    if not known:
        return BLANK, "", ""
    total = sum(value for _name, value in known)
    scale, unit = (1e6, "MCi") if abs(total) >= 1e6 else ((1e3, "kCi") if abs(total) >= 1e3 else (1.0, "Ci"))
    if len(known) == 1:
        split = f"{known[0][0]} only"
    else:
        split = " · ".join(f"{name} {format_number(value / scale)}" for name, value in known)
    sentence = " and ".join(f"{name} {format_activity(value)}" for name, value in known)
    return f"{format_number(total / scale)} {unit}", split, sentence


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n:,}{suffix}"
