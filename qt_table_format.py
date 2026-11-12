"""
qt_table_format.py -- how a value and a column name are written for a person.

One place for the two things every table and every chart needs:

- `humanize(name)`: a column identifier ("WasteSiteId", "Total_kg",
  "User_search_score_proxy") as a readable label with its unit ("Tank",
  "Total (kg)", "Search score (proxy)"). Table headers, axis labels and
  legend titles all read the same table, so a quantity has one name
  everywhere it appears.
- `format_number(value)` / `format_cell(value)`: about four significant
  figures, thousands separators, scientific notation only for the very
  small and the very large, blank for a missing value.

Pure Python on purpose (no Qt import): the plot layers use it too, and
they must stay importable without a QApplication.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column labels
# ---------------------------------------------------------------------------

# Identifier -> label. Only names whose generic reading (see humanize)
# would be wrong, clumsy or unit-less need an entry here.
COLUMN_LABELS: Dict[str, str] = {
    # identity
    "WasteSiteId": "Tank",
    "Name": "Tank",
    "BaseTank": "Base tank",
    "PartnerTank": "Partner tank",
    "TankFarm": "Farm",
    "TankSystem": "System",
    "TankType": "Tank type",
    "TankStatus": "Status",
    "TankIntegrity": "Integrity",
    "WastePhase": "Waste phase",
    "WasteType": "Waste type",
    "PhaseFamily": "Phase family",
    "AnalyteClass": "Analyte class",
    "ElementList": "Elements",
    "ElementListPadded": "Elements (padded)",
    "MassNumber": "Mass number",
    "IsMetastable": "Metastable",
    "IsDST": "Double-shell",
    "IsSST": "Single-shell",
    "IsLeakerOrAssumedLeaker": "Leaker or assumed leaker",
    "IsHighlighted": "Highlighted",
    "HasWIStatus": "Has WI status",
    "HasTankAttributes": "Has attributes",
    "Member": "Member",
    "Component": "Component",
    "Reported_as": "Reported as",
    "PartnerElement": "Partner element",
    "ExcludedElement": "Excluded element",
    "Element_A": "Element A",
    "Element_B": "Element B",
    "PC": "Component",
    "Cluster": "Cluster",
    "Dominant waste phase": "Dominant waste phase",
    "Dominant_waste_type": "Dominant waste type",
    "Dominant_waste_phase": "Dominant waste phase",
    "RelativeActivityBin": "Relative activity",
    "Important_warning": "Note",
    "topic": "Topic",
    "note": "Note",
    "kind": "Kind",
    "text": "Text",
    "column": "Column",
    "dtype": "Type",
    "min": "Minimum",
    "mean": "Mean",
    "median": "Median",
    "max": "Maximum",
    "sum": "Sum",
    "WastePhase_list": "Waste phases",
    "WasteType_list": "Waste types",
    "source_file": "Source file",
    "cache_used": "Cache used",
    "tank_attributes_file": "Attributes file",
    "tank_attributes_rows": "Attribute rows",
    "load_seconds": "Load time (s)",
    "estimated_size_mb": "Size in memory (MB)",
    # inventory
    "Units": "Unit",
    "Inventory": "Inventory",
    "Inventory_sum": "Inventory",
    "Inventory_RSD": "Rel. std. dev.",
    "VolumeRSD": "Volume rel. std. dev.",
    "VolumeUnits": "Volume unit",
    "DensityUnits": "Density unit",
    "ComponentDensity": "Density",
    "AdjustedConcentration": "Adjusted concentration",
    "AdjustedConcentrationUnits": "Concentration unit",
    "DecayDate": "Decay date",
    "PublishedDate": "Published",
    "CCBLog": "CCB log",
    "TotalInventory": "Total inventory",
    "MinInventory": "Smallest inventory",
    "MaxInventory": "Largest inventory",
    "Total_g": "Total (g)",
    "Total_kg": "Total (kg)",
    "Total_Ci": "Total (Ci)",
    "Total_wt%": "Concentration (wt%)",
    "Total_g/L": "Concentration (g/L)",
    "Total_kg_inventory": "Chemical inventory (kg)",
    "Total_Ci_inventory": "Radiological inventory (Ci)",
    "Total_inventory_kg": "Total inventory (kg)",
    "Mean_inventory_per_tank_kg": "Mean inventory per tank (kg)",
    "Tank_total_same_unit": "Tank total, same unit",
    "Context_total": "Tank total",
    "Context_TotalInventory_same_unit": "Tank total, same unit",
    "Target_Inventory_sum": "Target inventory",
    "Fraction_of_tank": "Fraction of tank",
    "Fraction_of_tank_unit_inventory": "Fraction of tank inventory",
    "Total_inventory_in_target_tanks": "Total in target tanks",
    "Mean_inventory_present_tanks_only": "Mean where present",
    "Median_inventory_present_tanks_only": "Median where present",
    "Std_inventory_present_tanks_only": "Std. dev. where present",
    "Max_inventory_in_one_tank": "Largest in one tank",
    "Mean_inventory_all_target_tanks_zero_filled": "Mean over all target tanks",
    "Mean_fraction_present_tanks_only": "Mean fraction where present",
    "Median_fraction_present_tanks_only": "Median fraction where present",
    "Std_fraction_present_tanks_only": "Std. dev. of fraction where present",
    "Max_fraction_in_one_tank": "Largest fraction in one tank",
    "Sum_fraction_present": "Sum of fractions",
    "Mean_fraction_all_target_tanks_zero_filled": "Mean fraction over all target tanks",
    "PresenceFraction_pct": "Tanks present (%)",
    "Mean_kg_present_tanks_only": "Mean where present (kg)",
    "Median_kg_present_tanks_only": "Median where present (kg)",
    "Std_kg_present_tanks_only": "Std. dev. where present (kg)",
    "Max_kg_in_one_tank": "Largest in one tank (kg)",
    "WasteMass_kg": "Waste mass (kg)",
    "Volume_kL": "Volume (kL)",
    "Unassigned_uncertainty_pct": "No uncertainty assigned (%)",
    "Default_100pct_uncertainty_pct": "Default 100 % uncertainty (%)",
    # counts
    "rows": "Rows",
    "columns": "Columns",
    "n_tanks": "Tanks",
    "n_analytes": "Analytes",
    "n_primary_elements": "Elements",
    "n_waste_phases": "Waste phases",
    "n_waste_types": "Waste types",
    "n_units": "Units",
    "n_null": "Missing",
    "pct_null": "Missing (%)",
    "n_nonnull": "Values",
    "N": "N",
    "N_rows": "Rows",
    "N_tanks": "Tanks",
    "N_analytes": "Analytes",
    "N_elements": "Elements",
    "N_PRIMARY_ELEMENTS": "Elements",
    "N_contributions": "Contributions",
    "N_waste_types": "Waste types",
    "N_waste_phases": "Waste phases",
    "N_tanks_present": "Tanks present",
    "N_tanks_total": "Tanks in scope",
    "N_tanks_positive": "Tanks above zero",
    "N_tanks_in_scope": "Tanks in scope",
    "N_target_tanks_total": "Target tanks",
    "N_leaker_or_assumed": "Leakers or assumed leakers",
    "N_overlap_nonzero_tanks": "Tanks with both",
    "N_both_present": "Tanks with both",
    "N_both_positive": "Rows with both above zero",
    "N_either_present": "Tanks with either",
    "N_A_present": "Tanks with A",
    "N_B_present": "Tanks with B",
    "N_tanks_used_for_corr": "Tanks used",
    "N_tanks_all_present": "Tanks with all",
    "N_tanks_any_present": "Tanks with any",
    "N_high": "High members",
    "High_members": "Members that are high",
    "N_edges": "Links",
    "N_selected_elements_present": "Elements present",
    # tank attributes
    "Capacity": "Capacity",
    "CapacityUnits": "Capacity unit",
    "DiameterUnits": "Diameter unit",
    "MaxOperatingDepth": "Max operating depth",
    "MaxOperatingDepthUnits": "Depth unit",
    "InServiceDate": "In service",
    "InServiceYear": "In-service year",
    "InterimStabilization": "Interim stabilization",
    "IntrusionPrevention": "Intrusion prevention",
    "OperationCapacityKGal": "Operating capacity (kgal)",
    "DIL_Gal": "Drainable interstitial liquid (gal)",
    "Total_capacity_kgal": "Total capacity (kgal)",
    "Mean_capacity_kgal": "Mean capacity (kgal)",
    "Total_operation_capacity_kgal": "Total operating capacity (kgal)",
    "Total_DIL_gal": "Drainable interstitial liquid (gal)",
    # association
    "r": "r",
    "abs_r": "|r|",
    "p_value": "p value",
    "Correlation_r": "Correlation r",
    "AbsCorrelation": "|r|",
    "Jaccard_presence": "Jaccard co-presence",
    "Metric_label": "Metric label",
    "Include_zeros": "Zeros included",
    "Rank_preferred_association": "Rank",
    "OverlapFraction_of_A_pct": "Overlap, share of A (%)",
    "OverlapFraction_of_B_pct": "Overlap, share of B (%)",
    "Total_A_kg": "Total A (kg)",
    "Total_B_kg": "Total B (kg)",
    "Shared_min_inventory_proxy_kg": "Shared inventory, proxy (kg)",
    "PreferredAssociationScore_proxy": "Association score (proxy)",
    "Fraction_all_present_pct": "Tanks with all (%)",
    "Mean_pairwise_correlation": "Mean pairwise r",
    "Min_pairwise_correlation": "Lowest pairwise r",
    "Max_pairwise_correlation": "Highest pairwise r",
    "ExplainedVarianceRatio": "Explained variance",
    "CumulativeVarianceRatio": "Cumulative variance",
    "LogTotalInventory": "log10 total inventory",
    "log10_inventory": "log10 inventory",
    "log10_plus1": "log10(inventory + 1)",
    "inventory": "inventory",
    "fraction": "fraction of tank inventory",
    # vitrification
    "Vitrification_screening_score_proxy": "Screening score (proxy)",
    "User_search_score_proxy": "Search score (proxy)",
    "Blend_complement_score_proxy": "Blend complement score (proxy)",
    "Ci_per_kg_proxy": "Activity per mass (Ci/kg, proxy)",
    "Ci_key_radionuclide_elements": "Key radionuclide elements (Ci)",
    "kg_glass_former_or_intermediate": "Glass formers and intermediates (kg)",
    "kg_alkali_alkaline_modifier": "Alkali and alkaline-earth modifiers (kg)",
    "kg_transition_redox_sensitive": "Redox-sensitive transition metals (kg)",
    "kg_volatile_halide_sulfate": "Volatile halides and sulfate (kg)",
    "kg_nuclear_key_radionuclide_elements": "Key radionuclide elements (kg)",
    "kg_platinum_group_or_noble": "Platinum-group and noble metals (kg)",
    "kg_potential_spinels_or_crystallizers": "Spinel and crystal formers (kg)",
    "kg_problem_elements_proxy": "Problem elements (kg, proxy)",
    "frac_glass_former_or_intermediate": "Glass formers and intermediates (fraction)",
    "frac_alkali_alkaline_modifier": "Alkali and alkaline-earth modifiers (fraction)",
    "frac_transition_redox_sensitive": "Redox-sensitive transition metals (fraction)",
    "frac_volatile_halide_sulfate": "Volatile halides and sulfate (fraction)",
    "frac_nuclear_key_radionuclide_elements": "Key radionuclide elements (fraction)",
    "frac_platinum_group_or_noble": "Platinum-group and noble metals (fraction)",
    "frac_potential_spinels_or_crystallizers": "Spinel and crystal formers (fraction)",
    "frac_problem_elements_proxy": "Problem elements (fraction, proxy)",
    "contrib_glass_former": "Score: glass formers",
    "contrib_modifier": "Score: modifiers",
    "contrib_problem": "Score: problem elements",
    "contrib_volatile": "Score: volatiles",
    "contrib_redox": "Score: redox",
    "UserTargetFraction_sum": "Target fraction",
    "UserTargetChemicalFraction_sum": "Target fraction, chemical",
    "UserTargetRadiologicalFraction_sum": "Target fraction, radiological",
    "UserPenaltyFraction_sum": "Penalty fraction",
    "UserPenaltyChemicalFraction_sum": "Penalty fraction, chemical",
    "UserPenaltyRadiologicalFraction_sum": "Penalty fraction, radiological",
    "CosineSimilarity_to_base_fraction_profile": "Similarity to base profile",
    "GlassFormerFraction_partner": "Glass formers, partner",
    "ProblemFraction_partner": "Problem elements, partner",
    "GlassFormerFraction_base": "Glass formers, base",
    "ProblemFraction_base": "Problem elements, base",
    "GlassFormerGain_vs_base": "Glass-former gain over base",
    "ProblemReduction_vs_base": "Problem-element reduction over base",
}

# How a label reads in the plural, for a sentence such as "top 30 of 177
# tanks shown". Anything absent takes an "s".
PLURALS: Dict[str, str] = {
    "Tank": "tanks",
    "Farm": "farms",
    "Waste phase": "waste phases",
    "Waste type": "waste types",
    "Phase family": "phase families",
    "Analyte": "analytes",
    "Element": "elements",
    "Member": "members",
    "Analyte class": "analyte classes",
    "Status": "statuses",
    "Integrity": "integrity classes",
    "System": "systems",
    "Tank type": "tank types",
}

# Words kept exactly as written when a name is split.
_UNITS = {"kg": "kg", "g": "g", "ci": "Ci", "kl": "kL", "l": "L", "gal": "gal", "kgal": "kgal",
          "mb": "MB", "pct": "%", "wt%": "wt%", "g/l": "g/L", "ci/kg": "Ci/kg"}
_ACRONYMS = {"dst": "DST", "sst": "SST", "dil": "DIL", "pca": "PCA", "rsd": "rel. std. dev.",
             "ccb": "CCB", "wi": "WI", "id": "ID", "bbi": "BBI", "toc": "TOC", "pc1": "PC1", "pc2": "PC2"}
_WORDS = {"n": "number of", "frac": "fraction", "std": "std. dev.", "max": "largest", "min": "smallest",
          "abs": "absolute", "corr": "correlation", "avg": "mean", "num": "number of", "proxy": "(proxy)"}

_TANK_RE = re.compile(r"^(\d{3}-[A-Z]{1,3}-\d{3})(?![0-9A-Za-z])")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9%])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_IDENT_RE = re.compile(r"(?<![\w%/])[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9%/]+)+(?![\w%])")


def tank_id_of(label: Any) -> Optional[str]:
    """The tank a label names, when it names one.

    "241-AW-106" is a tank; "241-AW-102/Saltcake Interstitial Liquid" is a
    layer of one. Anything else is not a tank."""
    if label is None:
        return None
    match = _TANK_RE.match(str(label).strip())
    return match.group(1) if match else None


def _looks_like_plain_symbol(name: str) -> bool:
    """Element symbols, isotopes, species and short codes are already the
    name a reader expects ("Na", "129I", "SO4", "PC1", "x")."""
    if len(name) <= 3:
        return True
    if re.fullmatch(r"(?:\d{1,3}m?/)?\d{1,3}m?[A-Z][a-z]?", name):            # 137Cs, 113mCd, 239/240Pu
        return True
    if re.fullmatch(r"(?:[A-Z][a-z]?\d*){1,4}", name) and len(name) <= 6:      # SO4, PO4, NO3, UO2
        return True
    return tank_id_of(name) is not None


def humanize(name: Any) -> str:
    """A column identifier as a label a reader would write.

    Known identifiers come from COLUMN_LABELS. Anything else is split on
    underscores and capital letters, written in sentence case, with a
    trailing unit moved into parentheses: "Total_capacity_kgal" becomes
    "Total capacity (kgal)". A name that already reads as a label (it
    contains a space, or is a symbol such as "Na" or "137Cs") is returned
    unchanged."""
    text = str(name)
    if text in COLUMN_LABELS:
        return COLUMN_LABELS[text]
    stripped = text.strip()
    if not stripped or " " in stripped or _looks_like_plain_symbol(stripped):
        return text
    if not ("_" in stripped or _CAMEL_RE.search(stripped)):
        return text
    tokens = []
    for chunk in stripped.split("_"):
        tokens.extend(part for part in _CAMEL_RE.split(chunk) if part)
    units = []
    words = []
    proxy = False
    for position, token in enumerate(tokens):
        low = token.lower()
        if low in _UNITS and position > 0:
            units.append(_UNITS[low])
        elif low == "proxy":
            proxy = True
        elif low in _ACRONYMS:
            words.append(_ACRONYMS[low])
        elif low in _WORDS and (position == 0 or low in ("std", "frac")):
            words.append(_WORDS[low])
        elif token.isupper() and len(token) > 1 and not token.isdigit():
            words.append(token)                      # an acronym not in the table
        elif len(token) <= 2 and token[:1].isupper() and position > 0:
            words.append(token)                      # an element symbol inside a name
        else:
            words.append(low)
    label = " ".join(words).strip()
    if label[:1].islower():
        label = label[:1].upper() + label[1:]
    qualifiers = units + (["proxy"] if proxy else [])
    if qualifiers:
        label = f"{label} ({', '.join(qualifiers)})"
    return label or text


def plural_label(name: Any) -> str:
    """The lower-case plural of a column's label, for running text."""
    label = humanize(name)
    if label in PLURALS:
        return PLURALS[label]
    low = label[:1].lower() + label[1:] if label[:1].isupper() and not label[:2].isupper() else label
    if low.endswith(")") or low.endswith("s"):
        return low
    return f"{low}s"


def humanize_text(text: Any) -> str:
    """Replace identifiers inside a sentence (a title, a note, an axis
    label) with their labels, leaving everything else alone.

    Only names with an underscore are touched, plus the handful of
    CamelCase column names that reach titles; an ordinary word is never a
    match."""
    if text is None:
        return ""
    out = str(text)
    if not out:
        return out

    def _swap(match: "re.Match[str]") -> str:
        token = match.group(0)
        label = humanize(token)
        if label == token:
            return token
        before = out[:match.start()].rstrip()
        mid_sentence = bool(before) and before[-1] not in ".:(|·—-\n"
        if mid_sentence and label[:1].isupper() and not label[:2].isupper():
            label = label[:1].lower() + label[1:]
        return label

    out = _IDENT_RE.sub(_swap, out)
    for token in _TEXT_CAMEL_NAMES:
        if token in out:
            out = re.sub(rf"(?<![\w]){re.escape(token)}(?![\w])", _swap, out)
    return out


_TEXT_CAMEL_NAMES = (
    "WasteSiteId", "WastePhase", "WasteType", "TankFarm", "TankType", "TankSystem", "TankStatus",
    "TankIntegrity", "PhaseFamily", "AnalyteClass", "TotalInventory", "PartnerTank", "BaseTank",
)


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

SMALL_LIMIT = 1e-3
LARGE_LIMIT = 1e9


def _clean_exponent(text: str) -> str:
    """'1.235e-04' -> '1.235e-4', '2.6e+10' -> '2.6e10'."""
    mantissa, _, exponent = text.partition("e")
    if not exponent:
        return text
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    sign = "-" if exponent.startswith("-") else ""
    return f"{mantissa}e{sign}{exponent.lstrip('+-').lstrip('0') or '0'}"


def number_style(values: Any) -> Optional[Dict[str, Any]]:
    """How ONE numeric column is written, so its cells compare at a glance.

    Formatting each cell on its own gives a column of ragged decimals
    ("-2.09", "0.06639", "1.366") and, around a billion, plain numbers
    beside scientific ones. A column is read downward, so the rule is
    chosen for the column:

    - {"mode": "sci"}: every cell in scientific notation, when the largest
      value reaches 1e9 or the largest is below 1e-3;
    - {"mode": "fixed", "decimals": d}: one number of decimals, giving
      about four significant figures on the largest value, when at least
      nine cells in ten then still show two significant figures (a cell
      too small for that is written on its own, never as a false zero);
    - None: no common rule fits (values spread over many decades): each
      cell keeps about four significant figures.
    """
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    arr = np.abs(arr[np.isfinite(arr)])
    arr = arr[arr > 0]
    if arr.size == 0:
        return None
    hi = float(arr.max())
    if hi >= LARGE_LIMIT or hi < SMALL_LIMIT:
        return {"mode": "sci"}
    if bool(np.all(arr == np.round(arr))):
        return {"mode": "fixed", "decimals": 0}
    decimals = int(min(6, max(0, 3 - math.floor(math.log10(hi)))))
    if float(np.percentile(arr, 10)) < 10.0 ** (1 - decimals):
        return None
    return {"mode": "fixed", "decimals": decimals}


def format_number(value: Any, sig: int = 4, style: Optional[Dict[str, Any]] = None) -> str:
    """A number for display: about `sig` significant figures, thousands
    separators, scientific notation only below 1e-3 or from 1e9 up.

    Integers are written in full. A missing value is an empty string.
    `style` is a column's rule from number_style(); without one the cell
    is formatted on its own."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    if math.isinf(number):
        return "∞" if number > 0 else "−∞"
    if number == 0.0:
        return "0"
    magnitude = abs(number)
    if style:
        if style.get("mode") == "sci":
            return _clean_exponent(f"{number:.{max(sig - 1, 0)}e}")
        decimals = int(style.get("decimals", 0))
        if style.get("mode") == "fixed" and (decimals == 0 or magnitude >= 10.0 ** (1 - decimals)):
            return f"{number:,.{decimals}f}"
    if magnitude < SMALL_LIMIT or magnitude >= LARGE_LIMIT:
        return _clean_exponent(f"{number:.{max(sig - 1, 0)}e}")
    if magnitude >= 1000.0:
        return f"{number:,.0f}"
    text = f"{number:.{sig}g}"
    if "e" in text or abs(float(text)) >= 1000.0:      # 999.96 rounds up into the next range
        return f"{float(text):,.0f}"
    return text


def format_full(value: Any) -> str:
    """A value at full precision (tooltips, copy, export)."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return "; ".join(str(v) for v in value)
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    if isinstance(value, (float, np.floating)):
        return "" if math.isnan(float(value)) else repr(float(value))
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def format_cell(value: Any, style: Optional[Dict[str, Any]] = None) -> str:
    """One table cell as text (`style`: the column's number_style())."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        # Polars list-columns reach pandas as numpy object arrays, and
        # pd.isna() on an array is an array: this branch must come first.
        return "; ".join(str(v) for v in value)
    if isinstance(value, (bool, np.bool_, int, np.integer)):
        return format_number(value)
    if isinstance(value, (float, np.floating)):
        return format_number(value, style=style)
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return ""
        return value.strftime("%Y-%m-%d") if value == value.normalize() else value.strftime("%Y-%m-%d %H:%M")
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def format_reading(value: Any) -> str:
    """A number as a tooltip states it: like format_number, except that a
    large value is written out ("1,238,412,345") instead of going
    scientific at a billion. One value read on its own is easier to take
    in as a number than as a power of ten."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return format_number(value)
    if not isinstance(value, (bool, np.bool_)) and LARGE_LIMIT <= abs(number) < 1e15:
        return f"{number:,.0f}"
    return format_number(value)


def with_unit(value: Any, unit: str = "") -> str:
    """'262,834,123 g' -- a formatted number followed by its unit."""
    text = format_reading(value)
    unit = (unit or "").strip()
    if not text or not unit:
        return text
    return f"{text}{'' if unit == '%' else ' '}{unit}"


_UNIT_IN_LABEL_RE = re.compile(r"\(([^()]*)\)\s*$")


def unit_from_label(label: Any) -> str:
    """The unit an axis label carries in its trailing parentheses, when it
    is one: 'Total mass (g)' -> 'g'. A parenthesis that qualifies rather
    than measures ('(proxy)', '(inventory, kg)') yields its last part only
    when that part is a known unit."""
    text = str(label or "").split(" — ")[0].strip()
    if text.lower().startswith(("log", "fraction", "share")):
        return ""                                 # a logarithm or a ratio of a quantity has no unit
    match = _UNIT_IN_LABEL_RE.search(text)
    if not match:
        return ""
    last = match.group(1).split(",")[-1].strip()
    known = {"g", "kg", "Ci", "wt%", "g/L", "%", "kL", "L", "gal", "kgal", "Ci/kg", "kg/kg", "MB", "s"}
    return last if last in known else ""
