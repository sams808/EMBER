"""
elements.py — element-symbol parsing from Hanford analyte labels
(framework-agnostic: no PySide6, no polars/pandas).

Ported from the original Hanford Tank Composition Visualizer (v1.4.5,
CACHE_SCHEMA_VERSION "parser5_external_attrs_iodinefix") with one
correctness fix: combined-isotope slash notation ("239/240Pu", "243/244Cm")
previously fell through every isotope regex and the formula-like fallback
(neither handles a bare "/" between digit groups), landing on the
has-lowercase rejection and silently dropping real Pu/Cm activity from
every element-keyed view. See the `combined_isotope` branch in
parse_analyte_elements below.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

ELEMENT_SYMBOLS: List[str] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
ELEMENT_SET = set(ELEMENT_SYMBOLS)
ELEMENT_PATTERN = re.compile("|".join(sorted(ELEMENT_SYMBOLS, key=len, reverse=True)))

# Query aliases used by element search boxes. The single-letter L/l alias is
# deliberate: an uppercase iodine symbol (I) is easily confused with a
# lowercase L in many fonts. L is not a valid element symbol, so an "auto"
# search for L/l is redirected to iodine; use analyte_contains mode for a
# literal substring search on "L".
ELEMENT_QUERY_ALIASES: Dict[str, str] = {
    "I": "I",
    "L": "I",
    "IODINE": "I",
    "IODIDE": "I",
    "IODATE": "I",
    "PERIODATE": "I",
    "129I": "I",
    "I129": "I",
    "I-129": "I",
    "129-I": "I",
}

# Exact-match aliases for non-formula analyte labels that matter most in
# Hanford tank inventories. Deliberately conservative: only mapped when the
# analyte text strongly implies a chemical element or carbon group. Mapping
# to [] blocks known false positives (TotalAlpha -> Al, Aroclors -> Ar, ...).
ANALYTE_ALIASES: Dict[str, List[str]] = {
    "UTOTAL": ["U"],
    "U_TOTAL": ["U"],
    "TOTALU": ["U"],
    "TOTAL_U": ["U"],
    "TOC": ["C"],
    "TIC": ["C"],
    "TICASCO3": ["C", "O"],
    "TC": ["Tc"],       # only exact uppercase TC maps to Technetium
    "TCO3": ["C", "O"],
    "FREEOH": ["O", "H"],
    "OH": ["O", "H"],
    "NO2": ["N", "O"],
    "NO3": ["N", "O"],
    "CO3": ["C", "O"],
    "SO4": ["S", "O"],
    "PO4": ["P", "O"],
    "NH3": ["N", "H"],
    "CN": ["C", "N"],
    "OXALATE": ["C", "O"],
    "FORMATE": ["C", "H", "O"],
    "ACETATE": ["C", "H", "O"],
    "GLYCOLATE": ["C", "H", "O"],
    "F": ["F"],
    "CL": ["Cl"],
    "BR": ["Br"],
    "I": ["I"],
    "IODINE": ["I"],
    "IODIDE": ["I"],
    "IODATE": ["I", "O"],
    "PERIODATE": ["I", "O"],
    "129I": ["I"],
    "I129": ["I"],
    # Measurements/classes, not chemical elements.
    "TOTALALPHA": [],
    "TOTALBETA": [],
    "TOTALGAMMA": [],
    "AROCLORSTOTALPCB": [],
    "AROCLOR": [],
    "PCB": [],
}

ANALYTE_PREFIX_ALIASES: List[Tuple[str, List[str]]] = [
    ("TICAS", ["C", "O"]),  # e.g. TIC as CO3
    ("TOCAS", ["C"]),
    ("TIC", ["C"]),
    ("TOC", ["C"]),
    ("FREEOH", ["O", "H"]),
    ("TOTALALPHA", []),
    ("TOTALBETA", []),
    ("TOTALGAMMA", []),
    ("AROCLOR", []),
]

# Common chemistry names mapped to broad formula elements, so e.g. "Acetate"
# resolves to C/H/O for search/co-occurrence purposes instead of matching
# the first letters as a metal symbol (Ac).
CHEMISTRY_NAME_ALIASES: Dict[str, List[str]] = {
    "NITRATE": ["N", "O"],
    "NITRITE": ["N", "O"],
    "CARBONATE": ["C", "O"],
    "BICARBONATE": ["C", "H", "O"],
    "SULFATE": ["S", "O"],
    "SULFITE": ["S", "O"],
    "THIOSULFATE": ["S", "O"],
    "SULFIDE": ["S"],
    "PHOSPHATE": ["P", "O"],
    "HYDROXIDE": ["O", "H"],
    "FLUORIDE": ["F"],
    "CHLORIDE": ["Cl"],
    "BROMIDE": ["Br"],
    "IODINE": ["I"],
    "IODIDE": ["I"],
    "IODATE": ["I", "O"],
    "PERIODATE": ["I", "O"],
    "OXALATE": ["C", "O"],
    "FORMATE": ["C", "H", "O"],
    "ACETATE": ["C", "H", "O"],
    "ACETONE": ["C", "H", "O"],
    "GLYCOLATE": ["C", "H", "O"],
    "AMMONIA": ["N", "H"],
    "CYANIDE": ["C", "N"],
    "TRIBUTYLPHOSPHATE": ["C", "H", "O", "P"],
    "CARBONTETRACHLORIDE": ["C", "Cl"],
    "VINYLCHLORIDE": ["C", "H", "Cl"],
}

_COMBINED_ISOTOPE_RE = re.compile(r"^\s*\d+\s*/\s*\d+\s*m?\s*([A-Za-z]{1,2})\s*$", re.IGNORECASE)
_ISOTOPE_START_RE = re.compile(r"^\s*\d+\s*m?\s*([A-Za-z]{1,2})(?:\b|$)", re.IGNORECASE)
_ISOTOPE_END_RE = re.compile(r"^\s*([A-Za-z]{1,2})\s*[-_]?\s*\d+\s*m?\s*$", re.IGNORECASE)
_FORMULA_LIKE_RE = re.compile(r"^\s*(?:\d+\s*)?(?:[A-Z][a-z]?\d*|[()\[\]{}+\-.·/\s])+$")
_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(?:\d*\.?\d*)?")


def normalize_element_symbol(query: object) -> Optional[str]:
    """Normalize a user query to a chemical element symbol.

    Tolerant of names/aliases ("iodine", "iodide", "129I") and maps a bare
    "L"/"l" to iodine (common visual typo for "I" in GUI fonts) — "Li" still
    resolves to lithium since the alias only fires on an exact 1-character
    match.
    """
    q_raw = str(query).strip()
    if not q_raw:
        return None

    compact = re.sub(r"[^A-Za-z0-9]+", "", q_raw).upper()
    if compact in ELEMENT_QUERY_ALIASES:
        return ELEMENT_QUERY_ALIASES[compact]

    # Currently unreachable with today's alias table: every hyphenated key
    # (e.g. "I-129") has a hyphen-free counterpart ("I129") that `compact`
    # above already matches first, since compact strips hyphens too. Kept
    # as defensive belt-and-suspenders in case a future alias only makes
    # sense in hyphenated form.
    hyphen_norm = re.sub(r"\s+", "", q_raw).upper()
    if hyphen_norm in ELEMENT_QUERY_ALIASES:  # pragma: no cover
        return ELEMENT_QUERY_ALIASES[hyphen_norm]

    q = q_raw[0].upper() + q_raw[1:].lower()
    return q if q in ELEMENT_SET else None


def parse_analyte_elements(analyte: object) -> List[str]:
    """Return plausible element symbols contained in an analyte label.

    Conservative by design: handles known aliases and isotope notation
    first, then parses only formula-like strings, so names like "Free OH"
    or "TIC as CO3" don't get misread by blind capitalized-substring
    scanning (which would otherwise yield Fr / I).
    """
    if analyte is None:
        return []
    s0 = str(analyte).strip()
    if not s0:
        return []

    s_upper_compact = re.sub(r"[^A-Za-z0-9]+", "", s0).upper()

    if s_upper_compact in ANALYTE_ALIASES:
        return ANALYTE_ALIASES[s_upper_compact]
    for prefix, els in ANALYTE_PREFIX_ALIASES:
        if s_upper_compact.startswith(prefix):
            return els
    if s_upper_compact in CHEMISTRY_NAME_ALIASES:
        return CHEMISTRY_NAME_ALIASES[s_upper_compact]

    # Combined-isotope slash notation: "239/240Pu", "243/244Cm". Must be
    # tried before the isotope_start/isotope_end/formula_like checks below,
    # none of which can consume a bare "/" between two digit groups.
    combined = _COMBINED_ISOTOPE_RE.match(s0)
    if combined:
        sym = normalize_element_symbol(combined.group(1))
        if sym:
            return [sym]

    # Isotope notation at the start/end, including metastable "m":
    # 137Cs, Cs137, 113mCd, 99Tc, 129I (case-tolerant, so 129i also resolves).
    isotope_start = _ISOTOPE_START_RE.match(s0)
    if isotope_start:
        sym = normalize_element_symbol(isotope_start.group(1))
        if sym:
            return [sym]

    isotope_end = _ISOTOPE_END_RE.match(s0)
    if isotope_end:
        sym = normalize_element_symbol(isotope_end.group(1))
        if sym:
            return [sym]

    # Exact element labels, including upper-case two-letter lab notation.
    if s0.isalpha() and 1 <= len(s0) <= 2:
        sym = normalize_element_symbol(s0)
        if sym:
            return [sym]

    # Don't parse natural-language analyte names by capital letters alone
    # (that would turn Acetone -> Ac, Aroclor -> Ar, Nitrobenzene -> Ni...).
    has_lowercase = bool(re.search(r"[a-z]", s0))
    formula_like = bool(_FORMULA_LIKE_RE.match(s0))
    if has_lowercase and not formula_like:
        return []

    tokens: List[str] = []
    for match in _TOKEN_RE.finditer(s0):
        sym = match.group(1)
        if sym in ELEMENT_SET and sym not in tokens:
            tokens.append(sym)
    return tokens


def primary_element_from_analyte(analyte: object) -> Optional[str]:
    """The first parsed element, which downstream aggregation attributes
    100% of a multi-element analyte's mass to (documented modeling choice,
    ported unchanged from the original app)."""
    els = parse_analyte_elements(analyte)
    return els[0] if els else None


def element_list_string(analyte: object) -> str:
    return ";".join(parse_analyte_elements(analyte))


def element_list_padded(analyte: object) -> str:
    els = parse_analyte_elements(analyte)
    return ";" + ";".join(els) + ";" if els else ";"


def tank_farm_from_id(waste_site_id: object) -> Optional[str]:
    """Extract the farm code from a Hanford tank id, e.g. "241-A-101" -> "A",
    "241-AN-104" -> "AN"."""
    if waste_site_id is None:
        return None
    s = str(waste_site_id).strip()
    m = re.match(r"^\s*\d+[-_ ]+([A-Za-z]+)[-_ ]+\d+", s)
    if m:
        return m.group(1).upper()
    parts = re.split(r"[-_ ]+", s)
    if len(parts) >= 2 and parts[1].isalpha():
        return parts[1].upper()
    return None


def classify_analyte(analyte: object) -> str:
    """Rough analyte classification: radionuclide / element / compound-formula
    / unparsed. Used for audit/overview breakdowns, not for numeric aggregation."""
    if analyte is None:
        return "unknown"
    s = str(analyte).strip()
    if not s:
        return "unknown"
    if re.match(r"^\d+\s*m?\s*[A-Z][a-z]?$", s) or re.match(r"^[A-Z][a-z]?\s*\d+\s*m?$", s):
        return "radionuclide"
    els = parse_analyte_elements(s)
    if len(els) == 1:
        return "element"
    if len(els) > 1:
        return "compound/formula"
    return "unparsed"
