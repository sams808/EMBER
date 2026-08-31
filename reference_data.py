"""
reference_data.py — bundled, citable physical reference tables (L0).

Three hand-curated tables plus values derived from them at import time:

- ATOMIC_MASSES: CIAAW 2021 standard atomic weights (abridged), g/mol.
  For elements with no stable isotope and no standard atomic weight
  (Tc, Pm, Po, Ra, Ac, Pa, Np, Pu, Am, Cm), the conventional mass number
  of the longest-lived or most common isotope is used, as is standard
  practice for bulk-mass bookkeeping.
- SPECIES_FORMULAS: element counts for the compound-species analytes in
  the BBI kg table (SO4, PO4, NO3, ... , Tributyl phosphate), including
  the BBI reporting conventions: "TIC as CO3" is carbonate mass, "TOC" is
  carbon mass, "UTOTAL" is elemental uranium mass.
- NUCLIDE_HALF_LIFE_YEARS: half-lives for every Ci-basis radionuclide in
  the dataset. Basis policy (adversarially verified 2026-08-30): ICRP
  Publication 107 (Eckerman & Endo 2008) values, UPDATED to the current
  ENSDF/NUBASE-adopted evaluation wherever the two differ by more than
  1% -- six nuclides: 229Th 7880 y (Essex 2018 IDMS), 59Ni 7.6e4 y
  (NUBASE2020), 79Se 3.27e5 y (Joerg 2010), 93Zr 1.61e6 y (Cassette
  2010), 151Sm 94.7 y (BIPM DDEP 2023), 63Ni 101.2 y (NIST/LNHB).
  Specific-activity spot checks against literature values (137Cs ~87
  Ci/g, 239Pu ~0.062 Ci/g, ...) agree within 0.8%. Combined-pair labels
  (239/240Pu, 243/244Cm, 144Ce/Pr) are None: the isotopic split within
  the pair is not reported, so no single specific activity exists and
  they are NOT mass-bridgeable (callers must disclose the exclusion).

Derived at import (single source of truth -- no hand-computed numbers
that can drift from the constants they came from):

- SPECIES_ELEMENT_MASS_FRACTIONS[analyte][element] = mass fraction of
  `element` in that species (e.g. S in SO4 = 32.06/96.06 ~= 0.3338).
- SPECIFIC_ACTIVITY_CI_PER_G[nuclide] = ln2 * N_A / (T_half_s * A) / 3.7e10,
  the Ci->gram bridge (A = mass number, the standard <0.1% approximation
  of the isotopic molar mass).

Citations: Prohaska et al., Pure Appl. Chem. 94 (2022) 573 (CIAAW 2021);
ICRP Publication 107, Ann. ICRP 38(3) (2008).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from elements import parse_isotope, primary_element_from_analyte

AVOGADRO = 6.02214076e23
CI_TO_BQ = 3.7e10
YEAR_S = 365.2422 * 86400.0

# CIAAW 2021 abridged standard atomic weights (g/mol); conventional values
# (marked *) for no-stable-isotope elements.
ATOMIC_MASSES: Dict[str, float] = {
    "H": 1.008, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45,
    "K": 39.098, "Ca": 40.078, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546,
    "Zn": 65.38, "As": 74.922, "Se": 78.971, "Br": 79.904, "Rb": 85.468,
    "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.95,
    "Tc": 98.0,      # * no stable isotope (99Tc-dominated in this waste)
    "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41,
    "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90, "Cs": 132.91,
    "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "Sm": 150.36, "Eu": 151.96, "Ta": 180.95, "W": 183.84, "Hg": 200.59,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98,
    "Ra": 226.0,     # *
    "Ac": 227.0,     # *
    "Th": 232.04, "Pa": 231.04, "U": 238.03,
    "Np": 237.0,     # *
    "Pu": 244.0,     # *
    "Am": 243.0,     # *
    "Cm": 247.0,     # *
}

# Element counts for the BBI compound-species analytes (exact labels).
# Reporting conventions applied: "TIC as CO3" = mass reported AS carbonate;
# "TOC" = mass reported AS carbon; "UTOTAL" = total elemental uranium;
# Formate = HCOO-, Acetate = CH3COO-, Glycolate = HOCH2COO-,
# Oxalate = C2O4^2-, Thiosulfate = S2O3^2-.
SPECIES_FORMULAS: Dict[str, Dict[str, int]] = {
    "SO4": {"S": 1, "O": 4},
    "PO4": {"P": 1, "O": 4},
    "NO3": {"N": 1, "O": 3},
    "NO2": {"N": 1, "O": 2},
    "NH3": {"N": 1, "H": 3},
    "CN": {"C": 1, "N": 1},
    "Free OH": {"O": 1, "H": 1},
    "TIC as CO3": {"C": 1, "O": 3},
    "TOC": {"C": 1},
    "Oxalate": {"C": 2, "O": 4},
    "Formate": {"C": 1, "H": 1, "O": 2},
    "Acetate": {"C": 2, "H": 3, "O": 2},
    "Glycolate": {"C": 2, "H": 3, "O": 3},
    "Sulfide": {"S": 1},
    "Thiosulfate": {"S": 2, "O": 3},
    "Tributyl phosphate": {"C": 12, "H": 27, "O": 4, "P": 1},
    "UTOTAL": {"U": 1},
}

# Half-lives in years (ICRP-107; combined pairs None -> not bridgeable).
NUCLIDE_HALF_LIFE_YEARS: Dict[str, Optional[float]] = {
    "3H": 12.32,
    "14C": 5700.0,
    "59Ni": 7.6e4,
    "60Co": 5.2713,
    "63Ni": 101.2,      # NIST/LNHB LSC standardization, ENSDF-adopted
    "79Se": 3.27e5,      # Joerg et al. 2010, ENSDF/NUBASE-adopted
    "90Sr": 28.79,
    "90Y": 64.10 / (24.0 * 365.2422),         # 64.10 h
    "93Zr": 1.61e6,
    "93mNb": 16.13,
    "94Nb": 2.03e4,
    "99Tc": 2.111e5,
    "106Ru": 371.8 / 365.2422,                # 371.8 d
    "113mCd": 14.1,
    "125Sb": 2.7586,
    "126Sn": 2.30e5,
    "129I": 1.57e7,
    "134Cs": 2.0648,
    "137Cs": 30.1671,
    "137mBa": 2.552 / (60.0 * 24.0 * 365.2422),  # 2.552 min
    "151Sm": 94.7,       # BIPM DDEP 2023 / He et al. 2009
    "152Eu": 13.537,
    "154Eu": 8.593,
    "155Eu": 4.7611,
    "226Ra": 1600.0,
    "227Ac": 21.772,
    "228Ac": 6.15 / (24.0 * 365.2422),        # 6.15 h
    "228Ra": 5.75,
    "228Th": 1.9116,
    "229Th": 7880.0,     # Essex et al. 2018 IDMS, NNDC-adopted
    "230Th": 7.538e4,
    "231Pa": 3.276e4,
    "232Th": 1.405e10,
    "232U": 68.9,
    "233U": 1.592e5,
    "234U": 2.455e5,
    "235U": 7.04e8,
    "236U": 2.342e7,
    "237Np": 2.144e6,
    "238Pu": 87.7,
    "238U": 4.468e9,
    "239Pu": 2.411e4,
    "240Pu": 6561.0,
    "241Am": 432.6,
    "241Pu": 14.329,
    "242Cm": 162.8 / 365.2422,                # 162.8 d
    "242Pu": 3.75e5,
    "243Am": 7370.0,
    "243Cm": 29.1,
    "244Cm": 18.1,
    # Combined-pair labels: isotopic split unreported -> no single specific
    # activity -> excluded from any Ci->gram bridge, with a disclosure.
    "239/240Pu": None,
    "243/244Cm": None,
    "144Ce/Pr": None,
}

# What each combined-pair label is the sum of. The BBI frequently reports
# BOTH the pair and its members for the same layer -- in the shipped data
# every one of the 181 layers carrying 239/240Pu also carries 239Pu and
# 240Pu, and the pair equals their sum to within 0.5%. Adding all three
# counts the same activity twice, so a caller must pick one
# representation; this table is what lets it.
COMBINED_PAIR_MEMBERS: Dict[str, Tuple[str, ...]] = {
    "239/240Pu": ("239Pu", "240Pu"),
    "243/244Cm": ("243Cm", "244Cm"),
    "144Ce/Pr": ("144Ce", "144Pr"),
}

# A reported TOTAL that already contains the itemisation the BBI reports
# beside it, mapped to the element it describes and the members it
# subsumes. Distinct from COMBINED_PAIR_MEMBERS in both direction and
# keep-rule: a pair EQUALS its members, so the itemisation is kept and
# the pair set aside; a total is a proper SUPERSET (TOC also holds
# organics the BBI never itemises -- the itemised anions are a median
# 18% of it), so the TOTAL is kept and the members set aside.
#
# TIC as CO3 is deliberately absent: it is INORGANIC carbon, which total
# organic carbon does not contain, so the two are correctly additive.
SPECIES_TOTAL_MEMBERS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "TOC": ("C", ("Oxalate", "Formate", "Acetate", "Glycolate", "Tributyl phosphate")),
}

# Mass number for the specific-activity molar-mass approximation, parsed
# by the SAME parser that builds the dataset's Isotope column (review
# finding: a second hand-rolled label parse here could silently drift
# from elements.parse_isotope). Combined-pair labels are None regardless
# of what the parser extracts -- 144Ce/Pr yields mass 144, but the pair
# carries no single specific activity.
NUCLIDE_MASS_NUMBER: Dict[str, Optional[int]] = {
    _label: (None if "/" in _label else parse_isotope(_label)[0])
    for _label in NUCLIDE_HALF_LIFE_YEARS
}


def canonical_nuclide_label(label: str) -> str:
    """Canonical "###m?Sym" key for any single-nuclide label form
    parse_isotope recognizes ("Cs-137" -> "137Cs", "Cd-113m" -> "113mCd").
    Combined/parent-daughter labels and non-isotope strings pass through
    unchanged, so combined pairs keep hitting their own explicit None
    entries rather than accidentally resolving to one member's activity."""
    s = str(label).strip()
    if "/" in s:
        return s
    mass, metastable, is_isotope = parse_isotope(s)
    if not is_isotope or mass is None:
        return s
    symbol = primary_element_from_analyte(s)
    if not symbol:
        return s
    return f"{mass}{'m' if metastable else ''}{symbol}"


def _species_mass_fractions() -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for analyte, counts in SPECIES_FORMULAS.items():
        molar_mass = sum(ATOMIC_MASSES[el] * n for el, n in counts.items())
        out[analyte] = {el: ATOMIC_MASSES[el] * n / molar_mass for el, n in counts.items()}
    return out


def _specific_activities() -> Dict[str, Optional[float]]:
    """Ci per gram: SA = ln2 * N_A / (T_half[s] * A[g/mol]) / 3.7e10."""
    out: Dict[str, Optional[float]] = {}
    for nuclide, t_half_y in NUCLIDE_HALF_LIFE_YEARS.items():
        a = NUCLIDE_MASS_NUMBER.get(nuclide)
        if t_half_y is None or a is None:
            out[nuclide] = None
            continue
        out[nuclide] = math.log(2) * AVOGADRO / (t_half_y * YEAR_S * a) / CI_TO_BQ
    return out


SPECIES_ELEMENT_MASS_FRACTIONS: Dict[str, Dict[str, float]] = _species_mass_fractions()
SPECIFIC_ACTIVITY_CI_PER_G: Dict[str, Optional[float]] = _specific_activities()


def nuclide_grams_per_ci(nuclide: str) -> Optional[float]:
    """Grams of the nuclide per curie, or None when the label carries no
    single specific activity (combined pairs, unknown labels). Accepts any
    label form parse_isotope recognizes ("Cs-137" == "137Cs")."""
    key = nuclide if nuclide in SPECIFIC_ACTIVITY_CI_PER_G else canonical_nuclide_label(nuclide)
    sa = SPECIFIC_ACTIVITY_CI_PER_G.get(key)
    if sa is None or sa <= 0:
        return None
    return 1.0 / sa


def species_element_fraction(analyte: str, element: str) -> Optional[float]:
    """Mass fraction of `element` in the species reported as `analyte`,
    or None when the analyte has no registered formula (organics, bare
    elements handled elsewhere)."""
    fractions = SPECIES_ELEMENT_MASS_FRACTIONS.get(analyte)
    if fractions is None:
        return None
    return fractions.get(element, 0.0)
