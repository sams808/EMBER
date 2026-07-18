"""
oxide_science.py — element -> oxide stoichiometric conversion, NBO/T, and
envelope comparison for the Vitrification "Oxide Chemistry" sub-tab.

The real depth upgrade over the old app's "glass calculation," which never
left raw elemental kg: this module is the missing first step (elemental
kg -> oxide wt%/mol%) feeding into glass_science.optical_basicity() and
glassnet_predict() (copied verbatim from Dataapp, already correctly
cited). Entirely new -- no old-app equivalent.

DEFAULT_OXIDE_MAP is a mechanism spec, not an assertion that every seeded
default is definitively correct for this waste stream: multivalent
elements (Fe/Cr/Mn/U/Np/Pu/Ce/Co) get one pre-selected default with
ALTERNATIVE_OXIDES offering others; halides and noble metals
(F/Cl/Br/I/Ru/Rh/Pd/Ag/Pt/Au/Tc) default to None ("not a network oxide" --
reported as elemental wt% instead). Every default is user-overridable via
the Qt layer's editable [Element, Assumed oxide] table. Defaults are
chosen, where possible, to already have a tabulated
glass_science.OPTICAL_BASICITY entry; Pu is the one unavoidable gap (no
PNNL-20184 Λrec value exists for any Pu oxide) -- composition_summary()
reports it as excluded rather than silently dropping it or crashing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

import glass_science as gscience
from data_model import HanfordDataset

# Element -> default oxide formula (None = "not a network oxide", shown as
# elemental wt% instead). Covers every element with a glass_science.
# OPTICAL_BASICITY entry (one representative oxide each) plus halides/noble
# metals explicitly flagged as non-oxide.
DEFAULT_OXIDE_MAP: Dict[str, Optional[str]] = {
    "Ag": None, "Al": "Al2O3", "As": "As2O5", "Au": None,
    "B": "B2O3", "Ba": "BaO", "Be": "BeO", "Bi": "Bi2O3",
    "Br": None, "C": "CO2", "Ca": "CaO", "Cd": "CdO", "Ce": "CeO2",
    "Cl": None, "Co": "CoO", "Cr": "Cr2O3", "Cs": "Cs2O",
    "Cu": "CuO", "Dy": "Dy2O3", "Eu": "Eu2O3", "F": None,
    "Fe": "Fe2O3", "Ga": "Ga2O3", "Gd": "Gd2O3", "Ge": "GeO2",
    "Hf": "HfO2", "Hg": "HgO", "Ho": "Ho2O3", "I": None,
    "In": "In2O3", "Ir": "IrO2", "K": "K2O", "La": "La2O3",
    "Li": "Li2O", "Lu": "Lu2O3", "Mg": "MgO", "Mn": "MnO",
    "Mo": "MoO3", "Na": "Na2O", "Nb": "Nb2O5", "Nd": "Nd2O3",
    "Ni": "NiO", "Np": "NpO2", "P": "P2O5", "Pb": "PbO",
    "Pd": None, "Pt": None, "Pu": "PuO2", "Rh": None, "Ru": None,
    "S": "SO3", "Sb": "Sb2O3", "Sc": "Sc2O3", "Se": "SeO2",
    "Si": "SiO2", "Sm": "Sm2O3", "Sn": "SnO2", "Sr": "SrO",
    "Ta": "Ta2O5", "Tb": "Tb2O3", "Tc": None, "Te": "TeO2",
    "Th": "ThO2", "Ti": "TiO2", "Tl": "Tl2O", "Tm": "Tm2O3",
    "U": "UO3", "V": "V2O5", "W": "WO3", "Y": "Y2O3",
    "Yb": "Yb2O3", "Zn": "ZnO", "Zr": "ZrO2",
    # actinide extras
    "Ac": "Ac2O3", "Am": "Am2O3", "Bk": "Bk2O3", "Cf": "Cf2O3",
    "Cm": "Cm2O3", "Pa": "PaO2",
}

# Multivalent elements offered a dropdown of alternatives in the Qt layer
# (the DEFAULT_OXIDE_MAP entry is always alternatives[0]).
ALTERNATIVE_OXIDES: Dict[str, List[str]] = {
    "Fe": ["Fe2O3", "FeO"], "Cr": ["Cr2O3", "CrO3"],
    "Mn": ["MnO", "MnO2", "Mn2O3", "Mn3O4"],
    "U": ["UO3", "UO2", "U3O8"], "Np": ["NpO2", "Np2O5"],
    "Pu": ["PuO2", "Pu2O3"], "Ce": ["CeO2", "Ce2O3"], "Co": ["CoO", "Co2O3"],
}

# Classical former/modifier sets for the simplified NBO/T calculation.
FORMER_OXIDES = {"SiO2", "B2O3", "P2O5", "Al2O3"}
MODIFIER_OXIDES = {
    "Li2O", "Na2O", "K2O", "Rb2O", "Cs2O",  # alkali
    "MgO", "CaO", "SrO", "BaO",  # alkaline earth
}
_T_CATIONS_PER_FORMULA_UNIT = {"SiO2": 1, "B2O3": 2, "P2O5": 2, "Al2O3": 2}


def default_oxide_role(oxide_formula: str) -> str:
    if oxide_formula in FORMER_OXIDES:
        return "former"
    if oxide_formula in MODIFIER_OXIDES:
        return "modifier"
    return "other"


def oxide_mass_from_element_mass(element: str, element_mass_g: float, oxide_formula: str) -> float:
    """Mass (g) of `oxide_formula` stoichiometrically equivalent to
    `element_mass_g` grams of `element`."""
    import xraydb
    counts = xraydb.chemparse(oxide_formula)
    n_el = counts.get(element, 0)
    if n_el <= 0:
        raise ValueError(f"{oxide_formula} does not contain {element}.")
    moles_el = element_mass_g / xraydb.atomic_mass(element)
    moles_oxide = moles_el / n_el
    oxide_molar_mass = sum(xraydb.atomic_mass(el) * n for el, n in counts.items())
    return moles_oxide * oxide_molar_mass


def convert_composition_to_oxides(
    element_kg: Dict[str, float], oxide_map: Optional[Dict[str, Optional[str]]] = None,
) -> pd.DataFrame:
    """Convert a {Element: kg} composition into an oxide (+ elemental, for
    unmapped/non-network elements) mass/mol/wt%/mol% table. Elements
    missing from `oxide_map` fall back to DEFAULT_OXIDE_MAP; elements
    missing from both are still reported (as elemental wt%), never
    silently dropped. Non-positive/missing inventories are skipped."""
    import xraydb
    combined_map = dict(DEFAULT_OXIDE_MAP)
    if oxide_map:
        combined_map.update(oxide_map)

    rows: Dict[str, Dict[str, object]] = {}
    for element, kg in element_kg.items():
        if kg is None or float(kg) <= 0:
            continue
        mass_g = float(kg) * 1000.0
        oxide_formula = combined_map.get(element, DEFAULT_OXIDE_MAP.get(element))
        if not oxide_formula:
            key, kind = element, "element"
            comp_mass_g = mass_g
            comp_mol = mass_g / xraydb.atomic_mass(element)
        else:
            key, kind = oxide_formula, "oxide"
            comp_mass_g = oxide_mass_from_element_mass(element, mass_g, oxide_formula)
            oxide_mw = sum(xraydb.atomic_mass(el) * n for el, n in xraydb.chemparse(oxide_formula).items())
            comp_mol = comp_mass_g / oxide_mw
        if key in rows:
            rows[key]["Mass_g"] += comp_mass_g
            rows[key]["Mol"] += comp_mol
            rows[key]["Source_elements"].add(element)
        else:
            rows[key] = {"Kind": kind, "Mass_g": comp_mass_g, "Mol": comp_mol, "Source_elements": {element}}

    if not rows:
        return pd.DataFrame(columns=["Component", "Kind", "Source_elements", "Mass_g", "Wt_pct", "Mol", "Mol_pct"])

    total_mass = sum(r["Mass_g"] for r in rows.values())
    total_mol = sum(r["Mol"] for r in rows.values())
    out_rows = []
    for component, r in rows.items():
        out_rows.append({
            "Component": component, "Kind": r["Kind"],
            "Source_elements": ";".join(sorted(r["Source_elements"])),
            "Mass_g": r["Mass_g"],
            "Wt_pct": 100.0 * r["Mass_g"] / total_mass if total_mass > 0 else np.nan,
            "Mol": r["Mol"],
            "Mol_pct": 100.0 * r["Mol"] / total_mol if total_mol > 0 else np.nan,
        })
    return pd.DataFrame(out_rows).sort_values("Wt_pct", ascending=False).reset_index(drop=True)


def tank_oxide_composition(
    dataset: HanfordDataset, tank_ids: Sequence[str], oxide_map: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, pd.DataFrame]:
    """Oxide composition for each of the given tanks (kg elemental
    inventory -> oxide/elemental wt%/mol%), plus one additional "Blend
    (selected tanks)" table when 2+ tanks are given -- built by summing raw
    kg across tanks FIRST and converting once, which is what physically
    combining the tanks' contents would actually produce (not an average
    of each tank's independently-computed percentages)."""
    tank_ids = [t for t in tank_ids if t]
    if not tank_ids:
        return {}
    df = dataset.require_df().filter(
        (pl.col("Units") == "kg") & pl.col("Element").is_not_null() & pl.col("WasteSiteId").is_in(tank_ids)
    )
    tables: Dict[str, pd.DataFrame] = {}
    if df.is_empty():
        return tables
    per_tank = df.group_by(["WasteSiteId", "Element"]).agg(pl.col("Inventory").sum().alias("Inventory")).to_pandas()
    for tank in tank_ids:
        sub = per_tank[per_tank["WasteSiteId"] == tank]
        if sub.empty:
            continue
        element_kg = dict(zip(sub["Element"], sub["Inventory"]))
        tables[tank] = convert_composition_to_oxides(element_kg, oxide_map)
    if len(tank_ids) > 1:
        blend_kg: Dict[str, float] = {}
        for element, inv in zip(per_tank["Element"], per_tank["Inventory"]):
            blend_kg[element] = blend_kg.get(element, 0.0) + float(inv)
        tables["Blend (selected tanks)"] = convert_composition_to_oxides(blend_kg, oxide_map)
    return tables


def nbo_over_t(oxide_mol: Dict[str, float], role_map: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    """NBO/T = (O_total - 2T) / T -- a simplified approximation (does not
    model Al/B charge-balance by alkali; verify against your preferred
    reference for publication use). T sums network-forming cations over
    former-role oxides only; O_total sums oxygen over former+modifier
    oxides. "Other"-role oxides (transition metals, halides, actinides,
    ...) are excluded from this calculation entirely -- their network role
    is composition-dependent and not captured by the classical
    former/modifier formalism (they still appear in the main oxide
    composition table and in the optical-basicity calculation)."""
    import xraydb
    roles = role_map or {}
    t_sum = 0.0
    o_total = 0.0
    for formula, mol in oxide_mol.items():
        role = roles.get(formula, default_oxide_role(formula))
        if role not in ("former", "modifier"):
            continue
        counts = xraydb.chemparse(formula)
        n_o = float(counts.get("O", 0.0))
        o_total += mol * n_o
        if role == "former":
            n_t = _T_CATIONS_PER_FORMULA_UNIT.get(formula)
            if n_t is None:
                # A former-role oxide outside the standard 4 (e.g. the user
                # reclassified GeO2 as a former) -- count every non-oxygen
                # atom as a T-cation, a reasonable generalization.
                n_t = sum(c for el, c in counts.items() if el != "O")
            t_sum += mol * n_t
    if t_sum <= 0:
        return {"T": t_sum, "O_total": o_total, "NBO_T": float("nan")}
    return {"T": t_sum, "O_total": o_total, "NBO_T": (o_total - 2.0 * t_sum) / t_sum}


def envelope_check(composition_wt_pct: Dict[str, float], envelope: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Per-oxide pass/fail against a user-supplied {oxide: (min_wt_pct,
    max_wt_pct)} envelope. Oxides in the envelope but absent from the
    composition are checked as 0 wt% (a nonzero Min then correctly fails).
    Oxides in the composition but absent from the envelope are reported
    with Status="Not specified", never silently skipped."""
    if not envelope:
        return pd.DataFrame(columns=["Oxide", "Wt_pct", "Min_wt_pct", "Max_wt_pct", "Status"])
    rows = []
    for oxide in sorted(set(composition_wt_pct) | set(envelope)):
        wt_pct = float(composition_wt_pct.get(oxide, 0.0))
        bounds = envelope.get(oxide)
        if bounds is None:
            rows.append({"Oxide": oxide, "Wt_pct": wt_pct, "Min_wt_pct": np.nan, "Max_wt_pct": np.nan, "Status": "Not specified"})
            continue
        lo, hi = bounds
        rows.append({
            "Oxide": oxide, "Wt_pct": wt_pct, "Min_wt_pct": lo, "Max_wt_pct": hi,
            "Status": "Pass" if lo <= wt_pct <= hi else "Fail",
        })
    return pd.DataFrame(rows)


def save_envelope_json(path: Union[str, Path], envelope: Dict[str, Tuple[float, float]]) -> None:
    payload = {oxide: {"min_wt_pct": lo, "max_wt_pct": hi} for oxide, (lo, hi) in envelope.items()}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_envelope_json(path: Union[str, Path]) -> Dict[str, Tuple[float, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {oxide: (float(v["min_wt_pct"]), float(v["max_wt_pct"])) for oxide, v in payload.items()}


def glassnet_input_row(oxide_table: pd.DataFrame) -> pd.DataFrame:
    """One-row DataFrame of oxide mol% (elemental/non-oxide components
    dropped -- GlassNet expects oxide formulas as columns), ready for
    glass_science.glassnet_predict()."""
    if oxide_table is None or oxide_table.empty:
        return pd.DataFrame()
    oxide_rows = oxide_table[oxide_table["Kind"] == "oxide"]
    if oxide_rows.empty:
        return pd.DataFrame()
    return pd.DataFrame([dict(zip(oxide_rows["Component"], oxide_rows["Mol_pct"]))])


def composition_summary(
    oxide_table: pd.DataFrame, role_map: Optional[Dict[str, str]] = None,
    envelope: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, object]:
    """Optical basicity + NBO/T + envelope pass/fail for one already-built
    oxide_table (as returned by convert_composition_to_oxides /
    tank_oxide_composition)."""
    if oxide_table is None or oxide_table.empty:
        return {
            "optical_basicity": float("nan"), "excluded_from_basicity": [],
            "T": 0.0, "O_total": 0.0, "NBO_T": float("nan"), "envelope_table": pd.DataFrame(),
        }
    oxide_rows = oxide_table[oxide_table["Kind"] == "oxide"]
    mol_map = dict(zip(oxide_rows["Component"], oxide_rows["Mol"]))

    components = [(formula, mol) for formula, mol in mol_map.items() if formula in gscience.OPTICAL_BASICITY]
    excluded_from_basicity = sorted(set(mol_map) - set(gscience.OPTICAL_BASICITY))
    basicity = float("nan")
    if components:
        try:
            basicity = gscience.optical_basicity(components, basis="mol")["basicity"]
        except ValueError:
            basicity = float("nan")

    nbo_result = nbo_over_t(mol_map, role_map)

    envelope_table = pd.DataFrame()
    if envelope:
        wt_map = dict(zip(oxide_table["Component"], oxide_table["Wt_pct"]))
        envelope_table = envelope_check(wt_map, envelope)

    return {
        "optical_basicity": basicity, "excluded_from_basicity": excluded_from_basicity,
        "T": nbo_result["T"], "O_total": nbo_result["O_total"], "NBO_T": nbo_result["NBO_T"],
        "envelope_table": envelope_table,
    }
