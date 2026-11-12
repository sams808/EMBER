"""
glass_rules.py -- published glass loading rules for low-activity waste
(LAW) glass, applied to the salt waste of each tank or waste layer
(framework-agnostic: no Qt, pure functions and dataclasses).

The question this answers: if the supernatant and saltcake of a tank were
the melter feed, WHICH published loading rule would set the sodium loading
of the glass, how far below the alkali limit would that be, and what would
it cost in glass?

Source of every rule and constant:

    J.D. Vienna, D.C. Skorski, D.S. Kim, J. Matyas, "Glass Property Models
    and Constraints for Estimating the Glass to be Produced at Hanford by
    Implementing Current Advanced Glass Formulation Efforts",
    PNNL-22631 Rev. 1 (ORP-58289), Pacific Northwest National Laboratory,
    2013.

Three rule sets are offered (all concentrations in wt% of glass):

Advanced LAW (2013) -- Table S.12, Eq. 3.15 to 3.17, Figure 3.4b
    alkali            Na2O + 0.66 K2O <= 24
    alkali + sulfur   Na2O + 0.66 K2O <= 33.94 - 11.69 SO3
    sulfur            SO3 <= 1.5
    halide            SO3 <= 1.65 - 0.725 (Cl + 0.3 F + 0.4 Cr2O3)
                      (the "optimistic" halide rule)

Advanced LAW, conservative halide -- the same alkali, alkali + sulfur and
sulfur rules, with the halide rule replaced by the conservative envelope
of Figure 3.4a:
    Cl + 0.607 F + 0.542 Cr2O3 + K2O <= 3.746 - 4.694 SO3
    or                               <= 1.243 - 0.4506 SO3
    (the boundary is the upper of the two lines), with SO3 <= 1.5.

WTP baseline -- Eq. 3.1 to 3.6
    sodium            Na2O <= 21
    alkali            Na2O + 0.66 K2O <= 21.5
    alkali + sulfur   Na2O + 42.5 SO3 <= 35.9
    sulfur            SO3 <= 0.77
    halide            NH = Cl + 0.3 F:
                      NH <= 1.4656 - 2.1111 SO3   for SO3 <= 0.59
                      NH <= 0.22                  for SO3 >  0.59
    Eq. 3.7 to 3.10 of the report (limits on Cr2O3 and K2O that depend on
    the P2O5 content of the glass) are NOT implemented.

How a rule becomes a loading. Every waste component enters the glass in
proportion to the sodium: with the waste mass ratios r_X = X / Na2O, a
glass loaded to w wt% Na2O holds w * r_X wt% of X. Each rule is therefore
an inequality in w alone and is solved for the largest w it allows, e.g.

    alkali + sulfur:  w = 33.94 / (1 + 0.66 r_K2O + 11.69 r_SO3)
    halide:           w = 1.65 / (r_SO3 + 0.725 (r_Cl + 0.3 r_F + 0.4 r_Cr2O3))

The loading of a unit is the smallest of its rule limits and the BINDING
rule is the one that gives it. Glass mass = Na2O mass / (w / 100). "Extra
glass" is the glass at the rule-limited loading minus the glass the same
sodium would need at the alkali limit alone.

Oxide conversions. The inventory reports Na, K, Cr, Cl and F as elements
and sulfur as the sulfate anion. Na -> Na2O, K -> K2O, Cr -> Cr2O3 and
SO4 -> SO3 use gravimetric factors computed from the atomic masses in
reference_data; Cl and F stay elements, as the rules state them.

Scope of validity. The rules were developed for LAW melter feeds. Applying
them to the supernatant and saltcake of a tank AS STORED ignores retrieval,
blending, pretreatment, the sodium added in processing and the wash
liquors of sludge. The result is a screening index of which rule would
bind, not a forecast of glass production. A unit holding no sodium has no
loading: it is left blank, never set to zero.

Blending. In the Advanced LAW (2013) set every rule is linear in the
component masses, so the glass needed by a blend never exceeds the sum of
the glass its parts need alone (equivalently: the blended loading is never
below the sodium-weighted HARMONIC mean of the unit loadings; it can be
well below their arithmetic mean). The conservative halide envelope and
the baseline halogen rule are piecewise, and for them a blend can need
more glass than its parts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import association_query as aq
from data_model import HanfordDataset
from reference_data import ATOMIC_MASSES

CITATION = (
    "J.D. Vienna, D.C. Skorski, D.S. Kim, J. Matyas, Glass Property Models and Constraints "
    "for Estimating the Glass to be Produced at Hanford by Implementing Current Advanced Glass "
    "Formulation Efforts, PNNL-22631 Rev. 1 (ORP-58289), Pacific Northwest National Laboratory, 2013."
)
SCOPE_OF_VALIDITY = (
    "The loading rules were developed for LAW melter feeds. Applied to waste as stored they ignore "
    "retrieval, blending, pretreatment, the sodium added in processing and sludge wash liquors: "
    "this is a screening index of which rule would bind, not a forecast of glass production."
)

# ---------------------------------------------------------------------------
# Inventory species -> glass components
# ---------------------------------------------------------------------------
# What the screen asks the inventory for, as the BBI reports it.
SPECIES: Tuple[str, ...] = ("Na", "K", "SO4", "Cl", "F", "Cr")


def _oxide_per_species(element: str, n_element: int, n_oxygen: int,
                       species_mass: Optional[float] = None) -> float:
    """kg of the oxide El_n O_m per kg of the reported species.

    The species is the element itself unless `species_mass` (g/mol of a
    species carrying ONE atom of the element, e.g. SO4) is given.
    """
    oxide = n_element * ATOMIC_MASSES[element] + n_oxygen * ATOMIC_MASSES["O"]
    source = species_mass if species_mass is not None else ATOMIC_MASSES[element]
    return oxide / (n_element * source)


_SO4_MASS = ATOMIC_MASSES["S"] + 4 * ATOMIC_MASSES["O"]

# glass component -> (inventory species, kg of component per kg of species)
COMPONENTS: Dict[str, Tuple[str, float]] = {
    "Na2O": ("Na", _oxide_per_species("Na", 2, 1)),
    "K2O": ("K", _oxide_per_species("K", 2, 1)),
    "SO3": ("SO4", _oxide_per_species("S", 1, 3, _SO4_MASS)),
    "Cl": ("Cl", 1.0),
    "F": ("F", 1.0),
    "Cr2O3": ("Cr", _oxide_per_species("Cr", 2, 3)),
}
RATIO_COMPONENTS: Tuple[str, ...] = ("K2O", "SO3", "Cl", "F", "Cr2O3")

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
ALKALI, ALKALI_SULFUR, SULFUR, HALIDE = "Alkali", "Alkali + sulfur", "Sulfur", "Halide"
# The four kinds of rule, in the order a diagram stacks them. Every rule of
# every set belongs to one; colours and the "not alkali-limited" share are
# defined on the class, so rule sets with different rule counts compare.
RULE_CLASSES: Tuple[str, ...] = (ALKALI, ALKALI_SULFUR, SULFUR, HALIDE)

# A unit loaded below this is called out in the summary (wt% Na2O).
LOW_LOADING_WTPCT = 15.0

# Advanced LAW rules (Table S.12, Eq. 3.15 to 3.17, Figure 3.4b)
ADV_ALKALI_MAX = 24.0
ADV_K2O_WEIGHT = 0.66
ADV_AS_INTERCEPT, ADV_AS_SLOPE = 33.94, 11.69
ADV_SO3_MAX = 1.5
ADV_HALIDE_INTERCEPT, ADV_HALIDE_SLOPE = 1.65, 0.725
ADV_HALIDE_WEIGHTS: Dict[str, float] = {"Cl": 1.0, "F": 0.3, "Cr2O3": 0.4}
# Conservative halide envelope (Figure 3.4a): two lines, the upper one holds
CONS_HALIDE_WEIGHTS: Dict[str, float] = {"Cl": 1.0, "F": 0.607, "Cr2O3": 0.542, "K2O": 1.0}
CONS_LINE_1 = (3.746, 4.694)      # halide term <= 3.746 - 4.694 SO3
CONS_LINE_2 = (1.243, 0.4506)     # halide term <= 1.243 - 0.4506 SO3
# WTP baseline rules (Eq. 3.1 to 3.6)
BASE_NA2O_MAX = 21.0
BASE_ALKALI_MAX = 21.5
BASE_K2O_WEIGHT = 0.66
BASE_AS_INTERCEPT, BASE_AS_SLOPE = 35.9, 42.5
BASE_SO3_MAX = 0.77
BASE_HALOGEN_WEIGHTS: Dict[str, float] = {"Cl": 1.0, "F": 0.3}
BASE_NH_INTERCEPT, BASE_NH_SLOPE = 1.4656, 2.1111
BASE_NH_SO3_BREAK = 0.59
BASE_NH_HIGH_SULFUR = 0.22


@dataclass(frozen=True)
class Rule:
    """One published inequality."""

    name: str            # label of the rule within its set (table column, binding rule)
    rule_class: str      # one of RULE_CLASSES
    equation: str        # as published, wt% of glass
    source: str          # where it is in the report


# (K2O / Na2O, SO3 / Na2O, halide term / Na2O) -> {rule name: largest Na2O loading, wt%}
LimitFunction = Callable[[np.ndarray, np.ndarray, np.ndarray], Dict[str, np.ndarray]]


@dataclass(frozen=True)
class RuleSet:
    """A named set of loading rules.

    Each set reduces a waste to three numbers -- K2O / Na2O, SO3 / Na2O
    and its own "halide term" per Na2O (`halide_weights` says which
    components it adds up) -- and `limit_function` turns those into the
    largest Na2O loading each rule allows. Two axes (SO3 and the halide
    term) therefore draw the whole rule set for a given potassium ratio.
    """

    name: str
    citation: str
    description: str
    rules: Tuple[Rule, ...]
    halide_weights: Mapping[str, float]
    limit_function: LimitFunction = field(repr=False, compare=False)

    @property
    def rule_names(self) -> List[str]:
        return [rule.name for rule in self.rules]

    @property
    def rule_classes(self) -> Dict[str, str]:
        return {rule.name: rule.rule_class for rule in self.rules}

    @property
    def halide_label(self) -> str:
        """The halide term as text, e.g. 'Cl + 0.3 F + 0.4 Cr2O3'."""
        parts = []
        for component, weight in self.halide_weights.items():
            parts.append(component if weight == 1.0 else f"{weight:g} {component}")
        return " + ".join(parts)

    def rules_of_class(self, rule_class: str) -> List[str]:
        return [rule.name for rule in self.rules if rule.rule_class == rule_class]

    def halide_term(self, ratios: pd.DataFrame) -> pd.Series:
        """This set's halide term per unit mass of Na2O."""
        filled = ratios.fillna(0.0)
        term = pd.Series(0.0, index=ratios.index)
        for component, weight in self.halide_weights.items():
            term = term + weight * filled[component]
        return term

    def limits(self, k2o, so3, halide) -> Dict[str, np.ndarray]:
        """Largest Na2O loading (wt% of glass) each rule allows, from the
        three mass ratios to Na2O. A rule that does not constrain a waste
        (no sulfate for the sulfur rule) returns infinity."""
        k = np.asarray(k2o, dtype=float)
        s = np.asarray(so3, dtype=float)
        h = np.asarray(halide, dtype=float)
        k, s, h = np.broadcast_arrays(k, s, h)
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.limit_function(k, s, h)


def _advanced_common(k: np.ndarray, s: np.ndarray) -> Dict[str, np.ndarray]:
    alkali = 1.0 + ADV_K2O_WEIGHT * k
    return {
        ALKALI: ADV_ALKALI_MAX / alkali,
        ALKALI_SULFUR: ADV_AS_INTERCEPT / (alkali + ADV_AS_SLOPE * s),
        SULFUR: ADV_SO3_MAX / s,
    }


def _advanced_limits(k: np.ndarray, s: np.ndarray, h: np.ndarray) -> Dict[str, np.ndarray]:
    out = _advanced_common(k, s)
    # SO3 <= 1.65 - 0.725 H, with SO3 = w s and H = w h
    out[HALIDE] = ADV_HALIDE_INTERCEPT / (s + ADV_HALIDE_SLOPE * h)
    return out


def _conservative_limits(k: np.ndarray, s: np.ndarray, h: np.ndarray) -> Dict[str, np.ndarray]:
    out = _advanced_common(k, s)
    # Allowed when H <= a1 - b1 SO3 OR H <= a2 - b2 SO3. Each line allows
    # every loading up to a / (h + b s); the union of the two intervals is
    # the larger of the two.
    (a1, b1), (a2, b2) = CONS_LINE_1, CONS_LINE_2
    out[HALIDE] = np.maximum(a1 / (h + b1 * s), a2 / (h + b2 * s))
    return out


def _baseline_limits(k: np.ndarray, s: np.ndarray, h: np.ndarray) -> Dict[str, np.ndarray]:
    # Piecewise halogen rule. Below the loading at which SO3 reaches 0.59
    # wt% the sloped line applies; above it, the flat 0.22 wt% cap. The
    # flat branch is reachable only if its own limit lies above that
    # loading; otherwise the sloped branch (cut at the break) is the answer.
    at_break = BASE_NH_SO3_BREAK / s
    sloped = np.minimum(at_break, BASE_NH_INTERCEPT / (h + BASE_NH_SLOPE * s))
    flat = BASE_NH_HIGH_SULFUR / h
    return {
        "Sodium": np.full(s.shape, BASE_NA2O_MAX),
        ALKALI: BASE_ALKALI_MAX / (1.0 + BASE_K2O_WEIGHT * k),
        ALKALI_SULFUR: BASE_AS_INTERCEPT / (1.0 + BASE_AS_SLOPE * s),
        SULFUR: BASE_SO3_MAX / s,
        HALIDE: np.where(flat > at_break, flat, sloped),
    }


ADVANCED = "Advanced LAW (2013)"
CONSERVATIVE = "Advanced LAW, conservative halide"
BASELINE = "WTP baseline"

_ADVANCED_BASE_RULES = (
    Rule(ALKALI, ALKALI, "Na2O + 0.66 K2O <= 24", "Table S.12; Eq. 3.15"),
    Rule(ALKALI_SULFUR, ALKALI_SULFUR, "Na2O + 0.66 K2O <= 33.94 - 11.69 SO3", "Table S.12; Eq. 3.16"),
    Rule(SULFUR, SULFUR, "SO3 <= 1.5", "Table S.12; Eq. 3.17"),
)

RULE_SETS: Dict[str, RuleSet] = {
    ADVANCED: RuleSet(
        name=ADVANCED,
        citation=CITATION,
        description="Advanced LAW glass loading rules with the optimistic halide rule "
                    "(Table S.12, Eq. 3.15 to 3.17, Figure 3.4b).",
        rules=_ADVANCED_BASE_RULES + (
            Rule(HALIDE, HALIDE, "SO3 <= 1.65 - 0.725 (Cl + 0.3 F + 0.4 Cr2O3)", "Table S.12; Figure 3.4b"),
        ),
        halide_weights=ADV_HALIDE_WEIGHTS,
        limit_function=_advanced_limits,
    ),
    CONSERVATIVE: RuleSet(
        name=CONSERVATIVE,
        citation=CITATION,
        description="Advanced LAW alkali and sulfur rules with the conservative halide envelope "
                    "(Eq. 3.15 to 3.17, Figure 3.4a).",
        rules=_ADVANCED_BASE_RULES + (
            Rule(HALIDE, HALIDE,
                 "Cl + 0.607 F + 0.542 Cr2O3 + K2O <= 3.746 - 4.694 SO3 or <= 1.243 - 0.4506 SO3",
                 "Figure 3.4a"),
        ),
        halide_weights=CONS_HALIDE_WEIGHTS,
        limit_function=_conservative_limits,
    ),
    BASELINE: RuleSet(
        name=BASELINE,
        citation=CITATION,
        description="WTP baseline LAW formulation rules as summarised in the same report "
                    "(Eq. 3.1 to 3.6; Eq. 3.7 to 3.10 are not implemented).",
        rules=(
            Rule("Sodium", ALKALI, "Na2O <= 21", "Eq. 3.1"),
            Rule(ALKALI, ALKALI, "Na2O + 0.66 K2O <= 21.5", "Eq. 3.2"),
            Rule(ALKALI_SULFUR, ALKALI_SULFUR, "Na2O + 42.5 SO3 <= 35.9", "Eq. 3.3"),
            Rule(SULFUR, SULFUR, "SO3 <= 0.77", "Eq. 3.4"),
            Rule(HALIDE, HALIDE,
                 "Cl + 0.3 F <= 1.4656 - 2.1111 SO3 for SO3 <= 0.59; Cl + 0.3 F <= 0.22 for SO3 > 0.59",
                 "Eq. 3.5 and 3.6"),
        ),
        halide_weights=BASE_HALOGEN_WEIGHTS,
        limit_function=_baseline_limits,
    ),
}
DEFAULT_RULE_SET = ADVANCED


def get_rule_set(rule_set) -> RuleSet:
    """A RuleSet from its name (or the RuleSet itself)."""
    if isinstance(rule_set, RuleSet):
        return rule_set
    if rule_set is None:
        return RULE_SETS[DEFAULT_RULE_SET]
    try:
        return RULE_SETS[str(rule_set)]
    except KeyError:
        raise ValueError(f"Unknown rule set {rule_set!r}; expected one of {list(RULE_SETS)}") from None


# ---------------------------------------------------------------------------
# Pure computation on a frame of kilograms
# ---------------------------------------------------------------------------
def oxide_masses(kg: pd.DataFrame) -> pd.DataFrame:
    """kg of each glass component (Na2O, K2O, SO3, Cl, F, Cr2O3) from kg of
    the inventory species (Na, K, SO4, Cl, F, Cr). A species a unit does
    not report counts as zero here; `screen` discloses how many."""
    out = {}
    for component, (species, factor) in COMPONENTS.items():
        column = kg[species] if species in kg.columns else pd.Series(0.0, index=kg.index)
        out[component] = column.astype(float).fillna(0.0) * factor
    return pd.DataFrame(out, index=kg.index)


def oxide_ratios(kg: pd.DataFrame) -> pd.DataFrame:
    """Mass of K2O, SO3, Cl, F and Cr2O3 per unit mass of Na2O in the
    waste. Blank where a unit holds no sodium."""
    masses = oxide_masses(kg)
    sodium = masses["Na2O"].where(masses["Na2O"] > 0)
    return masses[list(RATIO_COMPONENTS)].div(sodium, axis=0)


def loading_limits(ratios: pd.DataFrame, rule_set=None) -> pd.DataFrame:
    """Largest Na2O loading (wt% of glass) each rule of the set allows,
    one column per rule. `ratios` carries K2O, SO3, Cl, F and Cr2O3 per
    Na2O; a row with a blank SO3 ratio (no sodium) is blank throughout."""
    rules = get_rule_set(rule_set)
    defined = ratios["SO3"].notna()
    filled = ratios.fillna(0.0)
    limits = rules.limits(filled["K2O"].to_numpy(), filled["SO3"].to_numpy(),
                          rules.halide_term(ratios).to_numpy())
    frame = pd.DataFrame({name: limits[name] for name in rules.rule_names}, index=ratios.index)
    return frame.where(defined, np.nan)


def screen_frame(kg: pd.DataFrame, rule_set=None) -> pd.DataFrame:
    """The loading-rule screen of waste units given kg of Na, K, SO4, Cl,
    F and Cr (one row per unit, index kept).

    Columns: Na2O_t; the mass ratios to Na2O and the set's halide term;
    Limit_<rule> for each rule; Na2O_loading_wtpct; Binding_rule;
    Glass_t, Glass_t_alkali_only, Extra_glass_t; and the SO3, Cl, F and
    Cr2O3 content of the glass at that loading. Tonnes are metric.
    """
    rules = get_rule_set(rule_set)
    masses = oxide_masses(kg)
    has_sodium = masses["Na2O"] > 0
    ratios = masses[list(RATIO_COMPONENTS)].div(masses["Na2O"].where(has_sodium), axis=0)
    limits = loading_limits(ratios, rules)

    out = pd.DataFrame(index=kg.index)
    out["Na2O_t"] = masses["Na2O"] / 1000.0
    for component in RATIO_COMPONENTS:
        out[f"{component}_per_Na2O"] = ratios[component]
    out["Halide_term_per_Na2O"] = rules.halide_term(ratios).where(has_sodium)
    for name in rules.rule_names:
        out[f"Limit_{name}"] = limits[name]
    loading = limits.min(axis=1).where(has_sodium)
    out["Na2O_loading_wtpct"] = loading
    binding = pd.Series([None] * len(kg), index=kg.index, dtype=object)
    if has_sodium.any():
        binding.loc[has_sodium] = limits.loc[has_sodium].idxmin(axis=1)
    out["Binding_rule"] = binding
    alkali_limit = limits[rules.rules_of_class(ALKALI)].min(axis=1)
    out["Glass_t"] = out["Na2O_t"] / (loading / 100.0)
    out["Glass_t_alkali_only"] = (out["Na2O_t"] / (alkali_limit / 100.0)).where(has_sodium)
    out["Extra_glass_t"] = out["Glass_t"] - out["Glass_t_alkali_only"]
    for component in ("SO3", "Cl", "F", "Cr2O3"):
        out[f"{component}_in_glass_wtpct"] = ratios[component] * loading
    return out


def rule_shares(table: pd.DataFrame, rule_set=None) -> pd.Series:
    """Share (%) of the Na2O held by the units each rule binds, indexed by
    rule name. Units without a loading are left out; an empty table gives
    blanks."""
    rules = get_rule_set(rule_set)
    bound = table.dropna(subset=["Binding_rule"])
    held = bound.groupby("Binding_rule")["Na2O_t"].sum().reindex(rules.rule_names).fillna(0.0)
    total = float(held.sum())
    if total <= 0:
        return pd.Series(np.nan, index=rules.rule_names)
    return 100.0 * held / total


# ---------------------------------------------------------------------------
# The screen through the engine
# ---------------------------------------------------------------------------
SALT_WASTE: Tuple[str, ...] = ("Supernatant", "Saltcake")
LEVELS = ("tank", "layer")
LEVEL_NOUNS = {"tank": "tank", "layer": "waste layer"}
UNIT_COLUMN = "Unit"


def phases_label(phases: Optional[Sequence[str]]) -> str:
    """'supernatant + saltcake', 'saltcake', 'all phases'."""
    if not phases:
        return "all phases"
    return " + ".join(str(p).lower() for p in phases)


@dataclass
class ScreenSummary:
    """The headline numbers of one screen."""

    n_units: int = 0                      # units offered (after the phase selection)
    n_with_sodium: int = 0                # of which hold sodium and so have a loading
    na2o_kt: float = float("nan")
    share_by_rule: Dict[str, float] = field(default_factory=dict)     # % of the Na2O, by binding rule
    not_alkali_limited_pct: float = float("nan")
    glass_alkali_only_kt: float = float("nan")
    glass_unit_by_unit_kt: float = float("nan")
    extra_glass_kt: float = float("nan")
    extra_glass_pct: float = float("nan")
    glass_blended_kt: float = float("nan")
    blend_loading_wtpct: float = float("nan")
    blend_binding_rule: Optional[str] = None
    k2o_per_na2o: float = float("nan")    # of the blended feed (sodium-weighted mean)
    n_below_low_loading: int = 0
    low_loading_wtpct: float = LOW_LOADING_WTPCT

    def as_table(self, unit_noun: str = "unit") -> pd.DataFrame:
        """Quantity / value / unit rows, for display and export. Every
        value is a number (the binding rule of the blended feed is named
        in the label of its loading), so the column sorts and aligns."""
        blend_rule = f" ({self.blend_binding_rule.lower()} rule)" if self.blend_binding_rule else ""
        rows = [
            (f"{unit_noun.capitalize()}s screened", self.n_units, ""),
            (f"{unit_noun.capitalize()}s holding sodium", self.n_with_sodium, ""),
            ("Sodium screened", self.na2o_kt, "kt Na2O"),
        ]
        rows += [(f"Sodium bound by the {name.lower()} rule", value, "% of Na2O")
                 for name, value in self.share_by_rule.items()]
        rows += [
            ("Sodium not alkali-limited", self.not_alkali_limited_pct, "% of Na2O"),
            ("Glass at the alkali limit", self.glass_alkali_only_kt, "kt"),
            (f"Glass, {unit_noun} by {unit_noun}", self.glass_unit_by_unit_kt, "kt"),
            ("Extra glass over the alkali limit", self.extra_glass_kt, "kt"),
            ("Extra glass over the alkali limit", self.extra_glass_pct, "%"),
            ("Glass, one blended feed", self.glass_blended_kt, "kt"),
            (f"Loading of the blended feed{blend_rule}", self.blend_loading_wtpct, "wt% Na2O"),
            ("K2O / Na2O of the blended feed", self.k2o_per_na2o, "mass ratio"),
            (f"{unit_noun.capitalize()}s below {self.low_loading_wtpct:g} wt% Na2O",
             self.n_below_low_loading, ""),
        ]
        table = pd.DataFrame(rows, columns=["Quantity", "Value", "Unit"])
        table["Value"] = table["Value"].astype(float)
        return table


@dataclass
class ScreenResult:
    """Everything one run of the screen produced."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)         # one row per unit
    layer_table: pd.DataFrame = field(default_factory=pd.DataFrame)   # the same screen, layer by layer
    blend: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))   # all units as one feed
    summary: ScreenSummary = field(default_factory=ScreenSummary)
    groups: pd.DataFrame = field(default_factory=pd.DataFrame)        # sodium by binding rule, by group
    rule_set: RuleSet = field(default_factory=lambda: RULE_SETS[DEFAULT_RULE_SET])
    level: str = "tank"
    phases: Optional[Tuple[str, ...]] = SALT_WASTE
    scope: Dict[str, List[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)       # the engine's, about the data
    disclosures: List[str] = field(default_factory=list)    # how each species was obtained
    notes: List[str] = field(default_factory=list)          # the screen's own statements, all of them
    remarks: List[str] = field(default_factory=list)        # of which: what is particular to THIS result

    @property
    def unit_noun(self) -> str:
        return LEVEL_NOUNS[self.level]

    @property
    def waste_label(self) -> str:
        return phases_label(self.phases)

    def screened(self) -> pd.DataFrame:
        """The units that hold sodium (the ones with a loading)."""
        if self.table.empty:
            return self.table
        return self.table.dropna(subset=["Binding_rule"])


def _tank_frame(layers: pd.DataFrame) -> pd.DataFrame:
    """Layers summed per tank. A species no layer of the tank reports
    stays blank; the tank's farm and system are carried along."""
    species = list(SPECIES)
    # Grouping on the tank id alone: pandas drops rows whose group key is
    # null, and a tank without attributes has a null farm or system.
    grouped = layers.groupby("WasteSiteId", sort=False)
    out = grouped[species].sum(min_count=1)
    out["N_layers"] = grouped.size()
    if "Volume_kL" in layers.columns:
        out["Volume_kL"] = grouped["Volume_kL"].sum(min_count=1)
    for column in ("TankFarm", "TankSystem"):
        if column in layers.columns:
            out[column] = grouped[column].first()
    return out.reset_index()


def _group_row(name: str, frame: pd.DataFrame, rules: RuleSet) -> Optional[dict]:
    bound = frame.dropna(subset=["Binding_rule"])
    if bound.empty:
        return None
    shares = rule_shares(bound, rules)
    alkali_only = float(bound["Glass_t_alkali_only"].sum()) / 1000.0
    unit_by_unit = float(bound["Glass_t"].sum()) / 1000.0
    row = {"Group": name, "N_units": int(len(bound)), "Na2O_kt": float(bound["Na2O_t"].sum()) / 1000.0}
    row.update({f"{rule}_pct": float(value) for rule, value in shares.items()})
    row["Not_alkali_limited_pct"] = 100.0 - float(shares[rules.rules_of_class(ALKALI)].sum())
    row["Glass_kt_alkali_only"] = alkali_only
    row["Glass_kt_unit_by_unit"] = unit_by_unit
    row["Extra_glass_pct"] = 100.0 * (unit_by_unit / alkali_only - 1.0) if alkali_only > 0 else np.nan
    return row


def screen(
    dataset: HanfordDataset,
    *,
    level: str = "tank",
    phases: Optional[Sequence[str]] = SALT_WASTE,
    rule_set=None,
    scope: Optional[Dict[str, Sequence[str]]] = None,
) -> ScreenResult:
    """Screen every tank (or waste layer) against a set of loading rules.

    level: 'tank' sums the selected layers of each tank; 'layer' keeps one
    row per (tank, waste phase, waste type) entry.
    phases: the phase families that count as the feed (Supernatant,
    Saltcake, Sludge); None takes every layer.
    scope: the query engine's scope ({column: [values]}), applied to the
    inventory before anything is summed.

    Kilograms come from the query engine through
    association_query.association_matrix, so they are the figures every
    other workspace shows; the engine's warnings and disclosures travel
    with the result.
    """
    if level not in LEVELS:
        raise ValueError(f"Unknown level {level!r}; expected one of {list(LEVELS)}")
    rules = get_rule_set(rule_set)
    selected = tuple(phases) if phases else None
    scope_used = {column: list(values) for column, values in (scope or {}).items()}

    matrix = aq.association_matrix(dataset, SPECIES, basis="kg", level="layer", scope=scope)
    layers = matrix.values
    if selected is not None and "PhaseFamily" in layers.columns:
        layers = layers[layers["PhaseFamily"].isin(selected)]
    layers = layers.reset_index(drop=True)

    layer_ids = [c for c in ("WasteSiteId", "WastePhase", "WasteType", "PhaseFamily", "TankFarm",
                             "TankSystem", "Volume_kL") if c in layers.columns]
    layer_table = pd.concat([layers[layer_ids], screen_frame(layers, rules)], axis=1)
    layer_table.insert(0, UNIT_COLUMN, [
        " | ".join(str(part) for part in parts)
        for parts in layers[[c for c in ("WasteSiteId", "WastePhase", "WasteType") if c in layers.columns]]
        .itertuples(index=False, name=None)
    ])

    if level == "tank":
        units = _tank_frame(layers) if len(layers) else layers.assign(N_layers=pd.Series(dtype=int))
        unit_ids = [c for c in ("WasteSiteId", "TankFarm", "TankSystem", "N_layers", "Volume_kL")
                    if c in units.columns]
        table = pd.concat([units[unit_ids], screen_frame(units, rules)], axis=1)
        table.insert(0, UNIT_COLUMN, units["WasteSiteId"].astype(str).to_list())
    else:
        units = layers
        table = layer_table.copy()

    species = list(SPECIES)
    feed = pd.DataFrame([units[species].sum(min_count=1)]) if len(units) else pd.DataFrame(columns=species)
    blend = screen_frame(feed, rules).iloc[0] if len(feed) else pd.Series(dtype=object)

    noun = LEVEL_NOUNS[level]
    bound = table.dropna(subset=["Binding_rule"]) if len(table) else table
    summary = ScreenSummary(n_units=int(len(table)), n_with_sodium=int(len(bound)))
    if len(bound):
        shares = rule_shares(bound, rules)
        alkali_only = float(bound["Glass_t_alkali_only"].sum()) / 1000.0
        unit_by_unit = float(bound["Glass_t"].sum()) / 1000.0
        summary.na2o_kt = float(bound["Na2O_t"].sum()) / 1000.0
        summary.share_by_rule = {rule: float(value) for rule, value in shares.items()}
        summary.not_alkali_limited_pct = 100.0 - float(shares[rules.rules_of_class(ALKALI)].sum())
        summary.glass_alkali_only_kt = alkali_only
        summary.glass_unit_by_unit_kt = unit_by_unit
        summary.extra_glass_kt = unit_by_unit - alkali_only
        summary.extra_glass_pct = 100.0 * (unit_by_unit / alkali_only - 1.0)
        summary.glass_blended_kt = float(blend["Glass_t"]) / 1000.0
        summary.blend_loading_wtpct = float(blend["Na2O_loading_wtpct"])
        summary.blend_binding_rule = blend["Binding_rule"]
        summary.k2o_per_na2o = float(blend["K2O_per_Na2O"])
        summary.n_below_low_loading = int((bound["Na2O_loading_wtpct"] < LOW_LOADING_WTPCT).sum())

    group_frames: List[Tuple[str, pd.DataFrame]] = [(f"All {noun}s", table)]
    if "TankSystem" in table.columns:
        single, double = (("Single-shell tanks", "Double-shell tanks") if level == "tank" else
                          ("Layers in single-shell tanks", "Layers in double-shell tanks"))
        group_frames.append((single, table[table["TankSystem"] == "SST"]))
        group_frames.append((double, table[table["TankSystem"] == "DST"]))
    if "PhaseFamily" in layer_table.columns:
        for family in ("Saltcake", "Supernatant", "Sludge"):
            if selected is None or family in selected:
                group_frames.append((f"{family} layers", layer_table[layer_table["PhaseFamily"] == family]))
    group_rows = [row for name, frame in group_frames
                  if len(frame) and (row := _group_row(name, frame, rules)) is not None]
    groups = pd.DataFrame(group_rows)

    notes = [
        SCOPE_OF_VALIDITY,
        f"Rule set: {rules.name}. " + "; ".join(f"{rule.name.lower()}: {rule.equation}" for rule in rules.rules)
        + " (wt% of glass).",
        f"Feed: the {phases_label(selected)} of each {noun}, as stored. Na2O, K2O, Cr2O3 and SO3 are "
        "computed from Na, K, Cr and SO4 with gravimetric factors; Cl and F are elements.",
        "Glass mass = Na2O mass / (Na2O loading / 100). Extra glass = glass at the rule-limited loading "
        "minus glass at the alkali limit alone.",
    ]
    remarks: List[str] = []
    if len(table):
        without = int(len(table) - len(bound))
        if without:
            remarks.append(f"{without} {noun}(s) hold no sodium in the selected waste: "
                           "no loading is stated for them.")
        for name in SPECIES:
            if name == "Na":
                continue
            blank = int(units.loc[units["Na"].fillna(0.0) > 0, name].isna().sum())
            if blank:
                remarks.append(f"{blank} {noun}(s) holding sodium do not report {name}; "
                               "it counts as zero in the rules.")
    else:
        remarks.append(f"No {noun} matches the selected waste and scope.")
    if rules.name != ADVANCED:
        remarks.append("This rule set is not linear in the component masses: a blended feed can need "
                       "more glass than the sum of its parts.")
    notes += remarks

    return ScreenResult(
        table=table.reset_index(drop=True), layer_table=layer_table.reset_index(drop=True),
        blend=blend, summary=summary, groups=groups, rule_set=rules, level=level, phases=selected,
        scope=scope_used, warnings=list(matrix.warnings), disclosures=list(matrix.disclosures), notes=notes,
        remarks=remarks,
    )


# ---------------------------------------------------------------------------
# The rule diagram
# ---------------------------------------------------------------------------
DIAGRAM_X_RANGE = (0.0025, 0.62)      # SO3 / Na2O
DIAGRAM_Y_RANGE = (0.0038, 0.105)     # halide term / Na2O
DIAGRAM_LEVELS = (5.0, 10.0, 15.0, 20.0, 23.0)


@dataclass
class RegionGrid:
    """A rule set evaluated over (SO3 / Na2O, halide term / Na2O)."""

    x: np.ndarray                 # SO3 / Na2O, shape (nx,)
    y: np.ndarray                 # halide term / Na2O, shape (ny,)
    loading: np.ndarray           # allowed Na2O loading, wt%, shape (ny, nx)
    rule_index: np.ndarray        # index into rule_names of the binding rule, shape (ny, nx)
    class_index: np.ndarray       # index into RULE_CLASSES of its class, shape (ny, nx)
    rule_names: Tuple[str, ...]
    k2o_ratio: float
    rule_set: RuleSet


def region_grid(
    rule_set=None,
    *,
    k2o_ratio: float = 0.0,
    x_range: Tuple[float, float] = DIAGRAM_X_RANGE,
    y_range: Tuple[float, float] = DIAGRAM_Y_RANGE,
    nx: int = 360,
    ny: int = 280,
) -> RegionGrid:
    """The binding rule and the allowed Na2O loading over a logarithmic
    grid of SO3 / Na2O (x) and the set's halide term per Na2O (y).

    The alkali rules also depend on potassium; the grid is drawn for ONE
    K2O / Na2O ratio (`k2o_ratio`, typically that of the blended feed).
    A unit with a different potassium ratio can therefore sit in a region
    other than its own binding rule near the alkali boundary.
    """
    rules = get_rule_set(rule_set)
    x = np.geomspace(x_range[0], x_range[1], int(nx))
    y = np.geomspace(y_range[0], y_range[1], int(ny))
    xx, yy = np.meshgrid(x, y)
    limits = rules.limits(np.full_like(xx, float(k2o_ratio)), xx, yy)
    stack = np.stack([limits[name] for name in rules.rule_names])
    rule_index = stack.argmin(axis=0)
    class_of_rule = np.array([RULE_CLASSES.index(rule.rule_class) for rule in rules.rules])
    return RegionGrid(
        x=x, y=y, loading=stack.min(axis=0), rule_index=rule_index,
        class_index=class_of_rule[rule_index], rule_names=tuple(rules.rule_names),
        k2o_ratio=float(k2o_ratio), rule_set=rules,
    )
