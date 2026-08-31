"""
query_templates.py — guided question templates (the "keep it easy" layer).

Each template is a plain-language question bound to a complete, runnable
QuerySpec (subject x basis x grouping) plus plot-grammar hints. Picking one
in the Query workspace fills the controls AND runs — a colleague who has
never seen the engine gets a correct, disclosed answer in one click, and
can then see exactly which knobs produced it (the controls are set, not
hidden).

Templates only reference shipped engine capabilities: BASKET_PRESETS names,
element symbols, exact analyte labels, and DIMENSIONS keys. A test pins
that every template runs end-to-end on a chemistry fixture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class QueryTemplate:
    key: str                       # stable identifier (persistence/telemetry)
    question: str                  # plain-language label shown in the picker
    explanation: str               # one line under/behind the picker
    subject_kind: str              # "basket" | "element" | "analyte"
    subject: str                   # basket preset name, element symbol, or exact analyte
    basis: str                     # engine basis
    group_by: Tuple[str, ...] = ()
    mark: str = "bar"              # plot-grammar mark (must be a plot_grammar.MARKS key)


TEMPLATES: List[QueryTemplate] = [
    QueryTemplate(
        key="salt_formers_wtpct",
        question="Where do salt formers concentrate? (wt%, tank × phase)",
        explanation="Halides + apportioned sulfate-S + Cr as wt% of each layer — the molten-salt risk map.",
        subject_kind="basket", subject="Salt formers", basis="wt%",
        group_by=("WasteSiteId", "WastePhase"),
    ),
    QueryTemplate(
        key="cs137_by_farm",
        question="Which farms hold the most ¹³⁷Cs activity?",
        explanation="Reported ¹³⁷Cs curies summed per tank farm.",
        subject_kind="analyte", subject="137Cs", basis="Ci",
        group_by=("TankFarm",),
    ),
    QueryTemplate(
        key="uranium_grams_per_tank",
        question="How much uranium is in each tank, in grams?",
        explanation="Reported uranium mass; where a layer gives only isotope activity, "
                    "that is bridged to grams via ICRP-107 rather than left out.",
        subject_kind="element", subject="U", basis="g",
        group_by=("WasteSiteId",),
    ),
    QueryTemplate(
        key="mobile_longlived_by_farm",
        question="Where are the mobile long-lived drivers?",
        explanation="⁹⁹Tc + ¹²⁹I (bridged) + nitrate, in grams, per farm — the groundwater set.",
        subject_kind="basket", subject="Mobile long-lived", basis="g",
        group_by=("TankFarm",),
    ),
    QueryTemplate(
        key="rcra_by_phase",
        question="How do RCRA metals split across waste phases?",
        explanation="The eight 40 CFR 261 toxicity-characteristic metals, kg, by waste phase.",
        subject_kind="basket", subject="RCRA metals", basis="kg",
        group_by=("WastePhase",),
    ),
    QueryTemplate(
        key="glass_formers_per_tank",
        question="Which tanks carry the most glass formers?",
        explanation="B, Si, P, Al, Zr, Ti in kg per tank — vitrification feed at a glance.",
        subject_kind="basket", subject="Glass formers", basis="kg",
        group_by=("WasteSiteId",),
    ),
    QueryTemplate(
        key="sulfate_by_tank",
        question="Where is the sulfate?",
        explanation="SO₄ as reported (full species mass, kg) per tank.",
        subject_kind="analyte", subject="SO4", basis="kg",
        group_by=("WasteSiteId",),
    ),
    QueryTemplate(
        key="chromium_gl",
        question="Chromium concentration per tank (g/L)",
        explanation="Apportioned Cr mass over layer volumes — the refractory-spinel driver.",
        subject_kind="element", subject="Cr", basis="g/L",
        group_by=("WasteSiteId",),
    ),
    QueryTemplate(
        key="iodine_by_phase",
        question="How much iodine, and in which phases?",
        explanation="¹²⁹I bridged to grams (the only iodine reporting — a lower bound, disclosed).",
        subject_kind="element", subject="I", basis="g",
        group_by=("WastePhase",),
    ),
]


def template_by_key(key: str) -> Optional[QueryTemplate]:
    for t in TEMPLATES:
        if t.key == key:
            return t
    return None
