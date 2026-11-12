"""
Tests for glass_rules: the published LAW glass loading rules (PNNL-22631
Rev. 1) and the screen that applies them to tanks and waste layers.

Expected values are worked by hand from the published inequalities, or
found by brute force (scan the Na2O loading upward and test the
inequalities as printed), never by calling the code under test twice. On
the real dataset the screen is recomputed from the raw records with plain
pandas and literal conversion factors.
"""
import numpy as np
import pandas as pd
import polars as pl
import pytest

import glass_rules as gr
from conftest import REAL_ATTRS_PATH, REAL_CSV_PATH, requires_real_data
from data_model import HanfordDataset

# Gravimetric factors from literal atomic masses (Na 22.990, K 39.098,
# S 32.06, Cr 51.996, O 15.999), written out so the module's are checked
# against numbers it did not produce.
NA2O_PER_NA = (2 * 22.990 + 15.999) / (2 * 22.990)
K2O_PER_K = (2 * 39.098 + 15.999) / (2 * 39.098)
SO3_PER_SO4 = (32.06 + 3 * 15.999) / (32.06 + 4 * 15.999)
CR2O3_PER_CR = (2 * 51.996 + 3 * 15.999) / (2 * 51.996)


def ratios(**values) -> pd.DataFrame:
    """One row of mass ratios to Na2O; anything not named is zero."""
    row = {component: 0.0 for component in gr.RATIO_COMPONENTS}
    row.update(values)
    return pd.DataFrame([row])


def limits_of(rule_set: str, **values) -> pd.Series:
    return gr.loading_limits(ratios(**values), rule_set).iloc[0]


def loading_and_rule(rule_set: str, **values):
    limits = limits_of(rule_set, **values)
    return float(limits.min()), limits.idxmin()


# ---------------------------------------------------------------------------
# The published inequalities, as printed, for the brute-force checks
# ---------------------------------------------------------------------------
def allowed_advanced(g: dict) -> bool:
    return (g["Na2O"] + 0.66 * g["K2O"] <= 24.0
            and g["Na2O"] + 0.66 * g["K2O"] <= 33.94 - 11.69 * g["SO3"]
            and g["SO3"] <= 1.5
            and g["SO3"] <= 1.65 - 0.725 * (g["Cl"] + 0.3 * g["F"] + 0.4 * g["Cr2O3"]))


def allowed_conservative(g: dict) -> bool:
    term = g["Cl"] + 0.607 * g["F"] + 0.542 * g["Cr2O3"] + g["K2O"]
    return (g["Na2O"] + 0.66 * g["K2O"] <= 24.0
            and g["Na2O"] + 0.66 * g["K2O"] <= 33.94 - 11.69 * g["SO3"]
            and g["SO3"] <= 1.5
            and (term <= 3.746 - 4.694 * g["SO3"] or term <= 1.243 - 0.4506 * g["SO3"]))


def allowed_baseline(g: dict) -> bool:
    nh = g["Cl"] + 0.3 * g["F"]
    halogen = nh <= 1.4656 - 2.1111 * g["SO3"] if g["SO3"] <= 0.59 else nh <= 0.22
    return (g["Na2O"] <= 21.0
            and g["Na2O"] + 0.66 * g["K2O"] <= 21.5
            and g["Na2O"] + 42.5 * g["SO3"] <= 35.9
            and g["SO3"] <= 0.77
            and halogen)


ALLOWED = {gr.ADVANCED: allowed_advanced, gr.CONSERVATIVE: allowed_conservative,
           gr.BASELINE: allowed_baseline}


def brute_force_loading(rule_set: str, row: dict, step: float = 0.005) -> float:
    """The largest Na2O loading at which the printed inequalities hold,
    found by scanning the loading upward from zero."""
    allowed = ALLOWED[rule_set]
    best = 0.0
    for w in np.arange(step, 25.0, step):
        glass = {"Na2O": w, **{component: w * row[component] for component in gr.RATIO_COMPONENTS}}
        if allowed(glass):
            best = w
    return best


class TestConversions:
    def test_factors_come_from_the_atomic_masses(self):
        assert gr.COMPONENTS["Na2O"] == ("Na", pytest.approx(NA2O_PER_NA, rel=1e-12))
        assert gr.COMPONENTS["K2O"] == ("K", pytest.approx(K2O_PER_K, rel=1e-12))
        assert gr.COMPONENTS["SO3"] == ("SO4", pytest.approx(SO3_PER_SO4, rel=1e-12))
        assert gr.COMPONENTS["Cr2O3"] == ("Cr", pytest.approx(CR2O3_PER_CR, rel=1e-12))
        assert gr.COMPONENTS["Cl"] == ("Cl", 1.0)
        assert gr.COMPONENTS["F"] == ("F", 1.0)

    def test_factor_values(self):
        # 61.979 / 45.980, 94.195 / 78.196, 80.057 / 96.056, 151.989 / 103.992
        assert NA2O_PER_NA == pytest.approx(1.347956, abs=1e-6)
        assert K2O_PER_K == pytest.approx(1.204601, abs=1e-6)
        assert SO3_PER_SO4 == pytest.approx(0.833441, abs=1e-6)
        assert CR2O3_PER_CR == pytest.approx(1.461545, abs=1e-6)

    def test_oxide_masses_and_ratios(self):
        kg = pd.DataFrame({"Na": [1000.0], "K": [100.0], "SO4": [50.0], "Cl": [20.0], "F": [10.0], "Cr": [5.0]})
        masses = gr.oxide_masses(kg).iloc[0]
        assert masses["Na2O"] == pytest.approx(1000.0 * NA2O_PER_NA)
        assert masses["SO3"] == pytest.approx(50.0 * SO3_PER_SO4)
        assert masses["Cl"] == 20.0
        r = gr.oxide_ratios(kg).iloc[0]
        assert r["K2O"] == pytest.approx(100.0 * K2O_PER_K / (1000.0 * NA2O_PER_NA))
        assert r["Cr2O3"] == pytest.approx(5.0 * CR2O3_PER_CR / (1000.0 * NA2O_PER_NA))
        assert r["F"] == pytest.approx(10.0 / (1000.0 * NA2O_PER_NA))

    def test_a_species_not_reported_counts_as_zero_and_no_sodium_is_blank(self):
        kg = pd.DataFrame({"Na": [100.0, 0.0, np.nan], "K": [np.nan, 5.0, 5.0], "SO4": [1.0, 1.0, 1.0],
                           "Cl": [np.nan] * 3, "F": [0.0] * 3, "Cr": [0.0] * 3})
        r = gr.oxide_ratios(kg)
        assert r.loc[0, "K2O"] == 0.0 and r.loc[0, "Cl"] == 0.0
        assert r.loc[1].isna().all() and r.loc[2].isna().all()


class TestRuleSets:
    def test_the_three_sets_and_their_rules(self):
        assert list(gr.RULE_SETS) == ["Advanced LAW (2013)", "Advanced LAW, conservative halide", "WTP baseline"]
        assert gr.RULE_SETS[gr.ADVANCED].rule_names == ["Alkali", "Alkali + sulfur", "Sulfur", "Halide"]
        assert gr.RULE_SETS[gr.CONSERVATIVE].rule_names == ["Alkali", "Alkali + sulfur", "Sulfur", "Halide"]
        assert gr.RULE_SETS[gr.BASELINE].rule_names == ["Sodium", "Alkali", "Alkali + sulfur", "Sulfur", "Halide"]

    def test_every_rule_belongs_to_one_of_the_four_classes_and_cites_its_source(self):
        for rule_set in gr.RULE_SETS.values():
            assert "PNNL-22631" in rule_set.citation
            for rule in rule_set.rules:
                assert rule.rule_class in gr.RULE_CLASSES
                assert rule.equation and rule.source

    def test_halide_terms(self):
        assert gr.RULE_SETS[gr.ADVANCED].halide_label == "Cl + 0.3 F + 0.4 Cr2O3"
        assert gr.RULE_SETS[gr.CONSERVATIVE].halide_label == "Cl + 0.607 F + 0.542 Cr2O3 + K2O"
        assert gr.RULE_SETS[gr.BASELINE].halide_label == "Cl + 0.3 F"
        r = ratios(Cl=0.02, F=0.1, Cr2O3=0.05, K2O=0.01)
        assert gr.RULE_SETS[gr.ADVANCED].halide_term(r).iloc[0] == pytest.approx(0.02 + 0.03 + 0.02)
        assert gr.RULE_SETS[gr.CONSERVATIVE].halide_term(r).iloc[0] == pytest.approx(0.02 + 0.0607 + 0.0271 + 0.01)
        assert gr.RULE_SETS[gr.BASELINE].halide_term(r).iloc[0] == pytest.approx(0.02 + 0.03)

    def test_unknown_rule_set_is_refused(self):
        with pytest.raises(ValueError, match="Unknown rule set"):
            gr.get_rule_set("nonsense")
        assert gr.get_rule_set(None).name == gr.ADVANCED


class TestAdvancedRules:
    def test_no_anions_no_potassium_is_the_alkali_limit(self):
        w, rule = loading_and_rule(gr.ADVANCED)
        assert w == pytest.approx(24.0) and rule == "Alkali"

    def test_potassium_lowers_the_alkali_limit(self):
        # Na2O + 0.66 K2O = w (1 + 0.066) <= 24
        assert limits_of(gr.ADVANCED, K2O=0.1)["Alkali"] == pytest.approx(24.0 / 1.066)

    def test_each_rule_solved_by_hand(self):
        limits = limits_of(gr.ADVANCED, K2O=0.1, SO3=0.05, Cl=0.02, F=0.1, Cr2O3=0.05)
        assert limits["Alkali + sulfur"] == pytest.approx(33.94 / (1.066 + 11.69 * 0.05))
        assert limits["Sulfur"] == pytest.approx(1.5 / 0.05)
        assert limits["Halide"] == pytest.approx(1.65 / (0.05 + 0.725 * 0.07))

    def test_alkali_gives_way_to_alkali_plus_sulfur_at_the_crossover(self):
        crossover = (33.94 / 24.0 - 1.0) / 11.69          # 0.03543 with no potassium
        assert crossover == pytest.approx(0.035429, abs=1e-6)
        assert loading_and_rule(gr.ADVANCED, SO3=crossover * 0.99)[1] == "Alkali"
        w, rule = loading_and_rule(gr.ADVANCED, SO3=crossover * 1.01)
        assert rule == "Alkali + sulfur" and w < 24.0
        at = limits_of(gr.ADVANCED, SO3=crossover)
        assert at["Alkali"] == pytest.approx(at["Alkali + sulfur"])

    def test_the_sulfur_cap_takes_over_at_higher_sulfate(self):
        # 33.94 / (1 + 11.69 s) = 1.5 / s  ->  s = 1.5 / (33.94 - 17.535)
        takeover = 1.5 / (33.94 - 11.69 * 1.5)
        assert takeover == pytest.approx(0.091436, abs=1e-6)
        assert loading_and_rule(gr.ADVANCED, SO3=takeover * 0.99)[1] == "Alkali + sulfur"
        w, rule = loading_and_rule(gr.ADVANCED, SO3=0.2)
        assert rule == "Sulfur" and w == pytest.approx(7.5)

    def test_a_halide_bound_waste(self):
        # s = 0.05, halide term = 0.05: halide 1.65 / 0.08625 = 19.13 is below
        # alkali + sulfur (21.42), sulfur (30) and alkali (24).
        w, rule = loading_and_rule(gr.ADVANCED, SO3=0.05, Cl=0.05)
        assert rule == "Halide" and w == pytest.approx(1.65 / (0.05 + 0.725 * 0.05))
        assert w == pytest.approx(19.130, abs=1e-3)

    def test_no_sulfate_means_no_sulfur_limit(self):
        assert np.isinf(limits_of(gr.ADVANCED)["Sulfur"])
        assert np.isinf(limits_of(gr.ADVANCED)["Halide"])


class TestConservativeEnvelope:
    def test_low_sulfur_side_uses_the_steep_line(self):
        # h = 0.2, s = 0.01: 3.746 / 0.24694 = 15.17 (steep) vs 1.243 / 0.204506 = 6.08
        w, rule = loading_and_rule(gr.CONSERVATIVE, SO3=0.01, Cl=0.2)
        assert rule == "Halide" and w == pytest.approx(3.746 / (0.2 + 4.694 * 0.01))

    def test_high_sulfur_side_uses_the_shallow_line(self):
        # h = 0.1, s = 0.1: 3.746 / 0.5694 = 6.58 vs 1.243 / 0.14506 = 8.57 (shallow)
        w, rule = loading_and_rule(gr.CONSERVATIVE, SO3=0.1, Cl=0.1)
        assert rule == "Halide" and w == pytest.approx(1.243 / (0.1 + 0.4506 * 0.1))

    def test_potassium_counts_in_the_halide_term(self):
        with_k = limits_of(gr.CONSERVATIVE, SO3=0.1, Cl=0.05, K2O=0.05)["Halide"]
        assert with_k == pytest.approx(limits_of(gr.CONSERVATIVE, SO3=0.1, Cl=0.1)["Halide"])

    def test_the_other_three_rules_are_the_advanced_ones(self):
        kwargs = dict(K2O=0.03, SO3=0.07, Cl=0.01, F=0.02, Cr2O3=0.004)
        adv, cons = limits_of(gr.ADVANCED, **kwargs), limits_of(gr.CONSERVATIVE, **kwargs)
        for rule in ("Alkali", "Alkali + sulfur", "Sulfur"):
            assert cons[rule] == pytest.approx(adv[rule])

    def test_the_two_lines_meet_where_the_figure_shows_the_kink(self):
        so3 = (3.746 - 1.243) / (4.694 - 0.4506)
        assert so3 == pytest.approx(0.59, abs=1e-3)
        assert 3.746 - 4.694 * so3 == pytest.approx(0.977, abs=1e-3)


class TestBaselineRules:
    def test_no_anions_is_the_sodium_limit(self):
        w, rule = loading_and_rule(gr.BASELINE)
        assert w == pytest.approx(21.0) and rule == "Sodium"

    def test_potassium_brings_in_the_alkali_rule(self):
        crossover = (21.5 / 21.0 - 1.0) / 0.66
        assert loading_and_rule(gr.BASELINE, K2O=crossover * 0.99)[1] == "Sodium"
        w, rule = loading_and_rule(gr.BASELINE, K2O=0.1)
        assert rule == "Alkali" and w == pytest.approx(21.5 / 1.066)

    def test_alkali_plus_sulfur_then_the_sulfur_cap(self):
        w, rule = loading_and_rule(gr.BASELINE, SO3=0.05)
        assert rule == "Alkali + sulfur" and w == pytest.approx(35.9 / (1 + 42.5 * 0.05))
        # the cap wins once 35.9 s / (1 + 42.5 s) > 0.77, i.e. s > 0.2425
        assert loading_and_rule(gr.BASELINE, SO3=0.24)[1] == "Alkali + sulfur"
        w, rule = loading_and_rule(gr.BASELINE, SO3=0.3)
        assert rule == "Sulfur" and w == pytest.approx(0.77 / 0.3)

    def test_halogen_rule_low_sulfur_branch(self):
        # s = 0.01, NH ratio 0.1: w = 1.4656 / (0.1 + 0.021111); SO3 there is 0.12 wt%
        limit = limits_of(gr.BASELINE, SO3=0.01, Cl=0.1)["Halide"]
        assert limit == pytest.approx(1.4656 / (0.1 + 2.1111 * 0.01))
        assert limit * 0.01 <= 0.59

    def test_halogen_rule_high_sulfur_branch(self):
        # s = 0.1, NH ratio 0.01: the flat 0.22 wt% cap, reached at SO3 = 2.2 wt%
        limit = limits_of(gr.BASELINE, SO3=0.1, Cl=0.01)["Halide"]
        assert limit == pytest.approx(0.22 / 0.01)
        assert limit * 0.1 > 0.59

    def test_halogen_rule_at_the_branch_boundary(self):
        # The branches meet where SO3 = 0.59 wt% and NH = 0.22 wt% at once:
        # NH ratio = 0.22 s / 0.59. On either side of it the limit is the
        # loading that puts SO3 at 0.59 wt%, and it does not jump.
        s = 0.1
        h_boundary = 0.22 * s / 0.59
        just_low = limits_of(gr.BASELINE, SO3=s, Cl=h_boundary * 0.999)["Halide"]
        just_high = limits_of(gr.BASELINE, SO3=s, Cl=h_boundary * 1.001)["Halide"]
        assert just_low == pytest.approx(0.59 / s, rel=2e-3)
        assert just_high == pytest.approx(0.59 / s, rel=2e-3)
        assert just_high <= just_low
        # the two printed branches agree at SO3 = 0.59 to the rounding of the constants
        assert 1.4656 - 2.1111 * 0.59 == pytest.approx(0.22, abs=1e-4)

    def test_fluorine_weighs_three_tenths_and_chromium_is_not_counted(self):
        base = limits_of(gr.BASELINE, SO3=0.01, Cl=0.05, F=0.1)["Halide"]
        assert base == pytest.approx(1.4656 / (0.08 + 2.1111 * 0.01))
        assert limits_of(gr.BASELINE, SO3=0.01, Cl=0.05, F=0.1, Cr2O3=0.2)["Halide"] == pytest.approx(base)

    def test_no_halogen_means_no_halogen_limit(self):
        assert np.isinf(limits_of(gr.BASELINE, SO3=0.3)["Halide"])
        assert np.isinf(limits_of(gr.BASELINE)["Halide"])


class TestAgainstThePrintedInequalities:
    """The closed-form loading must be the largest loading at which the
    inequalities, evaluated as printed on the glass composition, hold."""

    CASES = [
        dict(K2O=0.0, SO3=0.0, Cl=0.0, F=0.0, Cr2O3=0.0),
        dict(K2O=0.02, SO3=0.03, Cl=0.01, F=0.01, Cr2O3=0.005),
        dict(K2O=0.2, SO3=0.01, Cl=0.002, F=0.0, Cr2O3=0.0),
        dict(K2O=0.01, SO3=0.12, Cl=0.009, F=0.1, Cr2O3=0.002),
        dict(K2O=0.0, SO3=0.45, Cl=0.004, F=0.13, Cr2O3=0.001),
        dict(K2O=0.05, SO3=0.02, Cl=0.15, F=0.05, Cr2O3=0.03),
        dict(K2O=0.01, SO3=0.1, Cl=0.03, F=0.02, Cr2O3=0.0),
        dict(K2O=0.0, SO3=0.06, Cl=0.0224, F=0.0, Cr2O3=0.0),     # on the baseline branch boundary
        dict(K2O=0.3, SO3=0.2, Cl=0.3, F=0.3, Cr2O3=0.3),
    ]

    @pytest.mark.parametrize("rule_set", list(gr.RULE_SETS))
    @pytest.mark.parametrize("case", CASES)
    def test_closed_form_equals_the_scan(self, rule_set, case):
        step = 0.005
        expected = brute_force_loading(rule_set, case, step)
        computed = float(gr.loading_limits(pd.DataFrame([case]), rule_set).iloc[0].min())
        assert computed == pytest.approx(expected, abs=2 * step)


class TestMonotonicity:
    @pytest.mark.parametrize("rule_set", list(gr.RULE_SETS))
    def test_more_sulfate_never_raises_the_loading(self, rule_set):
        rng = np.random.default_rng(11)
        n = 400
        frame = pd.DataFrame({
            "K2O": rng.uniform(0, 0.3, n), "Cl": rng.uniform(0, 0.2, n),
            "F": rng.uniform(0, 0.2, n), "Cr2O3": rng.uniform(0, 0.1, n),
        })
        previous = np.full(n, np.inf)
        for so3 in np.geomspace(1e-4, 1.0, 120):
            frame["SO3"] = so3
            loading = gr.loading_limits(frame, rule_set).min(axis=1).to_numpy()
            assert (loading <= previous + 1e-12).all()
            previous = loading

    @pytest.mark.parametrize("rule_set", list(gr.RULE_SETS))
    @pytest.mark.parametrize("component", ["K2O", "Cl", "F", "Cr2O3"])
    def test_no_component_ever_raises_the_loading(self, rule_set, component):
        rng = np.random.default_rng(5)
        n = 300
        frame = pd.DataFrame({name: rng.uniform(0, 0.2, n) for name in gr.RATIO_COMPONENTS})
        previous = np.full(n, np.inf)
        for value in np.geomspace(1e-4, 1.0, 80):
            frame[component] = value
            loading = gr.loading_limits(frame, rule_set).min(axis=1).to_numpy()
            assert (loading <= previous + 1e-12).all()
            previous = loading


def _random_units(seed: int, n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    na = rng.uniform(100, 5000, n)
    return pd.DataFrame({
        "Na": na, "K": na * rng.uniform(0, 0.2, n), "SO4": na * rng.uniform(0, 0.5, n),
        "Cl": na * rng.uniform(0, 0.1, n), "F": na * rng.uniform(0, 0.2, n), "Cr": na * rng.uniform(0, 0.05, n),
    })


class TestBlending:
    """What holds and what does not when units are blended into one feed.

    Found by working it through: with the Advanced rules, glass per rule
    is linear in the component masses, so glass (the largest of them) is
    subadditive -- a blend never needs more glass than its parts. That is
    a statement about glass mass, i.e. about the sodium-weighted HARMONIC
    mean of the loadings. The blended loading is NOT bounded below by the
    sodium-weighted arithmetic mean, and with the piecewise rules of the
    other two sets even the glass statement fails.
    """

    @pytest.mark.parametrize("seed", range(8))
    def test_advanced_blend_never_needs_more_glass_than_its_parts(self, seed):
        units = _random_units(seed)
        parts = gr.screen_frame(units, gr.ADVANCED)
        blend = gr.screen_frame(pd.DataFrame([units.sum()]), gr.ADVANCED).iloc[0]
        assert blend["Glass_t"] <= parts["Glass_t"].sum() * (1 + 1e-12)
        harmonic = parts["Na2O_t"].sum() / (parts["Na2O_t"] / parts["Na2O_loading_wtpct"]).sum()
        assert blend["Na2O_loading_wtpct"] >= harmonic * (1 - 1e-12)

    def test_blended_loading_can_fall_below_the_arithmetic_mean(self):
        # Equal sodium: one unit alkali-bound at 24 wt%, one sulfur-bound at 4 wt%.
        na = 1000.0
        so4 = (1.5 / 4.0) * na * NA2O_PER_NA / SO3_PER_SO4       # SO3 / Na2O = 0.375
        units = pd.DataFrame({"Na": [na, na], "K": [0.0, 0.0], "SO4": [0.0, so4],
                              "Cl": [0.0, 0.0], "F": [0.0, 0.0], "Cr": [0.0, 0.0]})
        parts = gr.screen_frame(units, gr.ADVANCED)
        assert parts["Na2O_loading_wtpct"].to_list() == pytest.approx([24.0, 4.0])
        blend = gr.screen_frame(pd.DataFrame([units.sum()]), gr.ADVANCED).iloc[0]
        # blended SO3 / Na2O = 0.1875 -> the sulfur cap, 1.5 / 0.1875 = 8 wt%
        assert blend["Na2O_loading_wtpct"] == pytest.approx(8.0)
        assert blend["Binding_rule"] == "Sulfur"
        assert blend["Na2O_loading_wtpct"] < parts["Na2O_loading_wtpct"].mean()      # 8 < 14
        assert blend["Glass_t"] <= parts["Glass_t"].sum()                              # 25 t < 29.2 t per t Na2O

    def test_conservative_envelope_blend_can_need_more_glass_than_its_parts(self):
        # A chloride-rich unit (steep line) and a sulfate-rich unit (sulfur
        # cap) each do better alone than their blend, which lands on the
        # shallow line of the envelope.
        na2o = 100.0 / NA2O_PER_NA
        units = pd.DataFrame({"Na": [na2o, na2o], "K": [0.0, 0.0], "SO4": [0.0, 100.0 / SO3_PER_SO4],
                              "Cl": [100.0, 0.0], "F": [0.0, 0.0], "Cr": [0.0, 0.0]})
        parts = gr.screen_frame(units, gr.CONSERVATIVE)
        # unit 1: h = 1 -> 3.746 wt%; unit 2: s = 1 -> sulfur cap 1.5 wt%
        assert parts["Na2O_loading_wtpct"].to_list() == pytest.approx([3.746, 1.5])
        blend = gr.screen_frame(pd.DataFrame([units.sum()]), gr.CONSERVATIVE).iloc[0]
        # blend: h = s = 0.5 -> max(3.746 / 2.847, 1.243 / 0.7253) = 1.7138 wt%
        assert blend["Na2O_loading_wtpct"] == pytest.approx(1.243 / (0.5 + 0.4506 * 0.5))
        assert blend["Glass_t"] > parts["Glass_t"].sum()


class TestScreenFrame:
    def _kg(self):
        return pd.DataFrame({
            "Na": [1000.0, 1000.0, 0.0, 500.0],
            "K": [0.0, 20.0, 5.0, np.nan],
            "SO4": [0.0, 300.0, 10.0, 20.0],
            "Cl": [0.0, 5.0, 1.0, np.nan],
            "F": [0.0, 10.0, 1.0, 2.0],
            "Cr": [0.0, 2.0, 1.0, 1.0],
        }, index=["clean", "sulfate", "no sodium", "blanks"])

    def test_columns(self):
        out = gr.screen_frame(self._kg())
        assert list(out.columns) == [
            "Na2O_t", "K2O_per_Na2O", "SO3_per_Na2O", "Cl_per_Na2O", "F_per_Na2O", "Cr2O3_per_Na2O",
            "Halide_term_per_Na2O", "Limit_Alkali", "Limit_Alkali + sulfur", "Limit_Sulfur", "Limit_Halide",
            "Na2O_loading_wtpct", "Binding_rule", "Glass_t", "Glass_t_alkali_only", "Extra_glass_t",
            "SO3_in_glass_wtpct", "Cl_in_glass_wtpct", "F_in_glass_wtpct", "Cr2O3_in_glass_wtpct",
        ]
        assert list(out.index) == ["clean", "sulfate", "no sodium", "blanks"]

    def test_clean_unit_loads_to_the_alkali_limit_with_no_extra_glass(self):
        row = gr.screen_frame(self._kg()).loc["clean"]
        assert row["Na2O_t"] == pytest.approx(NA2O_PER_NA)
        assert row["Na2O_loading_wtpct"] == pytest.approx(24.0)
        assert row["Binding_rule"] == "Alkali"
        assert row["Glass_t"] == pytest.approx(NA2O_PER_NA / 0.24)
        assert row["Extra_glass_t"] == pytest.approx(0.0, abs=1e-12)

    def test_sulfate_unit_by_hand(self):
        row = gr.screen_frame(self._kg()).loc["sulfate"]
        na2o = 1000.0 * NA2O_PER_NA
        s = 300.0 * SO3_PER_SO4 / na2o
        k = 20.0 * K2O_PER_K / na2o
        h = (5.0 + 0.3 * 10.0 + 0.4 * 2.0 * CR2O3_PER_CR) / na2o
        expected = {
            "Alkali": 24.0 / (1 + 0.66 * k), "Alkali + sulfur": 33.94 / (1 + 0.66 * k + 11.69 * s),
            "Sulfur": 1.5 / s, "Halide": 1.65 / (s + 0.725 * h),
        }
        for rule, value in expected.items():
            assert row[f"Limit_{rule}"] == pytest.approx(value)
        binding = min(expected, key=expected.get)
        assert binding == "Sulfur"
        assert row["Binding_rule"] == binding
        assert row["Na2O_loading_wtpct"] == pytest.approx(expected[binding])
        assert row["Glass_t"] == pytest.approx(na2o / 1000.0 / (expected[binding] / 100.0))
        assert row["Glass_t_alkali_only"] == pytest.approx(na2o / 1000.0 / (expected["Alkali"] / 100.0))
        assert row["Extra_glass_t"] == pytest.approx(row["Glass_t"] - row["Glass_t_alkali_only"])
        # the glass sits exactly on the rule that binds it
        assert row["SO3_in_glass_wtpct"] == pytest.approx(1.5)
        assert row["Cl_in_glass_wtpct"] == pytest.approx(expected[binding] * 5.0 / na2o)

    def test_a_unit_without_sodium_has_no_loading_not_a_zero(self):
        row = gr.screen_frame(self._kg()).loc["no sodium"]
        assert row["Na2O_t"] == 0.0
        assert pd.isna(row["Na2O_loading_wtpct"])
        assert row["Binding_rule"] is None
        assert pd.isna(row["Glass_t"]) and pd.isna(row["Extra_glass_t"])
        assert pd.isna(row["Limit_Alkali"]) and pd.isna(row["SO3_per_Na2O"])

    def test_blank_species_count_as_zero(self):
        row = gr.screen_frame(self._kg()).loc["blanks"]
        assert row["K2O_per_Na2O"] == 0.0 and row["Cl_per_Na2O"] == 0.0
        assert row["Limit_Alkali"] == pytest.approx(24.0)

    def test_baseline_alkali_only_glass_uses_the_tighter_of_its_two_alkali_rules(self):
        kg = pd.DataFrame({"Na": [1000.0, 1000.0], "K": [0.0, 200.0], "SO4": [0.0, 0.0],
                           "Cl": [0.0, 0.0], "F": [0.0, 0.0], "Cr": [0.0, 0.0]})
        out = gr.screen_frame(kg, gr.BASELINE)
        na2o_t = NA2O_PER_NA
        k = 200.0 * K2O_PER_K / (1000.0 * NA2O_PER_NA)
        assert out["Binding_rule"].to_list() == ["Sodium", "Alkali"]
        assert out["Glass_t_alkali_only"].to_list() == pytest.approx(
            [na2o_t / 0.21, na2o_t / (0.215 / (1 + 0.66 * k))])
        assert out["Extra_glass_t"].to_list() == pytest.approx([0.0, 0.0], abs=1e-12)

    def test_rule_shares_are_weighted_by_sodium(self):
        table = pd.DataFrame({"Na2O_t": [30.0, 10.0, 60.0, 5.0],
                              "Binding_rule": ["Alkali", "Sulfur", "Alkali", None]})
        shares = gr.rule_shares(table, gr.ADVANCED)
        assert shares.to_dict() == pytest.approx({"Alkali": 90.0, "Alkali + sulfur": 0.0, "Sulfur": 10.0,
                                                  "Halide": 0.0})
        assert gr.rule_shares(table.iloc[0:0], gr.ADVANCED).isna().all()


@pytest.fixture
def salt_dataset() -> HanfordDataset:
    """Three tanks in two farms:

    241-A-101  supernatant (clean) over saltcake (sulfate-rich) over sludge
    241-A-102  sludge only (no salt waste)
    241-B-101  one saltcake layer, chloride-rich, that does not report potassium
    """
    rows = {"WasteSiteId": [], "Analyte": [], "WastePhase": [], "WasteType": [], "Inventory": [], "Units": []}

    def add(tank, phase, waste_type, **kg):
        for analyte, value in kg.items():
            rows["WasteSiteId"].append(tank)
            rows["Analyte"].append(analyte)
            rows["WastePhase"].append(phase)
            rows["WasteType"].append(waste_type)
            rows["Inventory"].append(float(value))
            rows["Units"].append("kg")

    add("241-A-101", "Supernatant", "NA (Liquid)", Na=2000, K=40, SO4=20, Cl=10, F=2, Cr=1)
    add("241-A-101", "Saltcake Solid", "A1-SltCk (Solid)", Na=1000, K=5, SO4=400, Cl=2, F=30, Cr=3)
    add("241-A-101", "Sludge Solid", "R1 (Solid)", Na=300, K=1, SO4=5, Cl=1, F=1, Cr=40)
    add("241-A-102", "Sludge Solid", "R1 (Solid)", Na=500, K=2, SO4=8, Cl=1, F=2, Cr=60)
    add("241-B-101", "Saltcake Solid", "B-SltCk (Solid)", Na=800, SO4=30, Cl=120, F=5, Cr=2)
    dataset = HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame(rows))
    dataset.report = None
    return dataset


def _hand_loading(na, k, so4, cl, f, cr):
    na2o = na * NA2O_PER_NA
    alkali = 1 + 0.66 * k * K2O_PER_K / na2o
    s = so4 * SO3_PER_SO4 / na2o
    h = (cl + 0.3 * f + 0.4 * cr * CR2O3_PER_CR) / na2o
    limits = {"Alkali": 24.0 / alkali, "Alkali + sulfur": 33.94 / (alkali + 11.69 * s),
              "Sulfur": 1.5 / s, "Halide": 1.65 / (s + 0.725 * h)}
    rule = min(limits, key=limits.get)
    return limits[rule], rule, na2o / 1000.0


class TestScreen:
    def test_tank_level_sums_the_salt_layers_and_leaves_the_sludge_out(self, salt_dataset):
        result = gr.screen(salt_dataset)
        table = result.table.set_index("WasteSiteId")
        assert sorted(table.index) == ["241-A-101", "241-B-101"]          # the sludge-only tank is not salt waste
        loading, rule, na2o_t = _hand_loading(3000, 45, 420, 12, 32, 4)
        row = table.loc["241-A-101"]
        assert row["Na2O_t"] == pytest.approx(na2o_t)
        assert row["Na2O_loading_wtpct"] == pytest.approx(loading)
        assert row["Binding_rule"] == rule
        assert row["N_layers"] == 2
        loading_b, rule_b, _ = _hand_loading(800, 0, 30, 120, 5, 2)
        assert table.loc["241-B-101", "Na2O_loading_wtpct"] == pytest.approx(loading_b)
        assert table.loc["241-B-101", "Binding_rule"] == rule_b == "Halide"
        assert result.level == "tank" and result.phases == ("Supernatant", "Saltcake")
        assert result.waste_label == "supernatant + saltcake"

    def test_all_phases_includes_the_sludge(self, salt_dataset):
        result = gr.screen(salt_dataset, phases=None)
        table = result.table.set_index("WasteSiteId")
        assert sorted(table.index) == ["241-A-101", "241-A-102", "241-B-101"]
        loading, rule, na2o_t = _hand_loading(3300, 46, 425, 13, 33, 44)
        assert table.loc["241-A-101", "Na2O_loading_wtpct"] == pytest.approx(loading)
        assert table.loc["241-A-101", "Na2O_t"] == pytest.approx(na2o_t)
        assert result.waste_label == "all phases"

    def test_one_phase_family(self, salt_dataset):
        result = gr.screen(salt_dataset, phases=("Saltcake",))
        table = result.table.set_index("WasteSiteId")
        loading, rule, _ = _hand_loading(1000, 5, 400, 2, 30, 3)
        assert table.loc["241-A-101", "Na2O_loading_wtpct"] == pytest.approx(loading)
        assert table.loc["241-A-101", "Binding_rule"] == rule

    def test_layer_level_keeps_one_row_per_layer(self, salt_dataset):
        result = gr.screen(salt_dataset, level="layer")
        assert len(result.table) == 3
        assert set(result.table["PhaseFamily"]) == {"Supernatant", "Saltcake"}
        layer = result.table[result.table["WastePhase"] == "Supernatant"].iloc[0]
        loading, rule, _ = _hand_loading(2000, 40, 20, 10, 2, 1)
        assert layer["Na2O_loading_wtpct"] == pytest.approx(loading)
        assert layer["Unit"] == "241-A-101 | Supernatant | NA (Liquid)"
        assert result.table["Unit"].is_unique

    def test_summary_totals_and_the_blended_feed(self, salt_dataset):
        result = gr.screen(salt_dataset)
        table, summary = result.table, result.summary
        assert summary.n_units == 2 and summary.n_with_sodium == 2
        assert summary.na2o_kt == pytest.approx(table["Na2O_t"].sum() / 1000.0)
        assert summary.glass_unit_by_unit_kt == pytest.approx(table["Glass_t"].sum() / 1000.0)
        assert summary.glass_alkali_only_kt == pytest.approx(table["Glass_t_alkali_only"].sum() / 1000.0)
        assert summary.extra_glass_kt == pytest.approx(table["Extra_glass_t"].sum() / 1000.0)
        loading, rule, na2o_t = _hand_loading(3800, 45, 450, 132, 37, 6)
        assert summary.blend_loading_wtpct == pytest.approx(loading)
        assert summary.blend_binding_rule == rule
        assert summary.glass_blended_kt == pytest.approx(na2o_t / (loading / 100.0) / 1000.0)
        assert sum(summary.share_by_rule.values()) == pytest.approx(100.0)
        alkali_share = summary.share_by_rule["Alkali"]
        assert summary.not_alkali_limited_pct == pytest.approx(100.0 - alkali_share)
        assert summary.n_below_low_loading == int((table["Na2O_loading_wtpct"] < 15.0).sum())
        shown = summary.as_table("tank")
        assert list(shown.columns) == ["Quantity", "Value", "Unit"]
        assert "Tanks screened" in set(shown["Quantity"])
        assert shown["Value"].dtype == float                   # numbers only: the column sorts and aligns
        by_name = shown.set_index("Quantity")["Value"]
        assert by_name["Tanks screened"] == 2
        assert by_name[f"Loading of the blended feed ({rule.lower()} rule)"] == pytest.approx(loading)
        assert by_name["Glass, tank by tank"] == pytest.approx(summary.glass_unit_by_unit_kt)

    def test_groups_split_the_sodium_by_binding_rule(self, salt_dataset):
        result = gr.screen(salt_dataset)
        groups = result.groups.set_index("Group")
        assert list(groups.index) == ["All tanks", "Saltcake layers", "Supernatant layers"]   # no SST/DST without attributes
        for name in groups.index:
            row = groups.loc[name]
            assert sum(row[f"{rule}_pct"] for rule in result.rule_set.rule_names) == pytest.approx(100.0)
        assert groups.loc["All tanks", "Na2O_kt"] == pytest.approx(result.summary.na2o_kt)
        layer_sodium = groups.loc["Saltcake layers", "Na2O_kt"] + groups.loc["Supernatant layers", "Na2O_kt"]
        assert layer_sodium == pytest.approx(result.summary.na2o_kt)

    def test_scope_reaches_the_engine(self, salt_dataset):
        result = gr.screen(salt_dataset, scope={"TankFarm": ["B"]})
        assert result.table["WasteSiteId"].to_list() == ["241-B-101"]
        assert result.scope == {"TankFarm": ["B"]}

    def test_nothing_in_scope_is_an_empty_result_with_a_note(self, salt_dataset):
        result = gr.screen(salt_dataset, scope={"WastePhase": ["Sludge Solid"]})
        assert result.table.empty and result.groups.empty
        assert result.summary.n_units == 0 and np.isnan(result.summary.na2o_kt)
        assert any("No tank matches" in note for note in result.notes)
        assert "Na2O_loading_wtpct" in result.table.columns

    def test_notes_state_the_scope_of_validity_and_the_blanks(self, salt_dataset):
        result = gr.screen(salt_dataset)
        text = " ".join(result.notes)
        assert "screening index" in text and "not a forecast" in text
        assert "do not report K" in text                 # 241-B-101
        # what is particular to this result is also available on its own
        assert result.remarks == [note for note in result.notes if "do not report" in note]
        assert set(result.remarks) <= set(result.notes)
        assert "SO3 <= 1.5" in text
        assert any("Na:" in d for d in result.disclosures)

    def test_a_layer_without_sodium_is_counted_and_left_blank(self, salt_dataset):
        df = salt_dataset.df
        extra = pl.DataFrame({
            "WasteSiteId": ["241-B-102"], "Analyte": ["SO4"], "WastePhase": ["Saltcake Solid"],
            "WasteType": ["B-SltCk (Solid)"], "Inventory": [12.0], "Units": ["kg"],
        })
        dataset = HanfordDataset()
        raw = pl.concat([df.select(extra.columns), extra])
        dataset.df = dataset._clean_dataframe(raw)
        result = gr.screen(dataset)
        row = result.table.set_index("WasteSiteId").loc["241-B-102"]
        assert pd.isna(row["Na2O_loading_wtpct"]) and row["Binding_rule"] is None
        assert result.summary.n_units == 3 and result.summary.n_with_sodium == 2
        assert any("hold no sodium" in note for note in result.notes)
        assert len(result.screened()) == 2

    def test_unknown_level_is_refused(self, salt_dataset):
        with pytest.raises(ValueError, match="Unknown level"):
            gr.screen(salt_dataset, level="farm")

    def test_other_rule_sets_run_and_name_their_rules(self, salt_dataset):
        for name, rule_set in gr.RULE_SETS.items():
            result = gr.screen(salt_dataset, rule_set=name)
            assert result.rule_set is rule_set
            assert set(result.table["Binding_rule"]) <= set(rule_set.rule_names)
            for rule in rule_set.rule_names:
                assert f"Limit_{rule}" in result.table.columns


class TestRegionGrid:
    def test_shape_and_axes(self):
        grid = gr.region_grid(gr.ADVANCED, nx=50, ny=40)
        assert grid.loading.shape == (40, 50) and grid.rule_index.shape == (40, 50)
        assert grid.x[0] == pytest.approx(gr.DIAGRAM_X_RANGE[0]) and grid.x[-1] == pytest.approx(gr.DIAGRAM_X_RANGE[1])
        assert grid.y[0] == pytest.approx(gr.DIAGRAM_Y_RANGE[0]) and grid.y[-1] == pytest.approx(gr.DIAGRAM_Y_RANGE[1])
        assert grid.rule_names == ("Alkali", "Alkali + sulfur", "Sulfur", "Halide")

    @pytest.mark.parametrize("rule_set", list(gr.RULE_SETS))
    def test_grid_agrees_with_the_unit_calculation(self, rule_set):
        k = 0.02
        grid = gr.region_grid(rule_set, k2o_ratio=k, nx=30, ny=24)
        rules = gr.RULE_SETS[rule_set]
        for iy in (0, 7, 15, 23):
            for ix in (0, 9, 18, 29):
                limits = rules.limits(k, grid.x[ix], grid.y[iy])
                values = {name: float(limits[name]) for name in rules.rule_names}
                assert grid.loading[iy, ix] == pytest.approx(min(values.values()))
                binding = grid.rule_names[grid.rule_index[iy, ix]]
                assert values[binding] == pytest.approx(min(values.values()))
                assert gr.RULE_CLASSES[grid.class_index[iy, ix]] == rules.rule_classes[binding]

    def test_the_four_regions_of_the_advanced_set_are_where_the_equations_put_them(self):
        grid = gr.region_grid(gr.ADVANCED, k2o_ratio=0.0, nx=200, ny=160)

        def rule_at(x, y):
            ix, iy = np.abs(grid.x - x).argmin(), np.abs(grid.y - y).argmin()
            return grid.rule_names[grid.rule_index[iy, ix]]

        assert rule_at(0.005, 0.005) == "Alkali"
        assert rule_at(0.06, 0.005) == "Alkali + sulfur"
        assert rule_at(0.4, 0.005) == "Sulfur"
        assert rule_at(0.2, 0.09) == "Halide"

    def test_baseline_sodium_and_alkali_share_the_alkali_class(self):
        grid = gr.region_grid(gr.BASELINE, k2o_ratio=0.1, nx=40, ny=30)
        low = grid.rule_names[grid.rule_index[0, 0]]
        assert low == "Alkali" and gr.RULE_CLASSES[grid.class_index[0, 0]] == "Alkali"


# ---------------------------------------------------------------------------
# The shipped dataset
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_dataset():
    dataset = HanfordDataset()
    dataset.load(REAL_CSV_PATH, use_cache=False, attributes_path=REAL_ATTRS_PATH)
    return dataset


@pytest.fixture(scope="module")
def real_screen(real_dataset):
    return gr.screen(real_dataset)


def _raw_recomputation(dataset: HanfordDataset) -> pd.DataFrame:
    """The advanced screen of each tank's supernatant + saltcake from the
    raw records: plain pandas, literal factors, formulas written inline."""
    raw = dataset.require_df().select(["WasteSiteId", "Analyte", "WastePhase", "Inventory", "Units"]).to_pandas()
    raw = raw[raw["Analyte"].isin(["Na", "K", "SO4", "Cl", "F", "Cr"]) & (raw["Units"] == "kg")]
    phase = raw["WastePhase"].astype(str)
    raw = raw[phase.str.startswith("Supernatant") | phase.str.startswith("Saltcake")]
    kg = raw.pivot_table(index="WasteSiteId", columns="Analyte", values="Inventory", aggfunc="sum").fillna(0.0)
    na2o = kg["Na"] * NA2O_PER_NA
    kg = kg[na2o > 0]
    na2o = na2o[na2o > 0]
    k = kg["K"] * K2O_PER_K / na2o
    s = kg["SO4"] * SO3_PER_SO4 / na2o
    h = (kg["Cl"] + 0.3 * kg["F"] + 0.4 * kg["Cr"] * CR2O3_PER_CR) / na2o
    with np.errstate(divide="ignore"):
        limits = pd.DataFrame({
            "Alkali": 24.0 / (1 + 0.66 * k),
            "Alkali + sulfur": 33.94 / (1 + 0.66 * k + 11.69 * s),
            "Sulfur": 1.5 / s,
            "Halide": 1.65 / (s + 0.725 * h),
        })
    out = pd.DataFrame({"Na2O_t": na2o / 1000.0, "loading": limits.min(axis=1), "rule": limits.idxmin(axis=1)})
    out["glass_t"] = out["Na2O_t"] / (out["loading"] / 100.0)
    out["glass_alkali_t"] = out["Na2O_t"] / (limits["Alkali"] / 100.0)
    return out


@requires_real_data
class TestRealDataset:
    def test_every_tank_agrees_with_a_recomputation_from_the_raw_records(self, real_dataset, real_screen):
        expected = _raw_recomputation(real_dataset)
        got = real_screen.screened().set_index("WasteSiteId")
        assert sorted(got.index) == sorted(expected.index)
        expected = expected.loc[got.index]
        np.testing.assert_allclose(got["Na2O_t"], expected["Na2O_t"], rtol=1e-6)
        np.testing.assert_allclose(got["Na2O_loading_wtpct"], expected["loading"], rtol=1e-6)
        np.testing.assert_allclose(got["Glass_t"], expected["glass_t"], rtol=1e-6)
        np.testing.assert_allclose(got["Glass_t_alkali_only"], expected["glass_alkali_t"], rtol=1e-6)
        assert got["Binding_rule"].to_list() == expected["rule"].to_list()

    def test_reference_totals(self, real_screen):
        s = real_screen.summary
        assert s.n_with_sodium == 133
        assert s.na2o_kt == pytest.approx(54.56, abs=0.005)
        assert s.share_by_rule["Alkali"] == pytest.approx(53.2, abs=0.05)
        assert s.share_by_rule["Alkali + sulfur"] == pytest.approx(36.6, abs=0.05)
        assert s.share_by_rule["Sulfur"] == pytest.approx(3.3, abs=0.05)
        assert s.share_by_rule["Halide"] == pytest.approx(6.9, abs=0.05)
        assert s.glass_alkali_only_kt == pytest.approx(230.3, abs=0.05)
        assert s.glass_unit_by_unit_kt == pytest.approx(276.9, abs=0.05)
        assert s.glass_blended_kt == pytest.approx(248.4, abs=0.05)
        assert s.blend_loading_wtpct == pytest.approx(21.97, abs=0.005)
        assert s.blend_binding_rule == "Alkali + sulfur"

    def test_reference_tanks(self, real_screen):
        table = real_screen.table.set_index("WasteSiteId")
        assert table.loc["241-B-107", "Na2O_loading_wtpct"] == pytest.approx(3.27, abs=0.005)
        assert table.loc["241-B-107", "Binding_rule"] == "Sulfur"
        assert table.loc["241-AW-106", "Na2O_loading_wtpct"] == pytest.approx(11.25, abs=0.005)
        assert table.loc["241-AW-106", "Binding_rule"] == "Halide"
        assert table.loc["241-TX-113", "Na2O_loading_wtpct"] == pytest.approx(9.91, abs=0.005)
        assert table.loc["241-TX-113", "Binding_rule"] == "Sulfur"

    def test_blend_of_the_advanced_set_needs_no_more_glass_than_tank_by_tank(self, real_screen):
        s = real_screen.summary
        assert s.glass_alkali_only_kt <= s.glass_blended_kt <= s.glass_unit_by_unit_kt

    def test_conservative_envelope_blend_needs_more_glass_than_tank_by_tank(self, real_dataset):
        # The non-linear envelope at work on real data: the blended feed
        # lands on the halide envelope and needs MORE glass than the tanks
        # taken one by one.
        s = gr.screen(real_dataset, rule_set=gr.CONSERVATIVE).summary
        assert s.blend_binding_rule == "Halide"
        assert s.glass_blended_kt > s.glass_unit_by_unit_kt

    def test_groups_and_layers_are_consistent(self, real_dataset, real_screen):
        groups = real_screen.groups.set_index("Group")
        assert list(groups.index) == ["All tanks", "Single-shell tanks", "Double-shell tanks",
                                      "Saltcake layers", "Supernatant layers"]
        assert groups.loc["Single-shell tanks", "Na2O_kt"] + groups.loc["Double-shell tanks", "Na2O_kt"] == \
            pytest.approx(groups.loc["All tanks", "Na2O_kt"])
        assert groups.loc["Saltcake layers", "Na2O_kt"] + groups.loc["Supernatant layers", "Na2O_kt"] == \
            pytest.approx(groups.loc["All tanks", "Na2O_kt"])
        layers = gr.screen(real_dataset, level="layer")
        assert layers.summary.na2o_kt == pytest.approx(real_screen.summary.na2o_kt)
        # the blended feed is the same waste whichever way it was cut
        assert layers.summary.glass_blended_kt == pytest.approx(real_screen.summary.glass_blended_kt)
        assert layers.summary.glass_unit_by_unit_kt >= real_screen.summary.glass_unit_by_unit_kt

    def test_a_scope_is_the_same_screen_on_fewer_tanks(self, real_dataset, real_screen):
        scoped = gr.screen(real_dataset, scope={"TankFarm": ["B"]})
        whole = real_screen.table[real_screen.table["TankFarm"] == "B"].set_index("WasteSiteId")
        part = scoped.table.set_index("WasteSiteId")
        assert sorted(part.index) == sorted(whole.index)
        np.testing.assert_allclose(part["Na2O_loading_wtpct"].astype(float),
                                   whole.loc[part.index, "Na2O_loading_wtpct"].astype(float), rtol=1e-12)
