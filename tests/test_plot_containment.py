"""
Containment: no chart is cut by the edge of its pane.

plot_helpers.overflow_report() is the measurable definition (every visible
artist with text or ink whose window extent lies outside the figure, or
outside its cell in a composed figure) and plot_helpers.contain_region()
the net every layout pass ends with. This file sweeps every plot function
of plot_helpers, every mark of plot_grammar.render_plot (with and without
facets) and the charts the pages draw themselves (Tank Explorer, Element
Explorer, Loading rules, Oxide chemistry), on small realistic fixtures
with long category names, 3 and 60 categories, a long title and a
three-line subtitle, at four canvas sizes, and asserts that the report is
empty after layout -- on screen, in an export, and in a composed figure.

At the two small sizes a chart that cannot be shown may be replaced by
one calm line (plot_helpers.ENLARGE_MESSAGE); never above 520 x 360.
"""
import struct

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt

import figure_composer as fcomp
import plot_helpers as ph
from composer_store import ComposerItem
from plot_grammar import PlotSpec, render_plot
from qt_widgets import PlotWidget

SIZES = [(384, 220), (520, 360), (760, 520), (1200, 760)]
SMALL_SIZES = {(384, 220), (520, 360)}          # where the one-line message is allowed

LONG_NAME = "241-AN-102 Saltcake Interstitial Liquid"
LONG_TITLE = ("Inventory of the selected analytes in every tank of the selection, "
              "ranked by the total attributed to the element")
SUBTITLE = " · ".join([
    "Apportioned (stoichiometric) attribution of every reported analyte to its element",
    "Scope: farm AN, AP, AW, AY, AZ, SY; waste phase Saltcake Interstitial Liquid, Sludge Solid, Supernatant",
    "decay date 2015 for chemicals and 2022 for radionuclides; 12 of 177 tanks omitted (no inventory reported)",
])


# ---------------------------------------------------------------------------
# Fixtures (plain functions: the cases below are module-level lambdas)
# ---------------------------------------------------------------------------
ELEMENTS = ["Na", "Al", "Fe", "Cr", "Bi", "Zr", "Si", "P", "S", "K", "Ca", "Mn", "Ni", "U", "Sr", "Cs",
            "Pb", "La", "F", "Cl", "Tc", "Hg", "Ce", "Mg", "Ti", "Zn", "Ag", "Cd", "Ba", "Co"]


def tanks(n):
    farms = ["A", "AN", "AP", "AW", "AX", "AY", "AZ", "B", "BX", "BY", "C", "S", "SX", "SY", "T", "TX", "TY", "U"]
    return [f"241-{farms[i % len(farms)]}-{101 + i // len(farms)}" for i in range(n)]


def long_names(n):
    if n == 1:
        return [LONG_NAME]
    return [f"{LONG_NAME} {i + 1:02d}" for i in range(n)]


def ranked(n, label="Label", long=True):
    rng = np.random.default_rng(n)
    names = long_names(n) if long else tanks(n)
    return pd.DataFrame({label: names, "Value": np.sort(rng.uniform(1.0, 5.0e5, n))[::-1]})


def wide_matrix(n_tanks, n_elements):
    rng = np.random.default_rng(n_tanks * 100 + n_elements)
    frame = pd.DataFrame(rng.uniform(0.0, 5.0e4, size=(n_tanks, n_elements)), columns=ELEMENTS[:n_elements])
    frame.insert(0, "WasteSiteId", tanks(n_tanks))
    return frame


def tank_profile(n_tanks, n_elements=12):
    rng = np.random.default_rng(7)
    rows = [{"Element": e, "WasteSiteId": t, "Inventory_sum": float(rng.uniform(1, 1e5)), "Units": "kg"}
            for t in tanks(n_tanks) for e in ELEMENTS[:n_elements]]
    return pd.DataFrame(rows)


def correlation_scan(n):
    rng = np.random.default_rng(3)
    r = rng.uniform(-1, 1, n)
    return pd.DataFrame({
        "PartnerElement": ELEMENTS[:n], "Correlation_r": r, "AbsCorrelation": np.abs(r),
        "N_overlap_nonzero_tanks": rng.integers(5, 170, n), "Units": "kg", "Metric": "log10_plus1",
    })


def square(names, seed=0, label="Element", low=-1.0):
    rng = np.random.default_rng(seed)
    n = len(names)
    values = rng.uniform(low, 1, size=(n, n))
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 1.0)
    frame = pd.DataFrame(values, columns=list(names))
    frame.insert(0, label, list(names))
    return frame


def workbench(n_elements=8, n_tanks=40):
    elements = ELEMENTS[:n_elements]
    names = tanks(n_tanks)
    rng = np.random.default_rng(11)
    raw = pd.DataFrame(rng.uniform(0, 1.0e5, size=(n_tanks, n_elements)), columns=elements)
    raw[raw < 8.0e3] = 0.0
    raw.insert(0, "WasteSiteId", names)
    metric = raw.copy()
    presence = raw.copy()
    for e in elements:
        metric[e] = np.log10(metric[e] + 1.0)
        presence[e] = (presence[e] > 0).astype(int)
    stats = pd.DataFrame({
        "Element": elements, "Units": "kg",
        "Total_inventory_kg": [float(raw[e].sum()) for e in elements],
        "PresenceFraction_pct": [float((raw[e] > 0).mean() * 100.0) for e in elements],
        "Max_kg_in_one_tank": [float(raw[e].max()) for e in elements],
    })
    pairs = []
    for i, a in enumerate(elements):
        for b in elements[i + 1:]:
            r = float(rng.uniform(-1, 1))
            pairs.append({
                "Element_A": a, "Element_B": b, "Correlation_r": r, "AbsCorrelation": abs(r),
                "Jaccard_presence": float(rng.uniform(0, 1)), "N_both_present": int(rng.integers(3, n_tanks)),
                "PreferredAssociationScore_proxy": float(rng.uniform(0, 20)),
            })
    return {
        "elements": elements, "raw": raw, "metric": metric, "presence": presence, "stats": stats,
        "pairs": pd.DataFrame(pairs), "corr": square(elements, 1), "jaccard": square(elements, 2, low=0.0),
        "similarity": square(names, 3, label="WasteSiteId"),
    }


def structure(n_tanks=40, n_elements=8, levels=18):
    from scipy.cluster.hierarchy import linkage

    names = tanks(n_tanks)
    rng = np.random.default_rng(5)
    farms = [name.split("-")[1] for name in names]
    summary = pd.DataFrame({
        "WasteSiteId": names, "PC1": rng.normal(0, 2, n_tanks), "PC2": rng.normal(0, 1, n_tanks),
        "Cluster": rng.integers(1, 5, n_tanks),
        "TankFarm": [farms[i % levels] for i in range(n_tanks)],
        "Dominant waste phase": [(LONG_NAME, "Sludge Solid", "Supernatant")[i % 3] for i in range(n_tanks)],
    })
    variance = pd.DataFrame({"PC": ["PC1", "PC2"], "ExplainedVarianceRatio": [0.62, 0.21]})
    elements = ELEMENTS[:n_elements]
    xy = rng.uniform(-1, 1, size=(n_elements, 2))
    nodes = pd.DataFrame({
        "Element": elements, "Total_inventory_kg": rng.uniform(10, 1e5, n_elements),
        "LogTotalInventory": rng.uniform(1, 5, n_elements), "x": xy[:, 0], "y": xy[:, 1],
        "N_edges": rng.integers(0, 4, n_elements),
    })
    edges = pd.DataFrame([
        {"Element_A": elements[i], "Element_B": elements[i + 1], "Correlation_r": 0.8 if i % 2 else -0.5,
         "AbsCorrelation": 0.8 if i % 2 else 0.5, "Jaccard_presence": 0.5,
         "Sign": "positive" if i % 2 else "negative"}
        for i in range(n_elements - 1)
    ])
    return {
        "summary": summary, "variance": variance, "linkage": linkage(rng.uniform(0, 1, size=(n_tanks, 4)), "ward"),
        "labels": names, "raw": square(elements, 6), "partial": square(elements, 7),
        "nodes": nodes, "edges": edges,
    }


def screening(n):
    from vitrification_science import SCORE_CONTRIB_TERMS

    rng = np.random.default_rng(9)
    frame = pd.DataFrame({
        "WasteSiteId": tanks(n), "Total_kg_inventory": rng.uniform(1e3, 1e6, n),
        "Total_Ci_inventory": rng.uniform(1e2, 1e6, n), "frac_problem_elements_proxy": rng.uniform(0, 0.4, n),
        "frac_glass_former_or_intermediate": rng.uniform(0, 1, n),
        "Vitrification_screening_score_proxy": rng.uniform(0, 100, n),
        "User_search_score_proxy": rng.uniform(0, 100, n),
    })
    for i, column in enumerate(SCORE_CONTRIB_TERMS):
        frame[column] = rng.uniform(-20, 30, n) if i % 2 else rng.uniform(0, 30, n)
    return frame


def blend(n):
    rng = np.random.default_rng(13)
    return pd.DataFrame({"PartnerTank": tanks(n), "BaseTank": "241-AN-102",
                         "Blend_complement_score_proxy": rng.uniform(0, 1, n)})


def target_total(n):
    rng = np.random.default_rng(15)
    return pd.DataFrame({
        "WasteSiteId": tanks(n), "Context_TotalInventory_same_unit": rng.uniform(1e3, 1e7, n),
        "Target_Inventory_sum": rng.uniform(1e-2, 1e4, n), "Units": ["kg" if i % 2 else "Ci" for i in range(n)],
    })


def association_values(n=60):
    rng = np.random.default_rng(17)
    return pd.DataFrame({
        "WasteSiteId": tanks(n), "WastePhase": [("Supernatant", "Saltcake Solid", "Sludge Solid")[i % 3] for i in range(n)],
        "Na": rng.uniform(0, 1e5, n), "Al": rng.uniform(0, 1e4, n),
        "Family": [("Supernatant", "Saltcake", "Sludge")[i % 3] for i in range(n)],
    })


def long_frame(n_categories, n_series=4, long=True):
    rng = np.random.default_rng(n_categories + n_series)
    names = long_names(n_categories) if long else tanks(n_categories)
    series = ["Saltcake Interstitial Liquid", "Saltcake Solid", "Sludge Solid", "Supernatant",
              "Sludge Interstitial Liquid", "Retained Gas"][:n_series]
    rows = [{"Category": name, "Series": s, "Total_kg": float(rng.uniform(1, 1e5)),
             "Other_kg": float(rng.uniform(1, 1e5)), "WasteSiteId": tanks(n_categories)[i]}
            for i, name in enumerate(names) for s in series]
    return pd.DataFrame(rows)


def rule_frames(n=40):
    import glass_rules as gr

    rules = gr.get_rule_set(gr.DEFAULT_RULE_SET)
    rng = np.random.default_rng(19)
    names = tanks(n)
    points = pd.DataFrame({
        "Unit": names, "WasteSiteId": names, "Na2O_t": rng.uniform(1, 2000, n),
        "SO3_per_Na2O": rng.uniform(0.004, 0.5, n), "Halide_term_per_Na2O": rng.uniform(0.005, 0.1, n),
        "Na2O_loading_wtpct": rng.uniform(5, 24, n), "Binding_rule": [rules.rule_names[i % len(rules.rule_names)] for i in range(n)],
        "Extra_glass_t": rng.uniform(0, 5000, n), "Glass_t": rng.uniform(100, 20000, n),
        "TankSystem": ["SST" if i % 3 else "DST" for i in range(n)],
    })
    blend_point = {"SO3_per_Na2O": 0.05, "Halide_term_per_Na2O": 0.02, "Na2O_loading_wtpct": 21.9,
                   "Binding_rule": rules.rule_names[1], "Glass_t": 248400.0}
    groups = pd.DataFrame({"Group": ["Single-shell tanks", "Double-shell tanks", "All tanks"],
                           "Na2O_kt": [30.1, 24.5, 54.6], "N_units": [105, 28, 133]})
    shares = rng.dirichlet(np.ones(len(rules.rule_names)), size=3) * 100.0
    for j, rule in enumerate(rules.rule_names):
        groups[f"{rule}_pct"] = shares[:, j]
    return {"name": rules.name, "points": points, "blend": blend_point, "groups": groups}


RULE_SUBTITLE = " · ".join(["Advanced LAW (2013)", "Supernatant + saltcake of each tank",
                            "133 tanks holding sodium", "Scope: farm AN, AP, AW, AY, AZ"])


def composition(n_tanks, n_elements=20):
    rng = np.random.default_rng(23)
    rows = [{"Element": e, "WasteSiteId": t, "Inventory_sum": float(rng.uniform(1e-2, 1e6)), "Units": "kg"}
            for t in tanks(n_tanks) for e in ELEMENTS[:n_elements]]
    return pd.DataFrame(rows)


def _spec(**fields):
    fields.setdefault("title", LONG_TITLE)
    fields.setdefault("subtitle", SUBTITLE)
    return PlotSpec(**fields)


def _tank_composition(panel, n):
    from qt_tank_explorer import plot_tank_composition
    plot_tank_composition(panel, composition(n), "Inventory_sum",
                          f"Composition of {', '.join(tanks(n)[:3])}: the 20 largest elements", top_n=20, unit="kg")


def _ranked_bars(panel, n):
    from qt_explorer import plot_ranked_bars
    plot_ranked_bars(panel, ranked(n, "WasteSiteId"), "WasteSiteId", "Value", LONG_TITLE,
                     "Se inventory (g)", subtitle=SUBTITLE, top_n=60, log_x=True)


def _rule_diagram(panel):
    from qt_vitrification_rules import plot_rule_diagram
    f = rule_frames()
    plot_rule_diagram(panel, f["points"], f["blend"], f["name"], 0.019, "TankSystem",
                      "Loading rules: where each tank sits", RULE_SUBTITLE, "tank")


def _rule_shares(panel):
    from qt_vitrification_rules import plot_rule_shares
    f = rule_frames()
    plot_rule_shares(panel, f["groups"], f["name"], "Sodium by the rule that sets its loading", RULE_SUBTITLE)


def _extra_glass(panel):
    from qt_vitrification_rules import plot_extra_glass
    f = rule_frames()
    plot_extra_glass(panel, f["points"], f["name"], "Largest extra glass over the alkali limit, tank by tank",
                     RULE_SUBTITLE, 25, "tank")


def _oxide(panel, n):
    table = ranked(n, "Component").rename(columns={"Value": "Wt_pct"})
    render_plot(panel, table, PlotSpec(
        mark="bar", x="Wt_pct", y="Component", top_n=25, title=f"Oxide-basis composition of {LONG_NAME}",
        xlabel="Share of the converted inventory (wt%)", ylabel="Oxide or element"))


W = workbench()
S = structure()

# name -> callable(panel). One entry per plot function and per way it draws.
CASES = {
    # -- plot_helpers ------------------------------------------------------
    "barh_3_long": lambda p: ph.plot_barh(p, ranked(3), "Label", "Value", LONG_TITLE, "Total inventory (kg)"),
    "barh_60_long": lambda p: ph.plot_barh(p, ranked(60), "Label", "Value", LONG_TITLE,
                                           "Total inventory (kg)", top_n=60, log_x=True),
    "barh_60_tanks": lambda p: ph.plot_barh(p, ranked(60, long=False), "Label", "Value", "Se inventory by tank",
                                            "Se inventory (g)", top_n=60),
    "heatmap_3": lambda p: ph.plot_heatmap(p, wide_matrix(3, 4), "kg", "inventory", LONG_TITLE),
    "heatmap_60": lambda p: ph.plot_heatmap(p, wide_matrix(60, 30), "kg", "log10_inventory", LONG_TITLE),
    "heatmap_fraction": lambda p: ph.plot_heatmap(p, wide_matrix(60, 30), "Ci", "fraction", "Inventory heatmap"),
    "grouped_profile_3": lambda p: ph.plot_grouped_tank_profile(p, tank_profile(3), "Inventory_sum", LONG_TITLE),
    "grouped_profile_8": lambda p: ph.plot_grouped_tank_profile(p, tank_profile(8, 25), "Inventory_sum", "Profile"),
    "correlation_scan": lambda p: ph.plot_correlation_scan(p, correlation_scan(30), "Cs"),
    "correlation_scan_3": lambda p: ph.plot_correlation_scan(p, correlation_scan(3), "Cs"),
    "corr_heatmap_mpl": lambda p: ph.plot_correlation_heatmap(p, square(ELEMENTS[:30]), LONG_TITLE),
    "corr_heatmap_mpl_annot": lambda p: ph.plot_correlation_heatmap(p, square(ELEMENTS[:6]), "Element correlation",
                                                                    annotate=True),
    "corr_heatmap_seaborn": lambda p: ph.plot_correlation_heatmap(p, square(ELEMENTS[:30]), LONG_TITLE,
                                                                  style="Seaborn lower triangle"),
    "corr_heatmap_projections": lambda p: ph.plot_correlation_heatmap(
        p, square(ELEMENTS[:30]), LONG_TITLE, style="Seaborn + total projections",
        totals={e: 10.0 ** (i % 6) for i, e in enumerate(ELEMENTS)}, unit="kg"),
    "pair_scatter_2": lambda p: ph.plot_pair_scatter(p, W["metric"], ["Na", "Al"], "kg", "log10_plus1"),
    "pair_scatter_3": lambda p: ph.plot_pair_scatter(p, W["metric"], ["Na", "Al", "Fe"], "kg", "log10_plus1"),
    "target_vs_total": lambda p: ph.plot_target_vs_total(p, target_total(60), LONG_TITLE),
    "sns_triangle": lambda p: ph.plot_seaborn_lower_triangle_matrix(p, W["corr"], LONG_TITLE, "Correlation r"),
    "sns_triangle_annot": lambda p: ph.plot_seaborn_lower_triangle_matrix(
        p, W["corr"], "Element correlation heatmap - kg only", "Correlation r", annotate=True),
    "sns_triangle_projections": lambda p: ph.plot_seaborn_lower_triangle_matrix(
        p, W["corr"], LONG_TITLE, "Correlation r", totals=W["stats"], projections=True),
    "sns_triangle_jaccard": lambda p: ph.plot_seaborn_lower_triangle_matrix(
        p, W["jaccard"], "Jaccard co-presence", "Jaccard co-presence", center=None, totals=W["stats"],
        projections=True, color_mode="Coherent colors"),
    "sns_top_preferred": lambda p: ph.plot_seaborn_top_associations(p, W["pairs"], top_n=30),
    "sns_top_positive": lambda p: ph.plot_seaborn_top_associations(p, W["pairs"], top_n=30, mode="positive"),
    "sns_top_negative": lambda p: ph.plot_seaborn_top_associations(p, W["pairs"], top_n=30, mode="negative",
                                                                   color_mode="Coherent colors"),
    "sns_top_jaccard": lambda p: ph.plot_seaborn_top_associations(p, W["pairs"], top_n=3, mode="jaccard"),
    "sns_pair_matrix_regression": lambda p: ph.plot_seaborn_pair_matrix(
        p, W["metric"], W["raw"], W["elements"][:4], "log10_plus1"),
    "sns_pair_matrix_scatter_8": lambda p: ph.plot_seaborn_pair_matrix(
        p, W["metric"], W["raw"], W["elements"], "log10_plus1", kind="scatter"),
    "sns_pair_matrix_kde": lambda p: ph.plot_seaborn_pair_matrix(
        p, W["metric"], W["raw"], W["elements"][:3], "log10_plus1", kind="kde"),
    "sns_joint_scatter": lambda p: ph.plot_seaborn_joint_first_two(p, W["metric"], ["Na", "Al"], "log10_plus1"),
    "sns_joint_regression": lambda p: ph.plot_seaborn_joint_first_two(p, W["metric"], ["Na", "Al"], "log10_plus1",
                                                                      kind="regression"),
    "sns_joint_kde": lambda p: ph.plot_seaborn_joint_first_two(p, W["metric"], ["Na", "Al"], "log10_plus1",
                                                               kind="kde"),
    "sns_tank_similarity": lambda p: ph.plot_seaborn_tank_similarity(p, W["similarity"], W["raw"], top_tanks=40),
    "sns_tank_similarity_annot": lambda p: ph.plot_seaborn_tank_similarity(p, W["similarity"], W["raw"],
                                                                           top_tanks=8, annotate=True),
    "sns_tank_element_map": lambda p: ph.plot_seaborn_tank_element_map(p, W["raw"], W["elements"], top_tanks=40),
    "sns_tank_element_map_fraction": lambda p: ph.plot_seaborn_tank_element_map(
        p, W["raw"], W["elements"], top_tanks=40, metric="fraction"),
    "tank_element_bars_stacked": lambda p: ph.plot_tank_element_bars(p, W["raw"], W["elements"], top_n=30),
    "tank_element_bars_grouped": lambda p: ph.plot_tank_element_bars(
        p, W["raw"], W["elements"][:3], top_n=30, mode="Grouped", log_scale=True),
    "sns_presence": lambda p: ph.plot_seaborn_presence_patterns(p, W["presence"], W["elements"], top_n=30,
                                                                color_mode="Coherent colors"),
    "sns_dashboard": lambda p: ph.plot_seaborn_stats_dashboard(p, W["stats"], W["pairs"], top_n=20),
    "pca_by_farm": lambda p: ph.plot_pca_scatter(p, S["summary"], "TankFarm", S["variance"]),
    "pca_by_phase": lambda p: ph.plot_pca_scatter(p, S["summary"], "Dominant waste phase", S["variance"]),
    "pca_plain": lambda p: ph.plot_pca_scatter(p, S["summary"], None, S["variance"]),
    "dendrogram": lambda p: ph.plot_dendrogram(p, S["linkage"], S["labels"]),
    "partial_correlation": lambda p: ph.plot_partial_correlation_comparison(p, S["partial"], S["raw"]),
    "partial_correlation_annot": lambda p: ph.plot_partial_correlation_comparison(p, S["partial"], S["raw"],
                                                                                 annotate=True),
    "network": lambda p: ph.plot_element_network(p, S["nodes"], S["edges"]),
    "score_breakdown": lambda p: ph.plot_vitrification_score_breakdown(p, screening(40), LONG_TITLE),
    "burden": lambda p: ph.plot_vitrification_burden(p, screening(60), LONG_TITLE),
    "candidate_scores": lambda p: ph.plot_candidate_scores(p, screening(60), top_n=30),
    "candidate_scores_3": lambda p: ph.plot_candidate_scores(p, screening(3), top_n=30),
    "blend_scores": lambda p: ph.plot_blend_scores(p, blend(60), top_n=30),
    "association_heatmap_6": lambda p: ph.plot_association_heatmap(
        p, square(ELEMENTS[:6], label="Component"), LONG_TITLE, SUBTITLE),
    "association_heatmap_long": lambda p: ph.plot_association_heatmap(
        p, square([f"{e} ({LONG_NAME})" for e in ELEMENTS[:5]], label="Component"), "Cross-correlation", SUBTITLE),
    "association_heatmap_34": lambda p: ph.plot_association_heatmap(
        p, square([f"{e}{i}" for i in range(2) for e in ELEMENTS[:17]], label="Component"), "Cross-correlation"),
    "association_scatter": lambda p: ph.plot_association_scatter(
        p, association_values(), "Na", "Al", "kg", LONG_TITLE, SUBTITLE, color_col="Family"),
    "association_scatter_plain": lambda p: ph.plot_association_scatter(p, association_values(), "Na", "Al"),
    # -- plot_grammar.render_plot ------------------------------------------
    "grammar_bar_3": lambda p: render_plot(p, long_frame(3, 1), _spec(mark="bar", x="Total_kg", y="Category")),
    "grammar_bar_60": lambda p: render_plot(p, long_frame(60, 1), _spec(
        mark="bar", x="Total_kg", y="Category", top_n=60, log_x=True)),
    "grammar_bar_plain_title": lambda p: render_plot(p, long_frame(60, 1), PlotSpec(
        mark="bar", x="Total_kg", y="Category", top_n=60, title=LONG_TITLE)),
    "grammar_bar_hue_3": lambda p: render_plot(p, long_frame(3, 4), _spec(
        mark="bar", x="Total_kg", y="Category", hue="Series")),
    "grammar_bar_hue_60": lambda p: render_plot(p, long_frame(60, 6), _spec(
        mark="bar", x="Total_kg", y="Category", hue="Series", top_n=60)),
    "grammar_stacked_3": lambda p: render_plot(p, long_frame(3, 4), _spec(
        mark="stackedbar", x="Total_kg", y="Category", hue="Series")),
    "grammar_stacked_60": lambda p: render_plot(p, long_frame(60, 6), _spec(
        mark="stackedbar", x="Total_kg", y="Category", hue="Series", top_n=60)),
    "grammar_heatmap_3": lambda p: render_plot(p, long_frame(3, 4), _spec(
        mark="heatmap", x="Total_kg", y="Category", hue="Series")),
    "grammar_heatmap_60": lambda p: render_plot(p, long_frame(60, 6), _spec(
        mark="heatmap", x="Total_kg", y="Category", hue="Series", log_x=True, normalize="row")),
    "grammar_heatmap_wide": lambda p: render_plot(
        p, long_frame(6, 6).rename(columns={"Category": "Series2", "Series": "Category2"}),
        _spec(mark="heatmap", x="Total_kg", y="Category2", hue="Series2")),
    "grammar_scatter": lambda p: render_plot(p, long_frame(60, 1), _spec(
        mark="scatter", x="Total_kg", y="Other_kg", log_x=True, log_y=True)),
    "grammar_scatter_hue": lambda p: render_plot(p, long_frame(20, 6), _spec(
        mark="scatter", x="Total_kg", y="Other_kg", hue="Category")),
    "grammar_box_3": lambda p: render_plot(p, long_frame(3, 6), _spec(mark="box", x="Total_kg", y="Category")),
    "grammar_box_60": lambda p: render_plot(p, long_frame(60, 6), _spec(
        mark="box", x="Total_kg", y="Category", top_n=60, log_x=True)),
    "grammar_facet_bar_3": lambda p: render_plot(p, long_frame(3, 3), _spec(
        mark="bar", x="Total_kg", y="Category", facet="Series")),
    "grammar_facet_bar_60": lambda p: render_plot(p, long_frame(60, 6), _spec(
        mark="bar", x="Total_kg", y="Category", facet="Series", top_n=60)),
    "grammar_facet_bar_long_facets": lambda p: render_plot(
        p, long_frame(6, 6, long=False).rename(columns={"Category": "Tank", "Series": "Phase"}).assign(
            Phase=lambda f: LONG_NAME + " / " + f["Phase"]),
        _spec(mark="bar", x="Total_kg", y="Tank", facet="Phase")),
    "grammar_facet_scatter": lambda p: render_plot(p, long_frame(30, 6), _spec(
        mark="scatter", x="Total_kg", y="Other_kg", facet="Series", log_x=True, log_y=True)),
    "grammar_annotation": lambda p: render_plot(p, long_frame(8, 1), PlotSpec(
        mark="bar", x="Total_kg", y="Category", title="With a limit",
        annotations=({"kind": "vline", "value": 5.0e4, "label": "regulatory limit"},))),
    # -- charts the pages draw themselves ----------------------------------
    "tank_composition_1": lambda p: _tank_composition(p, 1),
    "tank_composition_3": lambda p: _tank_composition(p, 3),
    "tank_composition_6": lambda p: _tank_composition(p, 6),
    "tank_composition_9": lambda p: _tank_composition(p, 9),
    "explorer_ranked_3": lambda p: _ranked_bars(p, 3),
    "explorer_ranked_60": lambda p: _ranked_bars(p, 60),
    "rule_diagram": _rule_diagram,
    "rule_shares": _rule_shares,
    "extra_glass": _extra_glass,
    "oxide_3": lambda p: _oxide(p, 3),
    "oxide_25": lambda p: _oxide(p, 25),
    # -- sentences ---------------------------------------------------------
    "message": lambda p: p.show_message(
        "No tank reports a positive value for both Na and Al.\n0 of 177 tanks drawn · 177 do not report both"),
    "empty": lambda p: p.show_empty(),
}


# ---------------------------------------------------------------------------
# Panels at an exact canvas size
# ---------------------------------------------------------------------------
def shown_panel(qtbot, width, height):
    """A PlotWidget, shown (never on the real screen), whose canvas is
    width x height pixels."""
    panel = PlotWidget()
    qtbot.addWidget(panel)
    panel.setAttribute(Qt.WA_DontShowOnScreen, True)
    panel.show()
    qtbot.wait(10)
    chrome = panel.height() - panel.canvas.height()
    panel.resize(width, height + chrome)
    qtbot.wait(10)
    for _ in range(3):
        got = (panel.canvas.width(), panel.canvas.height())
        if got == (width, height):
            break
        panel.resize(panel.width() + width - got[0], panel.height() + height - got[1])
        qtbot.wait(10)
    dpi = panel._logical_dpi()                            # per logical pixel: a scaled screen has more
    if (panel.canvas.width(), panel.canvas.height()) != (width, height):
        # A platform whose toolbar is wider than the pane (the off-screen
        # platform's fallback font): the figure is given the size directly.
        panel.figure.set_size_inches(width / dpi, height / dpi, forward=False)
    # The room a plot function is told it has is the pane itself. (In the
    # application the widget sits in a window and reserves 280 px of that
    # window's width for the panel beside it; alone, as here, that reserve
    # would leave a 384 px pane one inch of room.)
    panel.available_content_width_inches = lambda reserve_px=280: width / dpi
    panel.available_content_height_inches = lambda: height / dpi
    return panel


def settle(panel):
    """Lay out and draw as the widget does after a render or a resize."""
    panel.relayout(draw=False)
    panel.canvas.draw()


def resize_figure(panel, width, height):
    """Give the figure the size of a width x height pane (logical pixels),
    as a resize of the widget does, without waiting for one."""
    dpi = panel._logical_dpi()
    panel.figure.set_size_inches(width / dpi, height / dpi, forward=False)


def canvas_size(panel):
    """The figure's size in logical pixels."""
    box, ratio = panel.figure.bbox, panel._pixel_ratio()
    return (int(round(box.width / ratio)), int(round(box.height / ratio)))


def assert_contained(panel, size, where=""):
    report = ph.overflow_report(panel.figure)
    assert not report, f"{where} at {size[0]}x{size[1]}: {ph.describe_overflow(report)}"
    if tuple(size) not in SMALL_SIZES:
        assert not ph.showing_enlarge_message(panel.figure), (
            f"{where} at {size[0]}x{size[1]}: replaced by the 'enlarge the window' line above 520 x 360")


def geometry(figure):
    """Everything a layout decides, rounded: axes boxes, text and sizes."""
    if ph.showing_enlarge_message(figure):
        return ["the one-line message"]                   # the hidden chart's layout is nobody's business
    out = []
    for ax in figure.axes:
        out.append(tuple(round(float(v), 4) for v in ax.get_position().bounds))
        out.append((ax.get_title(), ax.get_xlabel(), ax.get_ylabel(), round(float(ax.title.get_fontsize()), 2)))
        out.append(tuple(t.get_text() for t in ax.get_yticklabels()))
        out.append(tuple(t.get_text() for t in ax.get_xticklabels()))
        legend = ax.get_legend()
        if legend is not None:
            out.append(("legend", getattr(legend, "_ncols", 1), round(float(legend.get_texts()[0].get_fontsize()), 2)
                        if legend.get_texts() else 0.0))
    out.append(figure.get_suptitle())
    out.append(tuple(round(float(v), 4) for v in (figure.subplotpars.left, figure.subplotpars.right,
                                                  figure.subplotpars.bottom, figure.subplotpars.top)))
    return out


UNCHANGED = 0.0006          # of the figure: half a pixel at the largest size, rounding noise


def same_layout(a, b, tolerance=0.004):
    """Two geometries with the same wording, and boxes equal to within
    `tolerance` of the figure (matplotlib's tight_layout settles within a
    pixel or two of itself on axes with a locked aspect ratio; UNCHANGED
    is the tolerance for "the same layout", rounding apart)."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        numbers = (isinstance(x, tuple) and isinstance(y, tuple) and len(x) == len(y)
                   and all(isinstance(v, float) for v in x + y))
        if numbers:
            if any(abs(u - v) > tolerance for u, v in zip(x, y)):
                return False
        elif x != y:
            return False
    return True


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
def test_the_sweep_draws_every_plot_function_and_every_mark():
    """A plot function added to plot_helpers, or a mark added to the plot
    grammar, has to be added to CASES above."""
    import inspect
    import sys

    import plot_grammar

    source = inspect.getsource(sys.modules[__name__])
    functions = [name for name, value in vars(ph).items() if name.startswith("plot_") and callable(value)]
    assert len(functions) >= 26
    assert [name for name in functions if f"ph.{name}(" not in source] == []
    assert [mark for mark in plot_grammar.MARKS if f'mark="{mark}"' not in source] == []
    for mark in ("bar", "scatter"):                       # the marks a facet can hold
        assert any(f'mark="{mark}"' in line and "facet=" in line for line in source.splitlines())


@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("name", sorted(CASES))
def test_chart_stays_inside_its_pane(qtbot, name, size):
    """Drawn at this size: nothing outside the pane, and laying it out
    again changes nothing (the layout is a function of the size)."""
    panel = shown_panel(qtbot, *size)
    try:
        CASES[name](panel)
        settle(panel)
        assert canvas_size(panel) == tuple(size)
        assert_contained(panel, size, name)
        first = geometry(panel.figure)
        settle(panel)
        assert_contained(panel, size, f"{name} (laid out again)")
        assert same_layout(geometry(panel.figure), first, UNCHANGED), (
            f"{name} at {size}: laying it out again changed the chart")
    finally:
        panel.hide()


@pytest.mark.parametrize("name", sorted(CASES))
def test_chart_stays_inside_through_a_resize(qtbot, name):
    """Drawn large, then taken twice through every size down and back up:
    contained at each step, back to the layout it started from (the same
    wording, the same boxes), and the second round trip ends exactly where
    the first did: nothing creeps."""
    panel = shown_panel(qtbot, *SIZES[-1])
    try:
        CASES[name](panel)
        settle(panel)
        fresh = geometry(panel.figure)
        rounds = []
        for _ in range(2):
            for size in SIZES[::-1][1:] + SIZES[1:]:
                resize_figure(panel, *size)
                settle(panel)
                assert_contained(panel, size, f"{name} (resized)")
            rounds.append(geometry(panel.figure))
        assert same_layout(rounds[0], fresh), f"{name}: not the layout it started from\n{rounds[0]}\n{fresh}"
        assert same_layout(rounds[1], rounds[0], UNCHANGED), f"{name}: the layout crept between two round trips"
    finally:
        panel.hide()


STABLE = ["barh_60_long", "grammar_bar_hue_60", "pca_by_farm", "rule_diagram", "association_heatmap_long",
          "sns_triangle_projections", "grammar_facet_bar_60", "heatmap_60", "tank_composition_3"]


@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("name", STABLE)
def test_laying_out_twice_changes_nothing(qtbot, name, size):
    panel = shown_panel(qtbot, *size)
    try:
        CASES[name](panel)
        settle(panel)
        first = geometry(panel.figure)
        for _ in range(3):
            settle(panel)
        assert same_layout(geometry(panel.figure), first, UNCHANGED)
    finally:
        panel.hide()


# ---------------------------------------------------------------------------
# The definition itself
# ---------------------------------------------------------------------------
class TestOverflowReport:
    def _bare(self, qtbot, size=(520, 360)):
        panel = shown_panel(qtbot, *size)
        ax = panel.reset_axes()
        ax.plot([0, 1], [0, 1])
        return panel, ax

    def test_a_title_wider_than_the_figure_is_reported_with_both_sides(self, qtbot, monkeypatch):
        monkeypatch.setattr(ph, "CONTAINMENT", False)
        panel, ax = self._bare(qtbot)
        ax.set_title("A title that is a good deal wider than a pane of five hundred and twenty pixels can hold")
        panel.canvas.draw()
        report = ph.overflow_report(panel.figure)
        titles = [entry for entry in report if entry["kind"] == "title"]
        assert len(titles) == 1 and titles[0]["left"] > 1 and titles[0]["right"] > 1
        assert titles[0]["bottom"] == 0 and titles[0]["top"] == 0
        assert "title" in ph.describe_overflow(report)

    def test_the_net_brings_that_title_back_inside(self, qtbot):
        panel, ax = self._bare(qtbot)
        text = "A title that is a good deal wider than a pane of five hundred and twenty pixels can hold"
        ax.set_title(text)
        settle(panel)
        assert ph.overflow_report(panel.figure) == []
        assert "\n" in ax.get_title() and ax.get_title().replace("\n", " ") == text
        assert panel.suggested_caption() == text          # the caption is the title as written

    def test_marks_clipped_by_their_axes_are_not_overflow(self, qtbot):
        panel, ax = self._bare(qtbot)
        ax.plot([-50, 50], [-50, 50])                     # far beyond the limits, under clip_on
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        settle(panel)
        assert ph.overflow_report(panel.figure) == []

    def test_an_unclipped_mark_outside_the_figure_is_overflow(self, qtbot, monkeypatch):
        monkeypatch.setattr(ph, "CONTAINMENT", False)
        panel, ax = self._bare(qtbot)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.scatter([3.0], [0.5], s=80, clip_on=False)
        panel.canvas.draw()
        assert [entry["kind"] for entry in ph.overflow_report(panel.figure)] == ["mark"]

    def test_an_empty_or_hidden_text_is_not_overflow(self, qtbot):
        panel, ax = self._bare(qtbot)
        ax.text(5.0, 0.5, "", clip_on=False)
        hidden = ax.text(5.0, 0.5, "far outside", clip_on=False)
        hidden.set_visible(False)
        settle(panel)
        assert ph.overflow_report(panel.figure) == []

    def test_an_inset_colour_bar_and_a_figure_legend_are_looked_at(self, qtbot, monkeypatch):
        monkeypatch.setattr(ph, "CONTAINMENT", False)
        panel, ax = self._bare(qtbot)
        panel.figure.subplots_adjust(right=0.97)
        image = ax.imshow(np.arange(9.0).reshape(3, 3), aspect="auto")
        cax = ax.inset_axes([1.03, 0.0, 0.05, 1.0])
        panel.figure.colorbar(image, cax=cax, label="A colour bar label")
        panel.figure.legend(handles=list(ax.lines) * 6, labels=["a series with a rather long name"] * 6, ncol=6,
                            loc="lower center")
        panel.canvas.draw()
        kinds = {entry["kind"] for entry in ph.overflow_report(panel.figure)}
        assert "figure legend" in kinds and "y label" in kinds

    def test_a_composed_cell_is_held_to_its_own_rectangle(self, qtbot, monkeypatch):
        monkeypatch.setattr(ph, "CONTAINMENT", False)
        panel = shown_panel(qtbot, 760, 520)
        subfigs = panel.figure.subfigures(1, 2)
        ax = subfigs[0].add_subplot(111)
        ax.set_title("A title that runs out of the left cell and well into the right one, inside the figure")
        subfigs[1].add_subplot(111)
        panel.canvas.draw()
        report = ph.overflow_report(panel.figure)
        assert report and all(entry["region"] is subfigs[0] for entry in report)
        assert all(entry["left"] > 0 or entry["right"] > 0 for entry in report)


# ---------------------------------------------------------------------------
# What the net does, and leaves alone
# ---------------------------------------------------------------------------
class TestCorrections:
    def test_a_grid_of_charts_says_how_small_it_can_be_read(self, qtbot):
        # The four-chart dashboard: one line in a pane it declared too
        # small, the charts in a larger one, never the line above 520 x 360.
        small = shown_panel(qtbot, 520, 360)
        CASES["sns_dashboard"](small)
        settle(small)
        assert ph.showing_enlarge_message(small.figure) and ph.overflow_report(small.figure) == []
        panel = shown_panel(qtbot, 760, 520)
        CASES["sns_dashboard"](panel)
        settle(panel)
        assert not ph.showing_enlarge_message(panel.figure)
        renderer = panel.canvas.get_renderer()
        top_left, _top_right, bottom_left, _bottom_right = panel.figure.axes[:4]
        # The rows clear each other: the axis label of the upper chart ends
        # above the title of the chart under it.
        assert top_left.xaxis.label.get_window_extent(renderer).y0 >= bottom_left.title.get_window_extent(renderer).y1

    def test_the_note_about_thinned_labels_is_wrapped_to_its_column(self, qtbot):
        panel = shown_panel(qtbot, 384, 300)
        ph.plot_barh(panel, ranked(60, long=False), "Label", "Value", "Ranking", "Total inventory (kg)", top_n=60)
        settle(panel)
        lines = panel.ax.get_xlabel().split("\n")
        assert lines[0] == "Total inventory (kg)" and len(lines) >= 3     # the note takes two lines at 384 px
        assert "Point at a bar for its name." in " ".join(lines[1:])
        assert ph.overflow_report(panel.figure) == []

    def test_a_chart_that_fits_is_not_touched(self, qtbot):
        panel = shown_panel(qtbot, 760, 520)
        ph.plot_candidate_scores(panel, screening(10), top_n=10)
        settle(panel)
        state = getattr(panel.figure, ph.CONTAIN_ATTR)
        assert state["undo"] == []
        assert panel.ax.get_title() == "Candidate tank ranking"
        assert [t.get_text() for t in panel.ax.get_yticklabels()] == tanks(10)[::-1] or len(
            panel.ax.get_yticklabels()) == 10

    def test_long_category_names_are_cut_and_the_tooltip_keeps_them(self, qtbot):
        panel = shown_panel(qtbot, 384, 220)
        frame = ranked(3)
        ph.plot_barh(panel, frame, "Label", "Value", "Ranking", "Total inventory (kg)")
        settle(panel)
        if ph.showing_enlarge_message(panel.figure):
            pytest.skip("replaced by the one-line message at this size")
        drawn = [t.get_text() for t in panel.ax.get_yticklabels()]
        assert any(text.endswith("…") for text in drawn)
        marks = getattr(panel.ax, ph.MARKS_ATTR)
        assert sorted(marks["categories"]) == sorted(frame["Label"])

    def test_the_full_labels_come_back_when_the_pane_grows(self, qtbot):
        panel = shown_panel(qtbot, 384, 220)
        frame = ranked(3)
        ph.plot_barh(panel, frame, "Label", "Value", LONG_TITLE, "Total inventory (kg)")
        settle(panel)
        resize_figure(panel, 1400, 760)
        settle(panel)
        assert not ph.showing_enlarge_message(panel.figure)
        assert sorted(t.get_text() for t in panel.ax.get_yticklabels()) == sorted(
            ph.elide_label(v) for v in frame["Label"])
        assert panel.ax.get_title() == LONG_TITLE

    def test_tick_labels_cut_on_an_undeclared_axis_stay_readable_in_full(self, qtbot):
        panel = shown_panel(qtbot, 384, 300)
        ax = panel.reset_axes()
        names = long_names(4)
        ax.barh(range(4), [4, 3, 2, 1])
        ax.set_yticks(range(4))
        ax.set_yticklabels(names)
        panel.figure.tight_layout()
        settle(panel)
        assert ph.overflow_report(panel.figure) == []
        assert ph.full_tick_labels(ax.yaxis) == names
        import qt_plot_interaction as interaction
        assert interaction._full_tick_labels(ax.yaxis) == names

    def test_a_wide_legend_row_is_folded_into_the_pane(self, qtbot):
        panel = shown_panel(qtbot, 520, 400)
        _rule_diagram(panel)
        settle(panel)
        assert ph.overflow_report(panel.figure) == []
        if not ph.showing_enlarge_message(panel.figure):
            legend = panel.figure.legends[0]
            assert legend.get_window_extent(panel.canvas.get_renderer()).width <= panel.figure.bbox.width

    def test_the_one_line_message_gives_way_when_the_pane_grows(self, qtbot):
        panel = shown_panel(qtbot, 384, 220)
        CASES["sns_dashboard"](panel)
        settle(panel)
        assert ph.overflow_report(panel.figure) == []
        resize_figure(panel, 1200, 760)
        settle(panel)
        assert not ph.showing_enlarge_message(panel.figure)
        assert all(ax.get_visible() for ax in panel.figure.axes)
        assert ph.overflow_report(panel.figure) == []

    def test_a_figure_nobody_finished_is_contained_at_its_first_draw(self, qtbot):
        panel = shown_panel(qtbot, 520, 360)
        ax = panel.reset_axes()
        ax.barh(range(3), [3, 2, 1])
        ax.set_yticks(range(3))
        ax.set_yticklabels(long_names(3))
        ax.set_xlabel("An axis label that is longer than the pane is wide, so it has to be wrapped to be read")
        panel.canvas.draw()                               # no finish_panel, no relayout
        assert ph.overflow_report(panel.figure) == []

    def test_switching_the_net_off_only_measures(self, qtbot, monkeypatch):
        monkeypatch.setattr(ph, "CONTAINMENT", False)
        panel = shown_panel(qtbot, 384, 220)
        ph.plot_correlation_heatmap(panel, square(ELEMENTS[:30]), LONG_TITLE)
        settle(panel)
        assert ph.overflow_report(panel.figure)           # the title of this one is cut without the net


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def _png_size(path):
    with open(path, "rb") as handle:
        head = handle.read(24)
    return struct.unpack(">II", head[16:24])


EXPORTED = ["barh_60_long", "grammar_bar_hue_60", "grammar_heatmap_60", "pca_by_farm", "rule_diagram",
            "association_heatmap_long", "sns_triangle_projections", "partial_correlation", "tank_composition_6",
            "grammar_facet_bar_long_facets", "explorer_ranked_60", "score_breakdown"]


@pytest.mark.parametrize("size", [(384, 220), (760, 520)], ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("name", EXPORTED)
def test_export_is_contained(qtbot, tmp_path, monkeypatch, name, size):
    """The exported file holds the whole chart. While it is being saved
    (at the export size and resolution) the figure shows the chart, never
    the "enlarge the window" line, and the image written is at least as
    large as everything the figure draws. From a pane of ordinary size
    nothing reaches outside the figure at all, so the image is the figure:
    its export size, less the white the save trims."""
    from qt_widgets import _PanelFigure

    panel = shown_panel(qtbot, *size)
    try:
        CASES[name](panel)
        settle(panel)
        seen = []
        original = _PanelFigure.draw

        def recording_draw(figure, renderer):
            result = original(figure, renderer)
            drawn = figure.get_tightbbox(renderer)         # inches
            seen.append({
                "dpi": figure.dpi, "report": ph.overflow_report(figure, renderer),
                "message": ph.showing_enlarge_message(figure),
                "drawn_px": (drawn.width * figure.dpi, drawn.height * figure.dpi),
            })
            return result

        monkeypatch.setattr(_PanelFigure, "draw", recording_draw)
        path = tmp_path / f"{name}.png"
        dpi = 150
        panel.export_figure(str(path), dpi=dpi)
        monkeypatch.setattr(_PanelFigure, "draw", original)

        assert seen, "the export never drew the figure"
        px_w, px_h = _png_size(path)
        export_w, export_h = panel.export_size_inches()
        pad = 2 * 0.1 * dpi + 2                            # savefig's bbox_inches="tight" padding
        for record in seen:
            assert record["dpi"] == dpi
            assert not record["message"], f"{name}: the export shows the 'enlarge the window' line"
            assert px_w >= record["drawn_px"][0] - 2 and px_h >= record["drawn_px"][1] - 2, (
                f"{name}: the image ({px_w} x {px_h}) is smaller than what was drawn {record['drawn_px']}")
            if tuple(size) not in SMALL_SIZES:
                assert not record["report"], f"{name} exported from {size}: {ph.describe_overflow(record['report'])}"
        if tuple(size) not in SMALL_SIZES:
            assert px_w <= export_w * dpi + pad and px_h <= export_h * dpi + pad
        # Back on screen at the pane's size, still contained.
        panel.canvas.draw()
        assert_contained(panel, canvas_size(panel), f"{name} after export")
    finally:
        panel.hide()


def test_a_pane_showing_the_one_line_message_still_exports_its_chart(qtbot, tmp_path):
    panel = shown_panel(qtbot, 384, 220)
    try:
        CASES["sns_pair_matrix_scatter_8"](panel)
        settle(panel)
        assert ph.showing_enlarge_message(panel.figure)
        # What the figure is for the time of a save (the hook savefig runs under).
        with panel.figure._ember_save_hook():
            assert not ph.showing_enlarge_message(panel.figure)
            assert all(ax.get_visible() for ax in panel.figure.axes)
        path = tmp_path / "small_pane.png"
        panel.export_figure(str(path), dpi=100)
        assert path.stat().st_size > 2000
        panel.canvas.draw()
        assert ph.showing_enlarge_message(panel.figure)   # the pane itself is as small as before
    finally:
        panel.hide()


def test_export_at_an_exact_size_is_contained(qtbot, tmp_path):
    panel = shown_panel(qtbot, 760, 520)
    try:
        CASES["grammar_bar_hue_60"](panel)
        settle(panel)
        path = tmp_path / "exact.png"
        panel.export_at_size_cm(str(path), 8.0, 6.0, dpi=200)
        width, height = _png_size(path)
        assert abs(width - 8.0 / 2.54 * 200) <= 1 and abs(height - 6.0 / 2.54 * 200) <= 1
        panel.canvas.draw()
        assert_contained(panel, (760, 520), "after an export at 8 x 6 cm")
    finally:
        panel.hide()


# ---------------------------------------------------------------------------
# Figure Composer
# ---------------------------------------------------------------------------
def _recipe(name):
    """The recipe a PlotWidget records for a case, as the composer replays it."""
    panel = PlotWidget()
    try:
        CASES[name](panel)
        fn, args, kwargs = panel._last_recipe
    finally:
        panel.deleteLater()
    return ComposerItem(render_fn=fn, render_args=args, render_kwargs=dict(kwargs), caption=name)


COMPOSED = {
    "two": ["barh_60_long", "pca_by_farm"],
    "two_matrices": ["association_heatmap_long", "partial_correlation"],
    "four": ["grammar_bar_hue_60", "rule_diagram", "grammar_heatmap_60", "tank_composition_3"],
    "four_dense": ["sns_dashboard", "grammar_facet_bar_60", "sns_triangle_projections", "score_breakdown"],
}


@pytest.mark.parametrize("size", [(520, 360), (760, 520), (1200, 760)], ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("layout", sorted(COMPOSED))
def test_composed_cells_hold_their_panels(qtbot, layout, size):
    panel = shown_panel(qtbot, *size)
    try:
        items = [_recipe(name) for name in COMPOSED[layout]]
        composed = fcomp.compose_figure(panel, items)
        assert composed is not None and len(panel.figure.subfigs) == len(items)
        # A draw alone keeps every cell inside its rectangle ...
        resize_figure(panel, *size)
        panel.canvas.draw()
        report = ph.overflow_report(panel.figure)
        assert not report, f"composed {layout} at {size}, drawn: {ph.describe_overflow(report)}"
        # ... and so does the layout the widget makes once a resize has settled.
        settle(panel)
        report = ph.overflow_report(panel.figure)
        assert not report, f"composed {layout} at {size}: {ph.describe_overflow(report)}"
        if len(items) == 2 and size == (1200, 760):
            assert not any(ph.showing_enlarge_message(sub) for sub in panel.figure.subfigs)
        first = [geometry_of_cell(sub) for sub in panel.figure.subfigs]
        panel.canvas.draw()
        settle(panel)
        again = [geometry_of_cell(sub) for sub in panel.figure.subfigs]
        assert all(same_layout(now, before, UNCHANGED) for now, before in zip(again, first))
        assert not ph.overflow_report(panel.figure)
    finally:
        panel.hide()


def geometry_of_cell(subfig):
    if ph.showing_enlarge_message(subfig):
        return ["the one-line message"]                   # the hidden chart's layout is nobody's business
    out = []
    for ax in subfig.axes:
        out.append(tuple(round(float(v), 4) for v in ax.get_position().bounds))
        out.append((ax.get_title(), ax.get_xlabel(), ax.get_ylabel()))
    return out


def test_a_title_override_in_a_composed_cell_survives_the_layout(qtbot):
    panel = shown_panel(qtbot, 760, 520)
    try:
        items = [_recipe("barh_60_long"), _recipe("candidate_scores")]
        composed = fcomp.compose_figure(panel, items)
        panel.canvas.draw()
        override = "A title somebody typed for this panel"
        from dataclasses import replace
        edited = replace(items[1], title_override=override)
        assert fcomp.apply_text_override(panel, composed, 1, edited, "title")
        panel.canvas.draw()
        resize_figure(panel, 1200, 760)
        settle(panel)
        assert ph.full_text(composed.cells[1].ax.title) == override
        assert not ph.overflow_report(panel.figure)
    finally:
        panel.hide()
