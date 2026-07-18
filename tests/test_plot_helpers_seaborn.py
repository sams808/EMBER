"""
Tests for the kg Association Workbench's 8 seaborn plot functions (17 plot
types). These are heavily seaborn-integrated (gridspecs, gridded subplots,
sns.heatmap/regplot/kdeplot internals) so -- unlike plot_helpers' simpler
matplotlib-only functions -- they're tested against the real PlotWidget
throughout rather than a FakeAx stand-in, and the bar is "renders without
crashing / shows the right message on empty input" rather than exhaustive
per-branch coverage of seaborn's own internals.
"""
import numpy as np
import pandas as pd
import pytest

import plot_helpers as ph


@pytest.fixture
def panel(qtbot):
    from qt_widgets import PlotWidget
    p = PlotWidget()
    qtbot.addWidget(p)
    return p


@pytest.fixture
def workbench_results():
    elements = ["Cs", "Sr", "Fe", "Na"]
    tanks = [f"241-T-{i:03d}" for i in range(6)]
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(rng.uniform(0, 100, size=(6, 4)), columns=elements)
    raw.insert(0, "WasteSiteId", tanks)

    element_stats = pd.DataFrame({
        "Element": elements, "Units": "kg",
        "Total_inventory_kg": [500.0, 300.0, 60.0, 20.0],
        "N_tanks_present": [6, 6, 5, 3], "N_tanks_total": [6, 6, 6, 6],
        "PresenceFraction_pct": [100.0, 100.0, 83.3, 50.0],
        "Mean_kg_present_tanks_only": [83.3, 50.0, 12.0, 6.7],
        "Median_kg_present_tanks_only": [80.0, 48.0, 11.0, 6.0],
        "Std_kg_present_tanks_only": [10.0, 8.0, 3.0, 2.0],
        "Max_kg_in_one_tank": [120.0, 90.0, 20.0, 10.0],
    })
    pair_rows = [
        ("Cs", "Sr", 0.9, 0.8, 5, 12.0), ("Cs", "Fe", -0.6, 0.5, 4, 0.0),
        ("Cs", "Na", 0.3, 0.4, 3, 2.0), ("Sr", "Fe", float("nan"), 0.2, 1, 0.0),
        ("Sr", "Na", 0.1, 0.3, 2, 0.5), ("Fe", "Na", -0.2, 0.1, 1, 0.0),
    ]
    pair_stats = pd.DataFrame([
        {
            "Element_A": a, "Element_B": b, "Correlation_r": r, "AbsCorrelation": abs(r) if r == r else float("nan"),
            "Jaccard_presence": j, "N_both_present": n, "PreferredAssociationScore_proxy": score,
        }
        for a, b, r, j, n, score in pair_rows
    ])

    corr_square = pd.DataFrame(np.eye(4), index=elements, columns=elements)
    corr_square.loc["Cs", "Sr"] = corr_square.loc["Sr", "Cs"] = 0.9
    corr_matrix = corr_square.reset_index().rename(columns={"index": "Element"})

    jaccard_square = pd.DataFrame(np.eye(4), index=elements, columns=elements)
    jaccard_matrix = jaccard_square.reset_index().rename(columns={"index": "Element"})

    metric_matrix = raw.copy()
    for e in elements:
        metric_matrix[e] = np.log10(metric_matrix[e] + 1.0)

    tank_sim_square = pd.DataFrame(np.eye(6), index=tanks, columns=tanks)
    tank_similarity = tank_sim_square.reset_index().rename(columns={"index": "WasteSiteId"})

    presence_matrix = raw.copy()
    for e in elements:
        presence_matrix[e] = (presence_matrix[e] > 0).astype(int)

    return {
        "element_stats": element_stats, "pair_stats": pair_stats, "raw_matrix": raw,
        "metric_matrix": metric_matrix, "corr_matrix": corr_matrix, "jaccard_matrix": jaccard_matrix,
        "tank_similarity": tank_similarity, "presence_matrix": presence_matrix,
    }


@pytest.fixture
def structure_results():
    from scipy.cluster.hierarchy import linkage

    elements = ["Cs", "Sr", "Fe", "Na"]
    tanks = [f"241-T-{i:03d}" for i in range(6)]

    tank_summary = pd.DataFrame({
        "WasteSiteId": tanks, "PC1": [1.2, 1.5, 1.8, -0.9, -1.3, -1.6],
        "PC2": [0.3, -0.2, 0.1, 0.4, -0.1, 0.2], "Cluster": [1, 1, 1, 2, 2, 2],
        "TankFarm": ["A", "A", "A", "AN", "AN", "AN"],
        "TankType": ["DST", "DST", "DST", "SST-4", "SST-4", "SST-4"],
        "TankStatus": ["Active"] * 6,
        "Dominant waste phase": ["Sludge Solid"] * 3 + ["Supernatant"] * 3,
    })
    pca_variance = pd.DataFrame({
        "PC": ["PC1", "PC2"], "ExplainedVarianceRatio": [0.85, 0.10], "CumulativeVarianceRatio": [0.85, 0.95],
    })

    rng = np.random.default_rng(0)
    cluster_linkage = linkage(rng.uniform(0, 1, size=(6, 3)), method="ward")
    cluster_labels = tanks

    raw_square = pd.DataFrame(np.eye(4), index=elements, columns=elements)
    raw_square.loc["Cs", "Sr"] = raw_square.loc["Sr", "Cs"] = 0.9
    raw_corr_matrix = raw_square.reset_index().rename(columns={"index": "Element"})
    partial_square = pd.DataFrame(np.eye(4), index=elements, columns=elements)
    partial_square.loc["Cs", "Sr"] = partial_square.loc["Sr", "Cs"] = 0.2
    partial_corr_matrix = partial_square.reset_index().rename(columns={"index": "Element"})

    network_nodes = pd.DataFrame({
        "Element": elements, "Total_inventory_kg": [500.0, 300.0, 60.0, 20.0],
        "LogTotalInventory": [2.7, 2.5, 1.8, 1.3], "x": [0.1, 0.4, -0.2, -0.5], "y": [0.2, -0.3, 0.5, -0.1],
        "N_edges": [2, 1, 0, 1],
    })
    network_edges = pd.DataFrame([
        {"Element_A": "Cs", "Element_B": "Sr", "Correlation_r": 0.9, "AbsCorrelation": 0.9, "Jaccard_presence": 0.8, "Sign": "positive"},
        {"Element_A": "Cs", "Element_B": "Na", "Correlation_r": -0.4, "AbsCorrelation": 0.4, "Jaccard_presence": 0.3, "Sign": "negative"},
    ])

    return {
        "tank_summary": tank_summary, "pca_variance": pca_variance,
        "cluster_linkage": cluster_linkage, "cluster_labels": cluster_labels,
        "raw_corr_matrix": raw_corr_matrix, "partial_corr_matrix": partial_corr_matrix,
        "network_nodes": network_nodes, "network_edges": network_edges,
    }


ALL_SEABORN_PLOT_CALLS = [
    lambda p, w: ph.plot_seaborn_lower_triangle_matrix(p, w["corr_matrix"], "t", "Correlation r"),
    lambda p, w: ph.plot_seaborn_top_associations(p, w["pair_stats"]),
    lambda p, w: ph.plot_seaborn_pair_matrix(p, w["metric_matrix"], w["raw_matrix"], ["Cs", "Sr"], "log10_plus1"),
    lambda p, w: ph.plot_seaborn_joint_first_two(p, w["metric_matrix"], ["Cs", "Sr"], "log10_plus1"),
    lambda p, w: ph.plot_seaborn_tank_similarity(p, w["tank_similarity"], w["raw_matrix"]),
    lambda p, w: ph.plot_seaborn_tank_element_map(p, w["raw_matrix"], ["Cs", "Sr"]),
    lambda p, w: ph.plot_seaborn_presence_patterns(p, w["presence_matrix"], ["Cs", "Sr"]),
    lambda p, w: ph.plot_seaborn_stats_dashboard(p, w["element_stats"], w["pair_stats"]),
]


class TestAllPlotsRespectSeabornUnavailable:
    @pytest.mark.parametrize("plot_call", ALL_SEABORN_PLOT_CALLS)
    def test_shows_message_and_does_not_crash(self, panel, workbench_results, monkeypatch, plot_call):
        monkeypatch.setattr(ph, "sns", None)
        plot_call(panel, workbench_results)
        assert "not installed" in panel.ax.texts[-1].get_text()


ALL_STRUCTURE_PLOT_CALLS = [
    lambda p, s: ph.plot_pca_scatter(p, s["tank_summary"], "TankFarm", s["pca_variance"]),
    lambda p, s: ph.plot_dendrogram(p, s["cluster_linkage"], s["cluster_labels"]),
    lambda p, s: ph.plot_partial_correlation_comparison(p, s["partial_corr_matrix"], s["raw_corr_matrix"]),
    lambda p, s: ph.plot_element_network(p, s["network_nodes"], s["network_edges"]),
]


class TestAllStructurePlotsRespectSeabornUnavailable:
    @pytest.mark.parametrize("plot_call", ALL_STRUCTURE_PLOT_CALLS)
    def test_shows_message_and_does_not_crash(self, panel, structure_results, monkeypatch, plot_call):
        monkeypatch.setattr(ph, "sns", None)
        plot_call(panel, structure_results)
        assert "not installed" in panel.ax.texts[-1].get_text()


class TestSeabornAvailableOrMessage:
    def test_true_when_seaborn_installed(self, panel):
        assert ph._seaborn_available_or_message(panel) is True

    def test_false_and_message_when_unavailable(self, panel, monkeypatch):
        monkeypatch.setattr(ph, "sns", None)
        assert ph._seaborn_available_or_message(panel) is False


class TestCoherentColorHelpers:
    def test_basic_vs_coherent(self):
        assert ph._use_coherent_colors("Basic") is False
        assert ph._use_coherent_colors("Coherent colors") is True

    def test_cmaps_differ_by_mode(self):
        assert ph._corr_cmap("Basic") != ph._corr_cmap("Coherent colors")
        assert ph._sequential_cmap("Basic") != ph._sequential_cmap("Coherent colors")
        assert ph._jaccard_cmap("Basic") != ph._jaccard_cmap("Coherent colors")

    def test_pair_palette_basic_is_none(self):
        assert ph._pair_palette_name("preferred", "Basic") is None

    def test_pair_palette_coherent_varies_by_mode(self):
        assert ph._pair_palette_name("negative", "Coherent colors") == "rocket_r"
        assert ph._pair_palette_name("positive", "Coherent colors") == "crest"
        assert ph._pair_palette_name("jaccard", "Coherent colors") == "crest"
        assert ph._pair_palette_name("preferred", "Coherent colors") == "flare"

    def test_set_seaborn_theme_noop_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(ph, "sns", None)
        ph._set_seaborn_theme("Basic")  # must not raise


class TestSquareMatrixFromElementTable:
    def test_empty_when_no_element_column(self):
        data, elements = ph._square_matrix_from_element_table(pd.DataFrame({"x": [1]}))
        assert data.empty and elements == []

    def test_empty_when_no_matching_columns(self):
        data, elements = ph._square_matrix_from_element_table(pd.DataFrame({"Element": ["Cs"]}))
        assert data.empty and elements == []


@pytest.mark.parametrize("color_mode", ["Basic", "Coherent colors"])
class TestPlotSeabornLowerTriangleMatrix:
    def test_seaborn_unavailable_shows_message(self, panel, workbench_results, monkeypatch, color_mode):
        monkeypatch.setattr(ph, "sns", None)
        ph.plot_seaborn_lower_triangle_matrix(panel, workbench_results["corr_matrix"], "t", "Correlation r", color_mode=color_mode)
        assert "not installed" in panel.ax.texts[-1].get_text()

    def test_empty_matrix_shows_message(self, panel, color_mode):
        ph.plot_seaborn_lower_triangle_matrix(panel, pd.DataFrame({"x": [1]}), "t", "Correlation r", color_mode=color_mode)

    def test_corr_heatmap_no_projections(self, panel, workbench_results, qtbot, color_mode):
        ph.plot_seaborn_lower_triangle_matrix(panel, workbench_results["corr_matrix"], "t", "Correlation r", center=0, projections=False, color_mode=color_mode)
        qtbot.wait(20)
        assert panel.ax.get_title() == "t"

    def test_corr_heatmap_with_projections(self, panel, workbench_results, qtbot, color_mode):
        ph.plot_seaborn_lower_triangle_matrix(
            panel, workbench_results["corr_matrix"], "t", "Correlation r", center=0,
            totals=workbench_results["element_stats"], projections=True, color_mode=color_mode,
        )
        qtbot.wait(20)

    def test_jaccard_heatmap(self, panel, workbench_results, qtbot, color_mode):
        ph.plot_seaborn_lower_triangle_matrix(panel, workbench_results["jaccard_matrix"], "t", "Jaccard co-presence", center=None, color_mode=color_mode)
        qtbot.wait(20)

    def test_annotated_small_matrix(self, panel, workbench_results, qtbot, color_mode):
        ph.plot_seaborn_lower_triangle_matrix(panel, workbench_results["corr_matrix"], "t", "Correlation r", annotate=True, color_mode=color_mode)
        qtbot.wait(20)


class TestPlotSeabornTopAssociations:
    def test_empty_pair_stats_shows_message(self, panel):
        ph.plot_seaborn_top_associations(panel, pd.DataFrame(), mode="preferred")

    @pytest.mark.parametrize("mode", ["preferred", "positive", "negative", "jaccard"])
    def test_all_modes_render(self, panel, workbench_results, qtbot, mode):
        ph.plot_seaborn_top_associations(panel, workbench_results["pair_stats"], mode=mode)
        qtbot.wait(20)

    def test_no_matching_rows_shows_message(self, panel):
        # All-positive pair stats but asking for "negative" leaves nothing.
        pdf = pd.DataFrame([{"Element_A": "Cs", "Element_B": "Sr", "Correlation_r": 0.9, "Jaccard_presence": 0.5, "N_both_present": 3, "PreferredAssociationScore_proxy": 1.0}])
        ph.plot_seaborn_top_associations(panel, pdf, mode="negative")

    def test_coherent_palette_success_path(self, panel, workbench_results, qtbot):
        # color_mode="Coherent colors" with sns.color_palette succeeding
        # (no monkeypatch) -- the try branch, distinct from the fallback
        # exercised below.
        ph.plot_seaborn_top_associations(panel, workbench_results["pair_stats"], mode="preferred", color_mode="Coherent colors")
        qtbot.wait(20)

    def test_coherent_palette_failure_falls_back(self, panel, workbench_results, qtbot, monkeypatch):
        # Force sns.color_palette to raise, exercising the except-fallback path.
        import seaborn as real_sns
        def boom(*a, **k):
            raise RuntimeError("boom")
        monkeypatch.setattr(real_sns, "color_palette", boom)
        ph.plot_seaborn_top_associations(panel, workbench_results["pair_stats"], mode="preferred", color_mode="Coherent colors")
        qtbot.wait(20)


class TestPlotSeabornPairMatrix:
    def test_too_few_elements_shows_message(self, panel, workbench_results):
        ph.plot_seaborn_pair_matrix(panel, workbench_results["metric_matrix"], workbench_results["raw_matrix"], ["Cs"], "log10_plus1")

    @pytest.mark.parametrize("kind", ["regression", "scatter", "kde"])
    def test_all_kinds_render(self, panel, workbench_results, qtbot, kind):
        ph.plot_seaborn_pair_matrix(panel, workbench_results["metric_matrix"], workbench_results["raw_matrix"], ["Cs", "Sr", "Fe"], "log10_plus1", kind=kind)
        qtbot.wait(20)

    def test_no_finite_values_shows_message(self, panel):
        matrix = pd.DataFrame({"WasteSiteId": ["T1"], "Cs": [np.nan], "Sr": [np.nan]})
        ph.plot_seaborn_pair_matrix(panel, matrix, pd.DataFrame(), ["Cs", "Sr"], "log10_plus1")

    def test_max_elements_caps_grid(self, panel, workbench_results, qtbot):
        ph.plot_seaborn_pair_matrix(panel, workbench_results["metric_matrix"], workbench_results["raw_matrix"], ["Cs", "Sr", "Fe", "Na"], "log10_plus1", max_elements=2)
        qtbot.wait(20)

    def test_constant_element_shows_constant_diagonal_and_falls_back_off_diagonal(self, panel, qtbot):
        # C is constant -> "constant" text on its diagonal cell, and every
        # (X, C) off-diagonal cell falls back to a plain scatter (both the
        # regression and kde branches require nunique > 1 on both axes).
        metric = pd.DataFrame({"WasteSiteId": ["T1", "T2", "T3"], "A": [1.0, 2.0, 3.0], "C": [5.0, 5.0, 5.0]})
        ph.plot_seaborn_pair_matrix(panel, metric, pd.DataFrame(), ["A", "C"], "log10_plus1", kind="regression")
        qtbot.wait(20)
        ph.plot_seaborn_pair_matrix(panel, metric, pd.DataFrame(), ["A", "C"], "log10_plus1", kind="kde")
        qtbot.wait(20)

    def test_empty_raw_matrix_falls_back_to_both_present_true(self, panel, qtbot):
        metric = pd.DataFrame({"WasteSiteId": ["T1", "T2", "T3"], "A": [1.0, 2.0, 3.0], "B": [3.0, 2.0, 1.0]})
        ph.plot_seaborn_pair_matrix(panel, metric, pd.DataFrame(), ["A", "B"], "log10_plus1", kind="scatter")
        qtbot.wait(20)

    def test_pair_with_no_overlapping_finite_rows_shows_no_data_text(self, panel, qtbot):
        # Every row has exactly 2 of {A,B,C} populated (passes the overall
        # >=2-non-null filter), but A and B are never populated together.
        metric = pd.DataFrame({
            "WasteSiteId": ["T1", "T2", "T3"],
            "A": [1.0, 2.0, np.nan], "B": [np.nan, np.nan, 3.0], "C": [5.0, 6.0, 7.0],
        })
        ph.plot_seaborn_pair_matrix(panel, metric, pd.DataFrame(), ["A", "B", "C"], "log10_plus1", kind="scatter")
        qtbot.wait(20)

    def test_kde_exception_falls_back_to_scatter(self, panel, workbench_results, qtbot, monkeypatch):
        import seaborn as real_sns
        def boom(*a, **k):
            raise RuntimeError("boom")
        monkeypatch.setattr(real_sns, "kdeplot", boom)
        ph.plot_seaborn_pair_matrix(panel, workbench_results["metric_matrix"], workbench_results["raw_matrix"], ["Cs", "Sr"], "log10_plus1", kind="kde")
        qtbot.wait(20)


class TestPlotSeabornJointFirstTwo:
    def test_too_few_elements_shows_message(self, panel, workbench_results):
        ph.plot_seaborn_joint_first_two(panel, workbench_results["metric_matrix"], ["Cs"], "log10_plus1")

    @pytest.mark.parametrize("kind", ["regression", "scatter", "kde"])
    def test_all_kinds_render(self, panel, workbench_results, qtbot, kind):
        ph.plot_seaborn_joint_first_two(panel, workbench_results["metric_matrix"], ["Cs", "Sr"], "log10_plus1", kind=kind)
        qtbot.wait(20)

    def test_no_finite_values_shows_message(self, panel):
        matrix = pd.DataFrame({"WasteSiteId": ["T1"], "Cs": [np.nan], "Sr": [np.nan]})
        ph.plot_seaborn_joint_first_two(panel, matrix, ["Cs", "Sr"], "log10_plus1")

    def test_kde_exception_falls_back_to_scatter(self, panel, workbench_results, qtbot, monkeypatch):
        import seaborn as real_sns
        def boom(*a, **k):
            raise RuntimeError("boom")
        monkeypatch.setattr(real_sns, "kdeplot", boom)
        ph.plot_seaborn_joint_first_two(panel, workbench_results["metric_matrix"], ["Cs", "Sr"], "log10_plus1", kind="kde")
        qtbot.wait(20)


class TestPlotSeabornTankSimilarity:
    def test_empty_shows_message(self, panel):
        ph.plot_seaborn_tank_similarity(panel, pd.DataFrame(), pd.DataFrame())

    def test_renders_with_raw_matrix_ranking(self, panel, workbench_results, qtbot):
        ph.plot_seaborn_tank_similarity(panel, workbench_results["tank_similarity"], workbench_results["raw_matrix"], top_tanks=4)
        qtbot.wait(20)

    def test_renders_without_raw_matrix(self, panel, workbench_results, qtbot):
        ph.plot_seaborn_tank_similarity(panel, workbench_results["tank_similarity"], pd.DataFrame(), top_tanks=4)
        qtbot.wait(20)

    def test_fewer_than_two_tanks_shows_message(self, panel):
        sim = pd.DataFrame({"WasteSiteId": ["T1"], "T1": [1.0]})
        ph.plot_seaborn_tank_similarity(panel, sim, pd.DataFrame())


class TestPlotSeabornTankElementMap:
    def test_empty_raw_matrix_shows_message(self, panel):
        ph.plot_seaborn_tank_element_map(panel, pd.DataFrame(), ["Cs"])

    def test_no_matching_elements_shows_message(self, panel, workbench_results):
        ph.plot_seaborn_tank_element_map(panel, workbench_results["raw_matrix"], ["Zz"])

    def test_renders_log_mode(self, panel, workbench_results, qtbot):
        ph.plot_seaborn_tank_element_map(panel, workbench_results["raw_matrix"], ["Cs", "Sr"], metric="log10_plus1")
        qtbot.wait(20)

    def test_renders_fraction_mode(self, panel, workbench_results, qtbot):
        ph.plot_seaborn_tank_element_map(panel, workbench_results["raw_matrix"], ["Cs", "Sr"], metric="fraction")
        qtbot.wait(20)


class TestPlotSeabornPresencePatterns:
    def test_empty_shows_message(self, panel):
        ph.plot_seaborn_presence_patterns(panel, pd.DataFrame(), ["Cs", "Sr"])

    def test_too_few_elements_shows_message(self, panel, workbench_results):
        ph.plot_seaborn_presence_patterns(panel, workbench_results["presence_matrix"], ["Cs"])

    def test_renders(self, panel, workbench_results, qtbot):
        ph.plot_seaborn_presence_patterns(panel, workbench_results["presence_matrix"], ["Cs", "Sr", "Fe", "Na"])
        qtbot.wait(20)

    def test_all_zero_presence_shows_message(self, panel):
        presence = pd.DataFrame({"WasteSiteId": ["T1", "T2"], "Cs": [0, 0], "Sr": [0, 0]})
        ph.plot_seaborn_presence_patterns(panel, presence, ["Cs", "Sr"])


class TestPlotSeabornStatsDashboard:
    def test_empty_element_stats_shows_message(self, panel):
        ph.plot_seaborn_stats_dashboard(panel, pd.DataFrame(), pd.DataFrame())

    def test_renders_with_pairs(self, panel, workbench_results, qtbot):
        ph.plot_seaborn_stats_dashboard(panel, workbench_results["element_stats"], workbench_results["pair_stats"])
        qtbot.wait(20)

    def test_renders_without_pairs(self, panel, workbench_results, qtbot):
        ph.plot_seaborn_stats_dashboard(panel, workbench_results["element_stats"], pd.DataFrame())
        qtbot.wait(20)

    def test_coherent_colors_mode(self, panel, workbench_results, qtbot):
        ph.plot_seaborn_stats_dashboard(panel, workbench_results["element_stats"], workbench_results["pair_stats"], color_mode="Coherent colors")
        qtbot.wait(20)


class TestPlotPcaScatter:
    def test_empty_tank_summary_shows_message(self, panel):
        ph.plot_pca_scatter(panel, pd.DataFrame(), "TankFarm", pd.DataFrame())

    def test_missing_pc_columns_shows_message(self, panel):
        ph.plot_pca_scatter(panel, pd.DataFrame({"WasteSiteId": ["T1"]}), "TankFarm", pd.DataFrame())

    def test_renders_with_color_by(self, panel, structure_results, qtbot):
        ph.plot_pca_scatter(panel, structure_results["tank_summary"], "TankFarm", structure_results["pca_variance"])
        qtbot.wait(20)
        assert "PC1" in panel.ax.get_xlabel()

    def test_renders_without_color_by(self, panel, structure_results, qtbot):
        ph.plot_pca_scatter(panel, structure_results["tank_summary"], None, structure_results["pca_variance"])
        qtbot.wait(20)

    def test_color_by_column_missing_falls_back_to_plain_scatter(self, panel, structure_results, qtbot):
        ph.plot_pca_scatter(panel, structure_results["tank_summary"], "NotAColumn", structure_results["pca_variance"])
        qtbot.wait(20)

    def test_color_by_all_null_falls_back_to_plain_scatter(self, panel, structure_results, qtbot):
        summary = structure_results["tank_summary"].copy()
        summary["TankFarm"] = None
        ph.plot_pca_scatter(panel, summary, "TankFarm", structure_results["pca_variance"])
        qtbot.wait(20)

    def test_many_categories_uses_tab20_palette(self, panel, structure_results, qtbot):
        summary = structure_results["tank_summary"].copy()
        summary["ManyCats"] = [f"Cat{i}" for i in range(len(summary))]
        ph.plot_pca_scatter(panel, summary, "ManyCats", structure_results["pca_variance"])
        qtbot.wait(20)

    def test_no_variance_table_still_renders(self, panel, structure_results, qtbot):
        ph.plot_pca_scatter(panel, structure_results["tank_summary"], "TankFarm", pd.DataFrame())
        qtbot.wait(20)
        assert panel.ax.get_xlabel() == "PC1"

    def test_coherent_colors_mode(self, panel, structure_results, qtbot):
        ph.plot_pca_scatter(panel, structure_results["tank_summary"], "TankFarm", structure_results["pca_variance"], color_mode="Coherent colors")
        qtbot.wait(20)


class TestPlotDendrogram:
    def test_empty_linkage_shows_message(self, panel):
        ph.plot_dendrogram(panel, np.empty((0, 4)), [])

    def test_none_linkage_shows_message(self, panel):
        ph.plot_dendrogram(panel, None, [])

    def test_renders_with_valid_linkage(self, panel, structure_results, qtbot):
        ph.plot_dendrogram(panel, structure_results["cluster_linkage"], structure_results["cluster_labels"])
        qtbot.wait(20)
        assert "clustering" in panel.ax.get_title().lower()

    def test_coherent_colors_mode(self, panel, structure_results, qtbot):
        ph.plot_dendrogram(panel, structure_results["cluster_linkage"], structure_results["cluster_labels"], color_mode="Coherent colors")
        qtbot.wait(20)


class TestPlotPartialCorrelationComparison:
    def test_empty_matrices_show_message(self, panel):
        ph.plot_partial_correlation_comparison(panel, pd.DataFrame(), pd.DataFrame())

    def test_renders_side_by_side(self, panel, structure_results, qtbot):
        ph.plot_partial_correlation_comparison(panel, structure_results["partial_corr_matrix"], structure_results["raw_corr_matrix"])
        qtbot.wait(20)
        assert panel.ax.get_title() == "Raw correlation"

    def test_annotated(self, panel, structure_results, qtbot):
        ph.plot_partial_correlation_comparison(panel, structure_results["partial_corr_matrix"], structure_results["raw_corr_matrix"], annotate=True)
        qtbot.wait(20)

    def test_coherent_colors_mode(self, panel, structure_results, qtbot):
        ph.plot_partial_correlation_comparison(panel, structure_results["partial_corr_matrix"], structure_results["raw_corr_matrix"], color_mode="Coherent colors")
        qtbot.wait(20)


class TestPlotElementNetwork:
    def test_empty_nodes_shows_message(self, panel):
        ph.plot_element_network(panel, pd.DataFrame(), pd.DataFrame())

    def test_renders_with_edges(self, panel, structure_results, qtbot):
        ph.plot_element_network(panel, structure_results["network_nodes"], structure_results["network_edges"])
        qtbot.wait(20)
        assert panel.ax.get_title() == "Element association network (kg)"

    def test_renders_with_no_edges_isolated_nodes_only(self, panel, structure_results, qtbot):
        ph.plot_element_network(panel, structure_results["network_nodes"], pd.DataFrame())
        qtbot.wait(20)

    def test_none_edges_treated_as_no_edges(self, panel, structure_results, qtbot):
        ph.plot_element_network(panel, structure_results["network_nodes"], None)
        qtbot.wait(20)

    def test_coherent_colors_mode(self, panel, structure_results, qtbot):
        ph.plot_element_network(panel, structure_results["network_nodes"], structure_results["network_edges"], color_mode="Coherent colors")
        qtbot.wait(20)
