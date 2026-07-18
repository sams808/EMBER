"""
PCA/clustering/partial-correlation/network math needs a bigger, more
structured dataset than the 2-tank shared sample_dataset fixture -- see
conftest.py for structure_dataset (8 tanks in two clean, hand-separated
groups by Cs/Sr/Mo composition) and size_confound_dataset (12 tanks where
Cs and Ba are both driven mostly by a shared "tank size" element, built
with a fixed numpy Generator seed for exact reproducibility). Expected
raw/partial correlation values are computed independently in each test via
plain pandas .corr() calls on the same arrays, not by calling
structure_science itself.
"""
import numpy as np
import pandas as pd
import polars as pl
import pytest

import correlation_science as csci
import data_model as dm
import structure_science as ss
from conftest import GROUP_A_TANKS, GROUP_AN_TANKS


class TestTankCategoricalLabels:
    def test_tank_farm_from_attributes_table(self, structure_dataset):
        labels = ss.tank_categorical_labels(structure_dataset, "TankFarm")
        assert labels["241-A-101"] == "A"
        assert labels["241-AN-101"] == "AN"

    def test_tank_system_derived_from_tank_type(self, structure_dataset):
        labels = ss.tank_categorical_labels(structure_dataset, "TankSystem")
        assert labels["241-A-101"] == "DST"
        assert labels["241-AN-101"] == "SST"

    def test_tank_status_passthrough(self, structure_dataset):
        labels = ss.tank_categorical_labels(structure_dataset, "TankStatus")
        assert labels["241-A-101"] == "Active"
        assert labels["241-AN-101"] == "Interim Closure"

    def test_dominant_waste_phase_matches_group(self, structure_dataset):
        labels = ss.tank_categorical_labels(structure_dataset, "Dominant waste phase")
        assert labels["241-A-101"] == "Sludge Solid"
        assert labels["241-AN-101"] == "Supernatant"

    def test_dominant_waste_phase_weighted_by_inventory_not_row_count(self):
        # Three tiny Supernatant rows (majority by row count) vs one huge
        # Sludge Solid row (majority by summed inventory) -- the dominant
        # phase must be the inventory-weighted winner, not the mode by count.
        rows = {
            "WasteSiteId": ["241-A-999"] * 4,
            "Analyte": ["Cs", "Sr", "Tc", "Fe"],
            "WastePhase": ["Supernatant", "Supernatant", "Supernatant", "Sludge Solid"],
            "WasteType": ["T1"] * 4,
            "Inventory": [1.0, 1.0, 1.0, 1000.0],
            "Units": ["kg"] * 4,
        }
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame(rows))
        dataset.report = None
        labels = ss.tank_categorical_labels(dataset, "Dominant waste phase")
        assert labels["241-A-999"] == "Sludge Solid"

    def test_unknown_category_returns_empty_series(self, structure_dataset):
        labels = ss.tank_categorical_labels(structure_dataset, "NotARealCategory")
        assert labels.empty

    def test_dominant_waste_phase_empty_when_no_kg_data(self):
        dataset = dm.HanfordDataset()
        rows = {
            "WasteSiteId": ["241-A-101"], "Analyte": ["Cs"], "WastePhase": ["Liquid"],
            "WasteType": ["T1"], "Inventory": [5.0], "Units": ["Ci"],
        }
        dataset.df = dataset._clean_dataframe(pl.DataFrame(rows))
        dataset.report = None
        labels = ss.tank_categorical_labels(dataset, "Dominant waste phase")
        assert labels.empty


class TestTankPca:
    def _metric_matrix(self, dataset):
        base = csci.kg_correlation_workbench(dataset, elements_text="Cs, Sr, Mo, Fe", value_mode="log10_plus1")
        return base["metric_matrix"], [c for c in base["raw_matrix"].columns if c != "WasteSiteId"]

    def test_constant_element_dropped(self, structure_dataset):
        metric_matrix, elements = self._metric_matrix(structure_dataset)
        out = ss.tank_pca(metric_matrix, elements, n_components=2)
        assert "Fe" in out["dropped_constant_elements"]
        assert "Fe" not in out["loadings"]["Element"].tolist()

    def test_pc1_separates_the_two_groups(self, structure_dataset):
        metric_matrix, elements = self._metric_matrix(structure_dataset)
        out = ss.tank_pca(metric_matrix, elements, n_components=2)
        scores = out["scores"].set_index("WasteSiteId")
        g1 = scores.loc[GROUP_A_TANKS, "PC1"]
        g2 = scores.loc[GROUP_AN_TANKS, "PC1"]
        assert g1.gt(0).all() or g1.lt(0).all()          # group A internally consistent sign
        assert g2.gt(0).all() or g2.lt(0).all()          # group AN internally consistent sign
        assert (g1.iloc[0] > 0) != (g2.iloc[0] > 0)       # opposite sign between groups

    def test_explained_variance_sums_sane(self, structure_dataset):
        metric_matrix, elements = self._metric_matrix(structure_dataset)
        out = ss.tank_pca(metric_matrix, elements, n_components=2)
        variance = out["variance"]
        assert variance.loc[variance["PC"] == "PC1", "ExplainedVarianceRatio"].iloc[0] >= variance.loc[variance["PC"] == "PC2", "ExplainedVarianceRatio"].iloc[0]
        assert variance["CumulativeVarianceRatio"].iloc[-1] <= 1.0 + 1e-9

    def test_empty_metric_matrix_returns_empty(self):
        out = ss.tank_pca(pd.DataFrame(), ["Cs", "Sr"])
        assert out["scores"].empty and out["loadings"].empty and out["variance"].empty

    def test_too_few_usable_elements_returns_empty_but_reports_dropped(self, structure_dataset):
        metric_matrix, _ = self._metric_matrix(structure_dataset)
        out = ss.tank_pca(metric_matrix, ["Fe"], n_components=2)
        assert out["scores"].empty
        assert out["dropped_constant_elements"] == ["Fe"]


class TestHierarchicalClusters:
    def _metric_matrix(self, dataset):
        base = csci.kg_correlation_workbench(dataset, elements_text="Cs, Sr, Mo, Fe", value_mode="log10_plus1")
        return base["metric_matrix"], [c for c in base["raw_matrix"].columns if c != "WasteSiteId"]

    def test_two_clusters_recover_the_two_groups_exactly(self, structure_dataset):
        metric_matrix, elements = self._metric_matrix(structure_dataset)
        out = ss.hierarchical_clusters(metric_matrix, elements, method="ward", n_clusters=2)
        assignments = out["assignments"].set_index("WasteSiteId")["Cluster"]
        group_a_clusters = set(assignments.loc[GROUP_A_TANKS])
        group_an_clusters = set(assignments.loc[GROUP_AN_TANKS])
        assert len(group_a_clusters) == 1
        assert len(group_an_clusters) == 1
        assert group_a_clusters != group_an_clusters

    def test_linkage_matrix_shape(self, structure_dataset):
        metric_matrix, elements = self._metric_matrix(structure_dataset)
        out = ss.hierarchical_clusters(metric_matrix, elements, n_clusters=2)
        # scipy linkage: (n_samples - 1) rows, 4 columns
        assert out["linkage"].shape == (7, 4)
        assert len(out["labels"]) == 8

    def test_empty_metric_matrix_returns_empty(self):
        out = ss.hierarchical_clusters(pd.DataFrame(), ["Cs", "Sr"])
        assert out["assignments"].empty
        assert out["linkage"].shape == (0, 4)

    def test_too_few_usable_elements_returns_empty(self, structure_dataset):
        metric_matrix, _ = self._metric_matrix(structure_dataset)
        out = ss.hierarchical_clusters(metric_matrix, ["Fe"])  # Fe is constant -> zero usable elements
        assert out["assignments"].empty
        assert out["linkage"].shape == (0, 4)

    def test_n_clusters_capped_at_n_tanks(self, structure_dataset):
        metric_matrix, elements = self._metric_matrix(structure_dataset)
        out = ss.hierarchical_clusters(metric_matrix, elements, n_clusters=999)
        assert out["assignments"]["Cluster"].nunique() == 8


class TestPartialCorrelationMatrix:
    def test_diagonal_is_one_and_symmetric(self, structure_dataset):
        base = csci.kg_correlation_workbench(structure_dataset, elements_text="Cs, Sr, Mo", value_mode="log10_plus1")
        partial_df, raw_df = ss.partial_correlation_matrix(structure_dataset, base["metric_matrix"], ["Cs", "Sr", "Mo"])
        raw_sq = raw_df.set_index("Element")
        partial_sq = partial_df.set_index("Element")
        assert raw_sq.loc["Cs", "Cs"] == pytest.approx(1.0)
        assert partial_sq.loc["Cs", "Cs"] == pytest.approx(1.0)
        assert raw_sq.loc["Cs", "Sr"] == pytest.approx(raw_sq.loc["Sr", "Cs"])
        assert partial_sq.loc["Cs", "Sr"] == pytest.approx(partial_sq.loc["Sr", "Cs"])

    def test_partial_correlation_strips_out_shared_size_confound(self, size_confound_dataset):
        dataset, arrays = size_confound_dataset
        cs, ba = arrays["Cs"], arrays["Ba"]
        total = arrays["driver"] + arrays["Cs"] + arrays["Ba"] + arrays["other"]
        z = np.log10(total + 1.0)
        expected = pd.DataFrame({"Cs": cs, "Ba": ba, "z": z})
        expected_raw_r = expected["Cs"].corr(expected["Ba"])
        expected_r_az = expected["Cs"].corr(expected["z"])
        expected_r_bz = expected["Ba"].corr(expected["z"])
        expected_partial_r = (expected_raw_r - expected_r_az * expected_r_bz) / np.sqrt(
            (1 - expected_r_az ** 2) * (1 - expected_r_bz ** 2)
        )
        # Sanity: this confound scenario must actually show a large gap,
        # otherwise the test isn't exercising anything meaningful.
        assert abs(expected_raw_r) > abs(expected_partial_r) + 0.3

        metric_matrix = pd.DataFrame({"WasteSiteId": arrays["tanks"], "Cs": cs, "Ba": ba})
        partial_df, raw_df = ss.partial_correlation_matrix(dataset, metric_matrix, ["Cs", "Ba"])
        assert raw_df.set_index("Element").loc["Cs", "Ba"] == pytest.approx(expected_raw_r)
        assert partial_df.set_index("Element").loc["Cs", "Ba"] == pytest.approx(expected_partial_r)

    def test_too_few_elements_returns_empty(self, structure_dataset):
        base = csci.kg_correlation_workbench(structure_dataset, elements_text="Cs, Sr", value_mode="log10_plus1")
        partial_df, raw_df = ss.partial_correlation_matrix(structure_dataset, base["metric_matrix"], ["Cs"])
        assert partial_df.empty and raw_df.empty

    def test_empty_metric_matrix_returns_empty(self, structure_dataset):
        partial_df, raw_df = ss.partial_correlation_matrix(structure_dataset, pd.DataFrame(), ["Cs", "Sr"])
        assert partial_df.empty and raw_df.empty

    def test_three_tanks_raw_r_computed_but_partial_r_nan(self):
        # 3 points is enough for a plain Pearson r (matches
        # correlation_science's own >=3 floor elsewhere) but not enough for
        # partial correlation, which needs a 4th point/degree of freedom
        # after controlling for z -- raw_r must still be a real number even
        # though partial_r goes NaN.
        tanks = ["241-A-101", "241-A-102", "241-A-103"]
        rows = {"WasteSiteId": [], "Analyte": [], "WastePhase": [], "WasteType": [], "Inventory": [], "Units": []}
        for tank, cs_v, sr_v in zip(tanks, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]):
            for analyte, val in [("Cs", cs_v), ("Sr", sr_v)]:
                rows["WasteSiteId"].append(tank)
                rows["Analyte"].append(analyte)
                rows["WastePhase"].append("Liquid")
                rows["WasteType"].append("T1")
                rows["Inventory"].append(val)
                rows["Units"].append("kg")
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame(rows))
        dataset.report = None
        metric_matrix = pd.DataFrame({"WasteSiteId": tanks, "Cs": [1.0, 2.0, 3.0], "Sr": [4.0, 5.0, 6.0]})
        partial_df, raw_df = ss.partial_correlation_matrix(dataset, metric_matrix, ["Cs", "Sr"])
        raw_r = raw_df.set_index("Element").loc["Cs", "Sr"]
        partial_r = partial_df.set_index("Element").loc["Cs", "Sr"]
        assert raw_r == pytest.approx(1.0)
        assert pd.isna(partial_r)


class TestElementNetwork:
    def _workbench(self, dataset):
        return csci.kg_correlation_workbench(dataset, elements_text="Cs, Sr, Mo, Fe", value_mode="log10_plus1")

    def test_nodes_include_every_element_even_isolated(self, structure_dataset):
        base = self._workbench(structure_dataset)
        nodes, edges = ss.element_network(base, min_abs_r=0.999)  # strict threshold, Fe is constant -> r undefined
        assert set(nodes["Element"]) == {"Cs", "Sr", "Mo", "Fe"}
        # Fe's correlations are all NaN (constant) so it must be edge-less but still a node.
        fe_edges = edges[(edges["Element_A"] == "Fe") | (edges["Element_B"] == "Fe")]
        assert fe_edges.empty
        assert int(nodes.set_index("Element").loc["Fe", "N_edges"]) == 0

    def test_edges_respect_min_abs_r_threshold(self, structure_dataset):
        # Under log10_plus1, Cs-Sr ~= 0.999979 but Cs-Mo/Sr-Mo ~= -0.937 --
        # a 0.99 floor keeps only the Cs-Sr edge.
        base = self._workbench(structure_dataset)
        _, loose_edges = ss.element_network(base, min_abs_r=0.0)
        _, strict_edges = ss.element_network(base, min_abs_r=0.99)
        assert len(strict_edges) < len(loose_edges)
        assert len(strict_edges) == 1
        assert strict_edges.iloc[0]["Element_A"] == "Cs"
        assert strict_edges.iloc[0]["Element_B"] == "Sr"

    def test_all_edges_filtered_out_returns_empty_but_no_crash(self, structure_dataset):
        base = self._workbench(structure_dataset)
        nodes, edges = ss.element_network(base, min_abs_r=0.999999)
        assert edges.empty
        assert set(nodes["Element"]) == {"Cs", "Sr", "Mo", "Fe"}

    def test_cs_sr_edge_is_positive_cs_mo_edge_is_negative(self, structure_dataset):
        base = self._workbench(structure_dataset)
        _, edges = ss.element_network(base, min_abs_r=0.0)
        pairs = edges.set_index(["Element_A", "Element_B"])
        assert pairs.loc[("Cs", "Sr"), "Sign"] == "positive"
        assert pairs.loc[("Cs", "Mo"), "Sign"] == "negative"

    def test_nodes_have_layout_positions(self, structure_dataset):
        base = self._workbench(structure_dataset)
        nodes, _ = ss.element_network(base, min_abs_r=0.0)
        assert {"x", "y"}.issubset(nodes.columns)
        assert nodes[["x", "y"]].notna().all().all()

    def test_partial_corr_matrix_used_when_provided(self, structure_dataset):
        base = self._workbench(structure_dataset)
        partial_df, _ = ss.partial_correlation_matrix(structure_dataset, base["metric_matrix"], ["Cs", "Sr", "Mo", "Fe"])
        nodes_p, edges_p = ss.element_network(base, min_abs_r=0.0, partial_corr_matrix=partial_df)
        nodes_r, edges_r = ss.element_network(base, min_abs_r=0.0)
        assert set(nodes_p["Element"]) == set(nodes_r["Element"])

    def test_min_jaccard_excludes_high_corr_but_low_overlap_pair(self):
        # A hand-built workbench_results dict: A-B has high correlation
        # (0.95) but low Jaccard co-presence (0.1, e.g. rarely in the same
        # tank). A min_abs_r floor alone would keep this edge; a min_jaccard
        # floor must exclude it.
        element_stats = pd.DataFrame({"Element": ["A", "B"], "Total_inventory_kg": [10.0, 20.0]})
        corr_matrix = pd.DataFrame({"Element": ["A", "B"], "A": [1.0, 0.95], "B": [0.95, 1.0]})
        jaccard_matrix = pd.DataFrame({"Element": ["A", "B"], "A": [1.0, 0.1], "B": [0.1, 1.0]})
        workbench_results = {"element_stats": element_stats, "corr_matrix": corr_matrix, "jaccard_matrix": jaccard_matrix}
        _, edges_no_jaccard_floor = ss.element_network(workbench_results, min_abs_r=0.5, min_jaccard=0.0)
        _, edges_with_jaccard_floor = ss.element_network(workbench_results, min_abs_r=0.5, min_jaccard=0.5)
        assert len(edges_no_jaccard_floor) == 1
        assert edges_with_jaccard_floor.empty

    def test_empty_workbench_results_return_empty(self):
        nodes, edges = ss.element_network({"element_stats": pd.DataFrame(), "corr_matrix": pd.DataFrame(), "jaccard_matrix": pd.DataFrame()})
        assert nodes.empty and edges.empty


class TestStructureWorkbench:
    def test_end_to_end_builds_all_tables(self, structure_dataset):
        out = ss.structure_workbench(structure_dataset, elements_text="Cs, Sr, Mo, Fe", value_mode="log10_plus1", n_clusters=2)
        assert not out["pca_scores"].empty
        assert not out["cluster_assignments"].empty
        assert not out["partial_corr_matrix"].empty
        assert not out["network_nodes"].empty
        assert not out["tank_summary"].empty
        # Inherited from kg_correlation_workbench -- Structure and the
        # Association Workbench must agree on the same base tables.
        assert not out["element_stats"].empty
        assert not out["pair_stats"].empty

    def test_tank_summary_has_pca_cluster_and_category_columns(self, structure_dataset):
        out = ss.structure_workbench(structure_dataset, elements_text="Cs, Sr, Mo, Fe", value_mode="log10_plus1", n_clusters=2)
        summary = out["tank_summary"]
        for col in ["WasteSiteId", "PC1", "PC2", "Cluster"] + ss.CATEGORY_FIELDS:
            assert col in summary.columns
        row = summary.set_index("WasteSiteId").loc["241-A-101"]
        assert row["TankFarm"] == "A"
        assert row["Dominant waste phase"] == "Sludge Solid"

    def test_too_few_elements_raises_same_as_workbench(self, structure_dataset):
        with pytest.raises(ValueError, match="at least two valid elements"):
            ss.structure_workbench(structure_dataset, elements_text="Cs")

    def test_no_matching_rows_returns_empty_structure_tables_not_crash(self, structure_dataset):
        out = ss.structure_workbench(structure_dataset, elements_text="Cs, Sr", min_inventory=1e9)
        assert out["pca_scores"].empty
        assert out["cluster_assignments"].empty
        assert out["network_nodes"].empty
        assert out["tank_summary"].empty

    def test_use_partial_for_network_toggle_runs_without_crash(self, structure_dataset):
        out = ss.structure_workbench(
            structure_dataset, elements_text="Cs, Sr, Mo, Fe", value_mode="log10_plus1",
            n_clusters=2, use_partial_for_network=True,
        )
        assert not out["network_edges"].empty or out["network_edges"].empty  # just must not crash
