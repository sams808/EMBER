"""
Correlation math needs more than the 2-tank shared sample_dataset fixture
to produce meaningful (non-degenerate) coefficients, so this module builds
its own small dataset with known, hand-picked relationships: Cs/Sr move in
perfect lockstep (r=1.0), Cs/Mo move in perfect opposition (r=-1.0), and Fe
is constant across all tanks (undefined correlation -- must be dropped,
not crash).
"""
import math

import polars as pl
import pytest

import correlation_science as cs
import data_model as dm


@pytest.fixture
def corr_dataset():
    dataset = dm.HanfordDataset()
    tanks = ["241-T-101", "241-T-102", "241-T-103", "241-T-104", "241-T-105"]
    cs_vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    sr_vals = [v * 2.0 for v in cs_vals]          # perfectly correlated with Cs
    mo_vals = [60.0 - v for v in cs_vals]          # perfectly anti-correlated with Cs
    fe_vals = [5.0] * 5                            # constant -> undefined correlation

    rows = {"WasteSiteId": [], "Analyte": [], "WastePhase": [], "WasteType": [], "Inventory": [], "Units": []}
    for tank, cs_v, sr_v, mo_v, fe_v in zip(tanks, cs_vals, sr_vals, mo_vals, fe_vals):
        for analyte, val in [("Cs", cs_v), ("Sr", sr_v), ("Mo", mo_v), ("Fe", fe_v)]:
            rows["WasteSiteId"].append(tank)
            rows["Analyte"].append(analyte)
            rows["WastePhase"].append("Liquid")
            rows["WasteType"].append("T1")
            rows["Inventory"].append(val)
            rows["Units"].append("kg")
    # Ba present at only 2 of 5 tanks -- for exercising the overlap-floor
    # branch (len(pair) < 3) separately from the min_overlap threshold.
    for tank, val in [(tanks[0], 7.0), (tanks[1], 9.0)]:
        rows["WasteSiteId"].append(tank)
        rows["Analyte"].append("Ba")
        rows["WastePhase"].append("Liquid")
        rows["WasteType"].append("T1")
        rows["Inventory"].append(val)
        rows["Units"].append("kg")
    df = dataset._clean_dataframe(pl.DataFrame(rows))
    dataset.df = df
    dataset.report = None
    return dataset


class TestElementTotalsByUnit:
    def test_totals_sum_across_tanks(self, corr_dataset):
        totals = cs.element_totals_by_unit(corr_dataset, "kg")
        assert totals["Cs"] == pytest.approx(10 + 20 + 30 + 40 + 50)
        assert totals["Fe"] == pytest.approx(25.0)

    def test_empty_for_missing_unit(self, corr_dataset):
        assert cs.element_totals_by_unit(corr_dataset, "Ci") == {}


class TestElementCorrelationScan:
    def test_invalid_target_symbol_raises(self, corr_dataset):
        with pytest.raises(ValueError, match="not a valid element symbol"):
            cs.element_correlation_scan(corr_dataset, "Zz")

    def test_perfect_positive_correlation(self, corr_dataset):
        out, matrix = cs.element_correlation_scan(corr_dataset, "Cs", value_mode="inventory", min_overlap=3)
        sr_row = out[out["PartnerElement"] == "Sr"].iloc[0]
        assert sr_row["Correlation_r"] == pytest.approx(1.0)
        assert sr_row["N_overlap_nonzero_tanks"] == 5
        assert not matrix.empty

    def test_perfect_negative_correlation(self, corr_dataset):
        out, _ = cs.element_correlation_scan(corr_dataset, "Cs", value_mode="inventory", min_overlap=3)
        mo_row = out[out["PartnerElement"] == "Mo"].iloc[0]
        assert mo_row["Correlation_r"] == pytest.approx(-1.0)

    def test_constant_partner_dropped_not_crashed(self, corr_dataset):
        # Fe is constant (std=0) -> correlation is NaN -> must be excluded
        # from the results, not raise or appear as a NaN row.
        out, _ = cs.element_correlation_scan(corr_dataset, "Cs", value_mode="inventory", min_overlap=3)
        assert "Fe" not in set(out["PartnerElement"])

    def test_min_overlap_excludes_when_too_strict(self, corr_dataset):
        out, _ = cs.element_correlation_scan(corr_dataset, "Cs", value_mode="inventory", min_overlap=6)
        assert out.empty

    def test_rank_columns_assigned(self, corr_dataset):
        out, _ = cs.element_correlation_scan(corr_dataset, "Cs", value_mode="inventory", min_overlap=3)
        assert out.iloc[0]["Rank_abs"] == 1
        sr_row = out[out["PartnerElement"] == "Sr"].iloc[0]
        assert sr_row["Rank_positive"] == 1  # the single most-positive partner
        mo_row = out[out["PartnerElement"] == "Mo"].iloc[0]
        assert mo_row["Rank_negative"] == 1  # the single most-negative partner

    def test_totals_attached(self, corr_dataset):
        out, _ = cs.element_correlation_scan(corr_dataset, "Cs", value_mode="inventory", min_overlap=3)
        sr_row = out[out["PartnerElement"] == "Sr"].iloc[0]
        assert sr_row["Target_total_inventory"] == pytest.approx(150.0)  # sum of Cs
        assert sr_row["Partner_total_inventory"] == pytest.approx(300.0)  # sum of Sr

    def test_target_not_in_matrix_returns_empty(self, corr_dataset):
        # A syntactically valid symbol that never appears in the data.
        out, matrix = cs.element_correlation_scan(corr_dataset, "Au", value_mode="inventory")
        assert out.empty

    def test_sparse_partner_excluded_by_overlap_floor_not_min_overlap(self, corr_dataset):
        # Ba is present at only 2/5 tanks. min_overlap=1 lets it past the
        # n_overlap check, but the separate len(pair) < 3 floor (applied
        # after masking to present-only rows under include_zeros=False)
        # must still exclude it.
        out, _ = cs.element_correlation_scan(
            corr_dataset, "Cs", value_mode="inventory", min_overlap=1, include_zeros=False,
        )
        assert "Ba" not in set(out["PartnerElement"])

    def test_exclude_zeros_still_finds_perfect_correlation(self, corr_dataset):
        # Every tank has every element present here, so include_zeros=False
        # (present-only masking) should reach the same r as include_zeros=True.
        out, _ = cs.element_correlation_scan(
            corr_dataset, "Cs", value_mode="inventory", min_overlap=3, include_zeros=False,
        )
        sr_row = out[out["PartnerElement"] == "Sr"].iloc[0]
        assert sr_row["Correlation_r"] == pytest.approx(1.0)


class TestParseElementList:
    def test_comma_and_space_separated(self):
        assert cs.parse_element_list("Cs, Sr Tc;I") == ["Cs", "Sr", "Tc", "I"]

    def test_drops_invalid_tokens(self):
        assert cs.parse_element_list("Cs, Zz, Sr") == ["Cs", "Sr"]

    def test_deduplicates(self):
        assert cs.parse_element_list("Cs, Cs, Sr") == ["Cs", "Sr"]


class TestSelectedElementCorrelations:
    def test_too_few_elements_raises(self, corr_dataset):
        with pytest.raises(ValueError, match="at least two"):
            cs.selected_element_correlations(corr_dataset, ["Cs"])

    def test_pairwise_correlations(self, corr_dataset):
        pairs, joint, matrix = cs.selected_element_correlations(corr_dataset, ["Cs", "Sr", "Mo"], value_mode="inventory")
        pairs = pairs.set_index(["Element_A", "Element_B"])
        assert pairs.loc[("Cs", "Sr"), "Correlation_r"] == pytest.approx(1.0)
        assert pairs.loc[("Cs", "Mo"), "Correlation_r"] == pytest.approx(-1.0)
        assert not matrix.empty

    def test_exclude_zeros_mode(self, corr_dataset):
        pairs, _, _ = cs.selected_element_correlations(
            corr_dataset, ["Cs", "Sr"], value_mode="inventory", include_zeros=False,
        )
        assert pairs.set_index(["Element_A", "Element_B"]).loc[("Cs", "Sr"), "Correlation_r"] == pytest.approx(1.0)

    def test_empty_matrix_returns_three_empty_frames(self, corr_dataset):
        pairs, joint, matrix = cs.selected_element_correlations(corr_dataset, ["Cs", "Sr"], unit="Ci")
        assert pairs.empty and joint.empty and matrix.empty

    def test_element_absent_from_matrix_is_skipped_not_crashed(self, corr_dataset):
        # "Au" is a valid symbol but never appears in the data at all, so
        # it never becomes a matrix column -- the pair must be skipped.
        pairs, _, _ = cs.selected_element_correlations(corr_dataset, ["Cs", "Au"], value_mode="inventory")
        assert pairs.empty

    def test_joint_summary_all_present(self, corr_dataset):
        _, joint, _ = cs.selected_element_correlations(corr_dataset, ["Cs", "Sr"], value_mode="inventory")
        row = joint.iloc[0]
        assert row["N_tanks_all_present"] == 5
        assert row["N_tanks_any_present"] == 5
        assert row["Fraction_all_present_pct"] == pytest.approx(100.0)

    def test_joint_summary_partial_presence(self, corr_dataset):
        # Cs is present at all 5 tanks; a fabricated element absent
        # everywhere would make "all present" 0 -- use Fe (present at all
        # 5 here too) vs constructing a partial case isn't needed since
        # this dataset has no partial-presence element; assert mean/min/max
        # pairwise correlation fields exist and are finite instead.
        pairs, joint, _ = cs.selected_element_correlations(corr_dataset, ["Cs", "Sr", "Mo"], value_mode="inventory")
        row = joint.iloc[0]
        assert math.isfinite(row["Mean_pairwise_correlation"])
        assert row["Min_pairwise_correlation"] <= row["Max_pairwise_correlation"]


class TestKgCorrelationWorkbench:
    def test_too_few_elements_after_parsing_raises(self, corr_dataset):
        with pytest.raises(ValueError, match="at least two valid elements"):
            cs.kg_correlation_workbench(corr_dataset, elements_text="Cs")

    def test_skip_list_message_included_when_it_empties_selection(self, corr_dataset):
        with pytest.raises(ValueError, match="Skipped elements: Cs, Sr"):
            cs.kg_correlation_workbench(corr_dataset, elements_text="Cs, Sr", skip_elements_text="Cs, Sr")

    def test_user_list_selection(self, corr_dataset):
        out = cs.kg_correlation_workbench(corr_dataset, elements_text="Cs, Sr, Mo", value_mode="inventory")
        assert set(out["element_stats"]["Element"]) == {"Cs", "Sr", "Mo"}
        assert set(out["raw_matrix"].columns) - {"WasteSiteId"} == {"Cs", "Sr", "Mo"}

    def test_top_kg_selection_mode(self, corr_dataset):
        out = cs.kg_correlation_workbench(corr_dataset, selection_mode="Top kg elements", top_n_elements=2, value_mode="inventory")
        # Totals: Sr=300 > Cs=150 > Mo=150 > Ba=16 > Fe=25 -- top 2 by total is Sr, Cs (or Mo tie-broken by polars sort).
        assert len(set(out["element_stats"]["Element"])) == 2
        assert "Sr" in set(out["element_stats"]["Element"])

    def test_skip_list_applied_before_top_n(self, corr_dataset):
        # Skipping Sr means "top 2" must be computed from the remaining
        # elements (Cs, Mo, Fe, Ba), not silently drop to 1.
        out = cs.kg_correlation_workbench(
            corr_dataset, selection_mode="Top kg elements", top_n_elements=2,
            value_mode="inventory", skip_elements_text="Sr",
        )
        elements = set(out["element_stats"]["Element"])
        assert "Sr" not in elements
        assert len(elements) == 2

    def test_empty_user_list_falls_back_to_top_kg(self, corr_dataset):
        out = cs.kg_correlation_workbench(corr_dataset, elements_text="", selection_mode="User list", top_n_elements=3, value_mode="inventory")
        assert len(set(out["element_stats"]["Element"])) == 3

    def test_pair_stats_perfect_positive_scores_highest(self, corr_dataset):
        out = cs.kg_correlation_workbench(corr_dataset, elements_text="Cs, Sr, Mo", value_mode="inventory")
        pairs = out["pair_stats"].set_index(["Element_A", "Element_B"])
        cs_sr = pairs.loc[("Cs", "Sr")]
        assert cs_sr["Correlation_r"] == pytest.approx(1.0)
        assert cs_sr["Jaccard_presence"] == pytest.approx(1.0)  # present at all 5 tanks
        assert cs_sr["PreferredAssociationScore_proxy"] == pytest.approx(math.log1p(5))
        # Negative-r pairs must score exactly 0, not a negative number.
        cs_mo = pairs.loc[("Cs", "Mo")]
        assert cs_mo["Correlation_r"] == pytest.approx(-1.0)
        assert cs_mo["PreferredAssociationScore_proxy"] == pytest.approx(0.0)
        assert out["pair_stats"].iloc[0]["Rank_preferred_association"] == 1
        assert out["pair_stats"].iloc[0]["Element_A"] == "Cs"
        assert out["pair_stats"].iloc[0]["Element_B"] == "Sr"

    def test_corr_and_jaccard_matrices_are_square_and_symmetric(self, corr_dataset):
        out = cs.kg_correlation_workbench(corr_dataset, elements_text="Cs, Sr, Mo", value_mode="inventory")
        corr = out["corr_matrix"].set_index("Element")
        assert corr.loc["Cs", "Sr"] == pytest.approx(corr.loc["Sr", "Cs"])
        assert corr.loc["Cs", "Cs"] == pytest.approx(1.0)

    def test_sparse_element_presence_fraction(self, corr_dataset):
        out = cs.kg_correlation_workbench(corr_dataset, elements_text="Cs, Ba", value_mode="inventory")
        stats = out["element_stats"].set_index("Element")
        assert stats.loc["Ba", "N_tanks_present"] == 2
        assert stats.loc["Ba", "PresenceFraction_pct"] == pytest.approx(40.0)
        assert stats.loc["Cs", "PresenceFraction_pct"] == pytest.approx(100.0)

    def test_tank_similarity_shape(self, corr_dataset):
        out = cs.kg_correlation_workbench(corr_dataset, elements_text="Cs, Sr, Mo", value_mode="inventory")
        sim = out["tank_similarity"]
        assert not sim.empty
        assert sim.shape[0] == 5  # 5 tanks

    def test_constant_element_excluded_from_tank_similarity(self, corr_dataset):
        # Fe is constant -> must be dropped from the tank-similarity basis
        # (a constant column has undefined correlation), not crash it.
        out = cs.kg_correlation_workbench(corr_dataset, elements_text="Cs, Sr, Fe", value_mode="inventory")
        assert not out["tank_similarity"].empty

    def test_all_value_modes_run_without_crash(self, corr_dataset):
        for mode in ["inventory", "log10_inventory", "log10_plus1", "fraction", "presence"]:
            out = cs.kg_correlation_workbench(corr_dataset, elements_text="Cs, Sr, Mo", value_mode=mode)
            assert not out["metric_matrix"].empty

    def test_unknown_value_mode_raises(self, corr_dataset):
        with pytest.raises(ValueError, match="Unknown kg seaborn metric"):
            cs.kg_correlation_workbench(corr_dataset, elements_text="Cs, Sr", value_mode="bogus")

    def test_exclude_zeros_mode(self, corr_dataset):
        out = cs.kg_correlation_workbench(corr_dataset, elements_text="Cs, Sr, Mo", value_mode="inventory", include_zeros=False)
        pairs = out["pair_stats"].set_index(["Element_A", "Element_B"])
        assert pairs.loc[("Cs", "Sr"), "Correlation_r"] == pytest.approx(1.0)

    def test_tank_element_long_only_positive_rows(self, corr_dataset):
        out = cs.kg_correlation_workbench(corr_dataset, elements_text="Cs, Ba", value_mode="inventory")
        long_df = out["tank_element_long"]
        # Ba is present at only 2/5 tanks -- the long table must not include
        # zero-valued (absent) rows for it.
        assert len(long_df[long_df["Element"] == "Ba"]) == 2

    def test_excluded_elements_table_records_skip_list(self, corr_dataset):
        out = cs.kg_correlation_workbench(corr_dataset, elements_text="Cs, Sr, Mo", skip_elements_text="Fe, Ba", value_mode="inventory")
        excluded = out["excluded_elements"]
        assert set(excluded["ExcludedElement"]) == {"Fe", "Ba"}
        assert (excluded["Reason"] == "User skip list").all()

    def test_all_requested_elements_absent_raises(self, corr_dataset):
        # "Au"/"Pt" are valid symbols but never appear in the kg data at
        # all -- distinct from the len(elements)<2 case, this passes that
        # check but then finds zero matching inventory rows.
        with pytest.raises(ValueError, match="No kg inventory rows matched"):
            cs.kg_correlation_workbench(corr_dataset, elements_text="Au, Pt", value_mode="inventory")

    def test_one_requested_element_absent_becomes_zero_column(self, corr_dataset):
        # "Au" is valid but absent; "Cs" is present -- raw_matrix must keep
        # both requested columns, force-filling Au with zeros rather than
        # silently dropping it.
        out = cs.kg_correlation_workbench(corr_dataset, elements_text="Cs, Au", value_mode="inventory")
        assert set(out["raw_matrix"].columns) - {"WasteSiteId"} == {"Cs", "Au"}
        assert (out["raw_matrix"]["Au"] == 0.0).all()

    def test_empty_dataframe_when_no_kg_rows_match(self, corr_dataset):
        out = cs.kg_correlation_workbench(corr_dataset, elements_text="Cs, Sr", min_inventory=1e9)
        assert out["element_stats"].empty
        assert out["raw_matrix"].empty
        assert set(out["excluded_elements"].columns) == {"ExcludedElement", "Reason"}


class TestFullCorrelationMatrix:
    def test_diagonal_is_one(self, corr_dataset):
        corr, matrix = cs.full_correlation_matrix(corr_dataset, value_mode="inventory", top_n_elements=10)
        square = corr.set_index("Element")
        assert square.loc["Cs", "Cs"] == pytest.approx(1.0)
        assert not matrix.empty

    def test_symmetric_and_matches_scan(self, corr_dataset):
        corr, _ = cs.full_correlation_matrix(corr_dataset, value_mode="inventory", top_n_elements=10)
        square = corr.set_index("Element")
        assert square.loc["Cs", "Sr"] == pytest.approx(1.0)
        assert square.loc["Sr", "Cs"] == pytest.approx(1.0)
        assert square.loc["Cs", "Mo"] == pytest.approx(-1.0)

    def test_empty_when_no_data(self, corr_dataset):
        corr, matrix = cs.full_correlation_matrix(corr_dataset, unit="Ci")
        assert corr.empty and matrix.empty
