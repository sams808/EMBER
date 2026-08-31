import math

import pandas as pd
import pytest

import tank_science as ts


class TestTankAttributesTable:
    def test_one_row_per_tank_sorted(self, sample_dataset):
        out = ts.tank_attributes_table(sample_dataset)
        assert out["WasteSiteId"].tolist() == ["241-A-101", "241-AN-104"]

    def test_derived_flags(self, sample_dataset):
        out = ts.tank_attributes_table(sample_dataset).set_index("WasteSiteId")
        a101 = out.loc["241-A-101"]
        assert a101["TankSystem"] == "SST"
        assert bool(a101["IsSST"]) is True
        assert bool(a101["IsLeakerOrAssumedLeaker"]) is False
        assert a101["Capacity"] == pytest.approx(1000.0)

        an104 = out.loc["241-AN-104"]
        assert an104["TankSystem"] == "DST"
        assert bool(an104["IsDST"]) is True
        assert bool(an104["IsLeakerOrAssumedLeaker"]) is True
        assert an104["DIL_Gal"] == pytest.approx(0.0)

    def test_empty_when_no_matching_columns(self):
        import data_model as dm
        import polars as pl
        dataset = dm.HanfordDataset()
        dataset.df = pl.DataFrame({"SomeOtherColumn": [1, 2]})
        assert ts.tank_attributes_table(dataset).empty


class TestTankAttributeAudit:
    def test_empty_when_category_missing(self, sample_dataset):
        assert ts.tank_attribute_audit(sample_dataset, "NoSuchColumn").empty

    def test_groups_by_tank_system(self, sample_dataset):
        out = ts.tank_attribute_audit(sample_dataset, "TankSystem").set_index("TankSystem")
        assert out.loc["SST", "N_tanks"] == 1
        assert out.loc["SST", "Total_capacity_kgal"] == pytest.approx(1000.0)
        assert out.loc["SST", "N_leaker_or_assumed"] == 0
        assert out.loc["DST", "N_leaker_or_assumed"] == 1
        assert out.loc["DST", "Total_DIL_gal"] == pytest.approx(0.0)


class TestTankAttributeNumericSummary:
    def test_populated_column_stats(self, sample_dataset):
        out = ts.tank_attribute_numeric_summary(sample_dataset).set_index("column")
        capacity = out.loc["Capacity"]
        assert capacity["n_nonnull"] == 2
        assert capacity["min"] == pytest.approx(1000.0)
        assert capacity["max"] == pytest.approx(1160.0)
        assert capacity["sum"] == pytest.approx(2160.0)

    def test_fully_null_column_reports_nan_sum_not_zero(self, sample_dataset):
        # Distinct from tank_attribute_audit's plain groupby-sum (which
        # would report 0.0 for an all-null column) -- this is ported
        # as-is from the old app's explicit notna().any() guard.
        out = ts.tank_attribute_numeric_summary(sample_dataset).set_index("column")
        diameter = out.loc["Diameter"]
        assert diameter["n_nonnull"] == 0
        assert math.isnan(diameter["sum"])

    def test_empty_when_no_attributes(self):
        import data_model as dm
        import polars as pl
        dataset = dm.HanfordDataset()
        dataset.df = pl.DataFrame({"SomeOtherColumn": [1, 2]})
        assert ts.tank_attribute_numeric_summary(dataset).empty

    def test_skips_columns_absent_from_attributes_table(self):
        # A frame with only WasteSiteId + Capacity (no Diameter/
        # MaxOperatingDepth/OperationCapacityKGal/DIL_Gal at all) must
        # summarize only the column that's actually there.
        import data_model as dm
        import polars as pl
        dataset = dm.HanfordDataset()
        dataset.df = pl.DataFrame({"WasteSiteId": ["241-A-101"], "Capacity": [1000.0]})
        out = ts.tank_attribute_numeric_summary(dataset)
        assert out["column"].tolist() == ["Capacity"]


class TestTankProfile:
    def test_single_tank_kg(self, sample_dataset):
        out = ts.tank_profile(sample_dataset, ["241-A-101"], unit_filter="kg").set_index("Element")
        fe = out.loc["Fe"]
        assert fe["Inventory_sum"] == pytest.approx(60.0)  # duplicate-key sum
        assert fe["Fraction_of_tank_unit_inventory"] == pytest.approx(60.0 / 60.5)
        cd = out.loc["Cd"]
        assert cd["Inventory_sum"] == pytest.approx(0.5)

    def test_sorted_by_inventory_desc_within_tank(self, sample_dataset):
        out = ts.tank_profile(sample_dataset, ["241-A-101"], unit_filter="kg")
        assert out.iloc[0]["Element"] == "Fe"
        assert out.iloc[1]["Element"] == "Cd"

    def test_multi_tank_all_units(self, sample_dataset):
        out = ts.tank_profile(sample_dataset, ["241-A-101", "241-AN-104"])
        # 7 (Units, WasteSiteId, Element) groups across both tanks, matching
        # the same context used in composition_stats (Total Alpha excluded).
        assert len(out) == 7

    def test_empty_when_no_matching_tanks(self, sample_dataset):
        assert ts.tank_profile(sample_dataset, ["241-Z-999"]).empty

    def test_top_n_applies_per_tank_and_unit(self, sample_dataset):
        """Top-N is "how many elements to show", per tank -- not a global
        row budget. It used to be head(top_n * n_tanks) over a tank-ordered
        sort, which let the alphabetically-first tanks spend the whole
        budget and dropped later tanks entirely (review finding)."""
        tanks = ["241-A-101", "241-AN-104"]
        out = ts.tank_profile(sample_dataset, tanks, top_n=1)
        # Both tanks, in both units they report: 2 tanks x 2 units x 1 element.
        assert len(out) == 4
        assert sorted(out["WasteSiteId"].unique()) == tanks
        assert out.groupby(["Units", "WasteSiteId"]).size().eq(1).all()


class TestRawRowsForTank:
    def test_returns_all_rows_for_tank_sorted(self, sample_dataset):
        out = ts.raw_rows_for_tank(sample_dataset, "241-A-101")
        assert len(out) == 6
        assert out.iloc[0]["Units"] == "Ci"
        assert out.iloc[0]["Inventory"] == pytest.approx(100.0)  # 137Cs, largest Ci row

    def test_unit_filter(self, sample_dataset):
        out = ts.raw_rows_for_tank(sample_dataset, "241-A-101", unit_filter="kg")
        assert set(out["Units"]) == {"kg"}
        assert len(out) == 3

    def test_limit(self, sample_dataset):
        out = ts.raw_rows_for_tank(sample_dataset, "241-A-101", limit=2)
        assert len(out) == 2

    def test_no_rows_for_unknown_tank(self, sample_dataset):
        assert ts.raw_rows_for_tank(sample_dataset, "241-Z-999").empty


class TestFarmHelpers:
    def test_available_farms_with_all(self, sample_dataset):
        assert ts.available_farms_with_all(sample_dataset) == ["All", "A", "AN"]

    def test_tanks_in_farm(self, sample_dataset):
        assert ts.tanks_in_farm(sample_dataset, "A") == ["241-A-101"]
        assert ts.tanks_in_farm(sample_dataset, "AN") == ["241-AN-104"]

    def test_tanks_in_all_farms(self, sample_dataset):
        assert ts.tanks_in_farm(sample_dataset, "All") == ["241-A-101", "241-AN-104"]

    def test_tanks_in_none_farm_returns_all(self, sample_dataset):
        assert ts.tanks_in_farm(sample_dataset, None) == ["241-A-101", "241-AN-104"]


class TestFarmInventoryRollup:
    def test_kg_totals_match_manual_sum_per_farm(self, sample_dataset):
        # Manual cross-check against the same rule the rest of this session
        # has applied to every science-layer addition: filter the raw frame
        # by hand and sum, then compare against the function's own output.
        df = sample_dataset.require_df()
        out = ts.farm_inventory_rollup(sample_dataset, unit="kg").set_index("TankFarm")

        for farm, tank in [("A", "241-A-101"), ("AN", "241-AN-104")]:
            manual_total = (
                df.filter((df["TankFarm"] == farm) & (df["Units"] == "kg") & (df["Inventory"] > 0))
                .get_column("Inventory").sum()
            )
            assert out.loc[farm, "Total_inventory_kg"] == pytest.approx(manual_total)

        assert out.loc["A", "Total_inventory_kg"] == pytest.approx(60.5)  # Fe 50 + Cd 0.5 + Fe 10
        assert out.loc["AN", "Total_inventory_kg"] == pytest.approx(500.0)  # Na

    def test_n_tanks_per_farm(self, sample_dataset):
        out = ts.farm_inventory_rollup(sample_dataset, unit="kg").set_index("TankFarm")
        assert out.loc["A", "N_tanks"] == 1
        assert out.loc["AN", "N_tanks"] == 1

    def test_dominant_waste_type_and_phase_by_inventory_not_row_count(self, sample_dataset):
        # Farm A's kg rows: Fe=50 (Solid), Cd=0.5 (Liquid), Fe=10 (Liquid).
        # Liquid has 2 rows vs Solid's 1, but Solid carries more inventory
        # (50 > 10.5) -- dominance must follow inventory, not row count.
        out = ts.farm_inventory_rollup(sample_dataset, unit="kg").set_index("TankFarm")
        assert out.loc["A", "Dominant_waste_phase"] == "Solid"
        assert out.loc["A", "Dominant_waste_type"] == "T1"
        assert out.loc["AN", "Dominant_waste_phase"] == "Sludge"
        assert out.loc["AN", "Dominant_waste_type"] == "T2"

    def test_mean_inventory_per_tank(self, sample_dataset):
        out = ts.farm_inventory_rollup(sample_dataset, unit="kg").set_index("TankFarm")
        assert out.loc["A", "Mean_inventory_per_tank_kg"] == pytest.approx(60.5)  # single tank in farm

    def test_kg_and_ci_are_never_summed_together(self, sample_dataset):
        kg_out = ts.farm_inventory_rollup(sample_dataset, unit="kg").set_index("TankFarm")
        ci_out = ts.farm_inventory_rollup(sample_dataset, unit="Ci").set_index("TankFarm")
        assert kg_out.loc["A", "Total_inventory_kg"] == pytest.approx(60.5)
        assert ci_out.loc["A", "Total_inventory_Ci"] == pytest.approx(100.012)  # Cs 100 + Pu 0.002 + TotalAlpha 0.01

    def test_column_names_carry_the_unit(self, sample_dataset):
        # Review finding: a bare "Total_inventory" header on an exported
        # table hides which unit it was computed in -- the name is the cue.
        out = ts.farm_inventory_rollup(sample_dataset, unit="kg")
        assert "Total_inventory_kg" in out.columns
        assert "Mean_inventory_per_tank_kg" in out.columns

    def test_sorted_by_total_inventory_descending(self, sample_dataset):
        out = ts.farm_inventory_rollup(sample_dataset, unit="kg")
        assert out["TankFarm"].tolist() == ["AN", "A"]  # 500.0 > 60.5

    def test_every_farm_listed_even_without_rows_in_the_unit(self):
        # Review finding: an audit table must never silently drop a farm
        # whose inventory lives entirely in the other unit.
        import data_model as dm
        import polars as pl
        dataset = dm.HanfordDataset()
        rows = {
            "WasteSiteId": ["241-A-101", "241-SX-101"],
            "Analyte": ["Fe", "137Cs"],
            "WastePhase": ["Solid", "Liquid"],
            "WasteType": ["T1", "T2"],
            "Inventory": [50.0, 200.0],
            "Units": ["kg", "Ci"],  # SX exists ONLY in Ci
        }
        dataset.df = dataset._clean_dataframe(pl.DataFrame(rows))
        dataset.report = None
        out = ts.farm_inventory_rollup(dataset, unit="kg").set_index("TankFarm")
        assert set(out.index) == {"A", "SX"}
        assert out.loc["SX", "Total_inventory_kg"] == pytest.approx(0.0)
        assert out.loc["SX", "N_tanks"] == 0
        assert pd.isna(out.loc["SX", "Dominant_waste_type"])

    def test_unknown_unit_lists_all_farms_with_zeros(self, sample_dataset):
        out = ts.farm_inventory_rollup(sample_dataset, unit="Bq")
        assert set(out["TankFarm"]) == {"A", "AN"}
        assert (out["Total_inventory_Bq"] == 0.0).all()

    def test_empty_when_no_tank_farm_column(self):
        import data_model as dm
        import polars as pl
        dataset = dm.HanfordDataset()
        dataset.df = pl.DataFrame({"WasteSiteId": ["241-A-101"], "Units": ["kg"], "Inventory": [1.0]})
        assert ts.farm_inventory_rollup(dataset, unit="kg").empty
