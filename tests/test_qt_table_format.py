"""qt_table_format: how a column name and a value are written for a reader.

The expected strings are written out by hand from the rules in the module
docstring, not derived by calling the functions under test."""
import math

import numpy as np
import pandas as pd
import pytest

import qt_table_format as tf


class TestHumanize:
    @pytest.mark.parametrize("name, label", [
        ("WasteSiteId", "Tank"),
        ("Total_kg", "Total (kg)"),
        ("Total_g", "Total (g)"),
        ("Inventory_RSD", "Rel. std. dev."),
        ("Inventory_sum", "Inventory"),
        ("User_search_score_proxy", "Search score (proxy)"),
        ("N_PRIMARY_ELEMENTS", "Elements"),
        ("log10_inventory", "log10 inventory"),
        ("log10_plus1", "log10(inventory + 1)"),
        ("TankFarm", "Farm"),
        ("WastePhase", "Waste phase"),
    ])
    def test_known_identifiers_come_from_the_table(self, name, label):
        assert tf.humanize(name) == label

    @pytest.mark.parametrize("name, label", [
        ("Some_new_column", "Some new column"),            # snake case, sentence case
        ("SomeNewColumn", "Some new column"),              # CamelCase
        ("Peak_mass_kg", "Peak mass (kg)"),                # trailing unit moves into parentheses
        ("Share_of_total_pct", "Share of total (%)"),
        ("Heat_load_proxy", "Heat load (proxy)"),
        ("Cs_kg_frac", "Cs fraction (kg)"),                # an element symbol keeps its capital
        ("MeanDSTVolume", "Mean DST volume"),              # an acronym keeps its capitals
    ])
    def test_unknown_identifiers_are_split_and_given_their_unit(self, name, label):
        assert tf.humanize(name) == label

    @pytest.mark.parametrize("name", ["Na", "137Cs", "239/240Pu", "SO4", "PC1", "r", "241-AW-106",
                                      "Dominant waste phase", "Element", ""])
    def test_names_that_already_read_as_labels_are_left_alone(self, name):
        assert tf.humanize(name) == name

    def test_every_label_in_the_table_is_free_of_identifier_syntax(self):
        for name, label in tf.COLUMN_LABELS.items():
            assert "_" not in label, f"{name!r} -> {label!r}"
            assert label.strip() == label and label


class TestHumanizeText:
    def test_identifier_inside_a_sentence_is_replaced_in_lower_case(self):
        assert tf.humanize_text("Tank count by TankType") == "Tank count by tank type"

    def test_identifier_at_the_start_keeps_its_capital(self):
        assert tf.humanize_text("Inventory_sum") == "Inventory"
        assert tf.humanize_text("User_search_score_proxy (dimensionless)") == "Search score (proxy) (dimensionless)"

    def test_metric_identifier_inside_parentheses(self):
        assert tf.humanize_text("Element correlation heatmap (log10_plus1, kg)") == (
            "Element correlation heatmap (log10(inventory + 1), kg)")

    @pytest.mark.parametrize("text", [
        "Salt formers — g", "F vs Cl (mol per mol Na)", "14 tanks, every one counted",
        "Top elements by inventory", "log10 Total mass (kg)", "",
    ])
    def test_ordinary_text_is_not_touched(self, text):
        assert tf.humanize_text(text) == text

    def test_plural_label_for_running_text(self):
        assert tf.plural_label("WasteSiteId") == "tanks"
        assert tf.plural_label("WastePhase") == "waste phases"
        assert tf.plural_label("Analyte") == "analytes"
        assert tf.plural_label("Cluster") == "clusters"


class TestTankId:
    @pytest.mark.parametrize("label, tank", [
        ("241-AW-106", "241-AW-106"),
        ("241-A-101", "241-A-101"),
        ("241-AW-102/Saltcake Interstitial Liquid", "241-AW-102"),
        ("241-BX-110, Sludge", "241-BX-110"),
        ("Na", None), ("", None), (None, None), ("2410-A-101", None), ("241-A-1010", None),
    ])
    def test_tank_id_of(self, label, tank):
        assert tf.tank_id_of(label) == tank


class TestFormatNumber:
    @pytest.mark.parametrize("value, text", [
        (0, "0"), (42, "42"), (46597, "46,597"), (np.int64(1234567), "1,234,567"),
        (0.0, "0"),
        (0.5, "0.5"), (12.25, "12.25"), (123.456, "123.5"),
        (1000.0, "1,000"), (46597.0, "46,597"), (1234567.891, "1,234,568"), (262800000.0, "262,800,000"),
        (999.96, "1,000"),                                   # rounds up into the thousands
        (0.001, "0.001"),                                    # the lower limit is still plain
        (0.0005, "5e-4"), (-0.0005, "-5e-4"),                # below it: scientific
        (2.6e10, "2.6e10"), (1e9, "1e9"),                    # from 1e9 up: scientific
        (-12345.6, "-12,346"),
    ])
    def test_rules(self, value, text):
        assert tf.format_number(value) == text

    def test_about_four_significant_figures(self):
        assert tf.format_number(3.14159265) == "3.142"
        assert tf.format_number(0.0123456) == "0.01235"
        assert tf.format_number(1.23456e-7) == "1.235e-7"

    def test_missing_is_blank(self):
        assert tf.format_number(float("nan")) == ""
        assert tf.format_number(None) == ""
        assert tf.format_number(np.float64("nan")) == ""

    def test_booleans_read_as_words(self):
        assert tf.format_number(True) == "Yes"
        assert tf.format_number(np.bool_(False)) == "No"

    def test_infinity(self):
        assert tf.format_number(math.inf) == "∞"

    def test_with_unit(self):
        assert tf.with_unit(1234.4, "kg") == "1,234 kg"
        assert tf.with_unit(12.5, "%") == "12.5%"
        assert tf.with_unit(float("nan"), "kg") == ""
        assert tf.with_unit(3.0, "") == "3"


class TestFormatCell:
    def test_none_and_nan(self):
        assert tf.format_cell(None) == ""
        assert tf.format_cell(float("nan")) == ""
        assert tf.format_cell(pd.NA) == ""
        assert tf.format_cell(pd.NaT) == ""

    def test_text_is_unchanged(self):
        assert tf.format_cell("Cs") == "Cs"

    def test_lists_are_joined(self):
        assert tf.format_cell(["A", "B", "C"]) == "A; B; C"
        assert tf.format_cell(np.array(["Cs", "Sr"], dtype=object)) == "Cs; Sr"
        assert tf.format_cell(np.array([], dtype=object)) == ""

    def test_dates_drop_a_midnight_time(self):
        assert tf.format_cell(pd.Timestamp("2015-07-01")) == "2015-07-01"
        assert tf.format_cell(pd.Timestamp("2015-07-01 13:45")) == "2015-07-01 13:45"

    def test_full_precision_for_copy_and_tooltips(self):
        assert tf.format_full(1234567.891) == "1234567.891"
        assert tf.format_full(float("nan")) == ""
        assert tf.format_full(np.float64(0.1)) == "0.1"


class TestUnitFromLabel:
    @pytest.mark.parametrize("label, unit", [
        ("Total mass (g)", "g"), ("Total mass (g) — log scale", "g"), ("Total activity (Ci)", "Ci"),
        ("Cs (inventory, kg)", "kg"), ("Search score (proxy)", ""), ("Correlation r", ""), ("", ""),
    ])
    def test_unit_from_label(self, label, unit):
        assert tf.unit_from_label(label) == unit


class TestNumberStyle:
    """One way of writing numbers per column, so a column reads downward."""

    def test_similar_magnitudes_share_one_number_of_decimals(self):
        values = [-2.09, 0.06639, 1.366, 3.483]
        style = tf.number_style(values)
        assert style == {"mode": "fixed", "decimals": 3}          # four significant figures on 3.483
        assert [tf.format_number(v, style=style) for v in values] == ["-2.090", "0.066", "1.366", "3.483"]

    def test_integral_floats_have_no_decimals(self):
        style = tf.number_style([1.0, 250.0, 1000000.0])
        assert style == {"mode": "fixed", "decimals": 0}
        assert tf.format_number(1000000.0, style=style) == "1,000,000"

    def test_a_cell_too_small_for_the_column_is_never_a_false_zero(self):
        style = {"mode": "fixed", "decimals": 3}
        assert tf.format_number(0.0004, style=style) == "4e-4"     # not "0.000"
        assert tf.format_number(0.0, style=style) == "0"

    def test_a_column_reaching_a_billion_is_scientific_throughout(self):
        style = tf.number_style([920821584.0, 2.82e9])
        assert style == {"mode": "sci"}
        assert [tf.format_number(v, style=style) for v in (920821584.0, 2.82e9)] == ["9.208e8", "2.82e9"]

    def test_values_spread_over_many_decades_are_formatted_cell_by_cell(self):
        assert tf.number_style([1e-5, 0.02, 3.0, 5e6]) is None

    def test_nothing_to_go_on(self):
        assert tf.number_style([]) is None
        assert tf.number_style([float("nan")]) is None
        assert tf.number_style([0.0, 0.0]) is None
        assert tf.number_style(["a", "b"]) is None


class TestFormatReading:
    def test_a_tooltip_writes_a_large_number_out(self):
        assert tf.format_reading(1238412345.0) == "1,238,412,345"
        assert tf.with_unit(1.238e9, "g") == "1,238,000,000 g"

    def test_everything_else_is_format_number(self):
        assert tf.format_reading(12.5) == "12.5"
        assert tf.format_reading(2e16) == "2e16"
        assert tf.format_reading(float("nan")) == ""

    def test_a_logarithm_or_a_ratio_carries_no_unit(self):
        assert tf.unit_from_label("log10 Total mass (kg)") == ""
        assert tf.unit_from_label("fraction of row total (Total mass (kg))") == ""
