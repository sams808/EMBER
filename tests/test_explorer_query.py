"""
Engine-backed Element Explorer answers.

The defect this replaces: Explorer's element mode matched every analyte
whose formula contains the symbol and summed those analytes' FULL
masses, so searching "S" booked all of every sulfate row as sulfur AND
searching "O" booked the same sulfate mass again as oxygen.
"""
import polars as pl
import pytest

import data_model as dm
import explorer_query as eq
import reference_data as rd

from conftest import REAL_CSV_PATH, requires_real_data


def _dataset() -> dm.HanfordDataset:
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-A-101", "241-A-101", "241-AN-104"],
        "Analyte": ["SO4", "Sulfide", "137Cs"],
        "WastePhase": ["Sludge", "Sludge", "Supernatant"],
        "WasteType": ["T1", "T1", "T2"],
        "Inventory": [100.0, 5.0, 300.0],
        "Units": ["kg", "kg", "Ci"],
    }))
    dataset.report = None
    return dataset


class TestResolveSubject:
    def test_auto_reads_an_element_symbol_as_an_element(self):
        r = eq.resolve_subject(_dataset(), "S", "auto")
        assert r.ok and r.kind == "element" and r.subject == "S"

    def test_auto_falls_back_to_an_exact_analyte(self):
        r = eq.resolve_subject(_dataset(), "SO4", "auto")
        assert r.ok and r.kind == "analyte" and r.subject == "SO4"

    def test_element_mode_refuses_a_non_element(self):
        r = eq.resolve_subject(_dataset(), "Sulfate", "element")
        assert not r.ok and "not an element" in r.error

    def test_empty_query_is_refused(self):
        assert not eq.resolve_subject(_dataset(), "   ", "auto").ok

    def test_contains_becomes_a_basket_of_what_it_matched(self):
        r = eq.resolve_subject(_dataset(), "Sul", "analyte_contains")
        assert r.ok and r.kind == "basket"
        assert set(r.subject.members) == {"Sulfide"}
        assert "1 analyte(s) matched" in r.note

    def test_regex_becomes_a_basket_too(self):
        r = eq.resolve_subject(_dataset(), r"^S", "regex")
        assert r.ok and set(r.subject.members) == {"SO4", "Sulfide"}

    def test_a_pattern_matching_nothing_is_refused_not_silently_empty(self):
        r = eq.resolve_subject(_dataset(), "zzz", "analyte_contains")
        assert not r.ok and "No analyte" in r.error

    def test_an_invalid_regex_says_so(self):
        r = eq.resolve_subject(_dataset(), "[unclosed", "regex")
        assert not r.ok and "regular expression" in r.error

    def test_every_resolution_carries_a_note_for_disclosure(self):
        for query, mode in (("S", "auto"), ("SO4", "analyte_exact"), ("Sul", "analyte_contains")):
            assert eq.resolve_subject(_dataset(), query, mode).note


class TestTargetTables:
    def test_sulfur_is_apportioned_not_booked_at_full_species_mass(self):
        resolved = eq.resolve_subject(_dataset(), "S", "element")
        out = eq.target_tables(_dataset(), resolved, basis="kg")
        expected = 100.0 * rd.SPECIES_ELEMENT_MASS_FRACTIONS["SO4"]["S"] + 5.0
        assert out.tables["Target by tank"]["Total_kg"].sum() == pytest.approx(expected)

    def test_oxygen_and_sulfur_no_longer_each_claim_the_whole_sulfate(self):
        dataset = _dataset()
        s = eq.target_tables(dataset, eq.resolve_subject(dataset, "S", "element"), basis="kg")
        o = eq.target_tables(dataset, eq.resolve_subject(dataset, "O", "element"), basis="kg")
        s_total = s.tables["Target by tank"]["Total_kg"].sum()
        o_total = o.tables["Target by tank"]["Total_kg"].sum()
        # Their sulfate shares must add to the sulfate mass, not double it.
        so4_share = (100.0 * rd.SPECIES_ELEMENT_MASS_FRACTIONS["SO4"]["S"]
                     + 100.0 * rd.SPECIES_ELEMENT_MASS_FRACTIONS["SO4"]["O"])
        assert so4_share == pytest.approx(100.0)
        assert s_total + o_total == pytest.approx(100.0 + 5.0)

    def test_all_three_views_share_one_subject_so_they_agree(self):
        resolved = eq.resolve_subject(_dataset(), "S", "element")
        out = eq.target_tables(_dataset(), resolved, basis="kg")
        totals = {name: t[out.value_column].sum() for name, t in out.tables.items()}
        assert len(set(round(v, 9) for v in totals.values())) == 1, totals

    def test_warnings_are_collected_without_duplicates(self):
        resolved = eq.resolve_subject(_dataset(), "Cs", "element")
        out = eq.target_tables(_dataset(), resolved, basis="kg")
        # The Ci-excluded warning arises in all three views; report it once.
        assert len(out.warnings) == len(set(out.warnings))
        assert any("excluded under the kg basis" in w for w in out.warnings)

    def test_the_resolution_is_disclosed_first(self):
        resolved = eq.resolve_subject(_dataset(), "S", "element")
        out = eq.target_tables(_dataset(), resolved, basis="kg")
        assert out.disclosures and out.disclosures[0].startswith("S:")
        assert "apportioned" in out.disclosures[0]

    def test_scope_reaches_every_view(self):
        resolved = eq.resolve_subject(_dataset(), "S", "element")
        out = eq.target_tables(_dataset(), resolved, basis="kg",
                               scope={"WastePhase": ["Supernatant"]})
        assert out.tables["Target by tank"].empty

    def test_an_unresolvable_subject_returns_its_reason_not_a_crash(self):
        resolved = eq.resolve_subject(_dataset(), "zzz", "analyte_contains")
        out = eq.target_tables(_dataset(), resolved, basis="kg")
        assert out.is_empty and out.warnings


class TestRawRows:
    def test_element_rows_include_every_species_containing_it(self):
        resolved = eq.resolve_subject(_dataset(), "S", "element")
        rows = eq.raw_target_rows(_dataset(), resolved)
        assert set(rows["Analyte"]) == {"SO4", "Sulfide"}

    def test_rows_are_as_reported_not_apportioned(self):
        # Provenance: the records behind the answer, unmodified.
        resolved = eq.resolve_subject(_dataset(), "S", "element")
        rows = eq.raw_target_rows(_dataset(), resolved)
        assert rows.set_index("Analyte").loc["SO4", "Inventory"] == pytest.approx(100.0)

    def test_pattern_subject_rows_come_from_the_matched_analytes(self):
        resolved = eq.resolve_subject(_dataset(), "^S", "regex")
        rows = eq.raw_target_rows(_dataset(), resolved)
        assert set(rows["Analyte"]) == {"SO4", "Sulfide"}

    def test_scope_applies(self):
        resolved = eq.resolve_subject(_dataset(), "Cs", "auto")
        rows = eq.raw_target_rows(_dataset(), resolved,
                                  scope={"WastePhase": ["Sludge"]})
        assert rows.empty


@requires_real_data
class TestOnRealData:
    @pytest.fixture(scope="module")
    def real_dataset(self):
        dataset = dm.HanfordDataset()
        dataset.load(REAL_CSV_PATH, use_cache=True)
        return dataset

    def test_explorer_agrees_with_the_engine_used_elsewhere(self, real_dataset):
        from query_engine import run_query
        resolved = eq.resolve_subject(real_dataset, "S", "element")
        out = eq.target_tables(real_dataset, resolved, basis="kg")
        direct = run_query(real_dataset, subject=("element", "S"), basis="kg")
        assert out.tables["Target by tank"]["Total_kg"].sum() == pytest.approx(
            direct.table["Total_kg"].sum())

    def test_a_pattern_search_answers_the_pattern(self, real_dataset):
        resolved = eq.resolve_subject(real_dataset, "^24[01]", "regex")
        assert resolved.ok and resolved.kind == "basket"
        out = eq.target_tables(real_dataset, resolved, basis="Ci")
        assert not out.tables["Target by tank"].empty
