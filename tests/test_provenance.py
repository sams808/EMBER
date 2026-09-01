"""
Provenance record tests (Stage 3): can someone holding only the export
folder tell which data, which question, and which assumptions produced
the number?
"""
import json

import polars as pl
import pytest

import data_model as dm
import provenance as prov

from conftest import REAL_CSV_PATH, requires_real_data


def _dataset() -> dm.HanfordDataset:
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-A-101"],
        "Analyte": ["SO4"],
        "WastePhase": ["Sludge"],
        "WasteType": ["T1"],
        "Inventory": [100.0],
        "Units": ["kg"],
    }))
    dataset.report = None
    return dataset


class TestFingerprint:
    def test_unloaded_dataset_reports_unknowns_not_a_crash(self):
        out = prov.dataset_fingerprint(dm.HanfordDataset())
        assert out["dataset_path"] == ""
        assert out["dataset_rows"] is None

    def test_in_memory_dataset_reports_its_shape(self):
        out = prov.dataset_fingerprint(_dataset())
        assert out["dataset_rows"] == 1
        assert out["dataset_columns"] > 0

    def test_a_missing_source_file_is_recorded_as_unknown(self, tmp_path):
        dataset = _dataset()
        dataset.path = tmp_path / "gone.csv"
        out = prov.dataset_fingerprint(dataset)
        assert out["dataset_sha256_head"] == ""  # no hash, but no exception

    @requires_real_data
    def test_real_file_gets_a_hash_and_a_size(self):
        dataset = dm.HanfordDataset()
        dataset.load(REAL_CSV_PATH, use_cache=True)
        out = prov.dataset_fingerprint(dataset)
        assert len(out["dataset_sha256_head"]) == 64
        assert out["dataset_bytes"] > 0
        assert out["dataset_rows"] == 46597  # 2026 PHOENIX re-export

    @requires_real_data
    def test_the_hash_distinguishes_a_changed_file(self, tmp_path):
        # A re-published BBI revision under the same name must not look
        # like the same data.
        lines = REAL_CSV_PATH.read_text(encoding="utf-8", errors="replace").splitlines(True)
        edited = tmp_path / "Hanford.csv"
        edited.write_text("".join(lines[:5000]), encoding="utf-8")  # whole rows only
        a = dm.HanfordDataset()
        a.load(REAL_CSV_PATH, use_cache=True)
        b = dm.HanfordDataset()
        b.load(edited, use_cache=False)
        assert (prov.dataset_fingerprint(a)["dataset_sha256_head"]
                != prov.dataset_fingerprint(b)["dataset_sha256_head"])
        assert (prov.dataset_fingerprint(a)["dataset_rows"]
                != prov.dataset_fingerprint(b)["dataset_rows"])


class TestProvenanceRecord:
    def _record(self, **kwargs):
        return prov.build_provenance(
            _dataset(), app_name="Ember", app_version="0.1.0",
            question={"subject": "S", "basis": "kg", "group_by": ["WasteSiteId"]},
            plot_spec={"mark": "bar", "x": "Total_kg", "y": "WasteSiteId"},
            scope="Farm AN", warnings=["1 row excluded"],
            disclosures=["S: apportioned from SO4 (kg)."], **kwargs,
        )

    def test_round_trips_through_json(self):
        record = self._record()
        assert json.loads(record.to_json())["question"]["basis"] == "kg"

    def test_records_the_reference_tables_in_force(self):
        record = self._record()
        assert "CIAAW" in record.reference_tables["atomic_weights"]
        assert "ICRP" in record.reference_tables["half_lives"]

    def test_flat_table_covers_nested_fields_and_lists(self):
        table = self._record().as_table().set_index("key")["value"]
        assert table["question.basis"] == "kg"
        assert table["warnings[0]"] == "1 row excluded"
        assert table["scope"] == "Farm AN"

    def test_records_the_environment(self):
        record = self._record()
        assert record.python_version and record.platform
        assert record.manifest_version == prov.MANIFEST_VERSION


class TestMethodsParagraph:
    def _text(self, **overrides):
        question = {"subject": "S", "basis": "kg", "group_by": ["WasteSiteId"]}
        question.update(overrides.pop("question", {}))
        record = prov.build_provenance(
            _dataset(), app_name="Ember", app_version="0.1.0", question=question,
            **overrides,
        )
        return prov.methods_paragraph(record)

    def test_states_subject_basis_and_grouping(self):
        text = self._text()
        assert "S was quantified" in text
        assert "as reported mass (kg)" in text
        assert "grouped by WasteSiteId" in text

    def test_explains_the_bridged_basis_including_the_overlap_rule(self):
        text = self._text(question={"basis": "g"})
        assert "ICRP-107" in text
        assert "not already reported as a mass" in text, (
            "the methods text must state WHY some activity rows were set aside"
        )

    def test_always_states_the_apportionment_convention(self):
        assert "apportioned stoichiometrically" in self._text()

    def test_carries_the_caveats_not_just_the_result(self):
        text = self._text(warnings=["26,108 Ci rows excluded under the kg basis"],
                          disclosures=["S: apportioned from SO4 (kg)."])
        assert "26,108 Ci rows excluded" in text
        assert "apportioned from SO4" in text

    def test_names_the_scope_when_one_is_active(self):
        assert "restricted to Farm AN" in self._text(scope="Farm AN")

    def test_omits_the_scope_sentence_when_unscoped(self):
        assert "restricted to" not in self._text()
