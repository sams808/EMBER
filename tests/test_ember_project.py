"""
.ember project file tests (Stage 3): saved work must round-trip, must
refuse a format it cannot read rather than silently dropping half of it,
and must say what has gone stale when reopened on different data.
"""
import json

import pytest

from ember_project import (
    PROJECT_SUFFIX,
    PROJECT_VERSION,
    EmberProject,
    SavedQuestion,
    dataset_mismatch,
)


def _project() -> EmberProject:
    project = EmberProject(name="Salt formers study", description="for the survey")
    project.add_basket("My halides", ["F", "Cl", "Br", "I"])
    project.add_question(SavedQuestion(
        name="halides by tank", subject_kind="basket", subject="My halides",
        basis="wt%", group_by=["WasteSiteId", "WastePhase"], mark="bar",
    ))
    project.tank_lists["leakers"] = ["241-A-101", "241-AX-104"]
    project.scope = {"TankFarm": ["AN"]}
    return project


class TestRoundTrip:
    def test_survives_json(self):
        restored = EmberProject.from_json(_project().to_json())
        assert restored.name == "Salt formers study"
        assert restored.baskets["My halides"] == ["F", "Cl", "Br", "I"]
        assert restored.questions[0].basis == "wt%"
        assert restored.questions[0].group_by == ["WasteSiteId", "WastePhase"]
        assert restored.scope == {"TankFarm": ["AN"]}
        assert restored.tank_lists["leakers"] == ["241-A-101", "241-AX-104"]

    def test_save_and_load_from_disk(self, tmp_path):
        path = _project().save(tmp_path / "study")
        assert path.suffix == PROJECT_SUFFIX
        assert EmberProject.load(path).questions[0].name == "halides by tank"

    def test_save_adds_the_suffix_but_does_not_double_it(self, tmp_path):
        assert _project().save(tmp_path / "a").name == "a" + PROJECT_SUFFIX
        assert _project().save(tmp_path / ("b" + PROJECT_SUFFIX)).name == "b" + PROJECT_SUFFIX


class TestForwardAndBackwardCompatibility:
    def test_unknown_keys_from_a_newer_build_are_ignored(self):
        data = json.loads(_project().to_json())
        data["something_new"] = {"a": 1}
        data["questions"][0]["future_field"] = True
        restored = EmberProject.from_dict(data)   # must not raise
        assert restored.questions[0].subject == "My halides"

    def test_a_newer_format_version_is_refused_not_half_read(self):
        data = json.loads(_project().to_json())
        data["version"] = PROJECT_VERSION + 1
        with pytest.raises(ValueError, match="newer Ember"):
            EmberProject.from_dict(data)

    def test_a_project_with_no_version_is_assumed_current(self):
        data = json.loads(_project().to_json())
        del data["version"]
        assert EmberProject.from_dict(data).version == PROJECT_VERSION


class TestBasketsBecomeEngineObjects:
    def test_saved_baskets_convert_for_querying(self):
        baskets = _project().as_baskets()
        assert baskets["My halides"].members == ("F", "Cl", "Br", "I")

    def test_a_saved_basket_actually_runs(self):
        import polars as pl
        import data_model as dm
        from query_engine import run_query

        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-A-101", "241-A-101"],
            "Analyte": ["Cl", "F"],
            "WastePhase": ["Sludge"] * 2,
            "WasteType": ["T1"] * 2,
            "Inventory": [3.0, 2.0],
            "Units": ["kg", "kg"],
        }))
        dataset.report = None
        basket = _project().as_baskets()["My halides"]
        result = run_query(dataset, subject=("basket", basket), basis="kg")
        assert result.table["Total_kg"].iloc[0] == pytest.approx(5.0)


class TestStaleProjectsAreReported:
    def test_a_bad_basis_is_reported_without_refusing_the_project(self):
        project = _project()
        project.questions[0].basis = "furlongs"
        problems = project.problems()
        assert any("furlongs" in p for p in problems)
        assert project.questions[0].name in problems[0]

    def test_an_unknown_grouping_is_reported(self):
        project = _project()
        project.questions[0].group_by = ["NotADimension"]
        assert any("NotADimension" in p for p in project.problems())

    def test_a_basket_that_no_longer_exists_is_reported(self):
        project = _project()
        del project.baskets["My halides"]
        assert any("neither saved" in p for p in project.problems())

    def test_a_shipped_preset_basket_is_not_reported_as_missing(self):
        project = EmberProject()
        project.add_question(SavedQuestion(
            name="q", subject_kind="basket", subject="Salt formers", basis="kg"))
        assert project.problems() == []

    def test_a_healthy_project_reports_nothing(self):
        assert _project().problems() == []


class TestDatasetMismatch:
    def test_silent_when_no_fingerprint_was_saved(self):
        assert dataset_mismatch(_project(), {"dataset_rows": 10}) is None

    def test_detects_a_changed_content_fingerprint(self):
        project = _project()
        project.dataset_fingerprint = {
            "dataset_sha256_head": "aaa", "dataset_rows": 100, "dataset_path": "old.csv"}
        message = dataset_mismatch(project, {"dataset_sha256_head": "bbb", "dataset_rows": 100})
        assert message and "different content fingerprint" in message

    def test_detects_a_changed_row_count_without_a_hash(self):
        project = _project()
        project.dataset_fingerprint = {"dataset_rows": 100}
        message = dataset_mismatch(project, {"dataset_rows": 120})
        assert message and "100" in message and "120" in message

    def test_silent_when_the_data_matches(self):
        project = _project()
        project.dataset_fingerprint = {"dataset_sha256_head": "aaa", "dataset_rows": 100}
        assert dataset_mismatch(project, {"dataset_sha256_head": "aaa",
                                          "dataset_rows": 100}) is None
