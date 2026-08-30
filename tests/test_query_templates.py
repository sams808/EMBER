"""
Guided-template tests: every shipped template must be internally valid
against the engine's registries, and must actually RUN — returning rows —
on the real dataset. A template that errors or comes back empty is a
broken promise in the easiest-mode UI.
"""
import pytest

import data_model as dm
from plot_grammar import MARKS
from query_engine import BASES, BASKET_PRESETS, DIMENSIONS, LAYER_COMPATIBLE_DIMS, run_query
from query_templates import TEMPLATES, template_by_key

from conftest import REAL_CSV_PATH, requires_real_data


class TestTemplateRegistry:
    def test_keys_are_unique(self):
        keys = [t.key for t in TEMPLATES]
        assert len(keys) == len(set(keys))

    def test_every_template_references_shipped_engine_capabilities(self):
        for t in TEMPLATES:
            assert t.subject_kind in ("basket", "element", "analyte"), t.key
            assert t.basis in BASES, t.key
            assert t.mark in MARKS, t.key
            for dim in t.group_by:
                assert dim in DIMENSIONS, f"{t.key}: {dim}"
            if t.subject_kind == "basket":
                assert t.subject in BASKET_PRESETS, t.key
            if t.basis in ("wt%", "g/L"):
                assert all(d in LAYER_COMPATIBLE_DIMS for d in t.group_by), (
                    f"{t.key}: concentration basis with non-layer grouping would be refused at runtime"
                )

    def test_lookup_by_key(self):
        assert template_by_key("salt_formers_wtpct") is TEMPLATES[0]
        assert template_by_key("nope") is None


@pytest.fixture(scope="module")
def real_dataset():
    dataset = dm.HanfordDataset()
    dataset.load(REAL_CSV_PATH, use_cache=True)
    return dataset


@requires_real_data
class TestTemplatesOnRealData:
    def test_every_template_runs_and_returns_rows(self, real_dataset):
        for t in TEMPLATES:
            subject = (
                ("basket", BASKET_PRESETS[t.subject]) if t.subject_kind == "basket"
                else (t.subject_kind, t.subject)
            )
            res = run_query(real_dataset, subject=subject, basis=t.basis, group_by=list(t.group_by))
            assert not res.table.empty, f"template {t.key} returned no rows on the real dataset"
            finite = res.table[res.value_column].dropna()
            assert (finite > 0).any(), f"template {t.key} returned no positive values"
