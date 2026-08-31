"""
provenance.py — what produced a number (Stage 3, L4).

Every figure and table Ember exports should be reconstructable a year
later by someone who has only the export folder. That needs three things
recorded next to the data:

- WHICH DATA: the source file, its size and row count, and a content
  fingerprint so a later reader can tell whether the CSV they hold is the
  one the figure came from. The BBI is re-published periodically; "I ran
  it on Hanford.csv" identifies nothing on its own.
- WHICH QUESTION: the query (subject, basis, grouping, scope) and the
  plot spec, both already declarative and therefore serializable.
- WHICH ASSUMPTIONS: the reference tables and app version in force, plus
  every warning and disclosure the engine raised. A number is only as
  good as the exclusions that produced it.

`methods_paragraph()` turns that record into prose a scientist can paste
into a methods section, because the alternative is that they write it
from memory.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from data_model import HanfordDataset

MANIFEST_VERSION = "1"
_FINGERPRINT_BYTES = 4 * 1024 * 1024  # enough to separate BBI revisions cheaply


@dataclass
class Provenance:
    """One export's full record. Plain data: to_dict() is JSON-safe."""

    created: str = ""
    app_name: str = ""
    app_version: str = ""
    python_version: str = ""
    platform: str = ""
    manifest_version: str = MANIFEST_VERSION

    dataset_path: str = ""
    dataset_sha256_head: str = ""
    dataset_bytes: Optional[int] = None
    dataset_rows: Optional[int] = None
    dataset_columns: Optional[int] = None
    attributes_path: str = ""
    attributes_rows: Optional[int] = None

    question: Dict[str, object] = field(default_factory=dict)
    plot_spec: Dict[str, object] = field(default_factory=dict)
    scope: str = ""
    warnings: List[str] = field(default_factory=list)
    disclosures: List[str] = field(default_factory=list)
    reference_tables: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    def as_table(self) -> pd.DataFrame:
        """Flat key/value rows, for writing beside the CSVs an export
        already produces."""
        rows: List[Dict[str, str]] = []
        for key, value in self.to_dict().items():
            if isinstance(value, dict):
                for sub, sub_value in value.items():
                    rows.append({"key": f"{key}.{sub}", "value": str(sub_value)})
            elif isinstance(value, list):
                for i, entry in enumerate(value):
                    rows.append({"key": f"{key}[{i}]", "value": str(entry)})
            else:
                rows.append({"key": key, "value": "" if value is None else str(value)})
        return pd.DataFrame(rows)


def dataset_fingerprint(dataset: HanfordDataset) -> Dict[str, object]:
    """Identify the loaded data. The hash covers the first few MB of the
    source file rather than all of it: enough to distinguish one BBI
    revision from another without making every export wait on a full
    read of a large CSV, and it is labelled as a head hash so nobody
    mistakes it for a whole-file checksum."""
    out: Dict[str, object] = {
        "dataset_path": "", "dataset_sha256_head": "", "dataset_bytes": None,
        "dataset_rows": None, "dataset_columns": None,
        "attributes_path": "", "attributes_rows": None,
    }
    report = getattr(dataset, "report", None)
    if report is not None:
        out["dataset_path"] = str(report.source_path)
        out["dataset_rows"] = report.rows
        out["dataset_columns"] = report.columns
        if report.attributes_path is not None:
            out["attributes_path"] = str(report.attributes_path)
        out["attributes_rows"] = report.attributes_rows
    elif getattr(dataset, "path", None) is not None:
        out["dataset_path"] = str(dataset.path)
    if out["dataset_rows"] is None and dataset.is_loaded():
        out["dataset_rows"] = dataset.require_df().height
        out["dataset_columns"] = dataset.require_df().width

    path = Path(out["dataset_path"]) if out["dataset_path"] else None
    if path is not None and path.is_file():
        try:
            out["dataset_bytes"] = path.stat().st_size
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                digest.update(handle.read(_FINGERPRINT_BYTES))
            out["dataset_sha256_head"] = digest.hexdigest()
        except OSError:
            pass  # an unreadable source is recorded as unknown, not fatal
    return out


def reference_versions() -> Dict[str, str]:
    """What the numbers were computed against. Hard-coded strings rather
    than a scrape, so the citation in an export is the one a human
    checked."""
    return {
        "atomic_weights": "CIAAW 2021 (Prohaska et al., Pure Appl. Chem. 94 (2022) 573)",
        "half_lives": "ICRP Publication 107 (2008), updated to current ENSDF/NUBASE "
                      "evaluations where they differ by more than 1%",
        "species_formulas": "BBI analyte labels, formulas bundled in reference_data.py",
    }


def build_provenance(
    dataset: HanfordDataset,
    *,
    app_name: str = "",
    app_version: str = "",
    question: Optional[Dict[str, object]] = None,
    plot_spec: Optional[Dict[str, object]] = None,
    scope: str = "",
    warnings: Optional[Sequence[str]] = None,
    disclosures: Optional[Sequence[str]] = None,
    created: Optional[str] = None,
) -> Provenance:
    return Provenance(
        created=created or datetime.now().isoformat(timespec="seconds"),
        app_name=app_name, app_version=app_version,
        python_version=sys.version.split()[0], platform=platform.platform(),
        question=dict(question or {}), plot_spec=dict(plot_spec or {}), scope=scope,
        warnings=list(warnings or []), disclosures=list(disclosures or []),
        reference_tables=reference_versions(),
        **dataset_fingerprint(dataset),
    )


def methods_paragraph(prov: Provenance) -> str:
    """Prose for a methods section. States the exclusions as plainly as
    the results: a methods paragraph that omits them is the thing this
    module exists to prevent."""
    question = prov.question or {}
    subject = question.get("subject") or "the selected subject"
    basis = question.get("basis") or ""
    group_by = question.get("group_by") or []

    basis_text = {
        "kg": "as reported mass (kg)",
        "Ci": "as reported activity (Ci)",
        "g": "as mass in grams, with activity rows converted using ICRP-107 specific "
             "activities where the element was not already reported as a mass",
        "wt%": "as a percentage of each layer's mass (layer volume x component density)",
        "g/L": "as grams per litre of layer volume",
    }.get(str(basis), f"on a {basis} basis" if basis else "")

    parts: List[str] = []
    lead = f"{subject} was quantified"
    if basis_text:
        lead += f" {basis_text}"
    if group_by:
        lead += ", grouped by " + " and ".join(str(g) for g in group_by)
    if prov.scope:
        lead += f", restricted to {prov.scope}"
    parts.append(lead + ".")

    parts.append(
        "Elemental quantities are apportioned stoichiometrically: an element receives "
        "its mass fraction of every species reporting it, computed from "
        f"{prov.reference_tables.get('atomic_weights', 'standard atomic weights')}."
    )

    source = prov.dataset_path or "the loaded dataset"
    detail = []
    if prov.dataset_rows:
        detail.append(f"{prov.dataset_rows:,} rows")
    if prov.dataset_sha256_head:
        detail.append(f"sha256(first {_FINGERPRINT_BYTES // (1024 * 1024)} MB) "
                      f"{prov.dataset_sha256_head[:12]}")
    parts.append(
        f"Data: {source}" + (f" ({', '.join(detail)})" if detail else "") + "."
    )

    if prov.warnings:
        parts.append("Caveats recorded at export: " + " ".join(prov.warnings))
    if prov.disclosures:
        parts.append("Source resolution: " + " ".join(prov.disclosures))
    if prov.app_name:
        parts.append(f"Produced with {prov.app_name} {prov.app_version}.".replace("  ", " "))
    return "\n\n".join(parts)
