"""
ember_project.py — saved work (.ember project files, Stage 3, L4).

A project holds the things a user builds up over a session and would
otherwise retype: custom baskets, the questions they care about, tank
lists, and the active scope. It is possible only because the query and
plot layers are declarative — a QuerySpec and a PlotSpec are plain data,
so a saved question is a fact rather than a closure.

The file is JSON with a version field. Loading is deliberately tolerant
of unknown keys (a newer Ember's project opens in an older one, minus
what it doesn't understand) but STRICT about a version it cannot read,
because silently dropping half a project is worse than refusing it.

Nothing here touches Qt: a project can be built, saved and replayed from
a script or a test.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from query_engine import BASES, Basket, DIMENSIONS

PROJECT_VERSION = 1
PROJECT_SUFFIX = ".ember"


@dataclass
class SavedQuestion:
    """One question, exactly as the Query workspace would ask it."""

    name: str = ""
    subject_kind: str = "element"       # "element" | "analyte" | "basket"
    subject: str = ""                   # symbol, analyte label, or basket name
    basis: str = "kg"
    group_by: List[str] = field(default_factory=list)
    scope: Dict[str, List[str]] = field(default_factory=dict)
    mark: str = "bar"
    note: str = ""

    def validate(self) -> List[str]:
        """Problems that would make this question un-runnable, as plain
        sentences. Returned rather than raised: a project with one stale
        question should still open, with that question flagged."""
        problems: List[str] = []
        if self.subject_kind not in ("element", "analyte", "basket"):
            problems.append(f"{self.name or 'question'}: unknown subject kind "
                            f"{self.subject_kind!r}")
        if not self.subject:
            problems.append(f"{self.name or 'question'}: no subject")
        if self.basis not in BASES:
            problems.append(f"{self.name or 'question'}: unknown basis {self.basis!r}")
        unknown = [d for d in self.group_by if d not in DIMENSIONS]
        if unknown:
            problems.append(f"{self.name or 'question'}: unknown grouping {unknown}")
        return problems


@dataclass
class EmberProject:
    version: int = PROJECT_VERSION
    name: str = ""
    description: str = ""
    baskets: Dict[str, List[str]] = field(default_factory=dict)
    questions: List[SavedQuestion] = field(default_factory=list)
    tank_lists: Dict[str, List[str]] = field(default_factory=dict)
    scope: Dict[str, List[str]] = field(default_factory=dict)
    # What the project was built against, so a mismatch is visible when
    # it is reopened on a different BBI revision.
    dataset_fingerprint: Dict[str, object] = field(default_factory=dict)

    # ---------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "EmberProject":
        version = int(data.get("version", PROJECT_VERSION) or PROJECT_VERSION)
        if version > PROJECT_VERSION:
            raise ValueError(
                f"This project was written by a newer Ember (format {version}; "
                f"this build reads {PROJECT_VERSION}). Opening it here would silently "
                f"drop whatever it added."
            )
        known = {f.name for f in fields(cls)}
        payload = {k: v for k, v in data.items() if k in known}
        payload["version"] = version
        raw_questions = payload.get("questions") or []
        question_fields = {f.name for f in fields(SavedQuestion)}
        payload["questions"] = [
            SavedQuestion(**{k: v for k, v in q.items() if k in question_fields})
            if isinstance(q, dict) else q
            for q in raw_questions
        ]
        return cls(**payload)

    @classmethod
    def from_json(cls, text: str) -> "EmberProject":
        return cls.from_dict(json.loads(text))

    # ---------------------------------------------------------------
    def save(self, path: Path | str) -> Path:
        out = Path(path)
        if out.suffix != PROJECT_SUFFIX:
            out = out.with_suffix(PROJECT_SUFFIX)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: Path | str) -> "EmberProject":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # ---------------------------------------------------------------
    def add_basket(self, name: str, members: Sequence[str]) -> None:
        self.baskets[name] = [str(m) for m in members]

    def as_baskets(self) -> Dict[str, Basket]:
        """The project's baskets as engine objects, ready to query."""
        return {
            name: Basket(name, tuple(members), "saved in this project")
            for name, members in self.baskets.items()
        }

    def add_question(self, question: SavedQuestion) -> None:
        self.questions.append(question)

    def problems(self) -> List[str]:
        """Everything in this project that would not run as saved --
        collected, so reopening a project on a changed dataset says what
        is stale instead of failing on the first one."""
        found: List[str] = []
        for question in self.questions:
            found.extend(question.validate())
            if question.subject_kind == "basket" and question.subject not in self.baskets:
                from query_engine import BASKET_PRESETS
                if question.subject not in BASKET_PRESETS:
                    found.append(
                        f"{question.name or 'question'}: basket {question.subject!r} "
                        f"is neither saved in this project nor a shipped preset"
                    )
        return found


def dataset_mismatch(project: EmberProject, current: Dict[str, object]) -> Optional[str]:
    """A sentence describing how the loaded data differs from what the
    project was built on, or None when they agree. Compares the content
    fingerprint first and falls back to the row count, so a re-published
    BBI revision is caught even if the filename never changed."""
    saved = project.dataset_fingerprint or {}
    if not saved:
        return None
    saved_hash = str(saved.get("dataset_sha256_head") or "")
    current_hash = str(current.get("dataset_sha256_head") or "")
    if saved_hash and current_hash and saved_hash != current_hash:
        return (
            f"This project was built on {saved.get('dataset_path') or 'another dataset'} "
            f"({saved.get('dataset_rows')} rows); the loaded data has a different content "
            f"fingerprint ({current.get('dataset_rows')} rows). Numbers may have moved."
        )
    saved_rows, current_rows = saved.get("dataset_rows"), current.get("dataset_rows")
    if saved_rows and current_rows and saved_rows != current_rows:
        return (
            f"This project was built on {saved_rows:,} rows; the loaded data has "
            f"{current_rows:,}. Numbers may have moved."
        )
    return None
