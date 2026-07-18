# Ember

A desktop app for exploring Hanford nuclear-waste tank composition data and
screening tanks for vitrification (glass immobilization) — search elements
and analytes across 177 tanks, build tank×element heatmaps and correlation
maps, and estimate oxide-based glass chemistry from elemental inventories.

Ember's concept — a focused, single-purpose data tool for a specific DOE
dataset — was inspired by PNNL's
[Phoenix platform](https://phoenix.pnnl.gov/phoenix/apps/gallery/index.html),
a gallery of internally-developed PNNL data-science applications. Ember is an
independent project, not produced by, affiliated with, or endorsed by PNNL or
Phoenix.

The app is PySide6/Qt-based: one main window with a left navigation rail of
workspaces. It is a from-scratch rewrite of an earlier Tkinter prototype
(`hanford_tank_gui_app.py`); that prototype's full history remains archived
alongside this repo for reference, not in this git history.

## Workspaces

| Workspace | What it does |
|---|---|
| **Overview** | Dataset audit: units, top elements/analytes, waste-phase/type/farm breakdowns, missing values, debug-bundle export. |
| **Element Explorer** | Search by element symbol, analyte (exact/contains/regex); co-elements/co-analytes present alongside a target, composition stats, 7 plot types. |
| **Tank Attributes** | Browse joined tank engineering metadata (type, capacity, integrity, status). |
| **Tank Explorer** | Multi-tank composition profiles, fraction-of-tank-total, raw-row drill-down. |
| **Heatmaps** | Tank×element inventory matrices (log/raw/fraction modes). |
| **Correlations** | Quick Scan (target/dual/triple/full-matrix correlation), Association Workbench (kg-only Jaccard co-presence + preferred-association scoring, 17 plot types), Structure (PCA/clustering, partial correlation, network graph, interactive Plotly export). |
| **Vitrification** | Screening (adjustable-weight glass-formulation heuristic), Oxide Chemistry (element→oxide conversion, NBO/T, composition envelope checking, optional GlassNet ML property prediction), Candidate Search, Blend Partners. |
| **Debug / Export** | Global debug bundle export, environment info. |

*(Table fills in as each workspace ships — see the project plan.)*

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt      # core science stack
pip install -r requirements-qt.txt   # PySide6
pip install -r requirements-dev.txt  # pytest + pytest-qt, for running the tests
```

Optional: `pip install glasspy` enables GlassNet ML property prediction in
the Oxide Chemistry workspace (needs PyTorch — not included in the packaged
`.exe`, Python-run only, same tradeoff as `xraylarch` in the sibling Dataapp
project).

## Running

```bash
python qt_main.py
```

Or double-click `Ember.bat`. Place `Hanford.csv` and `Tank_attributes.csv`
next to the script (or the built `.exe`) — Ember auto-detects them; neither
file is distributed with the app or committed to this repo.

A standalone `Ember.exe` (no Python needed) can be built with
`build_exe.bat` (PyInstaller).

## Repository layout

Flat, one file per concern (mirrors the sibling Dataapp project's
convention): `qt_*.py` modules are the UI, `*_science.py` modules are
framework-agnostic analysis logic with no PySide6 imports and full unit-test
coverage. See `tests/` for the test suite.

## Data

`Hanford.csv` / `Tank_attributes.csv` are not included in this repository —
supply your own copies of the tank composition and tank-attributes datasets.
*(Provenance/citation for these datasets: TODO — confirm exact source and
citation wording.)*
