# Ember

A desktop app for exploring Hanford nuclear-waste tank composition data and
screening tanks for vitrification (glass immobilization) — search elements
and analytes across 177 tanks, build tank×element heatmaps and correlation
maps, convert compositions to real oxide-basis glass chemistry (optical
basicity, NBO/T, GlassNet ML property prediction), and combine any of the
resulting plots into one labeled multi-panel figure for a report or slide
deck.

The app is PySide6/Qt-based: one main window with a header (dataset
summary, global search, help, light/dark theme), a grouped navigation
sidebar of workspaces, and a scope strip that filters every scope-aware
workspace by farm, waste phase, waste type and tank system at once. It is a
from-scratch rewrite of an earlier Tkinter prototype
(`hanford_tank_gui_app.py`); that prototype's full history remains archived
alongside this repo for reference, not in this git history.

**Ember is an independent project — not produced by, affiliated with, or
endorsed by PNNL or the U.S. Department of Energy.** See [`NOTICE.md`](NOTICE.md)
for the full user manual (every workspace, every formula, every source
citation) and the exact data provenance.

## Workspaces

| Workspace | What it does |
|---|---|
| **Overview** | Site overview: headline figures, an interactive tank-farm map (each tank a disc, coloured by a chosen quantity, click to select, double-click to open), a linked ranking, and the dataset audit tables (units, top elements/analytes, waste-phase/type/farm breakdowns, missing values). |
| **Query** | One flexible question over the query engine: any element, analyte or basket × any basis (kg, Ci, bridged grams, wt%, g/L) × any grouping, with guided questions that fill the controls and run. |
| **Element Explorer** | Search by element symbol, analyte (exact/contains/regex); co-elements/co-analytes present alongside a target, composition stats, 7 plot types. |
| **Tank Attributes** | Browse joined tank engineering metadata (type, capacity, integrity, status). |
| **Tank Explorer** | Multi-tank composition profiles, fraction-of-tank-total, raw-row drill-down. |
| **Heatmaps** | Tank×element inventory matrices (log/raw/fraction modes). |
| **Correlations** | Cross-correlation (engine-backed: which components occur together on a concentration basis such as mol/mol Na, with species and Ci-only nuclides beside elements, each distinct composition counted once, global scope honored), Quick Scan (target/dual/triple/full-matrix correlation, partial-correlation tank-size control), Association Workbench (kg-only Jaccard co-presence + preferred-association scoring, 18 plot types incl. publication-style tank inventory bar charts), Structure (PCA/clustering, partial correlation, network graph, interactive Plotly export). |
| **Vitrification** | Screening (adjustable-weight glass-formulation heuristic), Oxide Chemistry (element→oxide stoichiometric conversion, optical basicity, NBO/T, composition envelope checking, optional GlassNet ML property prediction), Candidate Search, Blend Partners, Loading rules (published LAW glass loading rules of PNNL-22631 Rev. 1 applied to each tank's salt waste: binding rule, sodium loading, glass mass, tank by tank and blended). |
| **Figure Composer** | Combine any plots captured from any workspace into one multi-panel publication figure, with automatic panel labels (A, B, C, ...) and exact-size/DPI export. |

Across workspaces:

- **Charts answer to the pointer.** Hovering a bar, a point or a heat-map
  cell names it and gives its value with unit; a click selects the matching
  table row where there is one; a double click on anything that stands for
  a tank opens that tank in the Tank Explorer. Each chart can be exported
  (PNG, SVG, PDF) or sent to the Figure Composer from its own toolbar.
- **Tables** have a filter box, readable headers, formatted numbers, sorting
  by value, and a context menu to copy cells or export CSV.
- **Search** (Ctrl+K) finds tanks, elements, analytes and workspaces.
- **Notes that qualify a result** (what was set aside, converted or left
  out, and why) appear as one line above it, with the full list one click
  away and in the workspace's Notes tab and exports.
- Rarely-needed controls (thresholds, score weights, per-plot-type tuning)
  sit behind an "Advanced options" toggle; a `?` button on each workspace
  opens that workspace's own section of `NOTICE.md`.
- **Help → Getting Started…** re-opens the first-launch picker at any time.
  Data loading options (local CSVs, parquet cache) are in the **Data** menu;
  **Help → Debug / Export…** holds the live log viewer and the debug-bundle
  export.

## Version 2.0

Version 2.0 is a rebuild on four layers, with the science modules and
their tests carried over:

- **A query engine** (`query_engine.py`): one question surface for any
  element, analyte or basket, on any basis, with stoichiometric
  apportionment of species to elements, activity-to-mass conversion, and
  every exclusion or conversion stated with the result.
- **A plot grammar** (`plot_grammar.py`) and one chart style; charts answer
  to hover, click and double click.
- **Engine-backed workspaces**: Query, Element Explorer, Heatmaps, the Site
  overview with its tank-farm map, the tank profile, Cross-correlation on a
  concentration basis, and published glass loading rules.
- **Provenance**: exports carry the dataset fingerprint, the question, the
  scope and a methods paragraph; `.ember` project files save baskets,
  questions and scope.

The interface was redesigned in the same release: header with search,
grouped sidebar, scope strip, one page layout for every workspace, light
and dark themes.

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

Or double-click `Ember.bat`. `Hanford.csv` and `Tank_attributes.csv` are
committed to this repo and auto-detected next to the script (or the built
`.exe`) on launch — no separate data download needed. Point **Load
CSV/Parquet…** at a different export if you want to analyze updated data.

### Standalone `.exe` (no Python needed)

Build with `build_exe.bat` (PyInstaller) — produces `dist\Ember\Ember.exe`
plus a portable `dist\Ember-portable.zip` colleagues who don't code can run
by unzipping and double-clicking, no install step at all. GlassNet
(§ above) is disabled in this build to keep the download a reasonable size;
everything else is fully functional.

## Repository layout

Flat, one file per concern (mirrors the sibling Dataapp project's
convention): `qt_*.py` modules are the UI, `*_science.py` modules are
framework-agnostic analysis logic with no PySide6 imports and full unit-test
coverage. See `tests/` for the test suite (`pytest`).

## Data

`Hanford.csv` / `Tank_attributes.csv` **are committed to this repository**
so the app works immediately after a clone — no separate download step.
They come from PNNL's [PHOENIX](https://phoenix.pnnl.gov) (Hanford Online
Information Exchange), the access mechanism for Tri-Party Agreement tank
waste databases:

> Brulotte, P.J., and Christensen, K.C.. "Tri-Party Agreement databases,
> access mechanism and procedures". United States. doi:10.2172/10112540.
> https://www.osti.gov/servlets/purl/10112540
>
> "PNNL Hanford Online Information Exchange (PHOENIX)", Pacific Northwest
> National Laboratory, Richland WA, U.S. Department of Energy.
> https://phoenix.pnnl.gov

See [`NOTICE.md`](NOTICE.md) for the full citation context and independence
disclaimer.

## License

MIT — see [`LICENSE`](LICENSE). If you use Ember in published work, a
citation of the repository is appreciated, alongside the PHOENIX data
citation above.

Sami Soudani — McCloy group, Washington State University.
