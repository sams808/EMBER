# Ember — detailed user notice

This notice describes the desktop application in depth: what each workspace
does, what each control means, the exact formulas behind every score and
property, how to interpret the plots, how to export data, and — most
importantly — what the vitrification screening tools can and cannot tell
you. If you only read one section beyond this page, read §12 (Vitrification)
before you rely on any glass-related number Ember produces.

Ember is a **visual exploration and screening tool** for Hanford tank
composition data. It is not a regulatory classifier, not a glass
formulation model, not a thermodynamic database, and not a waste
loading/property predictor. Its goal is to help you quickly answer
questions like:

- Which tanks contain a given element or analyte, and how much?
- What other elements tend to appear in the same tanks?
- Which elements correlate across tanks, and how much of that is just
  tank-size effects?
- Which tanks look chemically/radiologically similar, or cluster together?
- What would a tank's composition look like converted to oxide (glass)
  chemistry, and roughly how "basic" or "polymerized" would that glass be?
- Which tanks may be interesting starting points for blending or
  vitrification screening — as a first pass, not a final answer?
- How do I turn several of these plots into one figure for a report or
  slide deck?

The bundled composition data (`Hanford.csv`, `Tank_attributes.csv`) comes
from PNNL's [PHOENIX](https://phoenix.pnnl.gov) (Hanford Online Information
Exchange), the access mechanism for Tri-Party Agreement tank waste
databases:

> Brulotte, P.J., and Christensen, K.C.. "Tri-Party Agreement databases,
> access mechanism and procedures". United States. doi:10.2172/10112540.
> https://www.osti.gov/servlets/purl/10112540
>
> "PNNL Hanford Online Information Exchange (PHOENIX)", Pacific Northwest
> National Laboratory, Richland WA, U.S. Department of Energy.
> https://phoenix.pnnl.gov

**Ember (this application) is an independent project and is not produced
by, affiliated with, or endorsed by PNNL or the U.S. Department of
Energy.**

Developed by Sami Soudani, in the McCloy group at Washington State
University, with support from the U.S. Department of Energy (DOE).
`github.com/sams808/EMBER` · MIT License.

---

## 1. Installation and launch

Three ways to run Ember, depending on who you are:

- **Standalone, no install** — unzip the portable build and double-click
  `Ember.exe`. Nothing else needs to be installed; the composition data is
  bundled inside the zip next to the executable. This is the form meant
  for colleagues who don't code.
- **From source, with Python** — `py -3.11 qt_main.py`, or double-click
  `Ember.bat` (tries `pyw` first for a windowless launch, falls back to
  `py`).
- **Build your own `.exe`** — `build_exe.bat` (PyInstaller). See the
  README for the exact command and the dependency-exclusion list.

### 1.1 Input files

Ember expects two files next to it (already the case for both the git
repo and the portable zip — you don't need to supply your own copies
unless you want to analyze a different export):

- `Hanford.csv` — composition data with columns `WasteSiteId, Analyte,
  WastePhase, WasteType, Inventory, Units` (required) plus optional
  `Inventory_RSD, Volume, VolumeUnits, VolumeRSD, ComponentDensity,
  DensityUnits, AdjustedConcentration, AdjustedConcentrationUnits,
  DecayDate, PublishedDate, CCBLog`.
- `Tank_attributes.csv` — tank metadata (`Name`/`WasteSiteId`, `TankType`,
  `Capacity`, `TankStatus`, `TankIntegrity`, ...), joined onto the
  composition data. Optional; Ember runs without it, just with less tank
  metadata available.

Ember can also open `.csv`, `.tsv`, `.txt`, and `.parquet`/`.pq` files via
**Load CSV/Parquet…**, so you can point it at an updated PHOENIX export
without editing anything.

### 1.2 Known anomalies in the bundled 2026 export

Two rows of the bundled BBI re-export (published 2025-12-31 / 2026-06-30)
look like upstream data problems, found by cross-checking the export
against itself and against the same pull's per-layer volumes product.
Ember deliberately does **not** drop or "correct" bundled data — it
computes what was published and warns — so both are listed here until the
BBI republishes:

- **241-AW-102 saltcake interstitial liquid ships twice.** The layer
  carries two (volume, density) cohorts under one label: a 2 kL / 1.31
  g/mL re-characterization, and a 52 kL / 1.30 cohort that is the
  *previous* export's 55 kL assessment rescaled by exactly 52/55 (all 76
  analytes match 1:1). The same pull's per-layer volumes export lists the
  layer at 2 kL only — but taking 2 kL alone would imply a physically
  impossible ~4,000 g/L of sodium, so neither product is self-evidently
  right. Ember sums the cohorts as published and warns on every query
  that touches the layer. Layer-scoped absolute totals there are
  suspect (~26× apart between the two readings); the effect on whole-tank
  AW-102 numbers is ~1.4%, and on site totals negligible.
- **241-AZ-102 supernatant radionuclide revision is physically
  impossible.** One revised layer (CCB-1842, RSD 0) raises ²²⁶Ra from
  8.9×10⁻⁵ to 793 Ci — 85,000× the site's previous total and ~262× the
  ceiling its own uranium chain could support at secular equilibrium —
  and similarly implausible ¹³⁴Cs (581 Ci, a ¹³⁴Cs/¹³⁷Cs ratio of
  9×10⁻⁴ is inconsistent with the layer's stated decay date) and ¹²⁵Sb
  values. Overview/Heatmap rankings of ²²⁶Ra, ¹³⁴Cs, ¹²⁵Sb and ¹⁵²Eu
  are driven by this one suspect layer. Treat those four site totals as
  provisional and consider reporting the layer to PHOENIX/BBI.

---

## 2. General architecture

The nav sidebar has 9 workspaces:

1. **Overview**
2. **Query** (§19) — one flexible question surface: any element/analyte/basket × any basis × any grouping
3. **Element Explorer**
4. **Tank Attributes**
5. **Tank Explorer**
6. **Heatmaps**
7. **Correlations** — Cross-correlation / Quick Scan / Association Workbench (kg) / Structure sub-tabs
8. **Vitrification** — Screening / Oxide Chemistry / Candidate Search / Blend Partners / Loading rules sub-tabs
9. **Figure Composer** — combine any of the plots above into one multi-panel publication figure

**Debug / Export** (§11) is not a workspace: it is a troubleshooting
utility (live log viewer, debug-bundle export), reached from **Help →
Debug / Export…**.

**The window.** A header band carries the name of the application, a
summary of the loaded dataset (file, records, tanks, publication years and
decay dates), a search box, Help and the light/dark switch. The sidebar
lists the nine workspaces in four groups (Explore, Tanks, Analyze, Report)
and can be collapsed to icons (Ctrl+B). Under the header, the scope strip
(§3) filters every scope-aware workspace at once. The status bar reports
what the last action did, and shows a busy indicator while a computation
runs in the background; the window stays responsive meanwhile.

**Every workspace is laid out the same way.** A title with one sentence
saying what the page answers; on the right of the title, **Export...** and
a **`?`** button that opens that workspace's own section of this notice; a
row with the few controls that are always needed, ending with **Run**
(which reads **Update** when the controls no longer match the result on
screen); then the result, as a chart beside its tables. Controls that are
needed rarely (thresholds, score weights, per-plot tuning) are collapsed
under **Advanced options**. **Help → Getting Started…** re-opens the picker
shown on first launch.

**Notes that qualify a result.** When a result depends on something that
was set aside, converted or left out (activity converted to mass, mixed
decay dates, analytes that cannot be attributed to an element, records
without an assigned uncertainty), one line above the result says how many
such notes apply; **Show all** lists them in place. The same notes, in
full, are in the workspace's Notes tab and in every export. They are
statements about the data, not error messages.

**Charts.**
- Pointing at a bar, a point, a line vertex or a heat-map cell names it
  and gives its value with its unit, in a tooltip and in the line under
  the chart.
- A click on a mark selects the row it was drawn from, where the page has
  a matching table.
- A double click on anything that stands for a tank (a bar, a point, a
  disc on the tank-farm map, a table row) opens that tank in the Tank
  Explorer (§8). The right-click menu offers the same, and copies the
  value or the label.
- The chart toolbar has home / back / forward, pan and zoom, **Send to
  Figure Composer** (§13) and **Export…** (PNG, SVG or PDF at 300 dpi).

**Tables.** Each table has a filter box (Ctrl+F) that keeps the rows
containing the typed text, a row count, readable column headers (the
column's name in the exported file is in the header's tooltip), numbers
formatted to a common precision and sorted by value. Right-click copies a
cell or the selection at full precision, or exports the table as CSV. An
empty table says what will appear in it.

**Search** (Ctrl+K, or the box in the header) finds tanks, elements,
analytes and workspaces: a tank opens its profile, an element or analyte
opens in the Element Explorer, a workspace is brought to the front.

Every workspace also exports the data tables behind its charts as CSV.
When a chart looks wrong, the exported tables (or the debug bundle) are
what to send, not the source CSV; they are also the right starting point
for reproducing a chart in other software.

---

## 3. Menus and the scope strip

**File**
- **Open data file…** opens a file picker for any supported file (CSV or
  Parquet).
- **Open default dataset** finds `Hanford.csv` / `Tank_attributes.csv` next to
  the app (or in the current folder). It also runs at startup.
- **Reload** (F5) reloads the current file; use it after replacing the
  input file on disk.
- **Open project… / Save project…** (§20), **Export current view…**
  (Ctrl+E, the same as the page's own Export...).

**Data**
- **Use parquet cache** reuses a cleaned Parquet cache if one exists next
  to the source CSV: reading the cache is much faster than parsing the
  46,597-row CSV again.
- **Refresh cache** ignores an existing cache and rebuilds it from the
  CSV. Use it after the source CSV changes.
- **Open output folder** opens `ember_outputs/` next to the app, where
  every workspace writes its exports.

**View** lists the workspaces, the guided questions (§19), Search, Collapse
sidebar and Dark mode. **Help** holds Getting Started, this notice (F1),
About, and Debug / Export.

**The scope strip** sits under the header: Farm, Waste phase, Waste type
and SST/DST. A choice there applies to every workspace that uses the
scope (Overview, Query, Element Explorer, Heatmaps, Cross-correlation,
Loading rules), and the scope a result was computed under is written on
its chart, in its Notes and in its export. A workspace that is not on
screen is not recomputed when the scope changes; it is brought up to date
when it is next opened. On a workspace that does not use the scope, the
strip says so instead of showing selectors that would do nothing.

---

## 4. Data conventions that matter everywhere

### 4.1 Units are never blindly mixed

The dataset carries two fundamentally different inventory units: `kg`
(chemical mass) and `Ci` (radioactivity, curies). These cannot be summed
or averaged together as if equivalent — 1 kg of stable sodium and 1 Ci of
cesium-137 are not comparable quantities in any physical sense. Most Ember
workspaces are unit-specific by design; the kg-only Correlations sub-tabs
(Association Workbench, Structure) and the entire Vitrification tab are
deliberately kg-only for exactly this reason — mixing Ci and kg into one
tank×element matrix would be scientifically meaningless, not just
inconvenient.

### 4.2 Element parsing from analytes

The source `Analyte` column is not a clean element list — it's a mix of
isotope notations (`137Cs`, `113mCd`, `239/240Pu`), organic-compound names
(`Total Alpha`, benzene, chloroform, various nitrosamines and phthalates),
and a few dead/placeholder columns. Ember's parser converts these to
element symbols conservatively, specifically to avoid false positives
(e.g. the string `TotalAlpha` does **not** parse to element `Al`, even
though it contains that substring). Combined-isotope slash notation
(`239/240Pu`, `243/244Cm`) is handled explicitly rather than silently
dropped.

When a multi-element analyte label is parsed, the **first** parsed element
receives the analyte's full inventory value — this attribution rule
matches the app's predecessor tool and is disclosed here because it
changes every downstream number for those specific rows. It's a
simplification, not a claim that 100% of that inventory chemically belongs
to only that element. Always check the raw analyte rows in Tank Explorer
(§9) when an element total looks surprising.

### 4.3 `(WasteSiteId, Analyte)` is not a unique key

The same tank/analyte pair can appear multiple times across different
`WastePhase` rows (e.g. separate entries for the liquid and solid phase of
the same tank/analyte). Every aggregation in Ember sums across duplicates
rather than assuming one row per tank×analyte — a naive `groupby().first()`
or a plain lookup would silently under-count.

### 4.4 The BBI describes the same inventory more than once

Some material is reported twice, in two different forms, and adding both
inflates the answer. Ember resolves this **per layer** (tank × phase ×
waste type), so a layer that reports only one of the two forms keeps it.
Three shapes exist, and every one of them is announced in the warnings
whenever it fires:

1. **A reported mass already includes its radionuclides.** The BBI gives
   `UTOTAL` in kilograms *and* the activity of each uranium isotope — the
   same metal, described chemically and radiologically. On a mass basis
   the bridged isotope grams are counted only in layers that report no
   mass for that element.
2. **A combined-nuclide label is the sum of its members.** `239/240Pu`
   appears alongside `239Pu` and `240Pu`. Exactly one representation
   survives: the itemised members when all of them are present, the
   combined label otherwise.
3. **A reported total already contains its itemisation.** `TOC` is total
   organic carbon and already holds the oxalate, formate, acetate and
   glycolate reported beside it. Here the rule inverts — the total is a
   *superset* (it also covers organics the BBI never itemises, a median
   82% of it), so the **total** is kept and the itemised members are set
   aside, and only for carbon: the phosphorus in tributyl phosphate is
   still booked in full.

Note that `TIC as CO3` is *inorganic* carbon, which `TOC` does not
contain, so those two are correctly added together.

---

## 5. Overview

The landing page, titled **Site overview**: the loaded dataset at a
glance, a map of the tank farms, and the dataset audit tables.

**Headline figures.** Tanks (single-shell and double-shell, when the tank
attributes are loaded), waste volume, records and waste layers, analytes
and elements, sodium, and the activity of ¹³⁷Cs plus ⁹⁰Sr. The activity
tile is the sum of those two nuclides and not a total activity: a total
would add daughters and gross-alpha entries to the nuclides they already
describe. All figures follow the scope strip; the tooltip of a tile gives
its full wording.

**Tank farms.** Each tank is a disc, grouped by farm and by area (200
West, 200 East). The arrangement is schematic: tanks are grouped by farm,
not drawn to scale or at their surveyed positions. A ring marks a
double-shell tank; a small disc is a tank holding less than 100 kL of
waste; a hollow disc has no record of the chosen quantity (unreported is
not shown as zero).
- **Colour by** chooses the quantity: waste volume, or any element,
  analyte or nuclide the dataset reports (type to search).
- **Basis** chooses how it is expressed: inventory (kg, with activity
  converted to mass where a layer reports only activity), activity (Ci),
  mol per mol Na, wt% of waste, or g/L. The last three are concentrations
  and compare tanks of different sizes fairly.
- **Scale** chooses the colour classes: quantile classes (equal numbers of
  tanks per class; the default, robust to a few extreme tanks),
  logarithmic or linear.
- Pointing at a disc names the tank, its farm and type, its waste volume,
  the value and its rank. A click selects it (Ctrl+click adds to the
  selection); a double click opens the tank in the Tank Explorer.

**Ranked tanks** lists the same quantity, largest first, with a bar per
row. The list and the map are linked both ways: selecting on one selects
on the other. With several tanks selected, **Compare in Tank explorer**
opens them side by side.

**Dataset tables** (below the map) are the audit of the data as loaded:
units in use and rows per unit, missing values, breakdowns by waste phase,
waste type and farm, and the top elements and analytes with a chart.
**Check the units audit before trusting anything downstream**: an
unexpected unit is a reason to stop and look.

**Top elements** comes from the query engine (§19), so this page agrees
with Heatmaps (§9) about what the waste contains: each element receives
its stoichiometric share of every species reporting it. `N_analytes`
therefore counts the species that carry the element — sulfur genuinely
arrives via SO₄ *and* sulfide — rather than being a property of one
label.

The **Basis** picker of the tables sets the unit for that ranking. `kg`
and `Ci` keep a single reported unit; **`All` means grams**, with curie
rows bridged onto the same mass axis through ICRP-107 specific
activities. It does *not* rank kilograms against curies, which compares
nothing. (The other tables on this page are audits of the data as
reported, and are unaffected.)

---

## 6. Element Explorer

The main search workspace. Its three **Target** views (by tank, phase and
type) and its raw rows come from the query engine (§19), so this page
agrees with Query, Heatmaps and Overview: an element receives its
stoichiometric share of every species reporting it. Previously a search
for `S` booked the full mass of every sulfate row as sulfur — and a
search for `O` booked that same sulfate mass again as oxygen.

The **Co-elements**, **Co-analytes** and **Composition** tabs still count
reported rows as they are, each analyte's full mass under its first
parsed element. That is a different and legitimate question — "what
appears alongside what" rather than "how much of X is there" — and the
Notes tab says which tab answers which.

Five search modes, so you can find what you're looking for whichever way
you think of it:

- **Element symbol** — exact match on the parsed element (e.g. `Cs`).
- **Exact analyte** — exact match on the raw, unparsed `Analyte` string
  (e.g. `137Cs`, useful when you specifically care about one isotope, not
  the whole element).
- **Analyte contains** — substring match on the raw analyte string.
- **Analyte regex** — full regular-expression match, for anything the
  first three modes can't express.

Results show: which tanks contain the target, co-elements/co-analytes
found alongside it in the same tanks, and composition statistics reported
**two different ways that answer two different questions** — don't
conflate them:

- Absolute mean/stdev inventory — "how much of this element is typically
  present, in the tanks that have it."
- Tank-normalized fraction mean/stdev — "what share of each tank's total
  inventory does this element typically represent."

Seven plot types are available for the current search result (bar charts,
scatter against total inventory, correlation-scan style views); each one
respects the same Top N / unit controls as the table.

## 7. Tank Attributes

Browses `Tank_attributes.csv` directly: tank type, capacity, integrity,
DST/SST status, and audits by category (counts and totals grouped by any
categorical column). This is metadata about the tanks themselves — not
composition — so it works even if the composition CSV can't answer a
question (e.g. "how many double-shell tanks are there").

## 8. Tank Explorer

One tank as a profile, or several tanks side by side. A tank can be
opened from anywhere: a double click on a map disc, a chart mark or a
table row that stands for it, or the search box.

**One tank** reads top to bottom: its name with farm, shell type,
capacity, integrity and status (from the tank attributes); headline
figures (waste volume, waste layers, sodium, ¹³⁷Cs and ⁹⁰Sr); the waste
layers it holds, with volume and density; its place on the tank-farm
schematic; and its composition as bars on a logarithmic axis, with the
full composition table and the **raw, unparsed rows** behind it in the
neighbouring tabs. The raw rows are the single most useful place to check
the §4.2 element-attribution rule when a number looks off.

**Several tanks** (select them in the list, filtered by farm or by typing
part of a name) are compared element by element. Up to six tanks are
drawn as bars per element; more are drawn as a range.

**Basis** chooses the unit (kg or Ci); Advanced options hold the number of
elements on the chart and the choice between inventory and the share of
each tank's own inventory.

## 9. Heatmaps

Tank × element inventory matrices, built by the query engine (§19) and
drawn through the plot grammar, so the global scope bar applies here and
every exclusion is disclosed.

**Attribution** — the control that decides what "how much X" means:

- **Apportioned (stoichiometric)** *(default)* — an element receives its
  mass fraction of every species reporting it, so sulfate contributes
  32.06/96.06 of its mass to S and the rest to O.
- **As reported (primary element)** — each analyte's full mass is booked
  to its first parsed element, the behavior of earlier Ember versions.
  Kept so an older figure can be reproduced exactly.

The two disagree by known factors. On the bundled dataset, switching to
apportioned attribution multiplies each element's total by:

| Element | Factor | Why |
|---|---|---|
| O | ×15.6 | oxygen is never a *first* element, so it was previously almost invisible; it becomes the largest element by mass |
| N | ×0.24 | nitrate's full mass used to count as nitrogen |
| S | ×0.33 | = 32.06/96.06, sulfate's sulfur fraction |
| P | ×0.33 | phosphate |
| C | ×0.29 | carbonate and TOC |
| Na, Al, Fe, Cr, … | ×1.00 | reported as bare elements — unchanged |

The **Scale** selector offers three views of the same matrix:
**Inventory** (as-is), **Log10 of inventory** (zeros become blank cells —
log10 of zero doesn't exist), and **Fraction of the tank's total** (each
tank's row divided by its own total, i.e. composition rather than
amount). The colorbar states which transform is active, and exports
record it as `inventory`, `log10_inventory` or `fraction`. A click on a
cell selects that tank and element in the Long table.

Restricted to the top-N elements by total (adjustable) to stay readable,
with any omission stated on the figure; the figure's own size scales with
how many tanks/elements are shown, capped at whatever's actually visible
in the panel so nothing gets silently clipped off-screen.

*"min inventory" is a row-level floor applied before summing. It belongs
to the as-reported path and is not applied to apportioned values; picking
both raises a warning rather than quietly ignoring the setting.*

---

## 10. Correlations

Four sub-tabs. **Cross-correlation** (§10.4), the first tab, is built on
the query engine (§19): it works on concentrations, accepts species and
radionuclides alongside elements, and is the only Correlations sub-tab
that honors the global scope bar. The other three — Quick Scan,
Association Workbench (kg) and Structure — are built on one
tank×element inventory pivot (`element_inventory_matrix`), so results
between them are directly comparable; they always use the whole
dataset, and the scope bar is hidden while one of them is showing.

### 10.1 Quick Scan

The **Correlate** selector chooses one of three questions, and **Run**
answers it: **One element against all others** (a target-element scan:
how does one element of interest correlate with every other element
across tanks), **Two or three elements** (a specific handful of elements
plotted against each other), and **Every pair (heat map)** (the full
correlation matrix, lower triangle only — a correlation matrix is
symmetric about its diagonal, so the upper triangle is pure redundancy
and only makes the plot harder to read).

Quick Scan works on one reported unit at a time (**Basis**). An element
that has no inventory in the chosen unit is named in the note above the
result together with the unit it is reported in: cesium, for example, is
reported only as activity, so it is scanned on the Ci basis.

Controls that change what the correlation is actually measuring:

- **Metric**: `inventory` (raw) / `log10_inventory` (zeros become
  undefined) / `log10_plus1` (zeros retained) / `fraction` (each element's
  share of that tank's total) / `presence` (1 if present, 0 if absent —
  turns "correlation" into "how often do these two elements co-occur").
- **Method**: `pearson` (linear correlation) / `spearman` (rank
  correlation — more robust to outliers and nonlinear-but-monotonic
  relationships).
- **include zeros** / **min overlap** / **min inv** — control which tanks
  and element pairs count at all, before any correlation is computed.

**Always check the overlap count before trusting a strong correlation** —
`r = 1.0` computed from only 3 overlapping tanks is a clue to investigate
further, not a conclusion you can act on.

**Control for tank size** checkbox: strips out the "everything correlates
because both elements just scale with how big the tank is" effect, via
partial correlation controlling for each tank's total same-unit inventory
(closed-form: `r_AB.Z = (r_AB − r_AZ·r_BZ) / sqrt((1−r_AZ²)(1−r_BZ²))`
with `Z` = each tank's total inventory, or log10 of it). When checked, the
table's `Correlation_r` column becomes this partial correlation and
`Raw_Correlation_r` shows the uncontrolled value alongside it, so you can
directly see how much of the raw correlation was a tank-size artifact.

### 10.2 Association Workbench (kg)

Kg-only element-association screening (Ci is deliberately excluded — see
§4.1): per-element stats, pairwise **Jaccard co-presence**
(`n_both / n_either` — what fraction of tanks that have *either* element
have *both*), and a `PreferredAssociationScore_proxy`
(`max(r, 0) · ln(1 + n_both) · jaccard`) that rewards element pairs that
are both positively correlated *and* frequently found together, and never
rewards a merely-negative correlation (a real absence relationship is
still informative, but it's not what this particular score is designed to
surface). 18 plot types (heatmaps, top-association bar charts, pair
scatter matrices, tank-similarity and tank×element maps, presence-pattern
plots, a combined stats dashboard, and **Tank inventory bars**), with a
Basic/Coherent-colors theme toggle for consistent color families across
all of them.

**Tank inventory bars** — the publication-figure workhorse: pick any N
elements (the same selection used everywhere else in this workbench), and
get one horizontal bar per tank showing their kg inventory, sorted by
total (descending/ascending) or tank ID. Two modes: **Stacked** (each
tank's bar splits into one segment per element — total bar length is that
tank's total inventory across the selected elements, the classic
side-by-side-panel summary-figure shape) or **Grouped** (one bar per
element per tank, for comparing magnitudes directly rather than a
cumulative total). Log scale is available for Grouped; it's left
available for Stacked too but not recommended there, since a log axis
makes stacked segment lengths visually non-additive. Uses the same "top
tanks" control as the tank-similarity/tank×element plots above.

### 10.3 Structure

New capability, no equivalent in earlier versions of this tool:

- **PCA + hierarchical clustering** of tanks over a standardized kg
  element matrix — a scatter of the first two principal components
  (color-by TankFarm/TankType/TankSystem/TankStatus/Dominant waste phase)
  and a cross-checkable dendrogram (do the two views agree on which tanks
  group together?).
- **Partial correlation matrix**, side-by-side with the raw correlation
  matrix, showing directly how much of each pairwise correlation is a
  tank-size artifact (same math as the Quick Scan checkbox, applied to the
  whole matrix at once).
- **Element-association network graph** — nodes sized by total kg, edges
  thresholded by minimum |r| and minimum Jaccard co-presence (both
  user-adjustable), edge color by correlation sign.
- **Export... → Interactive HTML views** — writes the correlation heatmap, PCA
  scatter, and network graph as self-contained interactive Plotly HTML
  files (the JavaScript is embedded, not loaded from a CDN, so they work
  fully offline — useful for sharing on a network without internet access)
  and opens the folder.

### 10.4 Cross-correlation

The first sub-tab. It answers one question: across the tank farm, which
components occur *together*? Enter a comma-separated list of components
— elements, species such as `SO4`, or radionuclides such as `129I` — or
fill it from a preset, choose a basis and a level, and press **Run**. The
result is a correlation matrix, the pair table behind it, and the tables
needed to judge how far each coefficient can be trusted.

**Basis — what is being correlated.**

| Basis | Meaning | What it compares |
|---|---|---|
| `mol/mol Na` *(default)* | moles of the component per mole of sodium in the same layer | composition; normalizing to sodium, the dominant cation of the waste, removes both the amount of waste and its water content |
| `wt%` | grams per 100 g of layer mass (layer volume × density) | composition by mass; follows the water content of the layer |
| `g/L` | grams per liter of layer volume | composition by volume; same caveat |
| `kg` | inventory, curie rows bridged to mass | *amount*, not composition — see below |

**Why inventories all correlate.** Every component scales with how much
waste a tank holds: a large tank has more fluoride, more sulfate and more
chromium than a small one, whatever its chemistry. On the `kg` basis the
matrix therefore describes tank size. With the default component list on
the bundled dataset, all 28 `kg` pairs are positive at either level
(Spearman 0.37 to 0.92 across tanks, 0.17 to 0.92 across layers); on
`mol/mol Na`, across layers, the same 28 pairs run from −0.50 to +0.69,
and only then does the sign of a coefficient say something about
chemistry. The `kg` basis is kept for comparison with the other
sub-tabs, not as a starting point.

**Level.** A *waste layer* is one (tank, waste phase, waste type) entry
of the inventory — the level at which a composition is actually stated,
and the default. A *tank* sums its layers, so a tank holding saltcake over
sludge averages two unrelated chemistries. At the layer level the
correlation can be restricted to one **phase family** (Supernatant,
Saltcake or Sludge) under Advanced options.

**Why iodine and selenium are entered as `129I` and `79Se`.** A kg-only
matrix cannot hold either element. Iodine is reported only as the
radionuclide ¹²⁹I, in curies; no stable-iodine mass exists in the
inventory. Selenium is reported site-wide as ⁷⁹Se in curies, and as an
elemental mass in only part of the layers (153 layers carry a reported
mass and 383 only the nuclide, on the bundled dataset). The engine
converts curies to grams with ICRP-107 specific activities (§19), which
is what lets a radionuclide sit beside sulfate in one matrix. Naming the
nuclide gives one consistently defined quantity in every layer. Naming
the element `Se` instead mixes two different quantities — a reported
total where one exists and, elsewhere, the ⁷⁹Se-implied mass, which is
only a lower bound on the element — and the tab raises a warning naming
the component. `I` resolves to the same ¹²⁹I rows as `129I`; the Notes
tab states that the mass is a lower bound on elemental iodine.

**Count each distinct composition once.** Many layers of the inventory
are not independent observations: one waste-type template composition was
multiplied by each layer's volume, so several tanks carry the same ratios
to three significant figures. Counting every copy inflates the number of
observations, and with it the apparent significance. With this option on
(the default), layers whose components *all* agree within 1.5% — blank
matching blank, zero matching zero — count as one observation. With the
default list on the bundled dataset, 557 layers reduce to 393 distinct
compositions. The figure states both numbers ("n = 393 distinct
compositions of 557 layers") and the Notes tab says how many layers were
set aside. Two consequences: the grouping depends on the component list
(two layers can agree on four components and differ on a fifth), and it
is unavailable on the `kg` basis, where one composition written into two
volumes gives two different inventories.

**Method.** `spearman` *(default)* is a rank correlation: insensitive to
the orders-of-magnitude spread of concentrations, and it keeps reported
zeros as the ties they are. `pearson (log10)` is the linear correlation
of log10 values; a pair then uses only the layers where both values are
positive. `kendall` is a rank correlation that is more conservative when
ties are frequent. Each pair uses the layers that report *both*
components. A pair with fewer shared observations than **min shared
observations** (default 8), or with a constant component, is left blank:
hatched in the figure, empty in the tables. The p-values are per pair and
are not corrected for the number of pairs examined.

**Tables.**

- **Pairs** — one row per pair, strongest first: `r`, `p_value`, `N`
  (observations used) and `N_both_positive`.
- **Correlation matrix** and **N per pair** — the same numbers as
  squares. The figure is drawn from the Correlation matrix table.
- **What each component rests on** — to be read *before* the matrix.
  `Reported_as` lists the analytes and units a component resolves to;
  `N_tanks`, `N_tanks_positive` and `N_tanks_in_scope` say how many tanks
  report it, report it above zero, and exist in the current scope;
  `Unassigned_uncertainty_pct` is the share of its rows with
  `Inventory_RSD = 0` (no uncertainty assigned — not an exact value) and
  `Default_100pct_uncertainty_pct` the share sitting at an RSD of 1.00 (a
  default of 100%, not a computed statistic). A component reported in a
  handful of tanks, or resting mostly on default or unassigned
  uncertainties, cannot carry a site-wide statement whatever its
  coefficient. Bromine is the example on the bundled dataset: 11 of 177
  tanks, about three quarters of its rows at the default.
- **Co-occurrence** — per layer (or tank), which components sit above
  their own quantile (default 0.75) across the matrix, and how many at
  once (`N_high`). The rows at the top are where the components actually
  coincide, which a coefficient only describes on average. This table
  covers every layer in scope, repeated compositions included.
- **Values** — the layer × component matrix itself, with the farm,
  SST/DST, phase family, volume and mass of each layer. A blank cell is a
  component the layer does not report; a zero is a reported zero.
- **Notes** — the scope the run was computed under, the engine's warnings
  and per-component disclosures, and the counts of repeated compositions
  and blank pairs.

The line above the results states in full only what bears on reading
this matrix (a component with no value, a component that is a total in
some layers and a lower bound in others, layers left blank for want of
sodium). Notes that are true of the dataset whatever the question are
given there as a count; their text is in the Notes tab, in the tooltip
of that line, and in every export.

**Plots.** *Correlation matrix* shows the lower triangle with the
coefficient written in each cell, on a fixed −1 to +1 scale: blue for
negative, red for positive, neutral gray at zero. *Pair scatter* is a
log–log scatter of the pair selected in the Pairs table (the strongest
pair when none is selected), drawn over the observations the coefficient
used and colored by phase family at the layer level. A value of zero has
no position on a logarithmic axis; the subtitle states how many
observations were left off for that reason and how many do not report
both components.

**Scope and export.** The global scope bar (§19) applies to this sub-tab:
a scope change re-runs the standing question, and the scope a result was
computed under is stated in the figure subtitle, in Notes and in the
export's `context.csv`. **Export** writes every table, the parameters of
the run and the current figure.

A coefficient here describes how two components vary together across the
layers of the inventory as published. It is not evidence of a mechanism,
and it inherits every limitation listed in the warnings above the
results.

---

## 11. Debug / Export

Reached from **Help → Debug / Export…**, not the sidebar — a
troubleshooting utility rather than a workspace.

- **Export global debug bundle** — writes a folder of small audit CSVs
  (overview, units/missing/phase/type/farm audits, top elements/analytes,
  a raw-row preview, an environment report, and a manifest). Send this
  instead of the full source CSV when something needs debugging.
- **Open output folder** — opens where every workspace's exports land.
- **Clear log** — clears the live app log shown below (file loads, cache
  writes, plot builds, exports, errors).

---

## 12. Vitrification

Five sub-tabs covering everything from a quick chemical/radiological
screening score to real oxide-basis glass chemistry and published glass
loading rules. Read this whole section before trusting a number from
here — **every score in the first four sub-tabs is a transparent
screening heuristic, built from ordinary chemistry and simple weighted
sums, not an official glass formulation model, thermodynamic
calculation, or waste classification** (§14 has the full, explicit list
of what these tools do *not* calculate). The fifth sub-tab, Loading
rules (§12.5), applies published loading constraints to the waste as
stored; it is a screening index, not a forecast of glass production.

The reason this workspace exists in this form: raw elemental kg inventory
(what the source CSV actually contains) is not what glass scientists
think in. Glass composition, structure, and most of the properties that
matter for vitrification (viscosity, durability, crystallization
tendency, melting behavior) are conventionally described in **oxide
weight or mole percent** — SiO₂, Na₂O, B₂O₃, and so on — not in kg of
elemental Fe or Na. The Oxide Chemistry sub-tab (§12.2) is the bridge
between what the data gives you and what the glass-science literature and
your own intuition actually reason about.

### 12.1 Screening

One row per tank, built entirely from **elemental** kg/Ci fractions (no
oxide conversion needed for this sub-tab — see §12.2 for that). Seven
element groups are defined, each with a specific glass-relevant reason for
being tracked separately:

| Group | Elements | Why it matters |
|---|---|---|
| `glass_former_or_intermediate` | B, Si, P, Al, Zr, Ti | The elements that can build the glass network itself (formers) or participate in it under the right conditions (intermediates) — more of this, generally, means an easier path to a stable glass. |
| `alkali_alkaline_modifier` | Li, Na, K, Mg, Ca, Sr, Ba | Network modifiers — they break up (depolymerize) the glass network, lowering viscosity and melting temperature, but too much can hurt durability. |
| `transition_redox_sensitive` | Fe, Cr, Mn, Ni, Cu, Ce, U, Np, Pu, Tc | Elements whose oxidation state (and therefore glass behavior, color, and sometimes volatility) depends on the melt's redox conditions — a flag for "this tank's glass chemistry may be redox-sensitive," not a redox calculation. |
| `volatile_halide_sulfate` | F, Cl, Br, I, S, Se, Tc, Cs | Elements/species prone to volatilizing out of the melt or having limited solubility (sulfate/halide/chalcogenide chemistry) — a real, well-known glass-processing headache at high concentration. |
| `nuclear_key_radionuclide_elements` | Cs, Sr, Tc, I, Se, U, Np, Pu, Am, Cm, Eu, Sm, Y, Ba | The elements most often tracked for radiological/dose reasons in Hanford tank waste discussions — used for the Ci-side of the screening score, not for any dose calculation itself. |
| `platinum_group_or_noble` | Ru, Rh, Pd, Ag, Re, Os, Ir, Pt, Au | Platinum-group/noble metals — chemically inert in most glass melts, but can form metallic/alloy precipitates (a real processing concern) rather than dissolving into the network. |
| `potential_spinels_or_crystallizers` | Cr, Fe, Ni, Mn, Zn, Zr, Mo, P, Al | Elements associated with spinel or other crystalline phases that can form on cooling — a coarse composition-based flag, not a crystallization-kinetics prediction. |

`problem_elements_proxy` is the union of `volatile_halide_sulfate` with
`{Cr, Mo, P, Ru, Rh, Pd, Ag}` — the single "elements that tend to make
vitrification harder" bucket used in the score below.

**The screening score**, with its actual default weights:

```
score = 60·frac_glass_former_or_intermediate
      + 25·frac_alkali_alkaline_modifier
      − 45·frac_problem_elements_proxy
      − 25·frac_volatile_halide_sulfate
      − 10·frac_transition_redox_sensitive
```

clipped to [−100, 100]. Every one of those five numbers (60, 25, −45, −25,
−10) is now an editable `QDoubleSpinBox` in the UI, with a "Reset to
defaults" button showing these original constants — they were buried in
code in the app's predecessor; here they're visible and adjustable next to
the score they produce, so you can see exactly how sensitive the ranking
is to your own judgment about what matters most.

`frac_*` values are each group's share of the tank's **total kg
inventory with a parsed primary element** (not oxide-converted; the rare
rows whose analyte parses to no element are excluded, and the tab names
the excluded mass beside its caveat note) — except when the **basis** toggle is
set to `oxide`, which swaps *only* the glass-former term to the oxide-wt%
glass-former fraction computed in §12.2 (SiO₂+B₂O₃+P₂O₅+Al₂O₃'s combined
share of the tank's oxide-converted composition); every other term stays
elemental. This matters because the elemental-kg glass-former fraction and
the oxide-wt% glass-former fraction can meaningfully disagree — a kg of Al
metal and a kg of Si both count equally in the elemental basis, but
convert to very different amounts of oxide (different molar masses,
different stoichiometry), so the oxide basis is arguably the more
chemically honest of the two once you have it available.

Also reported: `Ci_per_kg_proxy` (total Ci ÷ total kg over the same
parsed-element rows, a coarse activity-density indicator) and `RelativeActivityBin` (the dataset's tanks
split into 4 quartiles — low/medium-low/medium-high/high — by that same
ratio, purely for quick relative sorting, not an absolute classification).

### 12.2 Oxide Chemistry

The real depth upgrade over the old app's "glass calculation," which never
left raw elemental kg. This sub-tab is the missing first step (elemental
kg → oxide wt%/mol%) that everything else in glass science — optical
basicity, NBO/T, melting/viscosity models, GlassNet — actually expects as
input.

**Workflow:** pick a farm (optional filter) → select one or more tanks
(selecting 2+ also builds a "Blend (selected tanks)" sample; the first
tank is preselected, so the first press already gives a result) → **Run**
→ review/edit the element→oxide assignment table →
**Apply assumptions** whenever you change an assignment, a
role, or the envelope table. That button re-runs the conversion itself,
so an edited element→oxide assignment really does change the numbers.

#### 12.2.1 What the input masses are

Element masses come from the same stoichiometric apportionment the Query
workspace uses: a kilogram of SO₄ contributes 32.06/96.06 kg of sulfur,
not a kilogram. Booking an anion's whole mass to its cation and *then*
multiplying by the oxide/element molar ratio does not conserve mass — it
reported more SO₃ than there was sulfate to make it from.

**Oxygen and hydrogen are not listed as components.** On an oxide basis
the oxygen is carried by the oxide formulas themselves, so booking the
apportioned free-oxygen pool as well would count the same oxygen twice;
hydrogen leaves as water when the waste is melted. This is why the oxide
table's total mass is *less* than the tank's reported kg inventory.

#### 12.2.2 Element → oxide conversion

For each element with a positive apportioned kg inventory in the selected
tank(s), Ember:

1. Converts kg to grams, and looks up (or lets you override) which oxide
   formula that element is assumed to form.
2. Computes the molar mass of the element and the oxide via `xraydb`
   (a standard, well-established X-ray/atomic-data library — not a
   hand-rolled periodic table).
3. Converts the element's mass to moles, divides by how many atoms of that
   element the oxide formula contains, and multiplies back up by the
   oxide's molar mass — ordinary stoichiometric mass-balance, nothing
   probabilistic or fitted.
4. Reports the result as both **wt%** and **mol%** of the total converted
   composition (both are commonly used in different corners of the glass
   literature, so both are always shown together).

The default element→oxide table covers essentially every element with a
tabulated optical-basicity value (§12.2.3) plus explicit non-oxide flags:

- **Multivalent elements** (Fe, Cr, Mn, U, Np, Pu, Ce, Co) get one
  pre-selected default oxide *and* a dropdown of chemically reasonable
  alternatives right there in the table — e.g. Fe defaults to Fe₂O₃ but
  can be switched to FeO, U defaults to UO₃ but can be switched to UO₂ or
  U₃O₈. Real tank waste redox state is not something this table can know;
  picking the right oxidation state for your specific waste stream is a
  judgment call the tool deliberately leaves to you, with the mechanism
  ready either way.
- **Halides and noble/platinum-group metals** (F, Cl, Br, I, Ru, Rh, Pd,
  Ag, Pt, Au, Tc) default to **no oxide** — they're reported as elemental
  wt%/mol% instead, because forcing them into an oxide framing that
  doesn't chemically apply would be actively misleading, not just
  imprecise.
- **Every cell in the table is directly editable**, including elements not
  in either special list — nothing here is a hard-coded assumption you
  can't see or change.

**Blend mode** (2+ tanks selected): the elemental kg from every selected
tank is summed together **first**, and the combined total is converted to
oxides **once** — this is what physically mixing those tanks' contents
would actually produce. It is deliberately *not* an average of each tank's
independently-computed oxide percentages, which would give the wrong
answer whenever the tanks have different total masses (a small tank and a
huge tank don't contribute equally to a real blend just because you
selected both).

#### 12.2.3 Optical basicity (Λ)

Optical basicity is a single number, conventionally between roughly 0.3
(strongly acidic network formers like B₂O₃/P₂O₅) and 1.5+ (strongly basic
modifiers like Cs₂O/BaO), that summarizes how much a glass's oxide mixture
behaves like an electron-density donor (basic) versus acceptor (acidic) —
originally developed to predict UV absorption edge shifts, and now used
more broadly as a quick composition-based proxy that tends to correlate
with things like sulfate/chromate retention, redox equilibria position,
and general "how forgiving is this melt chemistry" intuition. It is a
useful *screening* number precisely because it's cheap to compute from
composition alone — it is not a substitute for a real thermodynamic redox
or solubility model.

Ember computes it via the standard oxygen-weighted (Duffy) mixing rule:

```
Λ = Σ(xᵢ · nO,ᵢ · Λᵢ) / Σ(xᵢ · nO,ᵢ)
```

— molar fraction `xᵢ` of each oxide, weighted by how many oxygen atoms
`nO,ᵢ` that oxide contributes per formula unit, times that oxide's own
tabulated basicity `Λᵢ`. The per-oxide Λᵢ values are the **recommended**
values (Λrec) from Table B.1 of Rodriguez & McCloy, *"Optical basicity and
nepheline crystallization in high alumina glasses,"* PNNL-20184 /
EMSP-RPT-003 (2011) — itself a careful reconciliation of five earlier
sources (Duffy & Ingram 1976; Duffy 2002–2006; Dimitrov & Sakka 1996;
Lebouteiller & Courtine 1998; Lenglet 2004; Mills 1995). This is real,
citable literature data, copied verbatim into Ember — not re-derived or
approximated.

Not every oxide Ember can convert to has a tabulated Λrec value (most
notably, no PNNL-20184 value exists for any plutonium oxide). Rather than
silently dropping those components or crashing, the summary panel lists
exactly which formulas were **excluded from the basicity calculation**
alongside the number itself, so you always know whether Λ represents the
whole composition or only part of it.

#### 12.2.4 NBO/T (non-bridging oxygens per network-forming cation)

NBO/T is a structural indicator, not a thermodynamic property: it
estimates, on average, how many of the oxygen atoms attached to each
network-forming cation (Si, B, P, Al in the classical formalism) are
"non-bridging" — bonded to only one network cation, versus "bridging" —
linking two network cations together into the connected 3D network that
gives silicate/borate/phosphate glasses their structural rigidity. Higher
NBO/T means a more broken-up, depolymerized network (generally lower
viscosity, lower melting/working temperature, faster crystallization
kinetics); lower NBO/T means a more fully connected, polymerized network
(generally higher viscosity, more sluggish/harder-to-crystallize melt).
It's a widely used first-order structural intuition in the glass
literature, not a viscosity or Tg prediction on its own.

Ember uses the classical **former/modifier** framework:

- **Formers** (contribute network-forming T-cations): SiO₂ (1 Si per
  formula unit), B₂O₃ (2 B), P₂O₅ (2 P), Al₂O₃ (2 Al).
- **Modifiers** (break the network, contribute non-bridging oxygen but no
  T-cations): the alkalis (Li₂O, Na₂O, K₂O, Rb₂O, Cs₂O) and alkaline
  earths (MgO, CaO, SrO, BaO).
- Every other oxide/element ("other" role — transition metals, halides,
  actinides, rare earths, ...) is excluded from the NBO/T calculation
  entirely, because its network role is genuinely composition- and
  context-dependent and isn't captured by this simple two-bucket
  formalism. Those components still appear in the main composition table
  and in the optical-basicity calculation; they're only left out of this
  one specific number.

Formula:

```
T       = Σ(molᵢ · T-cations per formula unit) over former-role oxides
O_total = Σ(molᵢ · oxygens per formula unit) over former + modifier oxides
NBO/T   = (2·O_total − 4·T) / T
```

Each tetrahedral cation carries four oxygens and each bridging oxygen is
shared between two of them, which is where the 2 and the 4 come from. As
a check: sodium metasilicate (Na₂O·SiO₂) is a chain silicate with two
non-bridging oxygens per silicon, and this formula returns exactly 2.

**Every oxide's role (former/modifier/other) is editable** in its own
table — if you disagree with a default classification for your specific
system, change it and recompute.

This is explicitly a **simplified approximation**: it does not model
charge-balance effects where Al³⁺ or B³⁺ can act as either a former or a
network-modifying species depending on how much alkali is available to
charge-compensate it (the well-known "aluminum avoidance" / boron
anomaly problems in real aluminosilicate and borosilicate glasses). Verify
against your preferred reference or a more detailed structural model
before using an NBO/T number from here in a publication.

#### 12.2.5 GlassNet property prediction

When the optional `glasspy` package is installed, **Predict properties (GlassNet)**
runs a pretrained machine-learning model (Cassar, *Ceramics International*
49 (2023) 36013) against the current sample's oxide mol% composition and
returns its predicted glass properties — including glass transition
temperature, density, refractive index, and viscosity at multiple
temperatures, among others GlassNet reports. The model was trained on
**SciGlass**, a large curated database of real measured glass properties;
its predictions are only as reliable as that training data's coverage of
compositions similar to yours — treat a GlassNet number for a composition
far outside typical oxide-glass chemistry (e.g. dominated by exotic
actinide oxides) with real skepticism, the same way you would any ML
model extrapolating outside its training distribution.

`glasspy` needs PyTorch, which is a large dependency deliberately
**excluded from the packaged `.exe`** to keep the standalone download a
reasonable size — this button is disabled (with an explanatory tooltip) in
that build. Run Ember from Python with `pip install glasspy` to enable it.
The first prediction after launch is slower than subsequent ones (the
model itself has to load into memory once).

#### 12.2.6 Envelope comparison

A per-oxide `[oxide, min wt%, max wt%]` table you build yourself — **Ember
ships this empty, with zero hardcoded DOE/PNNL/WTP composition bounds**.
This is a deliberate choice, not an oversight: baking in a specific
regulatory or process-envelope specification from memory would risk
silently embedding an outdated or simply misremembered number into a
domain where that could matter. Build your own envelope from whatever
specification you're actually screening against (an internal WTP glass
formulation envelope, a DOE contract requirement, your own group's target
window, ...), save it as JSON (**Save JSON…**) so you can reuse it across
sessions and share it with colleagues, and load it back with **Load
JSON…**.

Once loaded, every oxide in either the envelope or the current
composition is checked and reported with an explicit status: **Pass**
(within bounds), **Fail** (outside bounds), or **Not specified** (present
in the composition but the envelope has no bound for it — reported
explicitly rather than silently skipped, since "no bound specified" and
"passes because it's low" are very different situations). An oxide
specified in the envelope but absent from the composition is checked as
0 wt%, so a nonzero minimum correctly fails.

### 12.3 Candidate Search

Ranks tanks by how well they match a target/penalty-element search, using
the same tank-category summary as Screening (§12.1) plus per-target/
per-penalty element fractions in both kg and Ci (so a search for, say,
Cs/Sr/Tc still works even when those nuclides live primarily in the Ci
table rather than the kg table). `required elements` filters out any tank
missing at least one required element entirely (in either unit); `min
total kg` filters out tanks below a minimum total inventory.

```
score = 100 · (0.50·target_fraction_sum + 0.25·glass_former_fraction
             − 0.20·problem_fraction − 0.30·penalty_fraction_sum)
```

— `target_fraction_sum`/`penalty_fraction_sum` are each the sum of the
kg-fraction and Ci-fraction across all your target/penalty elements for
that tank. Same editable-weight treatment as Screening (0.50, 0.25, −0.20,
−0.30 are all `QDoubleSpinBox` controls with "Reset to defaults"), and the
same elemental/oxide **basis** toggle for the glass-former term.

### 12.4 Blend Partners

Given one base tank, ranks every other tank by how good a **blend
partner** it might be — i.e. "which other tank's composition might dilute
or balance out this one?" Not a blend optimization: it does not compute a
final blended composition, a mixing ratio, or any melt property
constraint on the result — for an actual blend composition, build it
yourself in Oxide Chemistry (§12.2) by selecting the tanks you're
considering together (which physically sums their kg before converting,
exactly as a real blend would).

```
complement_score = 100 · (0.35·max(glass_former_gain, 0)
                         + 0.35·max(problem_reduction, 0)
                         + 0.20·(1 − cosine_similarity)
                         + 0.10·glass_former_fraction_partner)
```

- `glass_former_gain` = partner's glass-former fraction − base tank's
  (only rewarded if positive — a partner that *lowers* the glass-former
  fraction doesn't help).
- `problem_reduction` = base tank's problem-element fraction − partner's
  (only rewarded if positive, i.e. the partner is comparatively cleaner).
- `cosine_similarity` is computed between the two tanks' full
  element-fraction profile vectors — `1 − similarity` is highest for the
  *most chemically different* partner, on the reasoning that a very
  similar tank has little to offer that the base tank doesn't already
  have.
- The glass-former fraction term uses the same elemental/oxide **basis**
  toggle as the other three sub-tabs.

### 12.5 Loading rules

For every tank (or waste layer), this sub-tab states which published
loading rule would set the sodium loading of a low-activity-waste (LAW)
glass made from its salt waste, how far below the alkali limit that
loading is, and what it costs in glass. Unlike the four sub-tabs above,
it is computed through the query engine (§19) and honors the scope bar:
the scope a result was computed under is stated on the figure, in the
Notes tab and in the export.

**Source.** Every rule and constant is taken from J.D. Vienna, D.C.
Skorski, D.S. Kim and J. Matyas, *Glass Property Models and Constraints
for Estimating the Glass to be Produced at Hanford by Implementing
Current Advanced Glass Formulation Efforts*, PNNL-22631 Rev. 1
(ORP-58289), Pacific Northwest National Laboratory, 2013.

**Rule sets.** Three sets can be selected. All concentrations are in wt%
of glass.

```
Advanced LAW (2013)              Table S.12, Eq. 3.15 to 3.17, Figure 3.4b
  alkali            Na2O + 0.66 K2O <= 24
  alkali + sulfur   Na2O + 0.66 K2O <= 33.94 - 11.69 SO3
  sulfur            SO3 <= 1.5
  halide            SO3 <= 1.65 - 0.725 (Cl + 0.3 F + 0.4 Cr2O3)

Advanced LAW, conservative halide          Eq. 3.15 to 3.17, Figure 3.4a
  alkali, alkali + sulfur and sulfur as above
  halide            Cl + 0.607 F + 0.542 Cr2O3 + K2O <= 3.746 - 4.694 SO3
                    or                               <= 1.243 - 0.4506 SO3
                    (the upper of the two lines is the boundary)

WTP baseline                                              Eq. 3.1 to 3.6
  sodium            Na2O <= 21
  alkali            Na2O + 0.66 K2O <= 21.5
  alkali + sulfur   Na2O + 42.5 SO3 <= 35.9
  sulfur            SO3 <= 0.77
  halide            Cl + 0.3 F <= 1.4656 - 2.1111 SO3   for SO3 <= 0.59
                    Cl + 0.3 F <= 0.22                  for SO3 >  0.59
```

The report calls the Table S.12 halide rule "optimistic" (the largest
loadings demonstrated) and the Figure 3.4a envelope "conservative" (no
salt formed in any melter test), and places the real limit between the
two. Eq. 3.7 to 3.10 of the report (limits on Cr₂O₃ and K₂O that depend
on the P₂O₅ content of the glass) are not implemented.

**From a rule to a loading.** Every waste component enters the glass in
proportion to the sodium. With the waste mass ratios `r_X = X / Na2O`, a
glass loaded to `w` wt% Na₂O holds `w · r_X` wt% of X, so each rule is an
inequality in `w` alone and is solved for the largest `w` it allows:

```
alkali            w = 24 / (1 + 0.66 r_K2O)
alkali + sulfur   w = 33.94 / (1 + 0.66 r_K2O + 11.69 r_SO3)
sulfur            w = 1.5 / r_SO3
halide            w = 1.65 / (r_SO3 + 0.725 (r_Cl + 0.3 r_F + 0.4 r_Cr2O3))
```

The conservative envelope allows the larger of `3.746 / (h + 4.694
r_SO3)` and `1.243 / (h + 0.4506 r_SO3)`, with `h` its own halide term
per Na₂O. The baseline halogen rule is piecewise: the sloped branch
applies up to the loading at which SO₃ reaches 0.59 wt%, the flat
0.22 wt% cap beyond it, and the result is the largest loading that
either branch allows.

The **loading** of a unit is the smallest of its rule limits and the
**binding rule** is the one that gives it. `Glass mass = Na2O mass /
(w / 100)`. **Extra glass** is the glass at the rule-limited loading
minus the glass the same sodium would need at the alkali limit alone.

**Oxide conversions.** The inventory reports Na, K, Cr, Cl and F as
elements and sulfur as the sulfate anion. The gravimetric factors are
computed from the atomic masses used everywhere else in Ember:

| Glass component | From | kg per kg |
|---|---|---|
| Na₂O | Na | 1.3480 |
| K₂O | K | 1.2046 |
| SO₃ | SO₄ | 0.8334 |
| Cr₂O₃ | Cr | 1.4615 |
| Cl, F | Cl, F | 1 (used as elements, as the rules state them) |

A species a unit does not report counts as zero in the rules; the Notes
tab states how many units that concerns.

**What "salt waste" means here.** The Waste selector decides which
layers count as the feed. *Supernatant + saltcake* (the default) takes
every layer whose waste phase begins with "Supernatant" or "Saltcake",
the waste that becomes LAW feed. *Saltcake* and *Supernatant* take one
family; *All phases* adds the sludge layers. With Unit set to *Tanks* the
selected layers of each tank are summed before the rules are applied;
with *Waste layers* each (tank, waste phase, waste type) entry is
screened on its own. A unit that holds no sodium in the selected waste
has no loading: it is left blank, not set to zero.

**What the tab shows.**

- Tiles: the sodium screened (kt Na₂O); the share of that sodium whose
  loading is not set by the alkali limit; the glass needed unit by unit,
  with its increase over the alkali-only figure; the glass needed if all
  units were one blended feed, with that feed's loading and binding rule;
  and the number of units loaded below 15 wt% Na₂O.
- Units: one row per tank or layer (loading, binding rule, glass, extra
  glass, the mass ratios, each rule's limit, and the SO₃, Cl, F and Cr₂O₃
  content of the glass at that loading), sorted by extra glass.
- Summary and Groups: the same totals, and the sodium split by binding
  rule for all units, single- and double-shell tanks, and the saltcake
  and supernatant layers.
- Notes: the scope, the method, the rules with their source, and every
  note the query engine raised about the data.

**Reading the rule diagram.** The horizontal axis is SO₃ / Na₂O in the
waste and the vertical axis is the rule set's halide term per Na₂O
(Cl + 0.3 F + 0.4 Cr₂O₃ for the advanced set), both logarithmic. The
background is tinted by the kind of rule that binds there and each
region is named in place. Contour lines give the Na₂O loading the rules
allow (5, 10, 15, 20 and 23 wt%). One marker stands for one tank or
layer, with an area proportional to its sodium, coloured by tank system
(tanks) or phase family (layers); the diamond is all units as one
blended feed. Moving right (more sulfate) or up (more halide and
chromium) lowers the loading. The alkali boundaries also depend on
potassium: the regions are drawn for the K₂O / Na₂O ratio of the blended
feed, so a unit with much more potassium can sit in a region other than
its own binding rule, which the tooltip and the table always state.
Pointing at a marker gives its values; a click selects its row in the
Units table and a click on a row rings its marker; a double click opens
the tank. The other two plots are the sodium split by binding rule for
each group, and the units that cost the most extra glass.

**Blending.** In the advanced set every rule is linear in the component
masses, so a blended feed never needs more glass than its parts taken
one by one; its loading is never below the sodium-weighted harmonic mean
of the unit loadings, but can be well below their arithmetic mean. The
conservative halide envelope and the baseline halogen rule are
piecewise, and with them a blend can need more glass than its parts.

**Scope of validity.** The rules were developed for LAW melter feeds.
Applying them to the supernatant and saltcake of a tank as stored
ignores retrieval, blending, pretreatment, the sodium added in
processing and the wash liquors of sludge. The result is a screening
index of which rule would bind, not a forecast of glass production, and
§14 applies to it in full: it is not a glass formulation, a flowsheet
waste loading or a property prediction.

**Check values.** On the bundled 2026 export, with the advanced set,
tanks, supernatant + saltcake and no scope: 133 tanks hold sodium
(54.56 kt Na₂O); the alkali rule binds 53.2 % of it, alkali + sulfur
36.6 %, sulfur 3.3 % and halide 6.9 %; the glass is 230.3 kt at the
alkali limit, 276.9 kt tank by tank and 248.4 kt for one blended feed
at 21.97 wt% Na₂O.

---

## 13. Figure Composer

A tool for combining several already-built plots from anywhere in Ember
into one multi-panel figure, laid out and labeled the way a publication
or slide-deck figure usually is (bold panel letters in the corner: A, B,
C, ...). This is a general-purpose panel arranger — it doesn't care what
kind of plot each panel is, so you can freely mix, say, a scatter plot
from Element Explorer with a heatmap from Correlations and a bar chart
from Vitrification into one figure.

**Workflow:**

1. Build any plot you want as a panel, anywhere in Ember.
2. Click **Send to Figure Composer** on that plot's toolbar. This captures a
   *recipe* — which internal plotting function drew it and the exact data/
   parameters used, not a picture of it — and adds it to the Figure
   Composer's gallery, along with a suggested caption drawn from the
   plot's own title.
3. Repeat for every panel you want in the final figure — from any
   workspace, in any order.
4. Switch to the **Figure Composer** workspace. The gallery on the left
   lists every captured panel with an editable caption, reorder
   (move up/down) and remove controls.
5. With a panel selected, edit its **Title** (or check **Hide** to remove
   it), **X axis** / **Y axis** label text, and any of its own **Parameters**
   — a table of every keyword the recipe was captured with (e.g. a
   heatmap's `style`, `annotate`, `method`; a bar chart's `top_n`,
   `log_x`). Editing a value re-renders that panel with the new setting;
   **Reset** clears all parameter overrides back to what was captured.
   Leaving Title/X axis/Y axis blank keeps whatever the plot itself
   produces.
6. Set the layout: number of columns (or leave on **auto** to let Ember
   pick a reasonable grid for however many panels you have), panel-label
   style (`A, B, C`, `a, b, c`, `(a), (b), (c)`, `1, 2, 3`, or **none**),
   and label size.
7. The preview updates live as you change the gallery, a panel's
   overrides, or the layout.
8. **Export** at a specific physical size and DPI (PNG/PDF/SVG/TIFF) —
   the same exact-size export path every other plot in Ember uses, so a
   composed figure prints at the size you actually specified, not
   whatever size happened to be on screen, and stays sharp at any size
   since every panel is redrawn as real vector artists, not scaled up
   from a fixed-resolution picture.

Each panel is *redrawn from its recipe* into its own region of the combined
figure at compose time (not composited as a raster image of what it looked
like at capture time) — the exact same plotting code that draws it in its
own workspace runs again here, including its own colorbars, legends, or
seaborn-specific styling, just placed into one cell of the combined grid
instead of its own standalone figure. That's what makes every panel's title,
axis labels, and parameters still editable after capture, and what keeps
the whole combined figure fully vector (SVG/PDF export contains real paths
and text, not an embedded picture) — a heatmap panel's own cells still
render as a raster image internally, exactly as they always have standalone,
since that's the efficient way to draw many cells; that's unrelated to
composition and unaffected by any of this.

The underlying *data* a panel was captured with is frozen at capture
time — recapture (send it to the composer again) if you've since reloaded
the dataset or want a genuinely different query/selection reflected. Purely
cosmetic and many display changes (title, axis labels, color mode, style,
annotate, top-N, and anything else the recipe exposes as a parameter) don't
require recapturing at all — edit them directly in Figure Composer.

---

## 14. What the Vitrification tools do NOT calculate

Worth restating plainly, in one place:

- Real glass formulation, waste loading limits, or liquidus temperature.
- Viscosity, electrical conductivity, or thermal properties in general
  (except where GlassNet provides an ML *estimate*, always clearly
  labeled as such and only when `glasspy` is installed).
- Sulfate solubility or halide volatility *limits* (optical basicity and
  the volatile-element flag are composition-based indicators that
  correlate with these behaviors in the literature, not a solubility or
  volatility calculation).
- Redox equilibrium, or spinel/nepheline crystallization risk, beyond the
  coarse composition-based element-group flags in §12.1.
- PCT durability or TCLP performance.
- Official HLW/LAW/LLW waste classification (see §15).
- A final blend composition or mixing ratio (Blend Partners ranks
  candidates; building the actual blend composition is a manual step in
  Oxide Chemistry, §12.2).

## 15. Why Ember does not assign HLW/LAW/LLW automatically

Waste class is not determined by the columns in this CSV alone. For
Hanford, low-activity waste (LAW) and high-level waste (HLW) are tied to
treatment flowsheets, pretreatment, separations, legal definitions, DOE
decisions, and disposal pathways — none of which live in a composition
table. A composition table cannot determine official class, no matter how
sophisticated the composition-based heuristic. Ember shows relative
screening indicators (total Ci, total kg, Ci/kg proxy, radionuclide-
associated element inventory, problem-element fraction) — nothing more,
and nothing that should be read as a classification.

---

## 16. Common problems

- **A correlation looks impressively strong but is based on very few
  tanks** — check the overlap/N-tanks column before trusting it, and try
  the "control for tank size" partial-correlation checkbox (§10.1) to see
  if it survives.
- **A heatmap or pair-matrix plot is unreadable** — reduce the
  element/tank count, or switch to log scaling.
- **Predict properties (GlassNet) is disabled** — `glasspy` (and the PyTorch it depends
  on) isn't installed in this environment, or you're running the packaged
  `.exe` (which deliberately excludes it). Run from Python with `pip
  install glasspy` to enable it.
- **Seaborn-based plots show "not installed"** — `pip install seaborn`;
  every seaborn-based plot type falls back to a clear message rather than
  crashing.
- **Optical basicity shows "Excluded from Λ"** for one of your
  components — that oxide has no tabulated PNNL-20184 value (plutonium
  oxides, most notably). The number shown is still computed correctly
  over everything that *does* have a value; the exclusion list tells you
  what's missing so you can judge whether that matters for your case.
- **NBO/T shows "n/a (no former-role oxide present)"** — the selected
  composition (or your edited role table) has no oxide classified as a
  network former, so `T` is zero and the ratio is undefined by
  construction, not a bug.
- **A Figure Composer panel looks stale after I changed something in the
  source workspace** — the composer holds the *data* from when you
  clicked Send to Figure Composer, not a live link, so a genuinely different
  query/selection/reload won't show up on its own. Title, axis labels,
  and any of the panel's own listed Parameters (§13) can be changed
  directly in Figure Composer without recapturing; for anything else,
  remove the old panel and send the plot again after changing it.

---

## 17. Source notes for waste terminology

Background for interpreting LAW/HLW terminology and Hanford vitrification
context:

- Hanford DFLAW overview: https://www.hanford.gov/page.cfm/DFLAW
- Washington State Department of Ecology, Hanford tank waste treatment:
  https://ecology.wa.gov/waste-toxics/nuclear-waste/hanford-cleanup/tank-waste-management/tank-waste-treatment
- NRC high-level waste overview: https://www.nrc.gov/waste/high-level-waste
- IAEA radioactive waste classification publication:
  https://www.iaea.org/publications/8154/classification-of-radioactive-waste

Use these for terminology context. Do not use this app alone to make
regulatory determinations.

---

## 18. Credits

Developed by Sami Soudani, in the McCloy group at Washington State
University, with support from the U.S. Department of Energy (DOE).
`github.com/sams808/EMBER` · MIT License.

Composition data via PNNL's PHOENIX (see the citation at the top of this
notice). Built on polars, pandas, numpy, matplotlib, seaborn, scipy,
scikit-learn, networkx, plotly, xraydb, and PySide6. Optical basicity
values: Rodriguez & McCloy, PNNL-20184/EMSP-RPT-003 (2011). GlassNet:
Cassar, *Ceramics International* 49 (2023) 36013, trained on SciGlass.

---

## 19. Query

The Query workspace is one flexible question surface over the whole
dataset — the first piece of Ember's query engine. Instead of hunting for
a workspace whose fixed plot list happens to include your question, you
compose it:

- **Subject** — an element symbol, an exact analyte label, or a *basket*
  (a named group of elements/analytes that queries as one species; starter
  baskets ship for salt formers, halides, glass formers, RCRA metals, and
  mobile long-lived nuclides).
- **Basis** — `kg` (kg rows only), `Ci` (activity only), `g` (kg **and**
  Ci on one mass axis: Ci rows are bridged to grams via ICRP-107 specific
  activities), `wt%` (grams per gram of layer mass), or `g/L` (grams per
  liter of layer volume). Layer masses/volumes come from the per-layer
  Volume and ComponentDensity columns, summed over every in-scope layer of
  the group.
- **Group by** — up to two dimensions: tank, farm, waste phase, waste
  type, SST/DST, tank type, integrity, element, isotope, analyte
  (species provenance: *which* species carry the subject), and more.
- **Plot** — how the result is drawn (the plot grammar): bars; with a
  second grouping dimension, grouped or **stacked bars** (second
  dimension becomes the colored series), a **heatmap** (second dimension
  becomes the columns), or a **box plot** (distribution of
  dimension-1 values within each dimension-2 level). Whenever a chart
  trims to a top-N, it says so on the figure itself.
- **Guided questions** — the leftmost picker holds one-click starter
  questions ("Where do salt formers concentrate?", "How much uranium is
  in each tank, in grams?"). Picking one fills every control and runs,
  so you see exactly which knobs produced the answer.
- **Scope** — the app-wide scope bar above the workspace (farm, waste
  phase, waste type, SST/DST) filters every query. The active scope is
  stamped on the plot subtitle, in the status bar, and into the export
  manifest — a figure filtered to one farm always says so on its face.
  The bar only appears above workspaces that honor it.

Three honesty rules the engine enforces everywhere:

1. **Stoichiometric apportionment.** An element receives its *mass
   fraction* of every species reporting it: S gets 32.06/96.06 of SO₄'s
   mass, N gets 14.007/62.005 of NO₃'s, computed from CIAAW 2021 atomic
   weights. (Elsewhere in the app, legacy views may still attribute a
   species' full mass to its first element — the Query engine never does.)
2. **Units are never mixed silently.** kg and Ci are separate worlds; the
   `g` basis unites them only through nuclide-specific activities
   (ICRP-107 half-lives, updated to current ENSDF/NUBASE evaluations
   where they differ >1%). Combined-pair labels (239/240Pu, 243/244Cm,
   144Ce/Pr) carry no single specific activity and are **excluded from
   mass bases with a visible warning** — never silently dropped.
3. **Every resolution is disclosed.** The Notes tab (and every export)
   states per member exactly what it resolved to — e.g. iodine: *"129I
   (Ci→g bridged); no stable-isotope reporting exists — mass is a lower
   bound."* Warnings above the results flag mixed decay baselines
   (2015-07-01 vs 2022-01-01 cohorts) and rows with no assigned
   uncertainty (Inventory_RSD = 0).
4. **A total and a lower bound are not one quantity.** On the mass bases
   an element is taken from its reported mass where a layer gives one,
   and from its bridged radionuclide where the layer gives only activity.
   That is right for a site-wide total, but the two kinds of layer are
   not comparable with each other: elemental selenium is reported in 60
   tanks at hundreds to thousands of times the 79Se-implied mass of the
   remaining 117. When a subject mixes the two, a warning names it and
   counts the layers of each kind (*"Not one quantity across layers:
   Se (153 layer(s) with a reported mass, 383 nuclide-only)"*). Twelve
   elements of the shipped dataset are affected. To rank or correlate
   like with like, query the nuclide itself (e.g. the analyte `79Se`).

Element Explorer's **Ask in Query** button jumps here with the same
subject — useful when you want the engine's apportioned/bridged numbers
for something you were just exploring. (Element Explorer itself sums
reported rows as-is; the two workspaces answer deliberately different
questions, and each says which.)

Every result exports as CSV with the warnings and disclosures alongside
the numbers.

---

## 20. Projects and provenance

Two things make a number defensible a year after you produced it: knowing
exactly what produced it, and being able to run it again.

### Provenance records

Every export can carry a `provenance` record (`provenance.py`) stating:

- **which data** — the source file, its row and column counts, and a
  SHA-256 of its first 4 MB. The BBI is re-published periodically, so
  "I ran it on Hanford.csv" identifies nothing on its own; the
  fingerprint distinguishes one revision from another. It is labelled a
  *head* hash and is not a whole-file checksum.
- **which question** — the subject, basis, grouping and scope, plus the
  plot spec. Both are declarative, so the record is the query itself
  rather than a description of it.
- **which assumptions** — the reference tables in force (CIAAW 2021
  atomic weights; ICRP-107 half-lives updated to current evaluations),
  the app and Python versions, and *every warning and disclosure the
  engine raised*. A number is only as good as the exclusions behind it.

`methods_paragraph()` renders that record as prose for a methods
section — including the caveats, which is the part a hand-written
methods paragraph usually loses.

### `.ember` project files

A project (`ember_project.py`) saves the work that would otherwise be
retyped: custom baskets, saved questions (subject × basis × grouping ×
scope × mark), tank lists, and the active scope. This is possible only
because the query and plot layers are declarative — a saved question is
plain data, not a closure.

Save and open them from **File → Save project…** / **Open project…**
(Ctrl+S / Ctrl+O). Every Query export also writes a `provenance.csv` and
a `methods.txt` beside the data, both listed in the manifest.

Project files are JSON with a version number. Loading is deliberately
tolerant of keys it does not recognise, so a project written by a newer
Ember still opens here minus what this build cannot use — but a *format
version* it cannot read is refused outright rather than half-loaded.

Reopening a project against different data does not fail silently:
`problems()` lists every saved question that would no longer run (a basis
that no longer exists, a grouping dimension that was renamed, a basket
that has gone), and a fingerprint mismatch is reported as a plain
sentence — *"this project was built on 46,894 rows; the loaded data has
47,102. Numbers may have moved."*
