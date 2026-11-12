import pytest
from PySide6.QtWidgets import QMainWindow, QMessageBox

from conftest import REAL_ATTRS_PATH, REAL_CSV_PATH, requires_real_data
from data_model import HanfordDataset
from qt_correlations import (
    ANALYSIS_COMPARE, ANALYSIS_HEATMAP, ANALYSIS_SCAN, DEFAULT_COMPARE, DEFAULT_TARGET, HEATMAP_STYLES,
    CorrelationsPage, QuickScanTab,
)
from qt_correlations_structure import StructureTab
from qt_correlations_workbench import WorkbenchTab, completion_items


def _make_quick_scan(qtbot):
    app_window = QMainWindow()
    qtbot.addWidget(app_window)
    tab = QuickScanTab(app_window)
    qtbot.addWidget(tab)
    return tab


def _status(tab) -> str:
    return tab.app_window.statusBar().currentMessage()


def _capture(monkeypatch, name):
    calls = []
    monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: calls.append(a) or QMessageBox.Ok))
    return calls


def _figure_texts(tab):
    return [t.get_text() for ax in tab.plot.figure.axes for t in ax.texts]


class TestQuickScanTab:
    def test_actions_without_dataset_say_so_in_the_status_bar(self, qtbot, monkeypatch):
        info = _capture(monkeypatch, "information")
        tab = _make_quick_scan(qtbot)
        for action in (tab.run_target_scan, tab.run_selected, tab.run_heatmap, tab.run):
            tab.app_window.statusBar().clearMessage()
            action()
            assert "no dataset is loaded" in _status(tab)
        assert not info                      # nothing to dismiss
        assert tab._result is None

    def test_on_dataset_changed_defaults_to_kg(self, qtbot, sample_dataset):
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(sample_dataset)
        assert tab.unit_combo.currentText() == "kg"

    def test_target_scan_populates_table(self, qtbot, sample_dataset):
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.unit_combo.setCurrentText("Ci")  # Cs (137Cs) is a Ci-unit element in the fixture
        tab.target_edit.setText("Cs")
        tab.min_overlap_spin.setValue(0)
        tab.run_target_scan()
        qtbot.wait(20)
        matrix = tab._table_views["Matrix"].dataframe()
        assert "Cs" in matrix.columns
        assert tab.warnings_label.isHidden()          # reported in Ci: nothing to qualify

    def test_invalid_target_symbol_shows_warning_not_crash(self, qtbot, sample_dataset, monkeypatch):
        warnings = _capture(monkeypatch, "warning")
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.target_edit.setText("Zz")
        tab.run_target_scan()
        assert len(warnings) == 1 and "Zz" in warnings[0][2]      # the refusal is the one thing worth a dialog
        assert tab._result is None and tab.run_btn.isEnabled()

    def test_blank_target_is_refused_without_a_dialog(self, qtbot, sample_dataset, monkeypatch):
        warnings = _capture(monkeypatch, "warning")
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.target_edit.setText("  ")
        tab.run()
        assert not warnings and "enter a target element" in _status(tab)

    def test_dual_triple_populates_tables(self, qtbot, sample_dataset):
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Cd")
        tab.include_zeros_check.setChecked(True)
        tab.run_selected()
        qtbot.wait(20)
        pairs = tab._table_views["Pairs"].dataframe()
        assert set(pairs.iloc[0][["Element_A", "Element_B"]]) == {"Fe", "Cd"}
        joint = tab._table_views["Joint"].dataframe()
        assert joint.iloc[0]["N_elements"] == 2
        assert tab.tables.currentWidget() is tab._table_views["Pairs"]

    def test_dual_triple_too_few_elements_shows_warning_not_crash(self, qtbot, sample_dataset):
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe")
        tab.run_selected()

    def test_all_heatmap_styles_render_without_crash(self, qtbot, sample_dataset):
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(sample_dataset)
        for style in HEATMAP_STYLES:
            tab.heatmap_style_combo.setCurrentText(style)
            assert tab.heatmap_style_combo.currentText() == style
            tab.run_heatmap()
            qtbot.wait(20)

    def test_heatmap_populates_projection_table(self, qtbot, sample_dataset):
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.run_heatmap()
        qtbot.wait(20)
        assert not tab._heatmap_projection_df.empty
        assert "Total_inventory_kg" in tab._heatmap_projection_df.columns
        # The matrix behind the heat map has its own table; the scan table is left alone.
        shown = tab._table_views["Correlation matrix"].dataframe()
        assert shown.equals(tab._heatmap_corr_df) and not shown.empty
        assert tab._table_views["Target scan"].dataframe().empty
        assert tab.tables.currentWidget() is tab._table_views["Correlation matrix"]

    def test_control_for_tank_size_checkbox_runs_without_crash(self, qtbot, sample_dataset):
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.control_for_size_check.setChecked(True)
        tab.target_edit.setText("Fe")
        tab.min_overlap_spin.setValue(0)
        tab.run_target_scan()
        qtbot.wait(20)
        scan = tab._table_views["Target scan"].dataframe()
        if not scan.empty:
            assert scan["ControlledForTotalInventory"].all()
        tab.elements_edit.setText("Fe, Cd")
        tab.run_selected()
        qtbot.wait(20)
        pairs = tab._table_views["Pairs"].dataframe()
        assert pairs.iloc[0]["ControlledForTotalInventory"]

    def test_export_without_results_says_so_in_the_status_bar(self, qtbot, monkeypatch):
        info = _capture(monkeypatch, "information")
        tab = _make_quick_scan(qtbot)
        tab._export_tables()
        assert not info and "nothing to export" in _status(tab)

    def test_export_writes_bundle(self, qtbot, sample_dataset, tmp_path):
        tab = _make_quick_scan(qtbot)
        sample_dataset.output_root = tmp_path
        tab.on_dataset_changed(sample_dataset)
        tab.run_heatmap()
        qtbot.wait(20)
        tab._export_tables()
        bundles = list(tmp_path.glob("correlations_*"))
        assert len(bundles) == 1
        assert (bundles[0] / "correlation_plot.png").exists()


class TestQuickScanPagePass:
    def test_opening_state(self, qtbot):
        tab = _make_quick_scan(qtbot)
        assert tab.target_edit.text() == DEFAULT_TARGET == "Na"
        assert tab.elements_edit.text() == DEFAULT_COMPARE == "Na, Al, Fe"
        assert tab.run_btn.objectName() == "Primary" and tab.run_btn.text() == "Run"
        assert tab.warnings_label is tab.notes_banner and tab.warnings_label.isHidden()
        assert not tab.advanced_section.is_expanded()
        assert _figure_texts(tab) == ["No result yet", "Choose what to correlate and press Run."]
        assert tab.header.title_label.text() == "Quick scan" and tab.header.subtitle_label.wordWrap()
        # Readable choices; the identifier the science takes rides as item data.
        assert [tab.metric_combo.itemData(i) for i in range(tab.metric_combo.count())] == [
            "log10_plus1", "log10_inventory", "fraction", "inventory", "presence"]
        assert tab.metric_combo.itemText(0) == "Log10 of inventory + 1"
        assert [tab.method_combo.itemText(i) for i in range(tab.method_combo.count())] == ["Pearson", "Spearman"]

    def test_the_selector_shows_the_field_its_analysis_needs(self, qtbot):
        tab = _make_quick_scan(qtbot)
        assert tab._analysis() == ANALYSIS_SCAN
        assert not tab.target_edit.isHidden() and tab.elements_edit.isHidden()
        tab._set_analysis(ANALYSIS_COMPARE)
        assert tab.target_edit.isHidden() and not tab.elements_edit.isHidden()
        tab._set_analysis(ANALYSIS_HEATMAP)
        assert tab.target_edit.isHidden() and tab.elements_edit.isHidden()

    def test_run_answers_the_selected_analysis(self, qtbot, sample_dataset):
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab._set_analysis(ANALYSIS_COMPARE)
        tab.elements_edit.setText("Fe, Cd")
        tab.run()
        qtbot.wait(20)
        assert tab._result["analysis"] == ANALYSIS_COMPARE and not tab._table_views["Pairs"].dataframe().empty
        tab._set_analysis(ANALYSIS_HEATMAP)
        tab.run()
        qtbot.wait(20)
        assert tab._result["analysis"] == ANALYSIS_HEATMAP and not tab._heatmap_corr_df.empty
        # A direct call brings the selector along, so the page never labels one analysis as another.
        tab.target_edit.setText("Fe")
        tab.min_overlap_spin.setValue(0)
        tab.run_target_scan()
        qtbot.wait(20)
        assert tab._analysis() == ANALYSIS_SCAN and tab._result["target"] == "Fe"
        # Two tanks give no coefficient: the chart says so instead of standing empty.
        assert tab._table_views["Target scan"].dataframe().empty
        assert _figure_texts(tab)[0] == "Nothing to rank for Fe in kg"
        assert tab.warnings_label.isHidden()          # iron IS reported in kg: nothing to qualify

    def test_run_button_says_update_when_the_controls_left_the_result(self, qtbot, sample_dataset):
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.target_edit.setText("Fe")
        tab.min_overlap_spin.setValue(0)
        assert tab.run_btn.text() == "Run"
        tab.run()
        qtbot.wait(20)
        assert tab.run_btn.text() == "Run"
        tab.unit_combo.setCurrentText("Ci")
        assert tab.run_btn.text() == "Update"
        tab.unit_combo.setCurrentText("kg")
        assert tab.run_btn.text() == "Run"
        tab.plot_top_n_spin.setValue(12)              # a display choice: nothing to update
        assert tab.run_btn.text() == "Run"
        tab.target_edit.setText("Na")
        assert tab.run_btn.text() == "Update"

    def test_an_element_without_rows_on_the_basis_is_explained_calmly(self, qtbot, sample_dataset, monkeypatch):
        warnings = _capture(monkeypatch, "warning")
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.target_edit.setText("Cs")                 # cesium is reported as activity only
        tab.run()
        qtbot.wait(20)
        assert not warnings
        assert tab._table_views["Target scan"].dataframe().empty
        note = "Cs has no inventory in kg in this dataset: it is reported in Ci only. Set Basis to Ci to scan it."
        assert list(tab.warnings_label.notes()) == [note]
        assert tab.warnings_label.summary_label.text() == note and not tab.warnings_label.isHidden()
        assert _figure_texts(tab) == ["Nothing to rank for Cs in kg", note]
        assert "no partner to rank for Cs in kg" in _status(tab)
        # Following the note gives a result and clears the banner.
        tab.unit_combo.setCurrentText("Ci")
        tab.min_overlap_spin.setValue(0)
        tab.run()
        qtbot.wait(20)
        assert tab.warnings_label.isHidden()
        assert "Cs" in tab._table_views["Matrix"].dataframe().columns

    def test_compare_names_what_was_left_out(self, qtbot, sample_dataset):
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Cd, Cs, Qq")
        tab.run_selected()
        qtbot.wait(20)
        notes = list(tab.warnings_label.notes())
        assert notes[0] == "Qq is not an element symbol and was left out."
        assert notes[1].startswith("Cs has no inventory in kg in this dataset: it is reported in Ci only.")
        assert "2 notes qualify this result" in tab.warnings_label.summary_label.text()

    def test_fields_offer_what_the_dataset_holds(self, qtbot, sample_dataset):
        tab = _make_quick_scan(qtbot)
        assert completion_items(tab.target_edit) == []
        tab.on_dataset_changed(sample_dataset)
        assert {"Fe", "Na", "Cs", "Cd"} <= set(completion_items(tab.target_edit))
        assert {"Fe", "Na", "Cs", "Cd"} <= set(completion_items(tab.elements_edit))
        completer = tab.elements_edit.completer()
        tab.elements_edit.setText("Fe, N")
        completer.setCompletionPrefix("Fe, N")
        assert completer.currentCompletion() == "Fe, Na"

    def test_clicking_a_bar_selects_its_partner(self, qtbot, structure_dataset):
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(structure_dataset)
        tab.target_edit.setText("Cs")
        tab.run()
        qtbot.wait(20)
        scan = tab._table_views["Target scan"].dataframe()
        assert {"Sr", "Mo"} <= set(scan["PartnerElement"])
        assert tab.plot.ax.get_title() == "Top correlations with Cs"
        partner = str(scan["PartnerElement"].iloc[-1])
        tab.tables.setCurrentWidget(tab._table_views["Matrix"])
        tab._on_mark_clicked({"label": f"{partner} (n=1)", "value": 0.5})
        view = tab._table_views["Target scan"]
        assert tab.tables.currentWidget() is view
        assert scan["PartnerElement"].iloc[view.selectedIndexes()[0].row()] == partner
        tab._on_mark_clicked({"label": "not a row"})          # no match: no error

    def test_clicking_a_point_selects_its_tank(self, qtbot, sample_dataset):
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na")
        tab.run_selected()
        qtbot.wait(20)
        tab._on_mark_clicked({"label": "241-AN-104", "tank_id": "241-AN-104"})
        view = tab._table_views["Matrix"]
        assert tab.tables.currentWidget() is view
        assert view.dataframe()["WasteSiteId"].iloc[view.selectedIndexes()[0].row()] == "241-AN-104"

    def test_a_display_choice_redraws_the_standing_result(self, qtbot, sample_dataset, monkeypatch):
        import qt_correlations as qc
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.run_heatmap()
        qtbot.wait(20)
        computed, drawn = [], []
        monkeypatch.setattr(qc.csci, "full_correlation_matrix", lambda *a, **k: computed.append(1))
        real = qc.ph.plot_correlation_heatmap
        monkeypatch.setattr(qc.ph, "plot_correlation_heatmap",
                            lambda *a, **k: (drawn.append(k["style"]), real(*a, **k))[1])
        tab.heatmap_style_combo.setCurrentText(HEATMAP_STYLES[0])
        tab.annotate_check.setChecked(True)
        qtbot.wait(20)
        assert drawn == [HEATMAP_STYLES[0], HEATMAP_STYLES[0]] and not computed

    def test_dataset_change_clears_the_previous_result(self, qtbot, sample_dataset, structure_dataset):
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.target_edit.setText("Cs")
        tab.run()
        tab.run_heatmap()
        qtbot.wait(20)
        tab.on_dataset_changed(structure_dataset)
        qtbot.wait(20)
        assert tab._result is None and tab._heatmap_corr_df.empty and tab._matrix_df.empty
        assert all(view.dataframe().empty for view in tab._table_views.values())
        assert tab.warnings_label.isHidden() and tab.run_btn.text() == "Run"
        assert _figure_texts(tab)[0] == "No result yet"


class TestCorrelationsPage:
    def _page(self, qtbot):
        app_window = QMainWindow()
        qtbot.addWidget(app_window)
        page = CorrelationsPage(app_window)
        qtbot.addWidget(page)
        return page

    def test_hosts_four_tabs_cross_correlation_first(self, qtbot):
        page = self._page(qtbot)
        titles = [page.tabs.tabText(i) for i in range(page.tabs.count())]
        assert titles == ["Cross-correlation", "Quick Scan", "Association Workbench (kg)", "Structure"]

    def test_on_dataset_changed_forwards_to_quick_scan(self, qtbot, sample_dataset):
        page = self._page(qtbot)
        page.on_dataset_changed(sample_dataset)
        assert page.quick_scan_tab.dataset is sample_dataset

    def test_on_dataset_changed_forwards_to_structure_tab(self, qtbot, sample_dataset):
        page = self._page(qtbot)
        page.on_dataset_changed(sample_dataset)
        assert page.structure_tab.dataset is sample_dataset

    def test_only_the_first_tab_is_built_with_the_page(self, qtbot):
        page = self._page(qtbot)
        assert page.tabs.widget(0) is page.association_tab
        assert not any(page.is_built(name) for name in ("quick_scan_tab", "workbench_tab", "structure_tab"))
        assert not page.findChildren(QuickScanTab) and not page.findChildren(WorkbenchTab)
        assert not page.findChildren(StructureTab)

    def test_asking_for_a_tab_builds_it_in_place(self, qtbot, sample_dataset):
        page = self._page(qtbot)
        page.on_dataset_changed(sample_dataset)           # before any legacy tab exists
        changes = []
        page.tabs.currentChanged.connect(changes.append)
        workbench = page.workbench_tab
        assert isinstance(workbench, WorkbenchTab) and page.workbench_tab is workbench       # built once
        assert page.tabs.widget(2) is workbench
        assert workbench.dataset is sample_dataset        # the dataset it missed is handed over
        titles = [page.tabs.tabText(i) for i in range(page.tabs.count())]
        assert titles == ["Cross-correlation", "Quick Scan", "Association Workbench (kg)", "Structure"]
        assert page.tabs.currentWidget() is page.association_tab and not changes   # the swap is silent
        assert not page.is_built("quick_scan_tab") and not page.is_built("structure_tab")

    def test_opening_a_tab_builds_it(self, qtbot, sample_dataset):
        page = self._page(qtbot)
        page.on_dataset_changed(sample_dataset)
        relevance = []
        page.scope_relevance_changed.connect(lambda: relevance.append(page.scope_applies()))
        page.tabs.setCurrentIndex(3)                      # what a click on the tab does
        assert page.is_built("structure_tab")
        assert page.tabs.currentIndex() == 3 and page.tabs.currentWidget() is page.structure_tab
        assert isinstance(page.structure_tab, StructureTab)
        assert page.structure_tab.dataset is sample_dataset
        assert relevance == [False]                       # told once, after the real tab stands there
        page.tabs.setCurrentIndex(1)
        assert isinstance(page.tabs.currentWidget(), QuickScanTab)
        assert page.tabs.count() == 4

    def test_a_later_dataset_reaches_built_tabs_and_waits_for_the_others(self, qtbot, sample_dataset,
                                                                         structure_dataset):
        page = self._page(qtbot)
        page.on_dataset_changed(sample_dataset)
        quick = page.quick_scan_tab
        page.on_dataset_changed(structure_dataset)
        assert quick.dataset is structure_dataset
        assert not page.is_built("workbench_tab")
        assert page.workbench_tab.dataset is structure_dataset

    def test_every_tab_follows_the_page_layout(self, qtbot):
        """Chart on the left, tables on the right, and a result area that
        asks for no more height than it needs: the page's preferred height
        is what the scroll area around it enforces, so a chart that would
        like 400 px turns an opened Advanced section into a page scroll bar."""
        from qt_widgets import PlotWidget
        page = self._page(qtbot)
        page.resize(1220, 748)                 # the page area of a 1440 x 900 window
        page.show()
        qtbot.waitExposed(page)
        for tab in (page.association_tab, page.quick_scan_tab, page.workbench_tab, page.structure_tab):
            page.tabs.setCurrentWidget(tab)
            qtbot.wait(30)
            splitter = tab.splitter
            assert isinstance(splitter.widget(0), PlotWidget) and splitter.widget(1) is tab.tables
            assert not splitter.childrenCollapsible()
            assert splitter.sizeHint().height() == splitter.minimumSizeHint().height()
            margins = tab.layout().contentsMargins()
            assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == (20, 16, 20, 16)
            assert tab.layout().spacing() == 12
            assert tab.header.title_label.text() and tab.header.subtitle_label.wordWrap()
            # Opening Advanced takes its room from the result area, not from the page.
            before = (tab.height(), splitter.height())
            tab.advanced_section.set_expanded(True)
            qtbot.wait(30)
            assert tab.height() == before[0] and splitter.height() < before[1]
            assert splitter.height() >= splitter.minimumSizeHint().height()
            tab.advanced_section.set_expanded(False)
            qtbot.wait(30)
        page.hide()


@pytest.fixture(scope="module")
def real_dataset():
    dataset = HanfordDataset()
    dataset.load(REAL_CSV_PATH, use_cache=False, attributes_path=REAL_ATTRS_PATH)
    return dataset


@requires_real_data
class TestDefaultsOnTheShippedDataset:
    """The page must answer as it opens: every default returns rows on Hanford.csv."""

    def test_default_target_scan_returns_partners(self, qtbot, real_dataset):
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(real_dataset)
        assert tab.unit_combo.currentText() == "kg"
        tab.run()
        qtbot.wait(20)
        scan = tab._table_views["Target scan"].dataframe()
        assert len(scan) > 20 and set(scan["TargetElement"]) == {"Na"}
        assert tab.warnings_label.isHidden()
        assert tab.plot.ax.get_title() == "Top correlations with Na"

    def test_default_comparison_returns_three_pairs(self, qtbot, real_dataset):
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(real_dataset)
        tab._set_analysis(ANALYSIS_COMPARE)
        tab.run()
        qtbot.wait(20)
        pairs = tab._table_views["Pairs"].dataframe()
        assert len(pairs) == 3 and pairs["Correlation_r"].notna().all()
        assert tab.warnings_label.isHidden()

    def test_cesium_is_pointed_to_the_activity_basis(self, qtbot, real_dataset):
        tab = _make_quick_scan(qtbot)
        tab.on_dataset_changed(real_dataset)
        tab.target_edit.setText("Cs")
        tab.run()
        qtbot.wait(20)
        assert tab._table_views["Target scan"].dataframe().empty
        assert "reported in Ci only" in tab.warnings_label.text()
        tab.unit_combo.setCurrentText("Ci")
        tab.run()
        qtbot.wait(20)
        assert len(tab._table_views["Target scan"].dataframe()) > 5 and tab.warnings_label.isHidden()
