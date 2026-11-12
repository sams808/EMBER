import pandas as pd
import polars as pl
import pytest
from PySide6.QtWidgets import QMainWindow, QMessageBox

import data_model as dm
import qt_vitrification_screening as qvs
import vitrification_science as vsci
from qt_vitrification_screening import BlendPartnersTab, CandidateSearchTab, ScreeningTab


def _window(qtbot):
    app_window = QMainWindow()
    qtbot.addWidget(app_window)
    return app_window


class TestScreeningTab:
    def test_actions_without_dataset_show_message_not_crash(self, qtbot):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.run_summary()
        tab.export_tables()

    def test_groups_and_notes_populated_on_construction(self, qtbot):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        assert not tab._table_views["Groups"].dataframe().empty
        assert not tab._table_views["Waste notes"].dataframe().empty

    def test_build_summary_populates_table_and_plot(self, qtbot, vitrification_dataset):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.run_summary()
        qtbot.wait(20)
        summary = tab._table_views["Screening summary"].dataframe()
        assert set(summary["WasteSiteId"]) == set(vitrification_dataset.available_tanks())

    def test_custom_persisted_weights_are_disclosed_not_hidden(self, qtbot):
        # Review finding: QSettings-persisted weight edits survive restarts;
        # with the weights section collapsed by default, a non-default score
        # would be silently applied. Custom weights must auto-expand the
        # section so the formula readout is on screen.
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.weight_spins["glass_former_weight"].setValue(5.0)  # persists via QSettings

        tab2 = ScreeningTab(_window(qtbot))  # fresh construction reads the persisted value
        qtbot.addWidget(tab2)
        assert tab2.advanced_section.is_expanded() is True

    def test_default_weights_leave_the_section_collapsed(self, qtbot):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        assert tab.advanced_section.is_expanded() is False

    def test_default_plot_is_score_breakdown(self, qtbot):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        assert tab.plot_type_combo.currentText() == "Score breakdown (top tanks)"

    def test_switching_plot_type_redraws_without_crash(self, qtbot, vitrification_dataset):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.run_summary()
        tab.plot_type_combo.setCurrentText("Screening map (kg vs Ci)")
        qtbot.wait(20)
        assert tab.plot.ax.get_title() == "Vitrification screening map"

    def test_double_clicking_a_row_emits_the_tank_id(self, qtbot, vitrification_dataset):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.run_summary()
        received = []
        tab.jump_to_blend_partners.connect(received.append)
        top_tank = tab._table_views["Screening summary"].dataframe().iloc[0]["WasteSiteId"]
        index = tab._table_views["Screening summary"].model().index(0, 0)
        tab._on_summary_row_double_clicked(index)
        assert received == [top_tank]

    def test_double_click_before_any_summary_is_a_noop(self, qtbot):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        received = []
        tab.jump_to_blend_partners.connect(received.append)
        index = tab._table_views["Screening summary"].model().index(0, 0)
        tab._on_summary_row_double_clicked(index)  # empty table -- must not raise or emit
        assert received == []

    def test_default_weight_spins_match_science_defaults(self, qtbot):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        for key, default in vsci.SCREENING_WEIGHT_DEFAULTS.items():
            assert tab.weight_spins[key].value() == default

    def test_reset_to_defaults_restores_edited_weight(self, qtbot):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.weight_spins["glass_former_weight"].setValue(999.0)
        from qt_vitrification_screening import _reset_weight_spins
        _reset_weight_spins(tab.weight_spins, vsci.SCREENING_WEIGHT_DEFAULTS)
        assert tab.weight_spins["glass_former_weight"].value() == vsci.SCREENING_WEIGHT_DEFAULTS["glass_former_weight"]

    def test_edited_weight_persists_to_a_freshly_constructed_tab(self, qtbot):
        tab_a = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab_a)
        tab_a.weight_spins["glass_former_weight"].setValue(12.5)

        tab_b = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab_b)
        assert tab_b.weight_spins["glass_former_weight"].value() == 12.5

        # Reset must persist the default too, not just the widget value.
        from qt_vitrification_screening import _reset_weight_spins
        _reset_weight_spins(tab_a.weight_spins, vsci.SCREENING_WEIGHT_DEFAULTS)
        tab_c = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab_c)
        assert tab_c.weight_spins["glass_former_weight"].value() == vsci.SCREENING_WEIGHT_DEFAULTS["glass_former_weight"]

    def test_custom_weight_changes_score(self, qtbot, vitrification_dataset):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.weight_spins["glass_former_weight"].setValue(1000.0)
        tab.run_summary()
        qtbot.wait(20)
        summary = tab._table_views["Screening summary"].dataframe().set_index("WasteSiteId")
        assert summary.loc["241-A-103", "Vitrification_screening_score_proxy"] == 100.0  # clipped

    def test_oxide_basis_runs_without_crash(self, qtbot, vitrification_dataset):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.basis_combo.setCurrentText("Oxide")
        tab.run_summary()
        qtbot.wait(20)
        summary = tab._table_views["Screening summary"].dataframe()
        assert "frac_glass_former_or_intermediate_oxide_basis" in summary.columns

    def test_formula_label_reflects_weight_changes(self, qtbot):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.weight_spins["glass_former_weight"].setValue(42.0)
        assert "42" in tab.formula_label.text()

    def test_export_writes_bundle(self, qtbot, vitrification_dataset, tmp_path):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        vitrification_dataset.output_root = tmp_path
        tab.on_dataset_changed(vitrification_dataset)
        tab.run_summary()
        qtbot.wait(20)
        tab.export_tables()
        bundles = list(tmp_path.glob("vitrification_screening_*"))
        assert len(bundles) == 1
        assert (bundles[0] / "tank_screening_summary.csv").exists()
        assert (bundles[0] / "group_definitions.csv").exists()
        assert (bundles[0] / "waste_class_notes.csv").exists()

    def test_export_survives_savefig_failure(self, qtbot, vitrification_dataset, tmp_path, monkeypatch):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        vitrification_dataset.output_root = tmp_path
        tab.on_dataset_changed(vitrification_dataset)
        tab.run_summary()
        qtbot.wait(20)
        monkeypatch.setattr(tab.plot.figure, "savefig", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        tab.export_tables()  # must not raise despite the plot PNG save failing


class TestWeightPersistenceIsIndependentPerTab:
    def test_screening_and_candidate_weights_do_not_collide(self, qtbot):
        # Both dicts share the key "glass_former_weight" -- settings_prefix
        # must keep them from clobbering each other in QSettings.
        screening_tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(screening_tab)
        screening_tab.weight_spins["glass_former_weight"].setValue(777.0)

        candidate_tab = CandidateSearchTab(_window(qtbot))
        qtbot.addWidget(candidate_tab)
        assert candidate_tab.weight_spins["glass_former_weight"].value() == vsci.CANDIDATE_WEIGHT_DEFAULTS["glass_former_weight"]


class TestCandidateSearchTab:
    def test_actions_without_dataset_show_message_not_crash(self, qtbot):
        tab = CandidateSearchTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.run_candidates()
        tab.export_tables()

    def test_target_and_penalty_ranking(self, qtbot, vitrification_dataset):
        tab = CandidateSearchTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.target_edit.setText("B")
        tab.penalty_edit.setText("Cl")
        tab.required_edit.setText("")
        tab.run_candidates()
        qtbot.wait(20)
        ranking = tab.table.dataframe()
        assert ranking.iloc[0]["WasteSiteId"] == "241-A-101"

    def test_required_elements_filters_results(self, qtbot, vitrification_dataset):
        tab = CandidateSearchTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.required_edit.setText("Si")
        tab.run_candidates()
        qtbot.wait(20)
        ranking = tab.table.dataframe()
        assert set(ranking["WasteSiteId"]) == {"241-A-103"}

    def test_min_kg_filters_results(self, qtbot, vitrification_dataset):
        tab = CandidateSearchTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.min_kg_spin.setValue(10.0)
        tab.run_candidates()
        qtbot.wait(20)
        ranking = tab.table.dataframe()
        assert set(ranking["WasteSiteId"]) == {"241-A-101"}

    def test_custom_weights_reflected_in_scores(self, qtbot, vitrification_dataset):
        tab = CandidateSearchTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.weight_spins["target_weight"].setValue(0.0)
        tab.weight_spins["glass_former_weight"].setValue(0.0)
        tab.weight_spins["problem_weight"].setValue(0.0)
        tab.weight_spins["penalty_weight"].setValue(0.0)
        tab.run_candidates()
        qtbot.wait(20)
        ranking = tab.table.dataframe()
        assert (ranking["User_search_score_proxy"] == 0.0).all()

    def test_oxide_basis_runs_without_crash(self, qtbot, vitrification_dataset):
        tab = CandidateSearchTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.basis_combo.setCurrentText("Oxide")
        tab.run_candidates()
        qtbot.wait(20)

    def test_export_writes_bundle(self, qtbot, vitrification_dataset, tmp_path):
        tab = CandidateSearchTab(_window(qtbot))
        qtbot.addWidget(tab)
        vitrification_dataset.output_root = tmp_path
        tab.on_dataset_changed(vitrification_dataset)
        tab.run_candidates()
        qtbot.wait(20)
        tab.export_tables()
        bundles = list(tmp_path.glob("vitrification_candidates_*"))
        assert len(bundles) == 1
        assert (bundles[0] / "candidate_ranking.csv").exists()

    def test_export_survives_savefig_failure(self, qtbot, vitrification_dataset, tmp_path, monkeypatch):
        tab = CandidateSearchTab(_window(qtbot))
        qtbot.addWidget(tab)
        vitrification_dataset.output_root = tmp_path
        tab.on_dataset_changed(vitrification_dataset)
        tab.run_candidates()
        qtbot.wait(20)
        monkeypatch.setattr(tab.plot.figure, "savefig", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        tab.export_tables()


class TestBlendPartnersTab:
    def test_actions_without_dataset_show_message_not_crash(self, qtbot):
        tab = BlendPartnersTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.run_blend()
        tab.export_tables()

    def test_on_dataset_changed_populates_base_tank_combo(self, qtbot, vitrification_dataset):
        tab = BlendPartnersTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        assert tab.base_tank_combo.count() == len(vitrification_dataset.available_tanks())
        assert tab.base_tank_combo.currentText() == vitrification_dataset.available_tanks()[0]

    def test_no_base_tank_selected_shows_warning(self, qtbot, vitrification_dataset):
        tab = BlendPartnersTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.base_tank_combo.setCurrentText("")
        tab.run_blend()  # said in the status bar; see TestStates

    def test_orthogonal_partner_ranking_hand_verified(self, qtbot, vitrification_dataset):
        tab = BlendPartnersTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.base_tank_combo.setCurrentText("241-A-101")
        tab.run_blend()
        qtbot.wait(20)
        result = tab.table.dataframe().set_index("PartnerTank")
        assert result.loc["241-A-103", "Blend_complement_score_proxy"] > result.loc["241-A-102", "Blend_complement_score_proxy"]

    def test_custom_weights_change_score(self, qtbot, vitrification_dataset):
        tab = BlendPartnersTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.base_tank_combo.setCurrentText("241-A-101")
        tab.weight_spins["dissimilarity_weight"].setValue(0.0)
        tab.run_blend()
        qtbot.wait(20)
        assert not tab.table.dataframe().empty

    def test_oxide_basis_runs_without_crash(self, qtbot, vitrification_dataset):
        tab = BlendPartnersTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.base_tank_combo.setCurrentText("241-A-101")
        tab.basis_combo.setCurrentText("Oxide")
        tab.run_blend()
        qtbot.wait(20)

    def test_re_setting_dataset_preserves_current_selection(self, qtbot, vitrification_dataset):
        tab = BlendPartnersTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.base_tank_combo.setCurrentText("241-A-103")
        tab.on_dataset_changed(vitrification_dataset)
        assert tab.base_tank_combo.currentText() == "241-A-103"

    def test_export_writes_bundle(self, qtbot, vitrification_dataset, tmp_path):
        tab = BlendPartnersTab(_window(qtbot))
        qtbot.addWidget(tab)
        vitrification_dataset.output_root = tmp_path
        tab.on_dataset_changed(vitrification_dataset)
        tab.base_tank_combo.setCurrentText("241-A-101")
        tab.run_blend()
        qtbot.wait(20)
        tab.export_tables()
        bundles = list(tmp_path.glob("vitrification_blend_*"))
        assert len(bundles) == 1
        assert (bundles[0] / "blend_partners.csv").exists()

    def test_export_survives_savefig_failure(self, qtbot, vitrification_dataset, tmp_path, monkeypatch):
        tab = BlendPartnersTab(_window(qtbot))
        qtbot.addWidget(tab)
        vitrification_dataset.output_root = tmp_path
        tab.on_dataset_changed(vitrification_dataset)
        tab.base_tank_combo.setCurrentText("241-A-101")
        tab.run_blend()
        qtbot.wait(20)
        monkeypatch.setattr(tab.plot.figure, "savefig", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        tab.export_tables()

    def test_on_dataset_changed_with_unloaded_dataset_is_noop(self, qtbot):
        import data_model as dm
        tab = BlendPartnersTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(dm.HanfordDataset())
        assert tab.base_tank_combo.count() == 0


# ----------------------------------------------------------------------
# The page pass: wording, states, notes, weights, links
# ----------------------------------------------------------------------
@pytest.fixture
def no_popups(monkeypatch):
    """Record any modal box used where the status bar should speak."""
    shown = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(lambda *a, _k=kind, **k: shown.append((_k, a[1:3]))))
    return shown


def _dataset(rows):
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": [r[0] for r in rows], "Analyte": [r[1] for r in rows],
        "WastePhase": ["Liquid"] * len(rows), "WasteType": ["T1"] * len(rows),
        "Inventory": [r[2] for r in rows], "Units": [r[3] for r in rows],
    }))
    dataset.report = None
    return dataset


ALL_TABS = (ScreeningTab, CandidateSearchTab, BlendPartnersTab)


class TestWording:
    @pytest.mark.parametrize("cls", ALL_TABS)
    def test_primary_button_is_run_and_export_sits_in_the_header(self, qtbot, cls):
        tab = cls(_window(qtbot))
        qtbot.addWidget(tab)
        assert tab.run_btn.text() == "Run" and tab.run_btn.objectName() == "Primary"
        assert tab.export_btn.text() == "Export..." and tab.export_btn.objectName() == "Secondary"
        assert tab.export_btn.parent() is tab.header
        assert tab.warnings_label is tab.notes_banner

    def test_the_screening_caveat_is_the_header_subtitle(self, qtbot):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        subtitle = tab.header.subtitle_label.text()
        assert "screening metric only" in subtitle
        assert "not an official waste classification or glass formulation model" in subtitle
        assert tab.header.subtitle_label.wordWrap()         # it can no longer widen the window

    @pytest.mark.parametrize("cls", ALL_TABS)
    def test_basis_reads_as_text_and_hands_the_science_its_own_value(self, qtbot, cls):
        tab = cls(_window(qtbot))
        qtbot.addWidget(tab)
        combo = tab.basis_combo
        assert [combo.itemText(i) for i in range(combo.count())] == ["Elemental", "Oxide"]
        assert [combo.itemData(i) for i in range(combo.count())] == ["elemental", "oxide"]
        assert qvs.basis_value(combo) == "elemental"
        combo.setCurrentText("Oxide")
        assert qvs.basis_value(combo) == "oxide"

    def test_the_science_call_receives_the_lower_case_basis(self, qtbot, vitrification_dataset, monkeypatch):
        seen = []
        original = vsci.tank_category_summary
        monkeypatch.setattr(vsci, "tank_category_summary",
                            lambda dataset, weights=None, basis="elemental": seen.append(basis)
                            or original(dataset, weights=weights, basis=basis))
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.run_summary()
        tab.basis_combo.setCurrentText("Oxide")
        tab.run_summary()
        assert seen == ["elemental", "oxide"]

    @pytest.mark.parametrize("cls", ALL_TABS)
    def test_every_input_has_a_tooltip(self, qtbot, cls):
        from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox
        tab = cls(_window(qtbot))
        qtbot.addWidget(tab)
        for kind in (QComboBox, QDoubleSpinBox, QSpinBox):
            for widget in tab.findChildren(kind):
                assert widget.toolTip(), f"{kind.__name__} without a tooltip in {cls.__name__}"
        for name in ("target_edit", "penalty_edit", "required_edit"):
            edit = getattr(tab, name, None)
            if isinstance(edit, QLineEdit):
                assert edit.toolTip()


class TestRunAndUpdate:
    def test_screening_button_reads_update_when_a_control_differs_from_the_result(self, qtbot, vitrification_dataset):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.basis_combo.setCurrentText("Oxide")
        assert tab.run_btn.text() == "Run"                  # nothing is standing yet
        tab.run_summary()
        assert tab.run_btn.text() == "Run"
        tab.basis_combo.setCurrentText("Elemental")
        assert tab.run_btn.text() == "Update"
        tab.basis_combo.setCurrentText("Oxide")
        assert tab.run_btn.text() == "Run"                  # back to what produced the result
        tab.weight_spins["redox_weight"].setValue(-11.0)
        assert tab.run_btn.text() == "Update"
        tab.run_summary()
        assert tab.run_btn.text() == "Run"

    def test_candidate_and_blend_buttons_follow_their_controls(self, qtbot, vitrification_dataset):
        candidate = CandidateSearchTab(_window(qtbot))
        qtbot.addWidget(candidate)
        candidate.on_dataset_changed(vitrification_dataset)
        candidate.run_candidates()
        candidate.target_edit.setText("Cs,Sr,  Tc, I")      # the same list, written differently
        assert candidate.run_btn.text() == "Run"
        candidate.target_edit.setText("B")
        assert candidate.run_btn.text() == "Update"
        blend = BlendPartnersTab(_window(qtbot))
        qtbot.addWidget(blend)
        blend.on_dataset_changed(vitrification_dataset)
        blend.run_blend()
        blend.base_tank_combo.setCurrentText("241-A-103")
        assert blend.run_btn.text() == "Update"
        blend.run_blend()
        assert blend.run_btn.text() == "Run"

    def test_a_late_answer_to_an_old_question_is_discarded(self, qtbot, vitrification_dataset):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        stale = qvs._Outcome(token=tab._run_token - 1, inputs=tab._read_inputs(),
                             frame=pd.DataFrame({"WasteSiteId": ["241-Z-999"]}))
        tab._on_run_done(stale)
        assert tab._table_views["Screening summary"].dataframe().empty

    @pytest.mark.parametrize("cls", ALL_TABS)
    def test_a_new_dataset_clears_the_standing_result(self, qtbot, vitrification_dataset, cls):
        tab = cls(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.run_btn.click()
        view = tab._result_view()
        assert not view.dataframe().empty
        tab.on_dataset_changed(vitrification_dataset)
        assert view.dataframe().empty
        assert tab.run_btn.text() == "Run"
        assert tab.notes_banner.notes() == ()


class TestStates:
    @pytest.mark.parametrize("cls,run,name", [
        (ScreeningTab, "run_summary", "Screening"),
        (CandidateSearchTab, "run_candidates", "Candidate search"),
        (BlendPartnersTab, "run_blend", "Blend partners"),
    ])
    def test_no_dataset_is_said_in_the_status_bar_without_a_popup(self, qtbot, no_popups, cls, run, name):
        window = _window(qtbot)
        tab = cls(window)
        qtbot.addWidget(tab)
        getattr(tab, run)()
        assert window.statusBar().currentMessage() == f"{name}: no dataset is loaded yet."
        tab.export_tables()
        assert "nothing to export yet" in window.statusBar().currentMessage()
        assert no_popups == []

    @pytest.mark.parametrize("cls", ALL_TABS)
    def test_idle_chart_and_table_say_what_to_do(self, qtbot, cls):
        tab = cls(_window(qtbot))
        qtbot.addWidget(tab)
        texts = {t.get_gid(): t.get_text() for t in tab.plot.ax.texts}
        assert texts["ember-empty-title"] == "No result yet"
        assert "Run" in texts["ember-empty-hint"]
        assert tab._result_view().empty_state.title_label.text() == "No result yet"

    def test_blend_refuses_an_unknown_or_missing_base_tank_in_the_status_bar(self, qtbot, no_popups,
                                                                            vitrification_dataset):
        window = _window(qtbot)
        tab = BlendPartnersTab(window)
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.base_tank_combo.setCurrentText("")
        tab.run_blend()
        assert window.statusBar().currentMessage() == "Blend partners: choose a base tank first."
        tab.base_tank_combo.setCurrentText("241-ZZ-999")
        tab.run_blend()
        assert window.statusBar().currentMessage() == "Blend partners: the dataset has no tank named 241-ZZ-999."
        assert tab.table.dataframe().empty and no_popups == []

    def test_a_search_no_tank_passes_says_so(self, qtbot, vitrification_dataset):
        tab = CandidateSearchTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.required_edit.setText("U")                      # no tank of the fixture reports uranium
        tab.run_candidates()
        assert tab.table.dataframe().empty
        assert tab.table.empty_state.title_label.text() == "No tank qualifies"


class TestDefaultsReturnRows:
    def test_default_candidate_search_ranks_tanks(self, qtbot, vitrification_dataset):
        tab = CandidateSearchTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.run_candidates()                                # targets as shipped: Cs, Sr, Tc, I
        ranking = tab.table.dataframe()
        assert not ranking.empty
        assert ranking["UserTargetFraction_sum"].max() > 0  # the targets are found, as activity

    def test_default_blend_search_ranks_partners(self, qtbot, vitrification_dataset):
        tab = BlendPartnersTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.run_blend()                                     # base tank as preselected: the first one
        assert not tab.table.dataframe().empty


class TestNotes:
    def test_unattributed_inventory_is_one_readable_sentence_per_unit(self):
        dataset = _dataset([
            ("241-A-101", "Na", 5.0, "kg"), ("241-A-101", "Total Alpha", 21417.2, "Ci"),
            ("241-A-101", "Aroclor-1254", 300.0, "kg"), ("241-A-102", "Aroclor-1260", 187.7, "kg"),
            ("241-A-102", "Na", 3.0, "kg"),
        ])
        import tank_science as tsci
        assert tsci.unattributed_mass_note(dataset, dataset.available_tanks()) is not None
        notes = qvs.unattributed_notes(dataset)
        assert len(notes) == 2
        ci, kg = notes
        assert "21,417 Ci reported as Total Alpha" in ci and "e+" not in ci
        assert "487.7 kg reported under 2 labels" in kg and "Aroclor-1254, Aroclor-1260" in kg
        for note in notes:
            assert note.endswith(".") and "Not shown" not in note

    def test_nothing_is_said_when_every_row_names_an_element(self, vitrification_dataset):
        import tank_science as tsci
        assert tsci.unattributed_mass_note(vitrification_dataset, vitrification_dataset.available_tanks()) is None
        assert qvs.unattributed_notes(vitrification_dataset) == []

    def test_notes_are_in_the_banner_after_a_run_and_never_in_a_wide_label(self, qtbot):
        from PySide6.QtWidgets import QLabel
        dataset = _dataset([("241-A-101", "Na", 5.0, "kg"), ("241-A-101", "Si", 5.0, "kg"),
                            ("241-A-101", "Total Alpha", 21417.2, "Ci")])
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(dataset)
        assert not tab.notes_banner.isVisibleTo(tab)
        tab.run_summary()
        assert tab.notes_banner.isVisibleTo(tab)
        assert "21,417 Ci reported as Total Alpha" in tab.warnings_label.text()
        assert all(label.minimumSizeHint().width() < 500 for label in tab.findChildren(QLabel))

    def test_custom_weights_are_stated_beside_the_result(self, qtbot, vitrification_dataset):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.run_summary()
        assert tab.notes_banner.notes() == ()
        tab.weight_spins["modifier_weight"].setValue(30.0)
        tab.run_summary()
        assert tab.notes_banner.notes() == (
            "The score weights differ from the documented defaults (see Advanced options).",)


class TestWeights:
    @pytest.mark.parametrize("cls,terms", [
        (ScreeningTab, qvs.SCREENING_TERMS), (CandidateSearchTab, qvs.CANDIDATE_TERMS),
        (BlendPartnersTab, qvs.BLEND_TERMS),
    ])
    def test_every_weight_is_labelled_and_says_what_it_multiplies(self, qtbot, cls, terms):
        from PySide6.QtWidgets import QLabel
        tab = cls(_window(qtbot))
        qtbot.addWidget(tab)
        assert set(terms) == set(cls.WEIGHT_DEFAULTS) == set(tab.weight_spins)
        captions = {label.text() for label in tab.advanced_section.findChildren(QLabel)}
        for key, spin in tab.weight_spins.items():
            label, term = terms[key]
            assert label in captions
            tip = spin.toolTip()
            assert tip.startswith(f"Multiplies {term}.")
            default = cls.WEIGHT_DEFAULTS[key]
            assert ("raises the score" if default > 0 else "lowers the score") in tip
            assert f"{default:g}" in tip

    def test_reset_button_restores_the_defaults(self, qtbot):
        tab = BlendPartnersTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.weight_spins["dissimilarity_weight"].setValue(0.9)
        assert tab.reset_btn.text() == "Reset to defaults"
        tab.reset_btn.click()
        assert qvs._current_weights(tab.weight_spins) == vsci.BLEND_WEIGHT_DEFAULTS

    def test_formula_helper_keeps_each_terms_own_sign(self):
        text = qvs._format_score_formula({"glass_former_weight": 60, "modifier_weight": 25, "problem_weight": 45,
                                          "volatile_weight": -25, "redox_weight": -10})
        assert text == ("score = 60*glass_former + 25*modifier + 45*problem - 25*volatile - 10*redox "
                        "(clipped to [-100, 100])")


class TestTablesAndLinks:
    def test_summary_leads_with_tank_farm_and_score_and_holds_the_science_table(self, qtbot, vitrification_dataset):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.run_summary()
        shown = tab._table_views["Screening summary"].dataframe()
        science = vsci.tank_category_summary(vitrification_dataset)
        assert list(shown.columns[:3]) == ["WasteSiteId", "TankFarm", "Vitrification_screening_score_proxy"]
        assert sorted(shown.columns) == sorted(science.columns)
        pd.testing.assert_frame_equal(shown[list(science.columns)], science, check_exact=True)
        pd.testing.assert_frame_equal(tab.summary_df, science, check_exact=True)    # what the export writes

    def test_group_names_read_as_words_and_the_export_keeps_the_identifiers(self, qtbot, vitrification_dataset,
                                                                            tmp_path):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        groups = tab._table_views["Groups"].dataframe()
        assert "Glass former or intermediate" in set(groups["Group"])
        assert not any("_" in name for name in groups["Group"])
        vitrification_dataset.output_root = tmp_path
        tab.on_dataset_changed(vitrification_dataset)
        tab.run_summary()
        tab.export_tables()
        exported = pd.read_csv(next(tmp_path.glob("vitrification_screening_*")) / "group_definitions.csv")
        assert "glass_former_or_intermediate" in set(exported["Group"])

    def test_a_click_on_a_chart_mark_selects_the_tank_row(self, qtbot, vitrification_dataset):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.run_summary()
        tab.tables.setCurrentIndex(1)
        view = tab._table_views["Screening summary"]
        frame = view.dataframe()
        tab._on_mark_clicked({"tank_id": "241-A-102", "label": "241-A-102"})
        assert tab.tables.currentWidget() is view
        rows = {index.row() for index in view.selectedIndexes()}
        assert rows == {list(frame["WasteSiteId"]).index("241-A-102")}
        assert tab.selected_tank() == "241-A-102"

    def test_blend_chart_click_selects_the_partner_row(self, qtbot, vitrification_dataset):
        tab = BlendPartnersTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        tab.base_tank_combo.setCurrentText("241-A-101")
        tab.run_blend()
        frame = tab.table.dataframe()
        assert list(frame.columns[:3]) == ["PartnerTank", "TankFarm", "Blend_complement_score_proxy"]
        tab._on_mark_clicked({"tank_id": "241-A-103"})
        assert {i.row() for i in tab.table.selectedIndexes()} == {list(frame["PartnerTank"]).index("241-A-103")}

    def test_open_tank_stays_reachable_beside_the_double_click_action(self, qtbot, vitrification_dataset):
        window = _window(qtbot)
        opened = []
        window.open_tank = opened.append
        tab = ScreeningTab(window)
        qtbot.addWidget(tab)
        assert not tab.open_tank_btn.isEnabled()            # nothing to open before a run
        tab.on_dataset_changed(vitrification_dataset)
        tab.run_summary()
        view = tab._table_views["Screening summary"]
        tank = str(view.dataframe().iloc[1]["WasteSiteId"])
        tab.open_tank_btn.click()                           # no row selected yet
        assert opened == [] and "select a tank" in window.statusBar().currentMessage()
        view.setCurrentIndex(view.model().index(1, 0))
        tab.open_tank_btn.click()
        assert opened == [tank]
        # ... and the row's context menu still offers it, although the
        # double click of this table is taken by "find blend partners".
        tab.open_tank = opened.append                       # the tab is its own window in this test
        entries = [action.text() for action in view.build_menu(view.model().index(1, 0)).actions()]
        assert f"Open tank {tank}" in entries
        assert "blend partners" in view.toolTip() and "Open tank" in view.toolTip()

    def test_candidate_and_blend_tables_keep_the_free_double_click(self, qtbot):
        for cls in (CandidateSearchTab, BlendPartnersTab):
            tab = cls(_window(qtbot))
            qtbot.addWidget(tab)
            assert not tab.table._has_external_listener()   # a double click opens the tank


class TestCompletions:
    def test_list_completer_completes_the_entry_being_typed(self, qtbot):
        from PySide6.QtWidgets import QLineEdit
        edit = QLineEdit()
        qtbot.addWidget(edit)
        qvs.set_list_completions(edit, ["Sr", "Cs", "S", "Na"])
        completer = edit.completer()
        assert completer.splitPath("Cs, S") == ["S"]
        assert completer.splitPath("Cs,") == [""]
        edit.setText("Cs, S")
        model = completer.model()
        names = [model.data(model.index(row, 0)) for row in range(model.rowCount())]
        assert names == ["Cs", "Na", "S", "Sr"]
        assert completer.pathFromIndex(model.index(names.index("Sr"), 0)) == "Cs, Sr"

    def test_element_fields_offer_the_elements_of_the_dataset(self, qtbot, vitrification_dataset):
        tab = CandidateSearchTab(_window(qtbot))
        qtbot.addWidget(tab)
        assert tab.target_edit.completer() is None
        tab.on_dataset_changed(vitrification_dataset)
        for edit in (tab.target_edit, tab.penalty_edit, tab.required_edit):
            model = edit.completer().model()
            offered = {model.data(model.index(row, 0)) for row in range(model.rowCount())}
            assert offered == {"B", "Na", "Cr", "Cs", "Cl", "Si"}

    def test_base_tank_field_offers_the_tanks_and_does_not_grow_its_list(self, qtbot, vitrification_dataset):
        tab = BlendPartnersTab(_window(qtbot))
        qtbot.addWidget(tab)
        tab.on_dataset_changed(vitrification_dataset)
        model = tab.base_tank_combo.lineEdit().completer().model()
        offered = [model.data(model.index(row, 0)) for row in range(model.rowCount())]
        assert offered == vitrification_dataset.available_tanks()
        tab.base_tank_combo.setCurrentText("241-ZZ-999")
        tab.run_blend()
        assert tab.base_tank_combo.count() == len(offered)


class TestLayout:
    @pytest.mark.parametrize("cls", ALL_TABS)
    def test_chart_left_tables_right_and_no_label_can_widen_the_window(self, qtbot, cls, vitrification_dataset):
        from PySide6.QtWidgets import QLabel
        tab = cls(_window(qtbot))
        qtbot.addWidget(tab)
        assert tab.splitter.widget(0) is tab.plot
        assert tab.splitter.indexOf(tab._result_view()) == 1 or tab.splitter.widget(1) is tab.tables
        assert not tab.splitter.childrenCollapsible()
        tab.on_dataset_changed(vitrification_dataset)
        tab.run_btn.click()
        tab.advanced_section.set_expanded(True)
        widest = max(label.minimumSizeHint().width() for label in tab.findChildren(QLabel))
        assert widest < 400, f"a label of {cls.__name__} asks for {widest} px"

    def test_waste_notes_are_readable_in_place(self, qtbot):
        tab = ScreeningTab(_window(qtbot))
        qtbot.addWidget(tab)
        view = tab._table_views["Waste notes"]
        table = view.table
        assert table.wordWrap()
        default = table.verticalHeader().defaultSectionSize()
        assert all(table.rowHeight(row) > 2 * default for row in range(len(view.dataframe())))
        narrow = [table.rowHeight(row) for row in range(len(view.dataframe()))]
        view.horizontalHeader().resizeSection(1, 600)       # a wider column needs fewer lines
        assert all(table.rowHeight(row) < before for row, before in enumerate(narrow))
