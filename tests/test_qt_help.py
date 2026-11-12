from pathlib import Path

import qt_help as qh


class TestAssetAndNoticePaths:
    def test_asset_path_points_next_to_module_when_not_frozen(self):
        path = Path(qh.asset_path("ember_logo.png"))
        assert path.parent.name == "assets"
        assert path.parent.parent == Path(qh.__file__).resolve().parent

    def test_notice_path_points_next_to_module_when_not_frozen(self):
        path = Path(qh.notice_path())
        assert path.name == "NOTICE.md"
        assert path.parent == Path(qh.__file__).resolve().parent

    def test_asset_path_uses_meipass_when_frozen(self, monkeypatch, tmp_path):
        monkeypatch.setattr(qh.sys, "_MEIPASS", str(tmp_path), raising=False)
        assert qh.asset_path("x.png") == str(tmp_path / "assets" / "x.png")

    def test_notice_path_uses_meipass_when_frozen(self, monkeypatch, tmp_path):
        monkeypatch.setattr(qh.sys, "_MEIPASS", str(tmp_path), raising=False)
        assert qh.notice_path() == str(tmp_path / "NOTICE.md")


class TestLoadNoticeMarkdown:
    def test_loads_real_notice_file(self):
        text = qh.load_notice_markdown()
        assert "Ember" in text
        assert "PHOENIX" in text

    def test_missing_file_returns_fallback_not_crash(self, monkeypatch, tmp_path):
        monkeypatch.setattr(qh, "notice_path", lambda: str(tmp_path / "does_not_exist.md"))
        text = qh.load_notice_markdown()
        assert "could not be found" in text


class TestLoadNoticeSection:
    _FAKE_NOTICE = (
        "# Notice\n\nIntro text before any numbered section.\n\n"
        "## 1. Installation\n\nInstall steps here.\n\n"
        "## 2. Overview\n\nOverview body, line one.\nLine two.\n\n"
        "## 3. Element Explorer\n\nExplorer body.\n\n"
        "## 10. Correlations\n\nCorrelations body.\n"
    )

    def test_extracts_only_the_requested_section(self, monkeypatch):
        monkeypatch.setattr(qh, "load_notice_markdown", lambda: self._FAKE_NOTICE)
        section = qh.load_notice_section(2)
        assert section.startswith("## 2. Overview")
        assert "Overview body, line one." in section
        assert "Installation" not in section
        assert "Explorer body" not in section

    def test_last_section_runs_to_end_of_file(self, monkeypatch):
        monkeypatch.setattr(qh, "load_notice_markdown", lambda: self._FAKE_NOTICE)
        section = qh.load_notice_section(10)
        assert section.startswith("## 10. Correlations")
        assert "Correlations body." in section

    def test_double_digit_number_does_not_partial_match(self, monkeypatch):
        # "## 1. " must not accidentally match inside "## 10. Correlations".
        monkeypatch.setattr(qh, "load_notice_markdown", lambda: self._FAKE_NOTICE)
        section = qh.load_notice_section(1)
        assert "Correlations" not in section

    def test_unknown_section_number_returns_a_clear_message_not_crash(self, monkeypatch):
        monkeypatch.setattr(qh, "load_notice_markdown", lambda: self._FAKE_NOTICE)
        section = qh.load_notice_section(999)
        assert "Not found" in section

    def test_every_workspace_section_number_resolves_in_the_real_notice(self):
        # These are the exact section numbers the per-workspace "?" help
        # buttons are wired to -- if NOTICE.md gets renumbered, this test
        # (not a silent blank help dialog) should be what catches it.
        real_sections = {
            5: "Overview", 6: "Element Explorer", 7: "Tank Attributes",
            8: "Tank Explorer", 9: "Heatmaps", 10: "Correlations",
            11: "Debug", 12: "Vitrification", 13: "Figure Composer",
            19: "Query",
        }
        for number, expected_word in real_sections.items():
            section = qh.load_notice_section(number)
            assert "Not found" not in section, f"section {number} missing from NOTICE.md"
            assert expected_word in section, f"section {number} doesn't look like {expected_word!r}: {section[:80]!r}"


class TestShowWorkspaceHelp:
    def test_opens_a_dialog_scoped_to_the_section(self, qtbot, monkeypatch):
        opened = {}

        class _FakeDialog:
            def __init__(self, parent, *, markdown, title):
                opened["markdown"] = markdown
                opened["title"] = title

            def exec(self):
                opened["exec_called"] = True

        monkeypatch.setattr(qh, "HelpDialog", _FakeDialog)
        monkeypatch.setattr(qh, "load_notice_section", lambda n: f"## {n}. Fake Section\n\nBody.")
        qh.show_workspace_help(None, 6, "Element Explorer")
        assert opened["markdown"].startswith("## 6. Fake Section")
        assert opened["title"] == "Element Explorer — Help"
        assert opened["exec_called"] is True


class TestGettingStartedDialog:
    def test_one_button_per_option(self, qtbot):
        dialog = qh.GettingStartedDialog()
        qtbot.addWidget(dialog)
        from PySide6.QtWidgets import QPushButton
        labels = [b.text() for b in dialog.findChildren(QPushButton)]
        for option_label, _explanation, _target in qh.GETTING_STARTED_OPTIONS:
            assert option_label in labels

    def test_clicking_an_option_emits_its_target_and_accepts(self, qtbot):
        dialog = qh.GettingStartedDialog()
        qtbot.addWidget(dialog)
        received = []
        dialog.workspace_chosen.connect(received.append)
        first_label, _explanation, first_target = qh.GETTING_STARTED_OPTIONS[0]
        from PySide6.QtWidgets import QPushButton
        button = next(b for b in dialog.findChildren(QPushButton) if b.text() == first_label)
        with qtbot.waitSignal(dialog.accepted, timeout=1000):
            button.click()
        assert received == [first_target]

    def test_skip_rejects_without_emitting(self, qtbot):
        from PySide6.QtWidgets import QPushButton
        dialog = qh.GettingStartedDialog()
        qtbot.addWidget(dialog)
        received = []
        dialog.workspace_chosen.connect(received.append)
        skip_button = next(b for b in dialog.findChildren(QPushButton) if "Skip" in b.text())
        with qtbot.waitSignal(dialog.rejected, timeout=1000):
            skip_button.click()
        assert received == []

    def test_every_option_target_is_a_real_nav_item(self):
        # qt_shell.py can't be imported from here without risking a
        # circular import at module scope (qt_shell imports qt_help), so
        # this is checked against a plain literal list of the real
        # NAV_ITEMS values instead of importing them.
        real_nav_items = {
            "Overview", "Query", "Element Explorer", "Tank Attributes", "Tank Explorer",
            "Heatmaps", "Correlations", "Vitrification", "Figure Composer",
        }
        for _label, _explanation, target in qh.GETTING_STARTED_OPTIONS:
            assert target in real_nav_items, f"{target!r} is not a real NAV_ITEMS value"


class TestHelpDialog:
    def test_renders_html_by_default(self, qtbot):
        dialog = qh.HelpDialog()
        qtbot.addWidget(dialog)
        assert "Ember" in dialog.browser.toPlainText()

    def test_renders_markdown_when_given(self, qtbot):
        dialog = qh.HelpDialog(markdown="# Heading\n\nBody text.")
        qtbot.addWidget(dialog)
        assert "Heading" in dialog.browser.toPlainText()
        assert "Body text" in dialog.browser.toPlainText()

    def test_custom_size_applied(self, qtbot):
        dialog = qh.HelpDialog(width=920, height=760)
        qtbot.addWidget(dialog)
        assert dialog.size().width() == 920
        assert dialog.size().height() == 760

    def test_dialog_has_a_title_band_and_a_close_button(self, qtbot):
        dialog = qh.HelpDialog(title="About Ember", subtitle="Hanford tank waste explorer")
        qtbot.addWidget(dialog)
        assert dialog.band.title_label.text() == "About Ember"
        assert dialog.band.subtitle_label.text() == "Hanford tank waste explorer"
        with qtbot.waitSignal(dialog.accepted, timeout=1000):
            dialog.close_btn.click()

    def test_band_hides_an_empty_second_line(self, qtbot):
        dialog = qh.HelpDialog(markdown="# Heading", title="Query — Help")
        qtbot.addWidget(dialog)
        assert dialog.band.subtitle_label.isHidden()


class TestAboutText:
    def test_about_states_that_the_project_is_independent(self):
        assert "independent project" in qh.ABOUT_HTML
        assert "not produced by, affiliated with, or" in qh.ABOUT_HTML
        assert "endorsed by PNNL or the U.S. Department of Energy" in qh.ABOUT_HTML

    def test_about_credits_the_author(self):
        assert "Sami Soudani" in qh.ABOUT_HTML

    def test_about_carries_no_colour_of_its_own(self):
        # Colours come from the active palette (document_css), so the text
        # reads the same in the light and in the dark theme.
        import re
        assert re.search(r"#[0-9a-fA-F]{6}\b", qh.ABOUT_HTML) is None
        assert "color:" not in qh.ABOUT_HTML

    def test_document_css_takes_its_colours_from_the_palette(self):
        import qt_theme
        for palette in (qt_theme.PALETTE, qt_theme.DARK_PALETTE):
            css = qh.document_css(palette)
            assert palette["link"] in css
            assert palette["ink"] in css
            assert palette["muted"] in css

    def test_product_names(self):
        assert qh.APP_NAME == "Ember"                 # settings key: must not change
        assert qh.APP_DISPLAY_NAME == "EMBER"
        assert qh.APP_SUBTITLE == "Hanford tank waste explorer"


class TestGettingStartedWording:
    def test_no_first_person_and_no_double_dashes_in_the_copy(self, qtbot):
        from PySide6.QtWidgets import QLabel, QPushButton
        dialog = qh.GettingStartedDialog()
        qtbot.addWidget(dialog)
        texts = [w.text() for w in dialog.findChildren(QLabel)] + [w.text() for w in dialog.findChildren(QPushButton)]
        for text in texts:
            assert " -- " not in text, text
            assert "I'll" not in text and " my " not in text, text
            assert "!" not in text, text

    def test_every_option_shows_its_explanation(self, qtbot):
        from PySide6.QtWidgets import QLabel
        dialog = qh.GettingStartedDialog()
        qtbot.addWidget(dialog)
        shown = {w.text() for w in dialog.findChildren(QLabel)}
        for _label, explanation, _target in qh.GETTING_STARTED_OPTIONS:
            assert explanation in shown

    def test_clicking_anywhere_on_a_row_chooses_it(self, qtbot):
        from PySide6.QtCore import QPoint, Qt
        dialog = qh.GettingStartedDialog()
        qtbot.addWidget(dialog)
        dialog.setAttribute(Qt.WA_DontShowOnScreen, True)
        dialog.show()
        received = []
        dialog.workspace_chosen.connect(received.append)
        rows = dialog.findChildren(qh._OptionRow)
        assert len(rows) == len(qh.GETTING_STARTED_OPTIONS)
        row = rows[1]
        with qtbot.waitSignal(dialog.accepted, timeout=1000):
            qtbot.mouseClick(row, Qt.LeftButton, pos=QPoint(row.width() - 6, row.height() // 2))
        assert received == [qh.GETTING_STARTED_OPTIONS[1][2]]
