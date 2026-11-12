"""Shared presentational components: behaviour that pages and exports rely on."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QSizePolicy, QVBoxLayout, QWidget

import qt_components as qc


class TestWrappedLabel:
    def test_long_sentence_cannot_widen_its_window(self, qtbot):
        host = QWidget()
        qtbot.addWidget(host)
        layout = QVBoxLayout(host)
        sentence = "Not shown: " + ", ".join(f"241-AN-{i}" for i in range(101, 181)) + "."
        label = qc.wrapped_label(sentence)
        layout.addWidget(label)
        assert label.wordWrap()
        assert label.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored
        assert host.minimumSizeHint().width() < 200

    def test_object_name_is_the_style_hook(self, qtbot):
        label = qc.wrapped_label("text", "PageSubtitle")
        qtbot.addWidget(label)
        assert label.objectName() == "PageSubtitle"


class TestPageHeader:
    def test_title_subtitle_and_actions(self, qtbot):
        header = qc.PageHeader("Site overview", "All tanks")
        qtbot.addWidget(header)
        assert header.title_label.text() == "Site overview"
        assert header.title_label.objectName() == "PageTitle"
        assert not header.subtitle_label.isHidden()
        button = header.add_action_widget(QPushButton("Export..."))
        assert button.parent() is header

    def test_empty_subtitle_takes_no_room(self, qtbot):
        header = qc.PageHeader("Query")
        qtbot.addWidget(header)
        assert header.subtitle_label.isHidden()
        header.set_subtitle("Scope: B farm")
        assert not header.subtitle_label.isHidden()
        header.set_subtitle("")
        assert header.subtitle_label.isHidden()


class TestStatTile:
    def test_value_and_hint(self, qtbot):
        tile = qc.StatTile("Tanks")
        qtbot.addWidget(tile)
        assert tile.value() == "—"
        assert tile.hint_label.isHidden()
        tile.set_value("177", "149 single-shell, 28 double-shell")
        assert tile.value() == "177"
        assert tile.hint_label.text() == "149 single-shell, 28 double-shell"
        assert not tile.hint_label.isHidden()

    def test_value_alone_keeps_the_hint(self, qtbot):
        tile = qc.StatTile("Waste volume", "200 ML", "53.0 million gallons")
        qtbot.addWidget(tile)
        tile.set_value("201 ML")
        assert tile.hint_label.text() == "53.0 million gallons"

    def test_long_label_and_hint_cannot_widen_the_tile(self, qtbot):
        long_hint = "22.0 wt% Na2O, limited by the alkali + sulfur rule in every scenario considered"
        tile = qc.StatTile("Cesium-137 + strontium-90, decayed to the dataset date", "248.4 kt", long_hint)
        qtbot.addWidget(tile)
        short = qc.StatTile("Glass", "248.4 kt", "blended")
        qtbot.addWidget(short)
        assert tile.minimumSizeHint().width() == short.minimumSizeHint().width()
        assert tile.hint_label.text() == long_hint
        tile.resize(160, tile.sizeHint().height())
        tile.show()
        qtbot.waitExposed(tile)
        assert tile.hint_label.is_elided()
        tile.grab()
        assert tile.hint_label.toolTip() == long_hint

    def test_click_is_reported(self, qtbot):
        tile = qc.StatTile("Records", "46,597")
        qtbot.addWidget(tile)
        tile.show()
        with qtbot.waitSignal(tile.clicked, timeout=1000):
            qtbot.mouseClick(tile, Qt.LeftButton)


class TestEmptyState:
    def test_text_can_be_replaced(self, qtbot):
        empty = qc.EmptyState("No result yet", "Choose a component and press Run.")
        qtbot.addWidget(empty)
        assert empty.title_label.text() == "No result yet"
        assert not empty.hint_label.isHidden()
        empty.set_text("Nothing matches the scope")
        assert empty.title_label.text() == "Nothing matches the scope"
        assert empty.hint_label.isHidden()


class TestNotesBanner:
    def test_hidden_without_notes(self, qtbot):
        banner = qc.NotesBanner()
        qtbot.addWidget(banner)
        assert banner.isHidden()
        banner.set_notes(["", "   "])
        assert banner.isHidden()
        assert banner.text() == ""

    def test_one_note_needs_no_toggle(self, qtbot):
        banner = qc.NotesBanner()
        qtbot.addWidget(banner)
        banner.set_notes(["Two layers carry more than one volume record."])
        assert not banner.isHidden()
        assert banner.summary_label.text() == "Two layers carry more than one volume record."
        assert banner.toggle.isHidden()

    def test_several_notes_collapse_to_one_line_with_the_rest_one_click_away(self, qtbot):
        banner = qc.NotesBanner()
        qtbot.addWidget(banner)
        notes = ["First note.", "Second note.", "Third note."]
        banner.set_notes(notes)
        assert banner.summary_label.text().startswith("First note.")
        assert "+2 more" in banner.summary_label.text()
        assert banner.detail_label.isHidden()
        banner.toggle.setChecked(True)
        assert not banner.detail_label.isHidden()
        for note in notes:
            assert note in banner.detail_label.text()
        assert banner.toggle.text() == "Hide"
        assert list(banner.notes()) == notes
        assert banner.text() == "\n".join(notes)
        assert banner.toolTip() == "\n\n".join(notes)

    def test_headline_replaces_the_first_note(self, qtbot):
        banner = qc.NotesBanner()
        qtbot.addWidget(banner)
        banner.set_notes(["A long engine disclosure."], headline="1 dataset note applies to this result.")
        assert banner.summary_label.text() == "1 dataset note applies to this result."
        assert not banner.toggle.isHidden()

    def test_result_notes_show_one_in_full_and_announce_several_by_count(self, qtbot):
        banner = qc.NotesBanner()
        qtbot.addWidget(banner)
        banner.set_result_notes(["Only one."])
        assert banner.summary_label.text() == "Only one."
        assert banner.toggle.isHidden()
        banner.set_result_notes(["a", " ", "b", "c"], subject="the figures on this page")
        assert banner.summary_label.text().startswith("3 notes qualify the figures on this page")
        assert not banner.toggle.isHidden()
        assert list(banner.notes()) == ["a", "b", "c"]
        banner.set_result_notes([])
        assert banner.isHidden()

    def test_new_notes_collapse_an_open_banner(self, qtbot):
        banner = qc.NotesBanner()
        qtbot.addWidget(banner)
        banner.set_notes(["a", "b"])
        banner.toggle.setChecked(True)
        banner.set_notes(["c", "d"])
        assert not banner.toggle.isChecked()
        assert banner.detail_label.isHidden()


class TestNoStrayWindows:
    """Showing a label before it has a parent makes it a top-level window
    for an instant: 0.1 s each, and a flash on screen. A component that was
    never shown must not have created a native window for any of its parts."""

    def test_components_build_without_creating_windows(self, qtbot):
        host = QWidget()                      # a page: the banner shows itself inside it
        qtbot.addWidget(host)
        layout = QVBoxLayout(host)
        banner = qc.NotesBanner()
        layout.addWidget(banner)
        banner.set_notes(["a", "b"])
        widgets = [
            qc.PageHeader("Query", "One sentence."),
            qc.StatTile("Tanks", "177", "149 single-shell"),
            qc.EmptyState("No result yet", "Press Run."),
            host,
        ]
        for widget in widgets:
            qtbot.addWidget(widget)
            for child in widget.findChildren(QWidget):
                assert child.parent() is not None
                assert not child.isWindow()
                assert not child.testAttribute(Qt.WA_WState_Created), child.objectName()


class TestPill:
    def test_is_a_named_label(self, qtbot):
        pill = qc.Pill("2026 export")
        qtbot.addWidget(pill)
        assert pill.objectName() == "Pill"
        assert pill.text() == "2026 export"
