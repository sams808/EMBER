"""
The global search: what the index finds for a tank fragment, an element,
an analyte or a workspace name, and how the box behaves on the keyboard.
Expected matches are stated by hand from the entries given to the index,
not derived from the matcher.
"""
import pytest
from PySide6.QtCore import Qt

import qt_search as qs
from qt_search import (
    KIND_ANALYTE, KIND_ELEMENT, KIND_TANK, KIND_WORKSPACE, GlobalSearch, SearchIndex,
)

TANKS = ["241-A-101", "241-AN-104", "241-AW-106", "241-B-107", "241-BX-107", "241-TX-106"]
ELEMENTS = ["Cs", "Sr", "S", "Se", "I", "Na", "C"]
ANALYTES = ["137Cs", "134Cs", "Cs", "90Sr", "SO4", "NO3", "129I", "239/240Pu", "113mCd", "Acetone", "Na"]
WORKSPACES = {"Overview": "", "Heatmaps": "", "Tank Explorer": "", "Element Explorer": ""}


@pytest.fixture
def index():
    idx = SearchIndex(WORKSPACES)
    idx.set_entries(TANKS, ELEMENTS, ANALYTES, {"241-AW-106": "AW", "241-B-107": "B"})
    return idx


def _keys(hits, kind=None):
    return [h.key for h in hits if kind is None or h.kind == kind]


class TestTankFragments:
    @pytest.mark.parametrize("text", ["AW-106", "aw106", "AW 106", "241-AW-106", "241aw106"])
    def test_a_tank_is_found_whatever_the_punctuation(self, index, text):
        hits = index.search(text)
        assert hits and hits[0].kind == KIND_TANK and hits[0].key == "241-AW-106"

    def test_b107_is_tank_b_107_not_bx_107(self, index):
        tanks = _keys(index.search("B107"), KIND_TANK)
        assert tanks == ["241-B-107"]

    def test_a_number_finds_every_tank_with_it(self, index):
        assert _keys(index.search("106"), KIND_TANK) == ["241-AW-106", "241-TX-106"]

    def test_a_farm_prefix_lists_its_tanks_first(self, index):
        tanks = _keys(index.search("AN"), KIND_TANK)
        assert tanks[0] == "241-AN-104"

    def test_the_detail_names_the_farm_when_known(self, index):
        hit = index.search("AW-106")[0]
        assert hit.detail == "Tank · farm AW"
        assert index.search("TX-106")[0].detail == "Tank"


class TestElementsAndAnalytes:
    def test_an_element_symbol_comes_first_then_its_isotopes(self, index):
        hits = index.search("Cs")
        assert (hits[0].kind, hits[0].key) == (KIND_ELEMENT, "Cs")
        assert _keys(hits, KIND_ANALYTE) == ["134Cs", "137Cs"]   # the analyte "Cs" is the element row

    def test_symbol_matching_ignores_case(self, index):
        assert (index.search("cs")[0].kind, index.search("cs")[0].key) == (KIND_ELEMENT, "Cs")

    def test_an_isotope_is_found_exactly(self, index):
        hits = index.search("137Cs")
        assert (hits[0].kind, hits[0].key) == (KIND_ANALYTE, "137Cs")
        assert "134Cs" not in _keys(hits)

    def test_a_species_is_found_exactly(self, index):
        hits = index.search("so4")
        assert (hits[0].kind, hits[0].key) == (KIND_ANALYTE, "SO4")

    def test_one_letter_offers_the_symbol_and_the_symbols_it_begins(self, index):
        elements = _keys(index.search("S"), KIND_ELEMENT)
        assert elements[0] == "S"
        assert set(elements) == {"S", "Se", "Sr"}

    def test_an_element_name_the_normaliser_knows_resolves_to_its_symbol(self, index):
        hits = index.search("iodine")
        assert (hits[0].kind, hits[0].key) == (KIND_ELEMENT, "I")

    def test_combined_and_metastable_isotopes_count_as_isotopes_of_the_symbol(self, index):
        assert "239/240Pu" in _keys(index.search("Pu"), KIND_ANALYTE)
        assert "113mCd" in _keys(index.search("Cd"), KIND_ANALYTE)

    def test_a_substring_finds_an_organic_by_name(self, index):
        assert _keys(index.search("ceto"), KIND_ANALYTE) == ["Acetone"]


class TestWorkspaces:
    def test_a_workspace_is_found_by_the_start_of_any_word(self, index):
        assert _keys(index.search("heat"), KIND_WORKSPACE) == ["Heatmaps"]
        assert set(_keys(index.search("explorer"), KIND_WORKSPACE)) == {"Tank Explorer", "Element Explorer"}

    def test_an_exact_name_ranks_above_everything_but_an_element(self, index):
        hits = index.search("Overview")
        assert (hits[0].kind, hits[0].key) == (KIND_WORKSPACE, "Overview")


class TestIndexEdges:
    def test_blank_text_finds_nothing(self, index):
        assert index.search("") == []
        assert index.search("   ") == []

    def test_no_match_is_an_empty_list(self, index):
        assert index.search("zzzz") == []

    def test_limit_is_honoured(self, index):
        assert len(index.search("1", limit=3)) == 3
        assert index.search("Cs", limit=0) == []

    def test_set_dataset_reads_tanks_elements_and_analytes(self, sample_dataset):
        idx = SearchIndex()
        idx.set_dataset(sample_dataset)
        counts = idx.counts()
        assert counts[KIND_TANK] == 2          # 241-A-101 and 241-AN-104 in the fixture
        hits = idx.search("AN-104")
        assert (hits[0].kind, hits[0].key, hits[0].detail) == (KIND_TANK, "241-AN-104", "Tank · farm AN")
        assert (idx.search("137Cs")[0].kind, idx.search("137Cs")[0].key) == (KIND_ANALYTE, "137Cs")
        assert (idx.search("Fe")[0].kind, idx.search("Fe")[0].key) == (KIND_ELEMENT, "Fe")

    def test_an_unloaded_dataset_empties_the_data_part(self, index):
        from data_model import HanfordDataset
        index.set_dataset(HanfordDataset())
        assert index.counts()[KIND_TANK] == 0
        assert _keys(index.search("heat")) == ["Heatmaps"]   # workspaces stay


class TestSearchBox:
    @pytest.fixture
    def box(self, qtbot, index):
        from PySide6.QtWidgets import QVBoxLayout, QWidget
        host = QWidget()
        host.resize(900, 600)
        layout = QVBoxLayout(host)
        search = GlobalSearch(index)
        layout.addWidget(search)
        layout.addStretch(1)
        qtbot.addWidget(host)
        host.setAttribute(Qt.WA_DontShowOnScreen, True)
        host.show()
        yield search          # `host` stays referenced (and alive) until the test ends
        host.hide()

    def test_typing_shows_the_matches(self, qtbot, box):
        qtbot.keyClicks(box, "AW-106")
        assert box.results_visible()
        assert box.popup.current_hit().key == "241-AW-106"
        assert box.popup.parent() is box.window()        # a child of the window, not a floating one

    def test_enter_opens_the_best_match_and_clears_the_box(self, qtbot, box):
        opened = []
        box.activated.connect(opened.append)
        qtbot.keyClicks(box, "Cs")
        qtbot.keyClick(box, Qt.Key_Return)
        assert [(h.kind, h.key) for h in opened] == [(KIND_ELEMENT, "Cs")]
        assert box.text() == ""
        assert not box.results_visible()

    def test_down_moves_to_the_next_match_and_wraps(self, qtbot, box):
        opened = []
        box.activated.connect(opened.append)
        qtbot.keyClicks(box, "Cs")           # Cs, 134Cs, 137Cs
        qtbot.keyClick(box, Qt.Key_Down)
        assert box.popup.current_hit().key == "134Cs"
        qtbot.keyClick(box, Qt.Key_Down)
        qtbot.keyClick(box, Qt.Key_Down)
        assert box.popup.current_hit().key == "Cs"
        qtbot.keyClick(box, Qt.Key_Up)
        assert box.popup.current_hit().key == "137Cs"
        qtbot.keyClick(box, Qt.Key_Return)
        assert opened[0].key == "137Cs"

    def test_escape_closes_the_list_first_then_clears(self, qtbot, box):
        qtbot.keyClicks(box, "Cs")
        qtbot.keyClick(box, Qt.Key_Escape)
        assert not box.results_visible()
        assert box.text() == "Cs"
        qtbot.keyClick(box, Qt.Key_Escape)
        assert box.text() == ""

    def test_no_match_says_so_and_enter_opens_nothing(self, qtbot, box):
        opened = []
        box.activated.connect(opened.append)
        qtbot.keyClicks(box, "zzzz")
        assert box.results_visible()
        assert box.popup.list.count() == 1
        assert "zzzz" in box.popup.list.item(0).text()
        assert box.popup.current_hit() is None
        qtbot.keyClick(box, Qt.Key_Return)
        assert opened == []

    def test_clicking_a_row_opens_it(self, qtbot, box):
        opened = []
        box.activated.connect(opened.append)
        qtbot.keyClicks(box, "106")
        row = box.popup.list.item(1)
        box.popup.list.itemClicked.emit(row)
        assert [h.key for h in opened] == ["241-TX-106"]

    def test_emptying_the_box_closes_the_list(self, qtbot, box):
        qtbot.keyClicks(box, "C")
        assert box.results_visible()
        qtbot.keyClick(box, Qt.Key_Backspace)
        assert not box.results_visible()

    def test_the_list_never_takes_the_keyboard_focus(self, qtbot, box):
        qtbot.keyClicks(box, "Cs")
        assert box.popup.focusPolicy() == Qt.NoFocus
        assert box.popup.list.focusPolicy() == Qt.NoFocus

    def test_kind_labels_cover_every_kind(self):
        for kind in (KIND_TANK, KIND_ELEMENT, KIND_ANALYTE, KIND_WORKSPACE):
            assert qs.KIND_LABELS[kind]
            assert qs.KIND_ICONS[kind]
