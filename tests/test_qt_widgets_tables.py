"""DataFrameTableModel / DataFrameTableView: what a table shows a reader
(labels, formatted numbers, no stray index), and what it must keep for
code and exports (the DataFrame as given, full precision)."""
import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from qt_widgets import NAME_ROLE, SORT_ROLE, DataFrameTableModel, DataFrameTableView


def _df():
    return pd.DataFrame({
        "WasteSiteId": ["241-A-101", "241-AW-106", "241-BX-110", "241-C-104"],
        "TankFarm": ["A", "AW", "BX", "C"],
        "Total_kg": [1234567.891, 0.00012345, 25.5, float("nan")],
        "N_tanks": [10, 2, 300, 1500],
    })


class _Host(QMainWindow):
    def __init__(self):
        super().__init__()
        self.opened = []

    def open_tank(self, tank_id):
        self.opened.append(tank_id)


@pytest.fixture
def host(qtbot):
    """Kept alive by the fixture: pytest-qt only holds a weak reference."""
    window = _Host()
    qtbot.addWidget(window)
    yield window


def _view(qtbot, df=None, host=None, **kwargs):
    view = DataFrameTableView(title="Result", **kwargs)
    if host is not None:
        host.setCentralWidget(view)
    else:
        qtbot.addWidget(view)
    if df is not None:
        view.set_dataframe(df)
    return view


def _shown_column(view, name):
    """The DISPLAYED text of one column, top to bottom, as the view shows it."""
    proxy = view.proxy_model()
    col = list(view.dataframe().columns).index(name)
    return [proxy.data(proxy.index(r, col), Qt.DisplayRole) for r in range(proxy.rowCount())]


class TestModelDisplay:
    def test_headers_are_labels_with_the_column_name_one_hover_away(self, qtbot):
        model = DataFrameTableModel(_df())
        assert [model.headerData(i, Qt.Horizontal) for i in range(4)] == ["Tank", "Farm", "Total (kg)", "Tanks"]
        assert [model.headerData(i, Qt.Horizontal, NAME_ROLE) for i in range(4)] == list(_df().columns)
        assert "WasteSiteId" in model.headerData(0, Qt.Horizontal, Qt.ToolTipRole)
        assert "Total_kg" in model.headerData(2, Qt.Horizontal, Qt.ToolTipRole)

    def test_the_dataframe_itself_is_untouched(self, qtbot):
        df = _df()
        model = DataFrameTableModel(df)
        assert model.dataframe() is df
        assert list(model.dataframe().columns) == ["WasteSiteId", "TankFarm", "Total_kg", "N_tanks"]
        assert model.dataframe()["Total_kg"].iloc[0] == 1234567.891

    def test_numbers_are_formatted_and_missing_is_blank(self, qtbot):
        model = DataFrameTableModel(_df())
        shown = [model.data(model.index(r, 2), Qt.DisplayRole) for r in range(4)]
        assert shown[0] == "1,234,568"
        assert shown[1] in ("1.234e-4", "1.235e-4")
        assert shown[2] == "25.5"
        assert shown[3] == ""
        assert [model.data(model.index(r, 3), Qt.DisplayRole) for r in range(4)] == ["10", "2", "300", "1,500"]

    def test_tooltip_and_sort_role_keep_full_precision(self, qtbot):
        model = DataFrameTableModel(_df())
        assert model.data(model.index(0, 2), Qt.ToolTipRole) == "1234567.891"
        assert model.data(model.index(0, 2), SORT_ROLE) == 1234567.891
        assert model.data(model.index(3, 3), SORT_ROLE) == 1500
        assert isinstance(model.data(model.index(3, 3), SORT_ROLE), int)

    def test_numbers_right_aligned_text_left_aligned(self, qtbot):
        model = DataFrameTableModel(_df())
        text = model.data(model.index(0, 0), Qt.TextAlignmentRole)
        number = model.data(model.index(0, 2), Qt.TextAlignmentRole)
        assert int(text) & int(Qt.AlignLeft) and not int(text) & int(Qt.AlignRight)
        assert int(number) & int(Qt.AlignRight)
        assert int(model.headerData(2, Qt.Horizontal, Qt.TextAlignmentRole)) & int(Qt.AlignRight)

    def test_booleans_are_not_numbers(self, qtbot):
        model = DataFrameTableModel(pd.DataFrame({"IsDST": [True, False]}))
        assert [model.data(model.index(r, 0), Qt.DisplayRole) for r in range(2)] == ["Yes", "No"]
        assert int(model.data(model.index(0, 0), Qt.TextAlignmentRole)) & int(Qt.AlignLeft)
        assert model.headerData(0, Qt.Horizontal) == "Double-shell"


class TestSorting:
    def test_sorting_is_by_value_not_by_the_displayed_text(self, qtbot):
        view = _view(qtbot, _df())
        view.sortByColumn(3, Qt.AscendingOrder)
        # As text "1,500" < "10" < "2" < "300"; by value 2 < 10 < 300 < 1500.
        assert view.dataframe()["N_tanks"].tolist() == [2, 10, 300, 1500]
        assert _shown_column(view, "N_tanks") == ["2", "10", "300", "1,500"]
        view.sortByColumn(2, Qt.DescendingOrder)
        assert view.dataframe()["WasteSiteId"].tolist()[:3] == ["241-A-101", "241-BX-110", "241-AW-106"]

    def test_mixed_type_column_sorts_as_text_instead_of_raising(self, qtbot):
        model = DataFrameTableModel(pd.DataFrame({"x": [3, "b", 1.5, "a"]}))
        model.sort(0, Qt.AscendingOrder)
        assert model.dataframe()["x"].tolist() == [1.5, 3, "a", "b"]

    def test_new_data_shows_no_stale_sort_arrow(self, qtbot):
        view = _view(qtbot, _df())
        assert view.horizontalHeader().sortIndicatorSection() == -1
        view.sortByColumn(3, Qt.AscendingOrder)
        assert view.horizontalHeader().sortIndicatorSection() == 3
        view.set_dataframe(_df())
        assert view.horizontalHeader().sortIndicatorSection() == -1
        assert view.dataframe()["N_tanks"].tolist() == [10, 2, 300, 1500]      # the page's own order


class TestIndexColumn:
    def test_unnamed_index_is_hidden(self, qtbot):
        view = _view(qtbot, _df())
        assert view.verticalHeader().isHidden() is True

    def test_named_index_is_shown(self, qtbot):
        df = _df().set_index("WasteSiteId")
        view = _view(qtbot, df)
        assert view.verticalHeader().isHidden() is False
        assert view.model().headerData(1, Qt.Vertical) == "241-AW-106"

    def test_switching_back_to_an_unnamed_index_hides_it_again(self, qtbot):
        view = _view(qtbot, _df().set_index("WasteSiteId"))
        view.set_dataframe(_df())
        assert view.verticalHeader().isHidden() is True


class TestFilter:
    def test_filter_box_identity(self, qtbot):
        view = _view(qtbot, _df())
        assert view.filter_edit.objectName() == "SearchBox"
        assert view.filter_edit.placeholderText() == "Filter rows"
        assert view.count_label.text() == "4 rows"

    def test_filter_matches_any_column_ignoring_case(self, qtbot):
        view = _view(qtbot, _df())
        view.set_filter_text("aw")
        assert _shown_column(view, "WasteSiteId") == ["241-AW-106"]
        assert view.count_label.text() == "1 of 4 rows"
        view.set_filter_text("241-")
        assert view.proxy_model().rowCount() == 4

    def test_every_word_must_match(self, qtbot):
        view = _view(qtbot, _df())
        view.set_filter_text("241 bx")
        assert _shown_column(view, "WasteSiteId") == ["241-BX-110"]

    def test_numbers_match_as_data_and_as_displayed(self, qtbot):
        view = _view(qtbot, _df())
        view.set_filter_text("1500")
        assert _shown_column(view, "WasteSiteId") == ["241-C-104"]
        view.set_filter_text("1,500")
        assert _shown_column(view, "WasteSiteId") == ["241-C-104"]

    def test_dataframe_keeps_every_row_and_visible_dataframe_the_filtered_ones(self, qtbot):
        view = _view(qtbot, _df())
        view.set_filter_text("bx")
        assert len(view.dataframe()) == 4
        assert view.visible_dataframe()["WasteSiteId"].tolist() == ["241-BX-110"]

    def test_clearing_the_filter_restores_every_row(self, qtbot):
        view = _view(qtbot, _df())
        view.set_filter_text("bx")
        view.set_filter_text("")
        assert view.proxy_model().rowCount() == 4 and view.count_label.text() == "4 rows"

    def test_filter_survives_a_sort_and_a_new_result(self, qtbot):
        view = _view(qtbot, _df())
        view.set_filter_text("241-a")
        view.sortByColumn(3, Qt.AscendingOrder)
        assert _shown_column(view, "WasteSiteId") == ["241-AW-106", "241-A-101"]
        view.set_dataframe(_df().iloc[::-1].reset_index(drop=True))
        assert _shown_column(view, "WasteSiteId") == ["241-AW-106", "241-A-101"]
        assert view.count_label.text() == "2 of 4 rows"

    def test_typing_filters_after_the_debounce(self, qtbot):
        view = _view(qtbot, _df())
        view.filter_edit.setText("bx")
        qtbot.waitUntil(lambda: view.proxy_model().rowCount() == 1, timeout=3000)

    def test_ctrl_f_in_the_table_goes_to_the_filter(self, qtbot, monkeypatch):
        view = _view(qtbot, _df())
        asked = []
        monkeypatch.setattr(view, "focus_filter", lambda: asked.append(True))
        qtbot.keyClick(view.table, Qt.Key_F, Qt.ControlModifier)
        assert asked == [True]
        qtbot.keyClick(view.table, Qt.Key_F)                 # a plain "f" is not the shortcut
        assert asked == [True]

    def test_escape_in_the_filter_clears_it(self, qtbot):
        view = _view(qtbot, _df())
        view.set_filter_text("bx")
        qtbot.keyClick(view.filter_edit, Qt.Key_Escape)
        assert view.filter_edit.text() == "" and view.proxy_model().rowCount() == 4

    def test_display_cap_is_stated(self, qtbot):
        df = pd.DataFrame({"x": range(50)})
        view = _view(qtbot, df, max_rows_display=10)
        assert len(view.dataframe()) == 10
        assert "first 10 of 50" in view.count_label.text()


class TestIndexesAreModelIndexes:
    """Page code reads `dataframe().iloc[index.row()]`: an index from this
    widget must mean a row of dataframe(), filter or no filter."""

    def test_current_index_maps_through_the_filter(self, qtbot):
        view = _view(qtbot, _df())
        view.set_filter_text("bx")
        view.table.setCurrentIndex(view.proxy_model().index(0, 0))      # the only visible row
        index = view.currentIndex()
        assert index.model() is view.model()
        assert view.dataframe().iloc[index.row()]["WasteSiteId"] == "241-BX-110"

    def test_set_current_index_accepts_a_model_index(self, qtbot):
        view = _view(qtbot, _df())
        view.set_filter_text("241-")
        view.setCurrentIndex(view.model().index(2, 0))
        assert view.currentIndex().row() == 2

    def test_double_click_signal_carries_a_model_index(self, qtbot):
        view = _view(qtbot, _df())
        view.set_filter_text("c-104")
        got = []
        view.doubleClicked.connect(got.append)
        view.table.doubleClicked.emit(view.proxy_model().index(0, 1))
        assert [(i.row(), i.column()) for i in got] == [(3, 1)]

    def test_selection_model_is_stable_across_new_data(self, qtbot):
        view = _view(qtbot, _df())
        selection = view.selectionModel()
        view.set_dataframe(_df())
        assert view.selectionModel() is selection

    def test_unknown_attributes_reach_the_inner_view(self, qtbot):
        view = _view(qtbot, _df())
        assert view.alternatingRowColors() is True                      # a QTableView method
        with pytest.raises(AttributeError):
            view.no_such_thing

    def test_lives_in_a_tab_widget_like_any_widget(self, qtbot):
        tabs = QTabWidget()
        qtbot.addWidget(tabs)
        view = DataFrameTableView(title="Result")
        tabs.addTab(view, "Result")
        tabs.setCurrentWidget(view)
        assert tabs.currentWidget() is view


class TestOpenTank:
    def test_double_click_on_a_tank_row_opens_the_tank(self, qtbot, host):
        view = _view(qtbot, _df(), host=host)
        view.table.doubleClicked.emit(view.proxy_model().index(1, 2))   # any cell of the row
        assert host.opened == ["241-AW-106"]

    def test_filtered_view_opens_the_row_that_was_clicked(self, qtbot, host):
        view = _view(qtbot, _df(), host=host)
        view.set_filter_text("bx")
        view.table.doubleClicked.emit(view.proxy_model().index(0, 0))
        assert host.opened == ["241-BX-110"]

    def test_a_table_with_no_tank_column_opens_nothing(self, qtbot, host):
        view = _view(qtbot, pd.DataFrame({"Element": ["Na"], "Total_kg": [1.0]}), host=host)
        view.table.doubleClicked.emit(view.proxy_model().index(0, 0))
        assert host.opened == []

    def test_a_page_that_handles_the_double_click_keeps_its_own_behaviour(self, qtbot, host):
        view = _view(qtbot, _df(), host=host)
        handled = []
        view.doubleClicked.connect(lambda index: handled.append(index.row()))
        view.table.doubleClicked.emit(view.proxy_model().index(1, 0))
        assert handled == [1] and host.opened == []

    def test_partner_tank_column_is_understood(self, qtbot, host):
        df = pd.DataFrame({"BaseTank": ["241-A-101"], "PartnerTank": ["241-C-104"], "Score": [1.0]})
        view = _view(qtbot, df, host=host)
        view.table.doubleClicked.emit(view.proxy_model().index(0, 2))
        assert host.opened == ["241-C-104"]                             # the partner, unless the base is clicked
        view.table.doubleClicked.emit(view.proxy_model().index(0, 0))
        assert host.opened == ["241-C-104", "241-A-101"]

    def test_no_shell_no_error(self, qtbot):
        view = _view(qtbot, _df())
        activated = []
        view.tankActivated.connect(activated.append)
        view.table.doubleClicked.emit(view.proxy_model().index(0, 0))
        assert activated == ["241-A-101"]


class TestContextMenu:
    def test_menu_entries(self, qtbot, host):
        view = _view(qtbot, _df(), host=host)
        menu = view.build_menu(view.model().index(1, 2))
        assert [a.text() for a in menu.actions() if a.text()] == [
            "Copy cell", "Copy selection", "Export CSV…", "Open tank 241-AW-106"]

    def test_no_open_tank_without_a_shell(self, qtbot):
        view = _view(qtbot, _df())
        menu = view.build_menu(view.model().index(1, 2))
        assert [a.text() for a in menu.actions() if a.text()] == ["Copy cell", "Copy selection", "Export CSV…"]

    def test_filtered_table_offers_both_exports(self, qtbot):
        view = _view(qtbot, _df())
        view.set_filter_text("bx")
        texts = [a.text() for a in view.build_menu(QModelIndex()).actions() if a.text()]
        assert texts == ["Copy selection", "Export filtered rows as CSV…", "Export all rows as CSV…"]

    def test_copy_cell_is_full_precision(self, qtbot):
        view = _view(qtbot, _df())
        menu = view.build_menu(view.model().index(0, 2))
        next(a for a in menu.actions() if a.text() == "Copy cell").trigger()
        assert QApplication.clipboard().text() == "1234567.891"

    def test_copy_selection_uses_column_names_and_raw_values(self, qtbot):
        view = _view(qtbot, _df())
        view.table.selectRow(0)
        view._copy_selection()
        lines = QApplication.clipboard().text().strip().splitlines()
        assert lines[0].split("\t") == ["WasteSiteId", "TankFarm", "Total_kg", "N_tanks"]
        assert lines[1].split("\t") == ["241-A-101", "A", "1234567.891", "10"]

    def test_csv_export_is_full_precision_with_original_names(self, qtbot, tmp_path, monkeypatch):
        from PySide6.QtWidgets import QFileDialog
        target = tmp_path / "out.csv"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), "CSV")))
        view = _view(qtbot, _df())
        view.set_filter_text("bx")
        view._export_csv()                                              # all rows, filter or not
        back = pd.read_csv(target)
        assert list(back.columns) == ["WasteSiteId", "TankFarm", "Total_kg", "N_tanks"]
        assert back["Total_kg"].iloc[0] == 1234567.891 and len(back) == 4
        view._export_csv(filtered=True)
        assert pd.read_csv(target)["WasteSiteId"].tolist() == ["241-BX-110"]


class TestEmptyState:
    def test_empty_table_says_so(self, qtbot):
        view = _view(qtbot)
        assert view.empty_state.isHidden() is False
        assert view.empty_state.title_label.text() == "Nothing to show yet"
        assert view.count_label.text() == ""

    def test_rows_remove_the_sentence(self, qtbot):
        view = _view(qtbot, _df())
        assert view.empty_state.isHidden() is True

    def test_a_filter_with_no_match_explains_itself(self, qtbot):
        view = _view(qtbot, _df())
        view.set_filter_text("zzz")
        assert view.empty_state.isHidden() is False
        assert view.empty_state.title_label.text() == "No rows match the filter"
        assert view.count_label.text() == "0 of 4 rows"

    def test_custom_empty_text(self, qtbot):
        view = _view(qtbot)
        view.set_empty_text("No tanks selected", "Pick one or more tanks, then Run.")
        assert view.empty_state.title_label.text() == "No tanks selected"

    def test_empty_frame_with_columns_keeps_its_headers(self, qtbot):
        view = _view(qtbot, pd.DataFrame({"WasteSiteId": [], "Total_kg": []}))
        assert view.model().columnCount() == 2 and view.empty_state.isHidden() is False


class TestColumnWidths:
    def test_columns_fit_content_up_to_a_cap(self, qtbot):
        df = pd.DataFrame({"short": ["a", "b"], "long": ["x" * 400, "y"], "tail": [1, 2]})
        view = _view(qtbot, df)
        header = view.horizontalHeader()
        assert header.sectionSize(0) < header.sectionSize(1) <= view.MAX_COLUMN_WIDTH
        assert header.stretchLastSection() is True           # the last column takes what is left

    def test_width_pass_reads_a_sample_not_every_row(self, qtbot, monkeypatch):
        import qt_widgets

        n = 46597
        df = pd.DataFrame({"WasteSiteId": ["241-A-101"] * n, "Inventory": np.arange(n, dtype=float)})
        view = _view(qtbot)
        calls = []
        real = qt_widgets.format_cell
        monkeypatch.setattr(qt_widgets, "format_cell", lambda value, style=None: calls.append(1) or real(value, style))
        view.set_dataframe(df)
        assert len(calls) <= 2 * view.SAMPLE_ROWS                       # two columns, a sample of rows each
        assert view.count_label.text() == "46,597 rows"


class TestColumnNumberStyle:
    def _shown(self, model):
        return [model.data(model.index(r, 0), Qt.DisplayRole) for r in range(model.rowCount())]

    def test_a_column_shares_one_number_of_decimals(self, qtbot):
        model = DataFrameTableModel(pd.DataFrame({"PC1": [-2.09, 0.06639, 1.366, 3.483]}))
        assert self._shown(model) == ["-2.090", "0.066", "1.366", "3.483"]

    def test_a_column_reaching_a_billion_is_scientific_throughout(self, qtbot):
        model = DataFrameTableModel(pd.DataFrame({"Context_total": [920821584.0, 2.82e9, float("nan")]}))
        assert self._shown(model) == ["9.208e8", "2.82e9", ""]

    def test_the_style_follows_the_data_not_the_first_table(self, qtbot):
        model = DataFrameTableModel(pd.DataFrame({"x": [1.5, 2.5]}))
        assert self._shown(model) == ["1.500", "2.500"]
        model.set_dataframe(pd.DataFrame({"x": [1500.0, 2500.0]}))
        assert self._shown(model) == ["1,500", "2,500"]

    def test_integers_are_never_given_decimals(self, qtbot):
        model = DataFrameTableModel(pd.DataFrame({"N_tanks": [1, 177]}))
        assert self._shown(model) == ["1", "177"]
