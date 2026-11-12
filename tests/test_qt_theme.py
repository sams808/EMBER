import os
import re

import pytest

import qt_theme as qt

# The palette keys the design system promises to every page.
REQUIRED_KEYS = [
    "bg", "bg_alt", "card", "ink", "muted", "faint", "border", "border_strong", "accent",
    "accent_hover", "accent_ink", "selection_bg", "chrome", "chrome_ink", "chrome_muted", "link",
    "good", "warn", "critical", "critical_bg", "note_bg", "note_ink",
]

# The object names pages set instead of writing inline styles.
STYLED_OBJECT_NAMES = [
    "Sidebar", "NavList", "Card", "SectionTitle", "SectionNote", "WarningNote", "ScopeBar",
    "ScopeTitle", "ScopeStamp", "CollapsibleHeader", "Primary", "HelpButton", "AppHeader",
    "AppTitle", "AppSubtitle", "DatasetChip", "NavGroup", "PageTitle", "PageSubtitle", "StatTile",
    "StatValue", "StatLabel", "StatHint", "EmptyTitle", "EmptyHint", "NotesBanner", "NotesSummary",
    "NotesDetail", "NotesToggle", "Pill", "SearchBox", "Secondary", "Toolbar", "FieldLabel",
    "PageHeader", "EmptyState",
]


def _luminance(hex_color: str) -> float:
    """WCAG 2 relative luminance, written out here rather than imported:
    the check must not share code with what it checks."""
    value = hex_color.lstrip("#")
    channels = []
    for i in (0, 2, 4):
        c = int(value[i:i + 2], 16) / 255.0
        channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


class TestBuildStylesheet:
    def test_light_palette_produces_css(self):
        css = qt.build_stylesheet(qt.PALETTE)
        assert qt.PALETTE["accent"] in css
        assert "QPushButton" in css
        assert "QPlainTextEdit" in css
        assert "QTableView, QTreeView, QTableWidget" in css

    def test_dark_palette_differs_from_light(self):
        light = qt.build_stylesheet(qt.PALETTE)
        dark = qt.build_stylesheet(qt.DARK_PALETTE)
        assert light != dark
        assert qt.DARK_PALETTE["accent"] in dark
        assert qt.DARK_PALETTE["accent"] not in light

    @pytest.mark.parametrize("name", STYLED_OBJECT_NAMES)
    def test_every_design_system_object_name_is_styled(self, name):
        css = qt.build_stylesheet(qt.PALETTE)
        assert re.search(rf"#{name}\b", css), f"no rule for #{name}"

    @pytest.mark.parametrize("selector", [
        "QComboBox::drop-down", "QComboBox::down-arrow", "QAbstractSpinBox::up-button",
        "QAbstractSpinBox::down-arrow", "QScrollBar::handle:vertical", "QTabBar::tab:selected",
        "QHeaderView::section", "QToolTip", "QMenu::item:selected", "QCheckBox::indicator:checked",
        "QRadioButton::indicator:checked", "QSplitter::handle", "QGroupBox::title", "QProgressBar::chunk",
        "NavigationToolbar2QT", "QPushButton:focus", "QPushButton:disabled", "QMenuBar::item",
        "QStatusBar",
    ])
    def test_widget_classes_the_brief_lists_have_rules(self, selector):
        assert selector in qt.build_stylesheet(qt.PALETTE)

    def test_tabs_are_underlined_not_boxed(self):
        css = qt.build_stylesheet(qt.PALETTE)
        tab_rule = css.split("QTabBar::tab:selected {")[1].split("}")[0]
        assert f"border-bottom: 2px solid {qt.PALETTE['accent']}" in tab_rule
        plain_rule = css.split("QTabBar::tab {")[1].split("}")[0]
        assert "border: none" in plain_rule and "background: transparent" in plain_rule

    def test_no_blanket_background_on_every_widget(self):
        # A background on the QWidget selector paints grey boxes behind
        # every label that sits on a white card.
        css = qt.build_stylesheet(qt.PALETTE)
        base_rule = css.split("QWidget {")[1].split("}")[0]
        assert "background" not in base_rule

    @pytest.mark.parametrize("palette", [qt.PALETTE, qt.DARK_PALETTE], ids=["light", "dark"])
    def test_arrow_and_check_images_exist_on_disk(self, palette):
        css = qt.build_stylesheet(palette)
        paths = re.findall(r'image: url\("([^"]+)"\)', css)
        assert len(paths) >= 8, "drop-down / step arrows and the check mark are drawn images"
        for path in paths:
            assert os.path.isfile(path), path
            assert os.path.isfile(path.replace(".png", "@2x.png")), "high-DPI variant missing"

    def test_the_stylesheet_parses_without_a_warning(self, qtbot, capfd):
        from PySide6.QtWidgets import QPushButton
        button = QPushButton("x")
        qtbot.addWidget(button)
        button.setStyleSheet(qt.build_stylesheet(qt.PALETTE))
        button.ensurePolished()
        assert "Could not parse" not in capfd.readouterr().err


class TestPalettes:
    def test_both_palettes_define_the_same_roles(self):
        assert set(qt.PALETTE) == set(qt.DARK_PALETTE)

    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_required_role_is_a_hex_colour_in_both(self, key):
        for palette in (qt.PALETTE, qt.DARK_PALETTE):
            assert re.fullmatch(r"#[0-9a-f]{6}", palette[key]), f"{key}: {palette[key]!r}"

    @pytest.mark.parametrize("palette", [qt.PALETTE, qt.DARK_PALETTE], ids=["light", "dark"])
    @pytest.mark.parametrize("fg,bg", [
        ("ink", "bg"), ("ink", "card"), ("ink", "bg_alt"), ("ink", "selection_bg"), ("ink", "hover"),
        ("muted", "bg"), ("muted", "card"), ("muted", "bg_alt"), ("muted", "hover"),
        ("accent_ink", "accent"), ("accent_ink", "accent_hover"), ("accent", "card"), ("accent", "bg"),
        ("accent", "selection_bg"),
        ("chrome_ink", "chrome"), ("chrome_muted", "chrome"), ("chrome_ink", "chrome_field"),
        ("chrome_muted", "chrome_field"), ("chrome_muted", "chrome_deep"),
        ("link", "card"), ("link", "bg"), ("link", "note_bg"),
        ("good", "card"), ("warn", "card"), ("critical", "card"), ("critical", "critical_bg"),
        ("note_ink", "note_bg"), ("tooltip_ink", "tooltip_bg"),
    ])
    def test_text_contrast_is_at_least_4_5(self, palette, fg, bg):
        assert _contrast(palette[fg], palette[bg]) >= 4.5, (fg, bg, palette[fg], palette[bg])

    @pytest.mark.parametrize("palette", [qt.PALETTE, qt.DARK_PALETTE], ids=["light", "dark"])
    @pytest.mark.parametrize("fg,bg", [
        ("faint", "card"), ("faint", "bg"), ("border_strong", "card"), ("border_strong", "bg"),
        ("focus", "card"), ("focus", "bg"), ("scroll_handle", "bg"), ("scroll_handle", "card"),
        ("chrome_border", "chrome"),
    ])
    def test_interface_marks_contrast_is_at_least_3(self, palette, fg, bg):
        assert _contrast(palette[fg], palette[bg]) >= 3.0, (fg, bg, palette[fg], palette[bg])

    def test_the_accent_is_warm(self):
        for palette in (qt.PALETTE, qt.DARK_PALETTE):
            r, g, b = (int(palette["accent"][i:i + 2], 16) for i in (1, 3, 5))
            assert r > g > b, "the accent is an orange: red over green over blue"


class TestDataColours:
    def test_categorical_order_is_fixed(self):
        assert qt.DATA["categorical"] == [
            "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

    def test_waste_phase_family(self):
        assert qt.DATA["phase"] == {"Supernatant": "#2a78d6", "Saltcake": "#eb6834", "Sludge": "#1baf7a"}

    def test_magnitude_ramp_is_one_blue_hue_light_to_dark(self):
        ramp = qt.DATA["sequential"]
        assert ramp == ["#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
        luminances = [_luminance(c) for c in ramp]
        assert luminances == sorted(luminances, reverse=True)
        assert qt.DATA["ramp"] == ramp and qt.DATA["blue_ramp"] == ramp

    def test_diverging_is_blue_grey_red(self):
        assert qt.DATA["diverging"] == ["#2a78d6", "#f0efec", "#e34948"]
        assert (qt.DATA["diverging_low"], qt.DATA["diverging_mid"], qt.DATA["diverging_high"]) == (
            "#2a78d6", "#f0efec", "#e34948")

    def test_chart_inks_and_the_single_series_colour(self):
        assert qt.DATA["ink"] == "#0b0b0b"
        assert qt.DATA["secondary"] == "#52514e"
        assert qt.DATA["grid"] == "#e1e0d9"
        assert qt.DATA["axis"] == "#c3c2b7"
        assert qt.DATA["single"] == "#2a78d6" == qt.DATA["bar"]
        assert qt.DATA["surface"] == "#ffffff"

    def test_a_single_series_is_never_the_interface_accent(self):
        assert qt.DATA["single"] not in (qt.PALETTE["accent"], qt.DARK_PALETTE["accent"])

    def test_lists_are_not_shared_between_aliases(self):
        # A caller that edits one list in place must not change the others.
        assert qt.DATA["ramp"] is not qt.DATA["sequential"]


class TestApplyTheme:
    def test_light_mode_sets_light_stylesheet(self, qtbot):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        qt.apply_theme(app, dark=False)
        assert qt.PALETTE["accent"] in app.styleSheet()

    def test_dark_mode_sets_dark_stylesheet(self, qtbot):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        qt.apply_theme(app, dark=True)
        assert qt.DARK_PALETTE["accent"] in app.styleSheet()
        qt.apply_theme(app, dark=False)  # restore for other tests sharing the QApplication singleton

    def test_current_palette_follows_the_applied_theme(self, qtbot):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        qt.apply_theme(app, dark=True)
        try:
            assert qt.current_palette() is qt.DARK_PALETTE
            assert qt.is_dark()
        finally:
            qt.apply_theme(app, dark=False)
        assert qt.current_palette() is qt.PALETTE
        assert not qt.is_dark()

    def test_the_application_palette_matches_the_theme(self, qtbot):
        # The stylesheet alone leaves palette-painted things (link colour in
        # rich text, item text) to the operating system's colour mode.
        from PySide6.QtGui import QPalette
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        for dark, palette in ((True, qt.DARK_PALETTE), (False, qt.PALETTE)):
            qt.apply_theme(app, dark=dark)
            applied = app.palette()
            assert applied.color(QPalette.Window).name() == palette["bg"]
            assert applied.color(QPalette.Base).name() == palette["card"]
            assert applied.color(QPalette.Text).name() == palette["ink"]
            assert applied.color(QPalette.WindowText).name() == palette["ink"]
            assert applied.color(QPalette.Link).name() == palette["link"]
            assert applied.color(QPalette.Disabled, QPalette.Text).name() == palette["faint"]

    def test_a_new_widget_reads_the_theme_palette(self, qtbot):
        from PySide6.QtGui import QPalette
        from PySide6.QtWidgets import QApplication, QTextBrowser
        app = QApplication.instance()
        qt.apply_theme(app, dark=False)
        browser = QTextBrowser()
        qtbot.addWidget(browser)
        assert browser.palette().color(QPalette.Link).name() == qt.PALETTE["link"]

    @pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
    def test_a_chart_toolbar_is_built_with_the_palette_of_the_theme(self, qtbot, dark):
        # matplotlib picks black or white toolbar icons from the luminance of
        # the toolbar's palette when it is built (white below a value of
        # 128). Without a QPalette that follows the theme, a light theme on
        # a dark-mode operating system got white icons on a light strip.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure
        from PySide6.QtWidgets import QApplication, QWidget
        app = QApplication.instance()
        qt.apply_theme(app, dark=dark)
        try:
            host = QWidget()
            qtbot.addWidget(host)
            canvas = FigureCanvasQTAgg(Figure())
            toolbar = NavigationToolbar2QT(canvas, host)
            background = toolbar.palette().color(toolbar.backgroundRole())
            foreground = toolbar.palette().color(toolbar.foregroundRole())
            if dark:
                assert background.value() < 128 and foreground.value() >= 128    # light icons on dark
            else:
                assert background.value() >= 128 and foreground.value() < 128    # dark icons on light
            css = app.styleSheet()
            strip_rule = css.split("NavigationToolbar2QT {")[1].split("}")[0]
            palette = qt.DARK_PALETTE if dark else qt.PALETTE
            assert f"background: {palette['card']}" in strip_rule
        finally:
            qt.apply_theme(app, dark=False)

    def test_theme_change_is_announced(self, qtbot):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        seen = []
        qt.theme_notifier().changed.connect(lambda: seen.append(qt.is_dark()))
        qt.apply_theme(app, dark=True)
        qt.apply_theme(app, dark=False)
        assert seen == [True, False]

    def test_qpalette_can_be_built_for_either_palette(self, qtbot):
        from PySide6.QtGui import QPalette
        for palette in (qt.PALETTE, qt.DARK_PALETTE):
            built = qt.build_qpalette(palette)
            assert built.color(QPalette.Highlight).name() == palette["selection_bg"]
            assert built.color(QPalette.HighlightedText).name() == palette["ink"]
            assert built.color(QPalette.PlaceholderText).name() == palette["faint"]
