"""
The entry point's splash screen is painted from the same mark, names and
colours as the application header.
"""
import qt_main


class TestSplash:
    def test_splash_is_painted_at_its_nominal_size(self, qtbot):
        pixmap = qt_main.build_splash_pixmap(1.0)
        assert not pixmap.isNull()
        assert (pixmap.width(), pixmap.height()) == qt_main.SPLASH_SIZE

    def test_high_dpi_splash_keeps_its_logical_size(self, qtbot):
        pixmap = qt_main.build_splash_pixmap(2.0)
        assert pixmap.devicePixelRatio() == 2.0
        assert (pixmap.width(), pixmap.height()) == (2 * qt_main.SPLASH_SIZE[0], 2 * qt_main.SPLASH_SIZE[1])

    def test_splash_ground_is_the_header_navy_and_the_mark_is_on_it(self, qtbot):
        import qt_theme
        from PySide6.QtGui import QColor
        image = qt_main.build_splash_pixmap(1.0).toImage()
        chrome = QColor(qt_theme.PALETTE["chrome"])
        corner = image.pixelColor(4, 4)
        assert (corner.red(), corner.green(), corner.blue()) == (chrome.red(), chrome.green(), chrome.blue())
        # The ember sits at the left: somewhere in its square a pixel is far
        # warmer (red over blue) than the navy ground.
        warm = [image.pixelColor(x, y) for x in range(64, 214, 6) for y in range(104, 254, 6)]
        assert any(c.red() > 180 and c.blue() < 140 for c in warm)

    def test_bundled_splash_image_matches_the_painted_size(self, qtbot):
        import os
        from PySide6.QtGui import QPixmap
        from qt_help import asset_path
        path = asset_path("ember_splash.png")
        assert os.path.isfile(path)
        bundled = QPixmap(path)
        assert (bundled.width(), bundled.height()) == qt_main.SPLASH_SIZE

    def test_make_splash_returns_a_splash_screen(self, qtbot):
        from PySide6.QtWidgets import QApplication, QSplashScreen
        splash = qt_main._make_splash(QApplication.instance())
        assert isinstance(splash, QSplashScreen)
        assert not splash.pixmap().isNull()

    def test_saved_theme_is_read_without_failing(self, qtbot):
        assert qt_main._saved_dark_mode() is False     # hermetic settings: nothing saved
