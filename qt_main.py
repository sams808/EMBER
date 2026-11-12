"""
qt_main.py — Ember application entry point.

Run:
    python qt_main.py

Import order matters here: the splash screen goes up FIRST, and only then
do the heavyweight imports (qt_shell pulls in matplotlib, scipy, sklearn,
polars, networkx, every workspace) run — so the user sees the product
mark within ~1 s instead of staring at nothing for the whole import phase.

The splash is painted here (build_splash_pixmap) from the same mark, names
and colours as the application header, so the two cannot drift apart;
assets/ember_splash.png is a rendering of it, used only if painting fails.
"""
from __future__ import annotations

import logging
import os
import sys

from PySide6.QtWidgets import QApplication

SPLASH_SIZE = (720, 420)
MIN_SPLASH_SECONDS = 3.0


def build_splash_pixmap(dpr: float = 1.0):
    """The startup splash: navy ground, the ember mark, the wordmark, one
    line saying what the product is, and the version."""
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QPainter, QPixmap

    from qt_help import APP_DISPLAY_NAME, APP_SUBTITLE, APP_VERSION
    from qt_icons import paint_ember_mark
    from qt_theme import PALETTE

    width, height = SPLASH_SIZE
    dpr = max(1.0, float(dpr))
    pixmap = QPixmap(int(width * dpr), int(height * dpr))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(QColor(PALETTE["chrome"]))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)

    # A deeper band along the bottom carries the small print.
    painter.fillRect(QRectF(0, height - 54, width, 54), QColor(PALETTE["chrome_deep"]))
    painter.fillRect(QRectF(0, height - 54, width, 1), QColor(PALETTE["chrome_field"]))

    paint_ember_mark(painter, QRectF(64, 104, 150, 150))

    title_font = QFont("Segoe UI")
    title_font.setPixelSize(64)
    title_font.setWeight(QFont.Bold)
    title_font.setLetterSpacing(QFont.AbsoluteSpacing, 6)
    painter.setFont(title_font)
    painter.setPen(QColor(PALETTE["chrome_ink"]))
    painter.drawText(QRectF(256, 110, width - 280, 84), Qt.AlignLeft | Qt.AlignVCenter, APP_DISPLAY_NAME)

    # One short rule in the accent: the only warm mark besides the ember.
    painter.fillRect(QRectF(260, 200, 56, 3), QColor(PALETTE["accent"]))

    subtitle_font = QFont("Segoe UI")
    subtitle_font.setPixelSize(20)
    painter.setFont(subtitle_font)
    painter.setPen(QColor(PALETTE["chrome_muted"]))
    painter.drawText(QRectF(258, 214, width - 280, 36), Qt.AlignLeft | Qt.AlignVCenter, APP_SUBTITLE)

    small_font = QFont("Segoe UI")
    small_font.setPixelSize(12)
    painter.setFont(small_font)
    painter.setPen(QColor(PALETTE["chrome_muted"]))
    painter.drawText(QRectF(28, height - 54, width - 56, 54), Qt.AlignLeft | Qt.AlignVCenter,
                     f"Version {APP_VERSION}")
    painter.drawText(QRectF(28, height - 54, width - 56, 54), Qt.AlignRight | Qt.AlignVCenter,
                     "Independent research software")
    painter.end()
    return pixmap


def _make_splash(app):
    """The splash screen, or None when neither painting nor the bundled
    image works: startup must never fail over a picture."""
    from PySide6.QtWidgets import QSplashScreen

    pixmap = None
    try:
        pixmap = build_splash_pixmap(app.devicePixelRatio())
    except Exception:
        logging.getLogger(__name__).exception("Could not paint the splash screen")
    if pixmap is None or pixmap.isNull():
        from PySide6.QtGui import QPixmap
        from qt_help import asset_path
        path = asset_path("ember_splash.png")
        if not os.path.isfile(path):
            return None
        pixmap = QPixmap(path)
    return QSplashScreen(pixmap)


def _saved_dark_mode() -> bool:
    """The theme of the last session, so the window is built in it rather
    than flashing light first."""
    try:
        from PySide6.QtCore import QSettings
        from qt_help import APP_NAME
        return bool(QSettings(APP_NAME, APP_NAME).value("dark_mode", False, type=bool))
    except Exception:
        return False


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)

    from qt_help import APP_NAME, asset_path
    app.setApplicationName(APP_NAME)
    icon_path = asset_path("ember_logo.png")
    if os.path.isfile(icon_path):
        from PySide6.QtGui import QIcon
        app.setWindowIcon(QIcon(icon_path))

    import time
    splash = _make_splash(app)
    splash_shown_at = 0.0
    if splash is not None:
        splash.show()
        splash_shown_at = time.time()
        app.processEvents()

    # Heavy imports happen behind the splash, not before it.
    from qt_shell import EmberMainWindow
    from qt_theme import apply_theme
    import qt_exception_hook

    apply_theme(app, dark=_saved_dark_mode())
    qt_exception_hook.install(app)

    window = EmberMainWindow()
    window.show()
    if splash is not None:
        # Keep the splash up for at least MIN_SPLASH_SECONDS (owner's request).
        while time.time() - splash_shown_at < MIN_SPLASH_SECONDS:
            app.processEvents()
            time.sleep(0.02)
        splash.finish(window)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
