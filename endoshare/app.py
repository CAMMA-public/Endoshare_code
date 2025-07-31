import os
import sys
import multiprocessing as mp
from PyQt5.QtCore import QCoreApplication, QSize, Qt
from PyQt5.QtGui import QPixmap, QSplashScreen
from PyQt5.QtWidgets import QApplication

from .utils.resources import resource_path
from .gui.main_window import MainApp


def run():
    if mp.current_process().name != "MainProcess":
        return

    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication(sys.argv)

    raw_pix = QPixmap(resource_path("icons/splash.png"))
    splash_size = QSize(400, 300)
    pix = raw_pix.scaled(splash_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    splash = QSplashScreen(pix, Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

    screen_geo = app.primaryScreen().availableGeometry()
    splash_geo = splash.geometry()
    x = screen_geo.left() + (screen_geo.width() - splash_geo.width()) // 2
    y = screen_geo.top() + (screen_geo.height() - splash_geo.height()) // 2
    splash.move(x, y)

    splash.show()
    app.processEvents()

    main_app = MainApp()
    main_app.show()

    splash.finish(main_app)

    sys.exit(app.exec_())
