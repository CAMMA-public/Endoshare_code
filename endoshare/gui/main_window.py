import os
import sys
import json
import platform
from pathlib import Path

import tensorflow as tf
from loguru import logger

from .video_merger import VideoMergerApp
from .settings import AppSettings
from .info import Info
from .help import Help
from .settings import AppSettings


import psutil
import webbrowser

from PyQt5.QtCore import (
    QPropertyAnimation,
    QSize,
    Qt,
)
from PyQt5.QtGui import (
    QIcon,
    QFontDatabase,
    QKeySequence,
)
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QToolBar,
    QMessageBox,
)

LOG_PERSIST = "PERSIST"
try:
    logger.level(LOG_PERSIST, no=25)  # between INFO and WARNING; ignore if already defined
except ValueError:
    pass

from ..utils.resources import (
    resource_path,
    tinted_icon,
    ICON_COLORS,
)
from ..utils.types import ProcessingMode
class MainApp(QMainWindow):
    

    def __init__(self):
        super().__init__()
         # Allow Cmd/Ctrl+Q to quit
        quit_act = QAction("Quit", self)
        quit_act.setShortcut(QKeySequence.Quit)          # platform‐aware (⌘Q on mac, Ctrl+Q on Win/Linux)
        qapp = QApplication.instance()
        # qapp.aboutToQuit.connect(self._on_app_about_to_quit)
        quit_act.triggered.connect(self.close)            # calls closeEvent()
        self.addAction(quit_act)

        self.central_widget = QWidget()
        self.stacked_widget = QStackedWidget(self.central_widget)
        self.frames = {}
        # self.initialize_frames()

        self.mainlogo = None

        self.setCentralWidget(self.central_widget)
        self.setGeometry(100, 100, 1000, 1000)
        self.setWindowTitle("Endoshare")
        if qapp is not None:
            qapp.setWindowIcon(
                QIcon(resource_path(os.path.join("icons", "icon_logo_app.svg")))
            )

        self.runtime_settings = {
            "mode": ProcessingMode.NORMAL,
            "fps": 25,
            "resolution": 720,
            "local_folder_path": "",
            "shared_folder_path": "",
            "purge_after": False,
        }
        self.load_settings()

        self.init_ui()

        logger.log(LOG_PERSIST, self.retrieve_system_hardware())

    def retrieve_system_hardware(self) -> str:
        return f"""
        {platform.platform()} {platform.system()} {platform.processor()} GPUs available={tf.config.list_physical_devices('GPU')} RAM={round(psutil.virtual_memory().total / (1024.0 **3))}GB
        """
    def closeEvent(self, event):
            # look up your merger frame and its thread
            merger = self.access_video_merger_frame()
            thread = getattr(merger, "video_process_thread", None)
            if thread and thread.isRunning():
                # ask the user if they really want to kill it
                resp = QMessageBox.question(
                    self,
                    "Still processing…",
                    "A processing job is still running.  Quit and terminate it?",
                    QMessageBox.Yes|QMessageBox.No
                )
                if resp == QMessageBox.Yes:
                    # exactly the same teardown you wrote in your terminate button
                    thread.requestInterruption()
                    thread.terminate()
                    thread.wait(2000)
                    merger._cleanup_after_termination()
                    event.accept()  # allow the application to quit
                else:
                    # prevent the application from actually quitting
                    event.ignore()
            else:
                event.accept()

    def load_settings(self):
        # read JSON (or use empty dict)
        settings_file = resource_path('settings.json')
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            settings = {}
        
        # ensure runtime_settings exists before we write into it
        if not hasattr(self, 'runtime_settings'):
            self.runtime_settings = {
                "mode": ProcessingMode.NORMAL,
                "fps": 25,
                "resolution": 720
            }

        # if blank, default to ~/Documents
        home_docs = os.path.expanduser('~/Documents')
        local_path  = settings.get('local_folder_path', '') or home_docs
        shared_path = settings.get('shared_folder_path', '') or home_docs
        local_path  = os.path.expanduser(local_path)
        shared_path = os.path.expanduser(shared_path)
        self.runtime_settings['purge_after'] = settings.get('purge_after', False)
        self.runtime_settings['local_folder_path'] = local_path
        self.runtime_settings['shared_folder_path'] = shared_path


        # now set up your logger into the shared folder
        log_path = Path(shared_path) / "endoshare_1.log"
        logger.add(sys.stdout, level="INFO")
        logger.add(str(log_path), rotation="50MB", level=LOG_PERSIST)
        
    def init_ui(self):
        self.create_mainlogo()
        self.create_toolbar()
        self.create_logobar()

        main_layout = QVBoxLayout()
        self.central_widget.setLayout(main_layout)

        self.stacked_widget = QStackedWidget()

        self.initialize_frames()
        main_layout.addWidget(self.stacked_widget)

    def create_mainlogo(self):
        self.mainlogo = QToolBar()
        self.mainlogo.setIconSize(QSize(120, 120))
        self.mainlogo.setMovable(False)
        # self.mainlogo.setFixedHeight(150)
        self.addToolBar(Qt.TopToolBarArea, self.mainlogo)
        self.mainlogo.setContextMenuPolicy(Qt.PreventContextMenu)
        self.mainlogo.setStyleSheet("QToolBar { border: 0px }")
        
        # Add icons to the toolbar
        home_button = QToolButton()
        home_button.setIcon(QIcon(resource_path(os.path.join('icons','icon_logo.svg'))))
        home_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        home_button.setCheckable(True) # Make the button checkable
        home_button.setStyleSheet("QToolButton:checked { background-color: transparent; }")
        home_button.clicked.connect(lambda checked: self.toggle_menu(checked, self.toolbar))
        self.mainlogo.addWidget(home_button)

    def create_toolbar(self):
        self.toolbar = QToolBar()
        self.toolbar.setIconSize(QSize(40, 40))
        self.toolbar.setMovable(False)
        self.toolbar.setFixedWidth(120)
        self.addToolBar(Qt.LeftToolBarArea, self.toolbar)
        self.toolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.toolbar.setStyleSheet("QToolBar { border: 0px }")
        
        home_button = QToolButton()
        home_button.setIcon(tinted_icon("home_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg", QSize(40, 40), ICON_COLORS['home']))
        home_button.setText('  Home')
        home_button.setFont(QFontDatabase.systemFont(QFontDatabase.GeneralFont))
        home_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        home_button.clicked.connect(lambda: self.show_frame(VideoMergerApp))
        self.toolbar.addWidget(home_button)
        
        settings_button = QToolButton()
        settings_button.setIcon(tinted_icon("settings_b_roll_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg", QSize(40, 40), ICON_COLORS['settings']))
        settings_button.setText('  Settings')
        settings_button.setFont(QFontDatabase.systemFont(QFontDatabase.GeneralFont))
        settings_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        settings_button.clicked.connect(lambda: self.show_frame(AppSettings))
        self.toolbar.addWidget(settings_button)

        Info_button = QToolButton()
        Info_button.setIcon(tinted_icon("info_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg", QSize(40, 40), ICON_COLORS['info']))
        Info_button.setText('  Info')
        Info_button.setFont(QFontDatabase.systemFont(QFontDatabase.GeneralFont))
        Info_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        Info_button.clicked.connect(lambda: self.show_frame(Info))
        self.toolbar.addWidget(Info_button)

        help_button = QToolButton()
        help_button.setIcon(tinted_icon("help_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg", QSize(40, 40), ICON_COLORS['help']))
        help_button.setText('  Help')
        help_button.setFont(QFontDatabase.systemFont(QFontDatabase.GeneralFont))
        help_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        help_button.clicked.connect(lambda: self.show_frame(Help))
        self.toolbar.addWidget(help_button)
    
    def create_logobar(self):
        logobar = QToolBar()
        # logobar.setIconSize(QSize(100, 60))
        logobar.setMovable(False)
        logobar.setFixedHeight(80)
        self.addToolBar(Qt.TopToolBarArea, logobar)
        logobar.setContextMenuPolicy(Qt.PreventContextMenu)
        logobar.setStyleSheet("QToolBar { border: 0px }")

        camma = "http://camma.u-strasbg.fr"
        unistra = "https://www.unistra.fr"
        ihu = "https://www.ihu-strasbourg.eu"
        def openweb(url):
            webbrowser.open(url)

        layout = QHBoxLayout()  # Horizontal layout for the logo bar
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addStretch()  # Add stretch to push buttons to the right

        # Add the buttons to the horizontal layout
        home1_button = QToolButton()
        home1_button.setIcon(QIcon(resource_path(os.path.join('icons','camma.png'))))
        home1_button.setIconSize(QSize(100, 60))
        home1_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        home1_button.clicked.connect(lambda: openweb(camma))
        layout.addWidget(home1_button)

        home2_button = QToolButton()
        home2_button.setIcon(QIcon(resource_path(os.path.join('icons','unistra.png'))))
        home2_button.setIconSize(QSize(100, 60))
        home2_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        home2_button.clicked.connect(lambda: openweb(unistra))
        layout.addWidget(home2_button)

        home3_button = QToolButton()
        home3_button.setIcon(QIcon(resource_path(os.path.join('icons','ihu.png'))))
        home3_button.setIconSize(QSize(100, 60))
        home3_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        home3_button.clicked.connect(lambda: openweb(ihu))
        layout.addWidget(home3_button)

        widget = QWidget()
        widget.setLayout(layout)
        logobar.addWidget(widget)
        
    def toggle_menu(self, checked, toolbar):
        if checked:
            toolbar.setFixedWidth(120)
            self.mainlogo.setFixedWidth(120)
            self.mainlogo.setFixedHeight(120)
            self.mainlogo.setContentsMargins(0, 0, 0, 0)
            self.animate_logo_size(QSize(120, 120))
        else:
            toolbar.setFixedWidth(50)  # Set the width to 0 to collapse
            self.mainlogo.setFixedWidth(100)
            self.mainlogo.setFixedHeight(100)
            self.mainlogo.setContentsMargins(0, 0, 100, 0)
            self.animate_logo_size(QSize(100, 100))
    
    def animate_logo_size(self, size):
        animation = QPropertyAnimation(self.mainlogo, b"iconSize")
        animation.setDuration(200)  # Duration in milliseconds
        animation.setStartValue(self.mainlogo.iconSize())
        animation.setEndValue(size)
        animation.start(animation.DeleteWhenStopped)
            
    def initialize_frames(self):
        for F in (VideoMergerApp, AppSettings, Info, Help):
            frame = F(self, self)
            self.frames[F] = frame
            self.stacked_widget.addWidget(frame)
        
    def show_frame(self, frame_class):
        frame = self.frames[frame_class]
        self.stacked_widget.setCurrentWidget(frame)
        if frame_class is AppSettings and hasattr(frame, "local_folder_entry"):
            rt = self.runtime_settings
            frame.local_folder_entry.setText(rt.get("local_folder_path", ""))
            frame.shared_folder_entry.setText(rt.get("shared_folder_path", ""))
            # Archive Mode checkbox is inverted relative to purge_after
            frame.purge_checkbox.setChecked(not rt.get("purge_after", False))
            frame.archive_entry_changed()
        if frame_class is AppSettings and hasattr(frame, "mode_changed"):
            frame.mode_changed.connect(self.frames[VideoMergerApp].update_ready_label)

    
    def access_video_merger_frame(self):
        # Access the VideoMergerApp frame
        video_merger_frame = self.stacked_widget.widget(0)
        return video_merger_frame

    def access_app_settings_frame(self):
        # Access the AppSettings frame
        app_settings_frame = self.stacked_widget.widget(1)
        return app_settings_frame
    
    