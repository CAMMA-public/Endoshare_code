#!/usr/bin/env python3

import os
import sys
import json
import time
import math
import shutil
import platform
import subprocess
import datetime
import secrets
import csv
from enum import Enum
from uuid import uuid4
from copy import deepcopy
from pathlib import Path

import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from loguru import logger
from vidgear.gears import WriteGear

from processing import deid

import psutil
import webbrowser

from PyQt5.QtCore import (
    QCoreApplication,
    QPropertyAnimation,
    QSize,
    QTimer,
    Qt,
    QThread,
    pyqtSignal,
)
from PyQt5.QtGui import QIcon, QPen, QPixmap, QFontDatabase, QColor, QKeySequence, QPainter
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QMainWindow,
    QSizePolicy,
    QSplashScreen,
    QSplitter,
    QStackedWidget,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QToolBar,
    QTreeView,
    QScrollArea,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QProgressBar,
    QLineEdit,
    QCheckBox,
    QMessageBox,
    QFileSystemModel,
)
import slider

# multiprocessing start method (must come after importing mp)
import multiprocessing as mp

try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass




tf.get_logger().setLevel("INFO")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
LOG_PERSIST = "PERSIST"
logger.level(LOG_PERSIST, no=25) # Between INFO and WARNING.

def _log_uncaught_exceptions(exctype, value, traceback):
    logger.opt(exception=(exctype, value, traceback)).error(f"An uncaught exception occurred.")

sys.excepthook = _log_uncaught_exceptions

def resource_path(relative_path):
    """
    Locate a bundled resource in Contents/Resources (or nested),
    or fall back to the source-tree path.
    """
    base_path = os.path.abspath(os.path.dirname(__file__))
    if getattr(sys, "frozen", False):
        # 1) The usual Resources folder
        bundle_dir = os.path.dirname(sys.executable)
        res_root  = os.path.abspath(os.path.join(bundle_dir, "..", "Resources"))
        # 2) Maybe you accidentally put them under Resources/Resources
        res_nested = os.path.join(res_root, "Resources")
        # 3) Or maybe you ended up in MacOS/ by using dest='.'
        res_macos = bundle_dir

        for candidate_root in (res_root, res_nested, res_macos):
            candidate = os.path.join(candidate_root, relative_path)
            if os.path.exists(candidate):
                return candidate

        # If nothing matched, return the primary location so error messages are clear
        return os.path.join(res_root, relative_path)

    # not frozen → running from source tree
    return os.path.join(base_path, relative_path)

ICON_DIR = "icons"

def load_icon(name: str) -> QIcon:
    """Load a Material icon from icons/material/<name> via resource_path."""
    path = resource_path(os.path.join(ICON_DIR, name))
    return QIcon(path)

def tinted_icon(name: str, size: QSize, hex_color: str) -> QIcon:
    """
    Load 'name' via load_icon, render it at 'size', tint every non‑transparent
    pixel to 'hex_color', then draw a thin 1px border in the same color.
    """
    # 1) render the base icon
    base = load_icon(name).pixmap(size)

    painter = QPainter(base)
    # 2) tint fill
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(base.rect(), QColor(hex_color))

    # 3) draw a 1px border inset so it stays inside the edges
    pen = QPen(QColor(hex_color))
    pen.setWidth(1)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    # inset by half the pen width
    inset = pen.width() // 2
    r = base.rect().adjusted(inset, inset, -inset, -inset)
    painter.drawRect(r)

    painter.end()
    return QIcon(base)

ICON_COLORS = {
    'home':    '#444446',
    'settings':'#444446',
    'info':    '#444446',
    'help':    '#444446'
}


def _ffmpeg_path():
    # this must match how you bundle ffmpeg in your .app
    base = resource_path("Externals")
    # if you shipped an ffmpeg.app bundle:
    app_bundle = os.path.join(base, "ffmpeg.app")
    cli_bin   = os.path.join(app_bundle, "Contents", "MacOS", "ffmpeg")

    # fallback: maybe you shipped the raw binary under Externals/ffmpeg
    raw_bin   = os.path.join(base, "ffmpeg")

    return cli_bin if os.path.isfile(cli_bin) else raw_bin

FFMPEG_BIN = _ffmpeg_path()
FFPROBE_BIN = FFMPEG_BIN 


class ProcessingMode(Enum):
    NORMAL = 0
    ADVANCED = 1

class ProcessingInterrupted(Exception):
    """Raised to abort processing when user hits Terminate."""
    pass

#####Main App Window###############
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
        self.load_settings()

        self.central_widget = QWidget()
        self.stacked_widget = QStackedWidget(self.central_widget)
        self.frames = {}
        # self.initialize_frames()

        self.mainlogo = None

        self.setCentralWidget(self.central_widget)
        self.setGeometry(100, 100, 1000, 1000)
        self.setWindowTitle("Endoshare")
        app.setWindowIcon(QIcon(resource_path(os.path.join('icons','icon_logo_app.svg'))))

        self.runtime_settings = {
            "mode": ProcessingMode.NORMAL,
            "fps": 25,
            "resolution": 720
        }

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
        self.stacked_widget.setCurrentWidget(self.frames[frame_class])
    
    def access_video_merger_frame(self):
        # Access the VideoMergerApp frame
        video_merger_frame = self.stacked_widget.widget(0)
        return video_merger_frame

    def access_app_settings_frame(self):
        # Access the AppSettings frame
        app_settings_frame = self.stacked_widget.widget(1)
        return app_settings_frame
    



##########################Video Browser Widget for handling all scenarios of input###################################
# List of video file extensions
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.mpeg', '.mpg', '.ts', '.m2ts']

class VideoBrowser(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Browser")
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ─── create your widgets ────────────────────────────────
        self.tree_view = QTreeView()
        self.selected_videos_list = QListWidget()
        self.selected_videos_list.setDragDropMode(QListWidget.InternalMove)
        self.selected_videos_list.setSelectionMode(QListWidget.SingleSelection)
        self.selected_videos_list.setAcceptDrops(True)
        self.selected_videos_list.setStyleSheet("""
            QListWidget {
                background: #f9f9f9;
            }
            QListWidget::item:selected {
                background: #BBDEFB;
            }
        """)

        self.add_all_button = QPushButton("Add All Videos")
        self.add_all_button.setToolTip("Add all videos from the selected folder and subfolders")
        self.add_all_button.setIcon(load_icon("playlist_add_check_circle_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg"))
        self.add_all_button.clicked.connect(self.add_all_videos)

        self.add_selected_button = QPushButton("Add Selected Video")
        self.add_selected_button.setToolTip("Add the selected video from the list")
        self.add_selected_button.setIcon(load_icon("play_arrow_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg"))
        self.add_selected_button.clicked.connect(self.add_single_video)

        self.preview_button = QPushButton("Preview")
        self.preview_button.setToolTip("Preview the selected video")
        self.preview_button.setIcon(load_icon("preview_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg"))
        self.preview_button.clicked.connect(self.preview_video)

        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.setToolTip("Remove the selected video from the list")
        self.remove_button.setIcon(load_icon("delete_forever_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg"))
        self.remove_button.clicked.connect(self.remove_selected_videos)

        # ─── wrap in GroupBoxes ────────────────────────────────
        avail_group = QGroupBox("📁  Available Videos")
        avail_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        avail_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #2196F3;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                color: #2196F3;
            }
        """)
        avail_layout = QVBoxLayout(avail_group)
        avail_layout.addWidget(self.tree_view)

        sel_group = QGroupBox("✅  Patient Videos")
        sel_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sel_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #4CAF50;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                color: #4CAF50;
            }
        """)
        sel_layout = QVBoxLayout(sel_group)
        sel_layout.addWidget(self.selected_videos_list)

        # ─── splitter so panels stay your default size but are resizable ───────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(avail_group)
        splitter.addWidget(sel_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        # ─── buttons row ────────────────────────────────────────
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_selected_button)
        button_layout.addWidget(self.add_all_button)
        button_layout.addWidget(self.preview_button)
        button_layout.addWidget(self.remove_button)

        # ─── assemble everything ─────────────────────────────────
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(splitter, 1)  # stretch=1
        main_layout.addLayout(button_layout, 0)  # stretch=0
        self.setLayout(main_layout)

        # ─── wire up the rest, unchanged ────────────────────────
        self.selected_videos_list.itemClicked.connect(self.on_item_clicked)
        self.selected_videos_list.clicked.connect(self.update_last_clicked_item)
        self.tree_view.doubleClicked.connect(self.add_video_from_tree)
        self.tree_view.clicked.connect(self.update_last_clicked_item)

        # track last-click
        self.last_clicked_item = None
    
    def populate_videos(self, folder):
        model = QFileSystemModel()
        # Set name filters like *.mp4, *.avi, etc.
        model.setNameFilters([f"*{ext}" for ext in VIDEO_EXTENSIONS])
        model.setNameFilterDisables(False)   # hide everything else
        # Now point it at the folder
        model.setRootPath(folder)
        self.tree_view.setModel(model)
        # And show that folder as the root
        header = self.tree_view.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree_view.setRootIndex(model.index(folder))
        self.tree_view.setColumnHidden(2, True)
        self.tree_view.setColumnHidden(3, True)

    def add_video_from_tree(self, index):
        file_path = self.tree_view.model().filePath(index)
        if (os.path.isfile(file_path) 
            and os.path.splitext(file_path)[1].lower() in VIDEO_EXTENSIONS
            and not os.path.basename(file_path).startswith('._')
            ):
            file_name = os.path.basename(file_path)
            if not self.selected_videos_list.findItems(file_name, Qt.MatchExactly):
                item = QListWidgetItem(file_name)
                item.setData(Qt.UserRole, file_path)  # Store full path as item data
                self.selected_videos_list.addItem(item)
    
    def add_single_video(self):
    # Grab the first selected index in the tree (only column-0)
        indexes = self.tree_view.selectionModel().selectedIndexes()
        for idx in indexes:
            if idx.column() == 0:
                # Reuse your existing single-add logic
                self.add_video_from_tree(idx)
                break

    def add_all_videos(self):
        # Create a list to store all videos
        all_videos = []

        # Add all videos from the selected folder and subfolders
        root_path = self.tree_view.model().rootPath()
        for root, dirs, files in os.walk(root_path):
            for file in files:
                file_path = os.path.join(root, file)
                if (os.path.splitext(file_path)[1].lower() in VIDEO_EXTENSIONS
                    and not os.path.basename(file_path).startswith('._')
                    ):
                    all_videos.append(file_path)

        # Sort the list of videos alphabetically
        all_videos.sort()

        # Clear the selected videos list
        self.selected_videos_list.clear()

        # Add sorted videos to the selected videos list
        for file_path in all_videos:
            file_name = os.path.basename(file_path)
            item = QListWidgetItem(file_name)
            item.setData(Qt.UserRole, file_path)  # Store full path as item data
            self.selected_videos_list.addItem(item)


    def remove_selected_videos(self):
        # Remove selected videos from the selected videos list
        for item in self.selected_videos_list.selectedItems():
            self.selected_videos_list.takeItem(self.selected_videos_list.row(item))

    def on_item_clicked(self, item):
        # Clear previous selection and select the clicked item
        self.selected_videos_list.clearSelection()
        item.setSelected(True)

    def update_last_clicked_item(self, index):
        # Update the last clicked item
        if isinstance(self.sender(), QTreeView):
            self.last_clicked_item = self.tree_view.model().filePath(index)
        elif isinstance(self.sender(), QListWidget):
            item = self.selected_videos_list.currentItem()
            if item:
                self.last_clicked_item = item.data(Qt.UserRole)

    def preview_video(self):
        # Get the last clicked video file path
        file_path = self.last_clicked_item

        if file_path and os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in VIDEO_EXTENSIONS:
            # Open the video with the default player
            # wmplayer_path = r'C:\Program Files (x86)\Windows Media Player\wmplayer.exe'
            logger.info(f"Previewing {file_path} in System Video Player ...")
            if platform.system() == "Darwin":
                subprocess.run(["open", file_path], check=True)
            elif platform.system() == "Windows":
                subprocess.run(["start", file_path], check=True)
            else: # Linux and Friends :)
                subprocess.run(["xdg-open", file_path], check=True)
            #subprocess.run(['open', '-a', 'QuickTime Player', file_path])

    def get_selected_video_list(self):
        return self.selected_videos_list
    
##########################Video Copy Thread for updating the video dictionary about the location; no need to save video################

class VideoCopyThread(QThread):
    update_progress = pyqtSignal(int, int, str, bool)

    def __init__(self, video_files, selected_folder):
        super().__init__()
        self.video_files = video_files  
        self.selected_folder = selected_folder
        self.video_dict = {}  # Dictionary to store the mapping of original file names to new names

    def run(self):
        total_videos = len(self.video_files)
        logger.info("total_videos:", total_videos)
        for i, video_file in enumerate(self.video_files):
            self.video_dict[video_file] = os.path.join(self.selected_folder, video_file)
            progress = int(((i + 1) / total_videos) * 100)
            self.update_progress.emit(i + 1, total_videos, f"Arranging file {video_file}... ({progress}%)", True)
        self.update_progress.emit(total_videos, total_videos, "Arranging completed successfully!" , True)

    def get_video_dict(self):
        return self.video_dict
    


#####################################Video Process Thread for OOB detection, merging and deidentification#######################


class VideoProcessThread(QThread):
    update_progress = pyqtSignal(int, int, str, bool)
    update_color = pyqtSignal(str, str)
    error           = pyqtSignal(str)

    def __init__(self, video_in_root_dir, shared_folder, local_folder, 
                 fps,
                 resolution,
                 mode,
                 purge_after=False
                 ):
        super().__init__()
        
        self.video_in_root_dir = video_in_root_dir 
        self.destination_folder = ""
        self.ckpt_path = resource_path(os.path.join('ckpt','oobnet_weights.h5'))   # "CKPT_PATH"  # "oobnet_weights.h5" ## needs to be changed with hone settings
        self.device = "/cpu:0"
        self.out_final = shared_folder  ## needs to be changed with hone settings
        self.name_translation_filename = os.path.join(local_folder, "./patientID_log.csv") ## needs to be changed from settings
        self.patient_name = ""
        self.crf = 20   ## needs to be changed from settings
        self.fps = fps
        self.resolution = resolution
        self.processing_mode = mode
        self.buffer_size=2048
        self.default_output_folder = local_folder
        self.purge_after = purge_after

    def preprocess(self, image, shape=[64, 64]):
        with tf.device(self.device):
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, shape)
            image = tf.reshape(image, shape + [3])
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            return tf.expand_dims(image, 0)

    def build_model(self, input_shape=[64, 64]):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(input_shape + [3], dtype=tf.float32))
        model.add(
            tf.keras.applications.MobileNetV2(
                input_shape=input_shape + [3], alpha=1.0, include_top=False, weights=None
            )
        )
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.Dropout(0))
        model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 0)))
        model.add(tf.keras.layers.LSTM(units=640, return_sequences=True, stateful=True))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        return model
    
    def terminate(self):
        """
        When the user hits Terminate:
          • In ADVANCED mode → soft‑stop so we don’t crash WriteGear.
          • In NORMAL mode   → fall back to the old hard kill.
        """
        if self.processing_mode == ProcessingMode.ADVANCED:
            # soft stop
            self.requestInterruption()

            # close WriteGear if it’s up
            vg = getattr(self, "_vg", None)
            if vg:
                try: vg.close()
                except: pass

            # close tqdm if it’s up
            pb = getattr(self, "_pbar", None)
            if pb:
                try: pb.close()
                except: pass

            # don’t call super().terminate() here
        else:
            # normal mode: do exactly what you had before
            super().terminate()

    def run_fast_inference(
        self,
        video_in_root_dir,
        video_out_root_dir,
        text_root_dir,
        ckpt_path,
        buffer_size,
        device,
        curr_progress,
        max_progress,
    ):
        video_names = list(video_in_root_dir.values())

        self.update_progress.emit(curr_progress, max_progress, "Processing started for " + self.patient_name , False)
        start_time = time.time()
        
        file_name, file_ext = os.path.splitext(video_names[0])
        out_name = file_name.split(".")[0]
        out_ext = file_ext[1:]
        out_video_path = os.path.join(
            video_out_root_dir, self.patient_name + "."+ out_ext
        )

        deid.process_video([Path(p) for p in video_names], Path(out_video_path), logger, self.update_progress, curr_progress, max_progress)
        end_time = time.time()

        # Video duration
        video      = cv2.VideoCapture(out_video_path)
        framecount = video.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        fps        = video.get(cv2.CAP_PROP_FPS) or 0
        if fps > 0:
            video_duration = framecount / fps
        else:
            logger.warning(f"FPS is zero for output video '{out_video_path}', skipping duration/speed calc.")
            video_duration = 0.0

        elapsed = end_time - start_time
        logger.log(LOG_PERSIST, f"total time spent: {elapsed:.2f} sec")
        if elapsed > 0:
            speed = video_duration / elapsed if video_duration > 0 else None
            if speed is not None:
                logger.log(LOG_PERSIST, f"processing speed: {speed:.2f}× real time")
            else:
                logger.log(LOG_PERSIST, "processing speed: N/A (could not compute)")
        else:
            logger.log(LOG_PERSIST, "processing speed: N/A (zero elapsed time)")

        # Emit the signal to update the progress bar in the main GUI thread
        self.update_progress.emit(curr_progress+len(video_names), max_progress, "Processing completed for " + self.patient_name , False)
        # Emit the signal to update the color of the patient in the name_list 
        self.update_color.emit(self.patient_name, "green")

    def run_advanced_inference(
        self,
        video_in_root_dir,
        video_out_root_dir,
        text_root_dir,
        ckpt_path,
        buffer_size,
        device,
        curr_progress,
        max_progress,
    ):
        videos_duration = 0
        write_out_video = True
        # build the model
        counter = 0
        with tf.device(device):
            model = self.build_model()
            model.load_weights(ckpt_path)
        init_once = True
        
        video_names = list(video_in_root_dir.values())
       
        self.update_progress.emit(curr_progress, max_progress, "Processing started for " + self.patient_name , False)
        start_time = time.time()
        rescaled_size = None
        total_chunks = 0
        for in_video_path in video_names:
            cap = cv2.VideoCapture(in_video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            total_chunks += math.ceil(frame_count / buffer_size)
        self._total_units = total_chunks
        self._processed_units = 0

        
        for i, in_video_path in enumerate(video_names):
            # ── interruption check before each file ───────────────
            if self.isInterruptionRequested():
                logger.info("Advanced inference interrupted before starting next video.")
                # gracefully close writer and progress bar
                video_out.close()
                pbar.close()
                raise ProcessingInterrupted()

            logger.info(f"Processing video {i+1} in advanced mode ...")

            try:
                video_in = cv2.VideoCapture(in_video_path)
                assert video_in.isOpened()
                fps_in = video_in.get(cv2.CAP_PROP_FPS)
                frame_count = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps_in
            except OSError:
                logger.error("Could not open/read file")
            if write_out_video and init_once:
                init_once = False
                os.makedirs(video_out_root_dir, exist_ok=True)
                
                width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
                orig_image_buffer = np.zeros(
                    (buffer_size, height, width, 3), dtype=np.uint8
                )
                
                file_name, file_ext = os.path.splitext(in_video_path)
                out_name = file_name.split(".")[0]
                out_ext = file_ext[1:]
                #out_name, out_ext = os.path.basename(in_video_path).split(".")
                out_video_path = os.path.join(
                    video_out_root_dir, self.patient_name + "." + out_ext
                )
                fps = video_in.get(cv2.CAP_PROP_FPS)
                logger.info(f"fps: {self.fps}, resolution: {self.resolution}p")
                
                ####Need to add to log################
                ##### Need to add resolution ###################

                w = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = video_in.get(cv2.CAP_PROP_FPS)

                # choose 0.10 for a balanced quality/size or 0.15 for high quality
                bpp = 0.10  
                bitrate_k = round(w * h * fps * bpp / 1000)

                output_params = {
                    "-pix_fmt": "yuv420p",
                    "-input_framerate": self.fps,
                }

                if sys.platform == "darwin":
                    output_params.update({
                        "-vcodec": "h264_videotoolbox",
                        "-b:v": f"{bitrate_k}k",
                        "-profile:v": "high",
                        "-tune": "zerolatency",
                    })
                else:
                    output_params.update({
                        "-vcodec": "libx264",
                        "-preset": "ultrafast",
                        "-crf": self.crf,
                        "-tune": "zerolatency",
                    })

                if self.resolution > 0:
                    rescaled_width = np.round(width*(self.resolution/height))
                    if rescaled_width%2 != 0:
                        rescaled_width += 1
                    rescaled_size = (int(rescaled_width), self.resolution)
                video_out = WriteGear(output=out_video_path, logging=False, compression_mode=True, **output_params) 

            video_nframes = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
            pred_history = []
            image_buffer = []
            image_count = 0
            pbar = tqdm(total=video_nframes // buffer_size)
            self._vg   = video_out
            self._pbar = pbar


            target_fps = self.fps
            frame_interval = fps / target_fps if fps > target_fps else target_fps / fps
            frame_index = 0
            if fps_in >= target_fps:
                while True:
                    # ── interruption check inside frame loop ─────────────
                    if self.isInterruptionRequested():
                        logger.info("Advanced inference interrupted during frame loop.")
                        video_out.close()
                        pbar.close()
                        raise ProcessingInterrupted()
                    ok, frame = video_in.read()
                    counter += 1
                    if ok:
                        preprocess_image = self.preprocess(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        image_buffer.append(preprocess_image)
                        if write_out_video:
                            orig_image_buffer[image_count] = frame
                        image_count += 1
                        if len(image_buffer) == buffer_size:
                            with tf.device(device):
                                image_batch = tf.concat(image_buffer, axis=0)
                                prediction = model(image_batch)
                            preds = np.round(prediction.numpy()[0, :, 0]).astype(np.uint8)
                            orig_image_buffer[preds.astype(bool)] = np.zeros_like(orig_image_buffer[preds.astype(bool)])
                            
                            for frame in orig_image_buffer:
                                if rescaled_size is not None:
                                    frame = cv2.resize(frame, rescaled_size, interpolation=cv2.INTER_AREA)
                                # Adjust for FPS-change.
                                if frame_index % frame_interval < 1:
                                    video_out.write(frame)
                                frame_index += 1
                            pred_history += preds.tolist()
                            image_buffer = []
                            image_count = 0
                            pbar.update(1)
                            progress = int((pbar.n / pbar.total * 100) if pbar.total != 0 else 0)
                            ####Need to add to log################
                            # Emit the signal to update the progress bar in the main GUI thread
                            self._processed_units += 1
                            self.update_progress.emit(self._processed_units,
                                                    self._total_units,
                                                    f"Processing {self.patient_name} ({i+1}/{len(video_names)})…",
                                                    False)
                    else:
                        # at end of file, also obey interruption
                        if self.isInterruptionRequested():
                            break
                        if len(image_buffer) > 0:
                            with tf.device(self.device):
                                image_batch = tf.concat(image_buffer, axis=0)
                                prediction = model(image_batch)
                            preds = np.round(prediction.numpy()[0, :, 0]).astype(np.uint8)
                            orig_image_buffer_write = deepcopy(
                                orig_image_buffer[:image_count]
                            )
                            orig_image_buffer_write[preds.astype(bool)] = (
                                orig_image_buffer_write[preds.astype(bool)]
                                .mean(1)
                                .mean(1)
                                .reshape(preds.sum(), 1, 1, 3)
                            )
                           
                            for frame in orig_image_buffer_write:
                                if rescaled_size is not None:
                                    frame = cv2.resize(frame, rescaled_size, interpolation=cv2.INTER_AREA)
                                video_out.write(frame)
                            pred_history += preds.tolist()
                        break
            else:
                while True:
                    if self.isInterruptionRequested():
                        logger.info("Advanced inference interrupted during frame loop.")
                        video_out.close()
                        pbar.close()
                        raise ProcessingInterrupted()
                    ok, frame = video_in.read()
                    counter += 1
                    if ok:
                        preprocess_image = self.preprocess(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        image_buffer.append(preprocess_image)
                        if write_out_video:
                            orig_image_buffer[image_count] = frame
                        image_count += 1
                        if len(image_buffer) == buffer_size:
                            with tf.device(device):
                                image_batch = tf.concat(image_buffer, axis=0)
                                prediction = model(image_batch)
                            preds = np.round(prediction.numpy()[0, :, 0]).astype(np.uint8)
                            orig_image_buffer[preds.astype(bool)] = np.zeros_like(orig_image_buffer[preds.astype(bool)])
                            #orig_image_buffer[preds.astype(bool)] = (
                            #    orig_image_buffer[preds.astype(bool)]
                            #    .mean(1)
                            #    .mean(1)
                            #    .reshape(preds.sum(), 1, 1, 3)
                            #)
                            
                            for frame in orig_image_buffer:
                                if rescaled_size is not None:
                                    frame = cv2.resize(frame, rescaled_size, interpolation=cv2.INTER_AREA)
                                # Don't adjust for FPS-change...
                                video_out.write(frame)
                            pred_history += preds.tolist()
                            image_buffer = []
                            image_count = 0
                            pbar.update(1)
                            progress = int((pbar.n / pbar.total * 100) if pbar.total != 0 else 0)
                            ####Need to add to log################
                            # Emit the signal to update the progress bar in the main GUI thread
                            self._processed_units += 1
                            self.update_progress.emit(self._processed_units,
                                                    self._total_units,
                                                    f"Processing {self.patient_name} ({i+1}/{len(video_names)})…",
                                                    False)
                    else:
                        if len(image_buffer) > 0:
                            with tf.device(self.device):
                                image_batch = tf.concat(image_buffer, axis=0)
                                prediction = model(image_batch)
                            preds = np.round(prediction.numpy()[0, :, 0]).astype(np.uint8)
                            orig_image_buffer_write = deepcopy(
                                orig_image_buffer[:image_count]
                            )
                            orig_image_buffer_write[preds.astype(bool)] = (
                                orig_image_buffer_write[preds.astype(bool)]
                                .mean(1)
                                .mean(1)
                                .reshape(preds.sum(), 1, 1, 3)
                            )
                           
                            for frame in orig_image_buffer_write:
                                if rescaled_size is not None:
                                    frame = cv2.resize(frame, rescaled_size, interpolation=cv2.INTER_AREA)
                                video_out.write(frame)
                            pred_history += preds.tolist()
                        break
            
            # ── per‑video cleanup if interrupted ────────────────
            if self.isInterruptionRequested():
                try: video_out.close()
                except: pass
                pbar.close()
                return

            framecount = video_in.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = video_in.get(cv2.CAP_PROP_FPS)
            duration = (framecount/fps)/1000
            videos_duration += duration
            video_in.release()
            pbar.update(1)
            self._processed_units += 1
            self.update_progress.emit(self._processed_units,
                                    self._total_units,
                                    f"Processing {self.patient_name} ({i+1}/{len(video_names)})…",
                                    False)
            progress = int((pbar.n / pbar.total * 100) if pbar.total != 0 else 0)
            self._vg   = None
            self._pbar = None


            ####Need to add to log################
            # Emit the signal to update the progress bar in the main GUI thread

            self.update_progress.emit(curr_progress+len(video_names), max_progress, f"Processing for {self.patient_name}, file {out_name}...", False)

        end_time = time.time()

        ####Need to add to log################
        logger.log(LOG_PERSIST, f"total time spent: {end_time - start_time:.2f} sec")
        elapsed = end_time - start_time
        if elapsed > 0:
            speed = videos_duration / elapsed
            logger.log(LOG_PERSIST, f"processing speed: {speed:.2f} video_duration/processing_time")
        else:
            logger.log(LOG_PERSIST, "processing speed: N/A (zero elapsed time)")

        video_out.close()
        pbar.close()
        # Emit the signal to update the progress bar in the main GUI thread
        self.update_progress.emit(curr_progress+len(video_names), max_progress, "Processing completed for " + self.patient_name , False)
        # Emit the signal to update the color of the patient in the name_list 
        self.update_color.emit(self.patient_name, "green")

    def setup_name_translation_file(self, name_translation_filename):
        """Creates a log file to record original to randomized video names.
        If no filename is specified, will create a log file named
        'patientID_log.csv'. If the file already exists, it will append
        new entries to it."""
        
        name_translation_file_path = Path(name_translation_filename)
        
        # Check if log file exists
        if name_translation_file_path.exists():
            return name_translation_file_path
        
        # If not, create the log file and write header
        with open(name_translation_file_path, mode='w', newline='') as name_translation_file:
            name_translation_writer = csv.writer(name_translation_file)
            name_translation_writer.writerow(["original", "anonymized"])
        
        return name_translation_file_path


    def name_generator(self,uuid=False, prefix="", start=0, width=3):
        """Returns functions that generate names. If uuid is True, will
        return a function that generates uuid4s. Otherwise, will return a
        function to generate incrementing names with a prefix added to the
        incrementing numbers that start at 'start' and padded to be 'width'
        wide, eg prefix = "video" then video001, video002, etc."""

        def incrementing_name():
            nonlocal n
            n+=1
            return prefix + f"{n:0{width}}"

        def uuid_name():
            # This is random. I thought, we don't want random ...
            return uuid4().hex[:-25]

        if uuid:
            return uuid_name
        else:
            # subtract 1 from start because incrementing name will increment it
            # by one before initial use
            n = start-1
            return incrementing_name


    def seq_width(self,num):
        """Returns one more than order of magnitude (base-10) for num which
        is equivalent to the number of digits-wide a sequential
        representation would need to be. E.g.: num = 103 return 3 so
        sequences would be 000, 001, ... 102, 103."""
        return math.floor(math.log10(num)) + 1


    def shuffle(self,some_list):
        """Returns the items in some_list in shuffled order."""
        # do a defensive copy so that the original list doesn't get
        # consumed by the 'pop()'
        items = some_list.copy()
        while items:
            yield items.pop(secrets.randbelow(len(items)))


    def randomize_paths(self,vid_paths, outdir, sequentialize):
        """Returns a dict of orig_name: random_name. When sequentialize is
        true, random_names will be like video000, video001, etc, otherwise
        returns a uuid4."""
        if sequentialize:
            generate_name = self.name_generator(prefix="video", width=self.seq_width(len(vid_paths)))
        else:
            generate_name = self.name_generator(uuid=True)
        orig_to_random = {}
        # shuffle the filenames so that sequentially generated filenames
        # don't mimic the order of the filenames in the input directory
        for orig_path in self.shuffle(vid_paths):
            randomized_path = Path(outdir).joinpath(generate_name() + orig_path.suffix)
            orig_to_random[orig_path] = randomized_path

        return orig_to_random


    def transpose_paths(self,paths, outdir):
        """Returns dict mapping each path in paths to a path from joining
        outdir with the basename of path."""
        return {path: Path(outdir).joinpath(path.name) for path in paths}


    def is_video_path(self,path):
        """Checks if path has a video extension and is a file."""
        #vid_exts = (".mp4", ".avi")
        return path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file()


    def get_video_paths(self,vid_dir):
        """Yield files with video extensions in vid_dir"""
        return [path for path in Path(vid_dir).rglob("*") if self.is_video_path(path)]
    

    def strip_metadata(self, input_vid, output_vid):
        """Strips metadata from input_vid and places stripped video in
        output_vid. If successful returns output_vid's path, otherwise
        returns 'FAILED'."""
        command = [
            FFMPEG_BIN,
            "-nostdin",
            # set input video
            "-i",
            str(input_vid),
            # select all video streams
            "-map",
            "0:v",
            # select all audio streams if present
            "-map",
            "0:a?",
            # just copy streams, do not transcode (much faster and lossless)
            "-c",
            "copy",
            # strip global metadata for the video container
            "-map_metadata",
            "-1",
            # strip metadata for video stream
            "-map_metadata:s:v",
            "-1",
            # strip metadata for audio stream
            "-map_metadata:s:a",
            "-1",
            # remove any chapter information
            "-map_chapters",
            "-1",
            # remove any disposition info
            "-disposition",
            "0",
            str(output_vid),
        ]

        try:
            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as perr:
            logger.error(f"ffmpeg failed to strip '{input_vid}' with output: {perr.stderr}")
            # File failed to process so delete it is ffmpeg made an
            # incomplete one
            if output_vid.is_file():
                output_vid.unlink()
            return "FAILED"

        return output_vid if output_vid.is_file() else None

    def anonymize(self, video_in_root_dir, video_out_root_dir, name_translation_filename):
        """Will strip metadata and optionally randomize filenames from a
        directory of videos."""

        vid_paths = self.get_video_paths(video_in_root_dir)
        outdir = Path(video_out_root_dir)
        outdir.mkdir(exist_ok=True)

        name_translation_file_path = self.setup_name_translation_file(name_translation_filename)

        vid_map = self.randomize_paths(vid_paths, outdir, sequentialize=False)
        final_name = None

        for orig_path, new_path in vid_map.items():
            # strip metadata then save into the csv log file:
            # orig_path,output (either new_path or "FAILED" if it was not successful)
            final_name = self.strip_metadata(orig_path, new_path)

            # Extract file names without extensions
            orig_name = orig_path.stem
            new_name = new_path.stem
            
            # Append to log file
            with open(name_translation_file_path, mode='a', newline='') as name_translation_file:
                name_translation_writer = csv.writer(name_translation_file)
                name_translation_writer.writerow([orig_name, new_name])
            logger.info(f"Anonymized into {new_path}.")

        return final_name
    
    def run(self):
        # ─── 0) Pre‑flight: verify every video can be opened ─────────────────
        all_paths = [
            p
            for vid_map in self.video_in_root_dir.values()
            for p in vid_map.values()
        ]
        for path in all_paths:
            cmd = [
                FFMPEG_BIN, "-v", "error",
                "-i", path,
                "-f", "null", "-"   # decode but throw away frames
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if proc.returncode != 0:
                # grab the first few lines of ffmpeg’s stderr to keep it concise
                err_lines = proc.stderr.strip().splitlines()[:5]
                snippet   = "\n".join(err_lines)
                self.error.emit(
                    f"Corrupt file detected: “{Path(path).name}”\n\n"
                    f"{snippet}\n\n"
                    "Processing aborted."
                )
                return

        name_translation_file_path = self.setup_name_translation_file(self.name_translation_filename)
        ###############Iteration happens for #of Patients############################
        n_all_videos = sum([len(videos_iter) for videos_iter in self.video_in_root_dir.values()])
        curr_n_videos = 0
        for patient_id, videos_iter in self.video_in_root_dir.items():
            if self.isInterruptionRequested():
                break
            temp_folder = self.default_output_folder
            os.makedirs(temp_folder, exist_ok=True)
            if not self.destination_folder:
                previous_patient_id = patient_id
                self.destination_folder = os.path.join(temp_folder, patient_id)
                os.makedirs(self.destination_folder, exist_ok=True)
            elif patient_id != previous_patient_id:
                previous_patient_id = patient_id
                self.destination_folder = os.path.join(temp_folder, patient_id)
                os.makedirs(self.destination_folder, exist_ok=True)
            else:
                self.destination_folder = os.path.join(self.destination_folder, patient_id)
                os.makedirs(self.destination_folder, exist_ok=True)
            self.patient_name = patient_id

            try:
                for path in videos_iter.values():
                    cap = cv2.VideoCapture(path)
                    if not cap.isOpened():
                        cap.release()
                        raise RuntimeError(f"Cannot open “{Path(path).name}”")
                    cap.release()
                    
                if self.processing_mode == ProcessingMode.ADVANCED:
                    self.run_advanced_inference(
                        video_in_root_dir=videos_iter,
                        video_out_root_dir=self.destination_folder,
                        text_root_dir=self.destination_folder,
                        ckpt_path=self.ckpt_path,
                        buffer_size=64,
                        device=self.device,
                        curr_progress=curr_n_videos,
                        max_progress=n_all_videos,
                    )
                elif self.processing_mode == ProcessingMode.NORMAL:
                    self.run_fast_inference(
                        video_in_root_dir=videos_iter,
                        video_out_root_dir=self.destination_folder,
                        text_root_dir=self.destination_folder,
                        ckpt_path=self.ckpt_path,
                        buffer_size=64,
                        device=self.device,
                        curr_progress=curr_n_videos,
                        max_progress=n_all_videos,
                    )
                curr_n_videos += len(videos_iter)
                anonymized_video = self.anonymize(self.destination_folder, self.out_final, self.name_translation_filename)
                if self.purge_after:
                    orig_folder = Path(self.default_output_folder) / self.patient_name
                    if orig_folder.exists():
                        try:
                            shutil.rmtree(orig_folder)
                            logger.info(f"Purged archive folder {orig_folder}")
                        except Exception as e:
                            logger.warning(f"Failed to purge archive folder {orig_folder}: {e}")
            
            except ProcessingInterrupted:
                logger.info(f"Processing aborted by user at patient {patient_id}")
                return
            except Exception as exc:
                # log full traceback
                logger.error(f"Error processing patient {patient_id}", exc_info=True)
                # notify UI
                self.error.emit(str(exc))
                return

######Home Window Widgets are part of this VideoMergerApp####################

class VideoMergerApp(QWidget):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        self.selected_folder = ""
        self.patient_dict = []
        self.video_items = []
        self.selected_videos = []
        self.shared_folder = ""
        self.local_folder = ""
        self.load_settings()
        self.video_dict = {}

        # Declare thread instances as class variables
        self.video_copy_thread = None
        self.video_merge_thread = None
        self.video_browser_thread = None
        self.finished_threads = 0
        self.total_threads = 0

        self.init_ui()

    def onFolderProvided(self, folder):
        self.output_folder = folder
        # Recheck if output folder is provided
        if self.output_folder:
            temp_folder = self.output_folder
            os.makedirs(temp_folder, exist_ok=True)
            self.destination_folder = os.path.join(temp_folder, self.patient_name)
            os.makedirs(self.destination_folder, exist_ok=True)

    def init_ui(self):
        layout = QVBoxLayout()

        # Textbox for entering patient name
        self.patient_name_label = QLabel("Enter Patient ID / Name:", self)
        layout.addWidget(self.patient_name_label)

        self.patient_name_input = QLineEdit(self)
        layout.addWidget(self.patient_name_input)

        self.folder_button = QPushButton("Browse Videos…", self)
        self.folder_button.setToolTip("Choose a folder containing patient videos")
        self.folder_button.setIcon(load_icon("folder_open_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg"))
        self.folder_button.clicked.connect(self.select_folder)
        self.folder_button.setFixedWidth(int(self.width() * 2))
        layout.addWidget(self.folder_button)

        self.video_list = QWidget(self)
        self.video_list.setMinimumHeight(250)
        self.video_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_list.setLayout(QVBoxLayout())
        self.video_list.layout().setContentsMargins(0, 0, 0, 0)
        self.video_browser = VideoBrowser()
        self.video_list.layout().addWidget(self.video_browser)
        layout.addWidget(self.video_list, 1)

        
        button_layout = QHBoxLayout()

        self.select_button = QPushButton("Confirm Patient Videos", self)
        self.select_button.setToolTip("Lock in this patient’s videos")
        self.select_button.setIcon(load_icon("playlist_add_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg"))
        self.select_button.clicked.connect(self.copy_selected_videos)
        button_layout.addWidget(self.select_button)

        self.add_button = QPushButton("Add new patient", self)
        self.add_button.setIcon(load_icon("person_add_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg"))
        self.add_button.clicked.connect(self.add_new_patient)

        button_layout.addWidget(self.add_button)

        layout.addLayout(button_layout)

        self.patient_list_label = QLabel("Ready to Process", self)
        layout.addWidget(self.patient_list_label)

        self.name_list = QListWidget(self)
        self.name_list.setFixedHeight(150)
        layout.addWidget(self.name_list)

        button_layout1 = QHBoxLayout()

        self.process_button = QPushButton("Start Processing", self)
        self.process_button.setToolTip("Begin de‑identification and merging")
        self.process_button.setIcon(load_icon("play_circle_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg"))
        self.process_button.clicked.connect(self.merge_files)
        self.process_button.setFixedWidth(int(self.width() * 2))
        button_layout1.addWidget(self.process_button)

        self.terminate_button = QPushButton("Terminate Processing", self)
        self.terminate_button.setIcon(load_icon("stop_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg"))
        self.terminate_button.setToolTip("Stop processing and reset application")
        self.terminate_button.clicked.connect(self.terminate_confirmation_dialog)
        self.terminate_button.setFixedWidth(int(self.width() * 2))
        button_layout1.addWidget(self.terminate_button)
        self.terminate_button.setEnabled(False)

        self.reset_button = QPushButton("Reset", self)
        self.reset_button.setToolTip("Reset the application to its initial state")
        self.reset_button.setIcon(load_icon("refresh_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg"))
        self.reset_button.clicked.connect(self.reset_application)
        self.reset_button.setFixedWidth(int(self.width() * 2))
        button_layout1.addWidget(self.reset_button)

        button_layout1.addStretch()

        self.remove_patient_button = QPushButton("Delete Patient Entry", self)
        self.remove_patient_button.setToolTip("Remove this patient from the queue")
        self.remove_patient_button.setIcon(load_icon("cancel_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg"))
        self.remove_patient_button.clicked.connect(self.remove_patient_confirmation_dialog)
        self.remove_patient_button.setFixedWidth(int(self.width() * 2))
        button_layout1.addWidget(self.remove_patient_button)

        layout.addLayout(button_layout1)

        self.progress_label = QLabel("", self)
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)
        self.setWindowTitle("Endoshare")
        self.resize(700, 600)
        self.move(100, 100)


    def reset_application(self):
        # Reset input fields and clear video list
        self.selected_folder = ""
        # self.destination_folder = ""
        self.patient_name = ""
        self.video_items = []
        self.video_dict = {}
        self.patient_name_input.clear()
        self.selected_videos.clear()
        self.video_dict.clear()
        self.name_list.clear()
        self.progress_label.clear()
        self.progress_bar.reset()
        self.folder_button.setEnabled(True)
        self.patient_name_input.setEnabled(True)
        self.select_button.setEnabled(True)
        self.add_button.setEnabled(True)
        self.remove_patient_button.setEnabled(True)
        self.terminate_button.setEnabled(False)
        self.process_button.setEnabled(False)
        if hasattr(self, 'video_browser'):
            # remove any model on the tree
            empty_model = QFileSystemModel()
            empty_model.setRootPath('')
            self.video_browser.tree_view.setModel(empty_model)
            self.video_browser.selected_videos_list.clear()
        self.load_settings()

        # Reset threads
        if hasattr(self, 'video_copy_thread') and self.video_copy_thread is not None and self.video_copy_thread.isRunning():
            self.video_copy_thread.quit()
            self.video_copy_thread.wait()
        if hasattr(self, 'video_process_thread') and self.video_process_thread is not None and self.video_process_thread.isRunning():
            self.video_process_thread.quit()
            self.video_process_thread.wait()

    def load_settings(self):
        try:
            with open(resource_path('settings.json'), 'r') as f:
                settings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.error("Settings file missing or invalid – using empty defaults")
            settings = {}

        # now pull the right keys
        self.local_folder  = settings.get('local_folder_path', '')
        self.shared_folder = settings.get('shared_folder_path', '')
    
    
    def add_new_patient(self):
        # Reset patient name
        self.patient_name = self.patient_name_input.text()
        self.patient_name_input.clear()
        self.video_browser.selected_videos_list.clear()

        # Clear the selected videos list
        self.selected_videos.clear()
        
        self.progress_label.clear()
        self.progress_bar.reset()

        # Check if the video_copy_thread is not None
        if self.video_copy_thread:
            existing_patient_data = self.video_copy_thread.get_video_dict()
            if existing_patient_data:
                self.video_dict[self.patient_name] = existing_patient_data
                # Patient is added to name_list and color is set as red
                item = QListWidgetItem(self.patient_name)
                item.setForeground(QColor("red"))
                self.name_list.addItem(item)
        self.folder_button.setEnabled(True)
        self.patient_name_input.setEnabled(True)
        self.select_button.setEnabled(True)

            
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if folder is not None:
            self.selected_folder = folder
            self.update_video_list()
       
    
    def update_video_list(self):
        if not self.selected_folder:
            self.progress_label.setText("Please enter a valid folder.")
            return
        self.video_browser.populate_videos(self.selected_folder)

    def _on_processing_error(self, message: str):
    # 1) show a critical dialog with the error
        QMessageBox.critical(
            self,
            "Processing Error",
            ("An unexpected error occurred during video processing:\n\n"
            f"{message}\n\n"
            "The application will reset so you can try again.")
        )
        # 2) reset UI state so the user can re‑start
        self.reset_application()
    
    def _gather_resolutions(self, paths: list[str]) -> dict[str, tuple[int, int] | None]:
        """
        Return {path: (w,h)} for every video.
        If a file cannot be opened, its value is None.
        """
        resolutions = {}
        for p in paths:
            cap = cv2.VideoCapture(p)
            if not cap.isOpened():
                resolutions[p] = None          # unreadable → flag as None
            else:
                res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                resolutions[p] = res
            cap.release()
        return resolutions

        
    def copy_selected_videos(self):
        # Video Copy Thread is called when they confirm selected videos#
        if not self.selected_folder:
            return
        self.patient_name = self.patient_name_input.text()

        if not self.patient_name:
            self.progress_label.setText("Please enter a patient name.")
            return
        self.folder_button.setEnabled(False)
        self.patient_name_input.setEnabled(False)
        self.select_button.setEnabled(False)
        self.process_button.setEnabled(True)
        video_list = self.video_browser.get_selected_video_list()
        pending      = [video_list.item(i).data(Qt.UserRole) for i in range(video_list.count())]

        # 🚦 collect resolutions
        res_map      = self._gather_resolutions(pending)
        unique_sizes = {r for r in res_map.values() if r is not None}

        if len(unique_sizes) > 1 or None in res_map.values():
            # build a pretty, multiline report
            lines = []
            for p, res in res_map.items():
                name = Path(p).name
                if res is None:
                    lines.append(f"{name}: unreadable ❌")
                else:
                    lines.append(f"{name}: {res[0]} × {res[1]}")
            detail = "\n".join(lines)

            QMessageBox.warning(
                self,
                "Resolution mismatch",
                ("The selected videos do not share the same resolution "
                "and cannot be processed together.\n\n"
                "<pre>{}</pre>").format(detail)
            )
            self.folder_button.setEnabled(True)
            self.select_button.setEnabled(True)
            return 
        if not video_list:
            self.progress_bar.reset()
            self.progress_label.setText("Please add videos to continue!")
            self.select_button.setEnabled(True)
            return
        for index in range(video_list.count()):
            item = video_list.item(index)
            self.selected_videos.append(item.data(Qt.UserRole))

        logger.info("selected_videos", len(self.selected_videos))
        
        # Initialize the video_copy_thread instance with selected videos
        self.video_copy_thread = VideoCopyThread(self.selected_videos, self.selected_folder)
        self.video_copy_thread.update_progress.connect(self.update_progress)

        # Start the thread
        self.video_copy_thread.start()
        if not self.patient_name:
            self.progress_label.setText("Please enter a patient name.")
            return
        self.video_dict[self.patient_name] = self.video_copy_thread.get_video_dict()
        # Patient is added to name_list and color is set as red
        n_item = QListWidgetItem(self.patient_name)
        n_item.setForeground(QColor("red"))
        self.name_list.addItem(n_item)
        
        #Clean up after connecting
        self.video_copy_thread.finished.connect(self.cleanup_after_copy)
        if not self.selected_folder:
            return
        self.patient_name = self.patient_name_input.text()

       
    def cleanup_after_copy(self):
        # This method will be called when the video_copy_thread finishes
        self.add_button.setEnabled(True)
        self.video_copy_thread.quit()
        self.video_copy_thread.wait()

        self.video_copy_thread = None

    def merge_files(self):
        
        logger.info(f"Processing started for {len(self.video_dict)} patients.")# Check if there are any videos to process
        # Keep track of the total number of process threads
        if self.video_dict == {}:
            self.progress_bar.reset()
            self.progress_label.setText("Please add videos to process.")
            return
        if not self.shared_folder:
            self.progress_bar.reset()
            self.progress_label.setText("Please add destination folders in the settings.")
            return
        if not self.local_folder:
            self.progress_bar.reset()
            self.progress_label.setText("Please add destination folders in the settings.")
            return
    
        self.video_process_thread = VideoProcessThread(self.video_dict,
                                                       self.shared_folder,
                                                       self.local_folder,
                                                       **self.controller.runtime_settings,
                                                       )
        self.select_button.setEnabled(False)
        self.add_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.terminate_button.setEnabled(True)
        self.remove_patient_button.setEnabled(False)
        self.process_button.setEnabled(False)
        self.video_process_thread.update_progress.connect(self.update_progress)
        self.video_process_thread.update_color.connect(self.update_color)
        self.video_process_thread.error.connect(self._on_processing_error)

        self._progress_start = time.time()
        self.video_process_thread.start()
        self.video_process_thread.finished.connect(self.on_process_thread_finished)
        


    def on_process_thread_finished(self):
        # Increment the count of finished threads
        self.finished_threads += 1
        self.reset_button.setEnabled(True)
        
        # Check if all threads have finished
        if self.finished_threads == self.total_threads:
            # All threads have finished, perform cleanup
            self.cleanup_after_merge()

    def cleanup_after_merge(self):
        # This method will be called when the video_merge_thread finishes
        self.process_button.setEnabled(True)
        self.video_process_thread.quit()
        self.video_process_thread.wait()
        self.video_process_thread = None

    def update_color(self, item_text, color):
        # Find the item with the specified text
        items = self.name_list.findItems(item_text, Qt.MatchExactly)
        if items:
            for item in items:
                # Update the item's text color
                item.setForeground(QColor(color))


    def update_progress(self, current, total, message, is_copying=True):
        # compute raw pct
        if total > 0:
            pct = int((current / total) * 100)
        else:
            pct = 0

        # clamp to [0,100]
        pct = max(0, min(100, pct))

        # update bar
        self.progress_bar.setValue(pct)

        if is_copying:
            self.progress_label.setText(message)
        else:
            # init timer if needed
            if not hasattr(self, "_progress_start") or self._progress_start is None:
                self._progress_start = time.time()
            now     = time.time()
            elapsed = now - self._progress_start

            if current > 0:
                total_est = elapsed * (total / current)
                eta       = total_est - elapsed
                eta_str   = str(datetime.timedelta(seconds=int(eta)))
                text = f"{message}\n{pct}%  •  ETA: {eta_str}"
            else:
                text = f"{message}\n{pct}%"

            self.progress_label.setText(text)

        # clear the timer when done
        if current >= total:
            self._progress_start = None


    
    def set_shared_folder(self, folder_path):
        self.shared_folder = folder_path
    
    def get_shared_folder(self):
        return self.shared_folder
    
    def set_local_folder(self, folder_path):
        self.local_folder = folder_path
    
    def get_local_folder(self):
        return self.local_folder
    
    def terminate_confirmation_dialog(self):
        confirmation = QMessageBox.question(
            self,
            "Confirmation",
            "Are you sure you want to terminate the process?",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirmation != QMessageBox.Yes:
            logger.info("Process continuation confirmed")
            return

        logger.info("Process termination requested")
        thread = self.video_process_thread
        if not thread or not thread.isRunning():
            return

        # politely flag interruption
        thread.requestInterruption()

        def _do_terminate():
            # stop any pending timer
            if hasattr(self, "_term_timer"):
                self._term_timer.stop()
                del self._term_timer
            thread.terminate()
            self._cleanup_after_termination()

        # see if ffmpeg is _right now_ in the process list
        ffmpegs = [
            p for p in psutil.process_iter(['name', 'cmdline'])
            if p.info['name'] == 'ffmpeg'
               or ('ffmpeg' in ' '.join(p.info.get('cmdline') or []))
        ]

        if ffmpegs:
            # case A: it's already running → kill immediately
            _do_terminate()
        else:
            # case B: no ffmpeg yet → wait for the _next_ spawn
            QMessageBox.information(
                self,
                "Termination Deferred",
                "I'll terminate as soon as the next encoding job starts."
            )
            self.progress_label.setText("Waiting to terminate the process...")
            self._term_timer = QTimer(self)
            self._term_timer.setInterval(1000)
            def _poll_for_ffmpeg():
                # once we see ffmpeg, do the kill
                for p in psutil.process_iter(['name', 'cmdline']):
                    if p.info['name'] == 'ffmpeg' \
                       or ('ffmpeg' in ' '.join(p.info.get('cmdline') or [])):
                        _do_terminate()
                        return
            self._term_timer.timeout.connect(_poll_for_ffmpeg)
            self._term_timer.start()

    def _cleanup_after_termination(self):
        # 1) remove any deid temp‐folders
        for patient_id, vid_map in self.video_process_thread.video_in_root_dir.items():
            # vid_map keys are full paths to each original video
            for orig_path in vid_map.keys():
                parent = Path(orig_path).parent
                for entry in parent.iterdir():
                    if entry.is_dir() and entry.name.startswith("tmp_"):
                        try:
                            shutil.rmtree(entry)
                            logger.info(f"Removed temp folder {entry}")
                        except Exception:
                            logger.exception(f"Failed to remove temp folder {entry}")

        # 2) now do your normal UI cleanup
        self.patient_name_input.clear()
        self.video_dict.clear()
        self.name_list.clear()
        self.progress_label.clear()
        self.progress_bar.reset()
        self.progress_label.setText("Process terminated")
        self.terminate_button.setEnabled(False)
    
    def remove_patient_confirmation_dialog(self):
        confirmation = QMessageBox.question(self, "Confirmation", "Are you sure you want to remove the patient?",
                                            QMessageBox.Yes | QMessageBox.No)
        if confirmation == QMessageBox.Yes:
            selected_item = self.name_list.currentItem()
            if selected_item:
                selected_patient_name = selected_item.text()
                if selected_patient_name in self.video_dict:
                    del self.video_dict[selected_patient_name]
                self.name_list.takeItem(self.name_list.row(selected_item))
                logger.info("Patient removed")
                self.progress_bar.reset()
                self.progress_label.setText("Patient removed.")
            else:
                logger.info("No patient selected")
                self.progress_bar.reset()
                self.progress_label.setText("No patient selected.")
        else:
            logger.info("Patient removal cancelled")

class AppSettings(QWidget):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.controller = controller
        self.videomerger = self.controller.access_video_merger_frame()

        # MAIN LAYOUT
        layout = QVBoxLayout(self)

        # DIRECTORIES FORM
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

        # --- ARCHIVE MODE ROW WITH INFO ICON ---
        archive_mode_row = QHBoxLayout()
        archive_mode_row.setContentsMargins(0, 0, 0, 0)
        archive_mode_row.setSpacing(4)  # tighten gap
        self.purge_checkbox = QCheckBox("Archive Mode")  # name kept for logic
        self.purge_checkbox.setChecked(True)  # ON by default
        self.purge_checkbox.stateChanged.connect(self._on_archive_mode_toggled)
        self.purge_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        archive_mode_row.addWidget(self.purge_checkbox)

        # info icon as hover-only label
        info_icon_lbl = QLabel()
        info_pix = load_icon("info_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg").pixmap(QSize(16, 16))
        info_icon_lbl.setPixmap(info_pix)
        info_icon_lbl.setToolTip(
            "Archive Mode ON: anonymized + ID copy \n"
            "Archive Mode OFF: only anonymized output"
        )
        info_icon_lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        archive_mode_row.addWidget(info_icon_lbl)
        form.addRow(archive_mode_row)


        # --- Archive (Raw Video Repository) row ---
        self.local_folder_entry = QLineEdit()
        self.local_folder_entry.setReadOnly(True)
        self.local_folder_entry.setMinimumWidth(400)
        browse_local_btn = QPushButton("Browse…")
        browse_local_btn.setToolTip("Select Archive Video Repository folder")
        browse_local_btn.setIcon(load_icon("folder_open_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg"))
        browse_local_btn.clicked.connect(self.select_folder)
        self.archive_usage_label = QLabel()
        self._archive_row_widgets = [self.local_folder_entry, browse_local_btn, self.archive_usage_label]  # for show/hide

        local_row = QHBoxLayout()
        local_row.addWidget(self.local_folder_entry)
        local_row.addWidget(browse_local_btn)
        local_row.addWidget(self.archive_usage_label)
        self.archive_repo_label = QLabel("Archive Video Repository:")
        form.addRow(self.archive_repo_label, local_row)


        # --- Export (De-identified Output) row ---
        self.shared_folder_entry = QLineEdit()
        self.shared_folder_entry.setReadOnly(True)
        self.shared_folder_entry.setMinimumWidth(400)
        browse_shared_btn = QPushButton("Browse…")
        browse_shared_btn.setToolTip("Select De-identified Output folder")
        browse_shared_btn.setIcon(load_icon("folder_open_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg"))
        browse_shared_btn.clicked.connect(self.select1_folder)
        self.shared_usage_label = QLabel()
        shared_row = QHBoxLayout()
        shared_row.addWidget(self.shared_folder_entry)
        shared_row.addWidget(browse_shared_btn)
        shared_row.addWidget(self.shared_usage_label)
        form.addRow("De-identified Output:", shared_row)

        # --- SAVE + ARCHIVE MODE ON SAME LINE ---
        save_button = QPushButton("Save Settings")
        save_button.setIcon(load_icon("save_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg"))
        save_button.setEnabled(False)
        save_button.clicked.connect(self.save_settings)

        bottom = QWidget()
        bottom_l = QHBoxLayout(bottom)
        bottom_l.setContentsMargins(0, 0, 0, 0)
        bottom_l.addWidget(self.purge_checkbox)  # keep for layout consistency
        bottom_l.addStretch()
        bottom_l.addWidget(save_button)
        form.addRow("", bottom)

        directories_box = QGroupBox("Directories")
        directories_box.setStyleSheet("""
        QGroupBox {
            font-size: 18px;
            font-weight: 600;
        }
        """)
        directories_box.setLayout(form)

        # --- hook validation ---
        self.local_folder_entry.textChanged.connect(self.archive_entry_changed)
        self.shared_folder_entry.textChanged.connect(self.archive_entry_changed)


        # MODE & SETTINGS

        mode_box = QGroupBox("Processing mode")
        mode_box.setStyleSheet("""
        QGroupBox {
            font-size: 18px;
            font-weight: 600;
        }
        """)
        mode_layout = QVBoxLayout(mode_box)
        mode_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)


        # 1) Combo row
        combo_row = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Fast", "Advanced"])
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.setFixedWidth(120)
        combo_row.addWidget(self.mode_combo)
        combo_row.addStretch()
        mode_layout.addLayout(combo_row)

        # 2) Stacked widget with mode-specific panels
        mode_stack = QStackedWidget()

        # — Fast page
        fast_page = QWidget()
        fast_layout = QVBoxLayout(fast_page)
        fast_label = QLabel(
            "Video merged, de-identified, and anonymized at original FPS & quality for improved speed."
        )
        fast_label.setWordWrap(True)
        fast_layout.addWidget(fast_label)
        mode_stack.addWidget(fast_page)

        # — Advanced page
        adv_page = QWidget()
        adv_form = QFormLayout(adv_page)
        # Framerate slider
        fps_vals = list(range(20, 61, 5))
        fps_slider = slider.LabeledSlider(1, len(fps_vals), labels=list(map(str, fps_vals)))
        fps_slider.sl.setTickInterval(1); fps_slider.sl.setSingleStep(1); fps_slider.sl.setPageStep(1)
        fps_slider.sl.setValue(fps_vals.index(self.controller.runtime_settings["fps"]) + 1)
        fps_slider.sl.valueChanged.connect(
            lambda i: self.controller.runtime_settings.__setitem__("fps", fps_vals[i-1])
        )
        adv_form.addRow("Framerate:", fps_slider)
        # Quality slider
        res_vals = {480:"Low (480p)", 720:"Medium (720p)", 1080:"High (1080p)", -1:"Original"}
        res_slider = slider.LabeledSlider(1, len(res_vals), labels=list(res_vals.values()))
        res_slider.sl.setTickInterval(1); res_slider.sl.setSingleStep(1); res_slider.sl.setPageStep(1)
        res_slider.sl.setValue(
            list(res_vals.keys()).index(self.controller.runtime_settings["resolution"]) + 1
        )
        res_slider.sl.valueChanged.connect(
            lambda i: self.controller.runtime_settings.__setitem__("resolution", list(res_vals.keys())[i-1])
        )
        adv_form.addRow("Quality:", res_slider)
        mode_stack.addWidget(adv_page)

        mode_layout.addWidget(mode_stack)

        # 3) Wire combo → stacked pages + runtime_settings
        self.mode_combo.currentIndexChanged.connect(lambda idx: (
            mode_stack.setCurrentIndex(idx),
            self.controller.runtime_settings.__setitem__(
                "mode",
                ProcessingMode.NORMAL if idx == 0 else ProcessingMode.ADVANCED
            )
        ))

        # Replace in your main layout:
        layout.addWidget(directories_box)
        layout.addSpacing(20)
        layout.addWidget(mode_box)
        layout.addStretch(1)
    
    def _on_archive_mode_toggled(self, state):
        archive_on = bool(state)  # checked means Archive Mode active
        # inverted: archive_on=True => do not purge; archive_on=False => purge
        self.controller.runtime_settings["purge_after"] = not archive_on

        # show/hide archive repository widgets
        for w in getattr(self, "_archive_row_widgets", []):
            w.setVisible(archive_on)

        # if archive mode is off, mirror shared into local under the hood
        if not archive_on:
            self.local_folder_entry.setText(self.shared_folder_entry.text())

        # Re-validate enabling logic
        self.archive_entry_changed()
        self.archive_repo_label.setVisible(archive_on)


    def archive_entry_changed(self):
        arch = self.local_folder_entry.text()
        exp = self.shared_folder_entry.text()
        valid_arch = os.path.isdir(arch)
        valid_exp = os.path.isdir(exp)

        if valid_arch:
            usage = shutil.disk_usage(arch)
            free_gb = usage.free // (1024**3)
            self.archive_usage_label.setText(f"{free_gb} GB free")
        else:
            self.archive_usage_label.setText("Invalid directory" if self.purge_checkbox.isChecked() else "")

        if valid_exp:
            usage = shutil.disk_usage(exp)
            free_gb = usage.free // (1024**3)
            self.shared_usage_label.setText(f"{free_gb} GB free")
        else:
            self.shared_usage_label.setText("Invalid directory")

        # Save logic: Archive Mode ON requires both, OFF only export
        if self.purge_checkbox.isChecked():  # Archive Mode ON
            save_enabled = valid_arch and valid_exp
        else:
            save_enabled = valid_exp

        for w in self.findChildren(QPushButton):
            if w.text() == "Save Settings":
                w.setEnabled(save_enabled)
                break

    def select_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Raw Video Repository folder")
        if path:
            self.local_folder_entry.setText(path)

    def select1_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select De-identified Output folder")
        if path:
            self.shared_folder_entry.setText(path)

    def save_settings(self):
        cfg = {
            'local_folder_path': self.local_folder_entry.text(),
            'shared_folder_path': self.shared_folder_entry.text(),
            'purge_after': self.purge_checkbox.isChecked() == False  # inverted: Archive Mode ON => purge_after=False
        }
        # If Archive Mode is OFF, mirror de-id output into both
        if not self.purge_checkbox.isChecked():
            cfg['local_folder_path'] = cfg['shared_folder_path']
            self.local_folder_entry.setText(cfg['local_folder_path'])

        with open(resource_path('settings.json'), 'w') as f:
            json.dump(cfg, f)
        # push into running frame: both get shared_folder if archive mode off
        self.videomerger.set_local_folder(cfg['local_folder_path'])
        self.videomerger.set_shared_folder(cfg['shared_folder_path'])
        QMessageBox.information(self, "Settings Saved", "Your directories have been updated.")


    def load_settings(self):
        try:
            cfg = json.load(open(resource_path('settings.json')))
        except (FileNotFoundError, json.JSONDecodeError):
            cfg = {}
        # populate fields
        self.local_folder_entry.setText(cfg.get('local_folder_path',''))
        self.shared_folder_entry.setText(cfg.get('shared_folder_path',''))
        self.purge_checkbox.setChecked(cfg.get('purge_after', False))
            
class Info(QWidget):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Formatted text
        text = """
        <h1>Endoshare: Streamlining Clinical Video Compliance and Accessibility</h1>
        <p><b>Introduction:</b><br>
        Endoscopic and laparoscopic procedures generate invaluable video recordings for clinical review, research, and education. However, the sharing and utilization of such videos present significant challenges, primarily due to stringent Institutional Review Board (IRB) regulations and the inherent sensitivity of patient data contained within these recordings. Endoshare addresses these challenges by providing a comprehensive solution for ensuring compliance with IRB regulations while facilitating seamless access to surgical videos for medical professionals.</p>
        
        <h2>Key Features:</h2>
        <ol>
            <li><b>Video Integration:</b><br>
            Endoshare seamlessly integrates video files sourced from operating room systems, overcoming the fragmentation often encountered in traditional Operating Room Information Technology (ORIT) systems. By consolidating fragmented recordings into a single cohesive file, Endoshare streamlines the video preparation process, saving valuable time for clinicians.</li>

            <li><b>Deep Learning Image Classification:</b><br>
            Utilizing cutting-edge deep learning technology, Endoshare employs an advanced image classifier to identify and blur sensitive patient information within surgical videos. This innovative approach ensures compliance with privacy regulations while preserving the educational and research value of the video content. The user-friendly interface of Endoshare empowers clinicians to effortlessly apply these privacy-enhancing measures, even without specialized technical expertise.</li>

            <li><b>Metadata Removal:</b><br>
            In addition to image-based privacy protection, Endoshare incorporates robust metadata removal capabilities to further safeguard patient privacy. By eliminating metadata associated with surgical videos, Endoshare ensures that no identifiable patient information remains embedded within the video files. This comprehensive approach to data anonymization enables clinicians to confidently utilize deidentified videos for research, educational, and training purposes.</li>
        </ol>

        <h2>Workflow:</h2>
        <p>Endoshare simplifies the video preparation process into three intuitive steps:</p>
        <ol>
            <li><b>Video Merging:</b><br>
            Clinicians can easily merge video files obtained from ORIT systems within Endoshare, consolidating fragmented recordings into a unified format.</li>

            <li><b>Image Classification and Privacy Enhancement:</b><br>
            Endoshare's deep learning-based image classifier automatically identifies and blurs sensitive patient information within the merged video. This step ensures compliance with privacy regulations while preserving the clinical relevance of the video content.</li>

            <li><b>Metadata Removal:</b><br>
            Endoshare removes metadata associated with the surgical videos, further enhancing patient privacy and compliance with regulatory requirements.</li>
        </ol>

        <p><b>Conclusion:</b><br>
        With Endoshare, clinicians can confidently prepare surgical videos for research, education, and training purposes while ensuring compliance with IRB regulations and protecting patient privacy. By streamlining the process of video compliance and accessibility, Endoshare empowers medical professionals to leverage the full educational and research potential of surgical video recordings.</p>
        
        <p><b>Contact:</b><br>
        For assistance, please contact: lorenzo.arboit@ext.ihu-strasbourg.eu</p>

        <p><b>Licensing:</b><br>
        This code is available for non-commercial scientific research purposes as defined in the CC BY-NC-SA 4.0.
        By downloading and using this code you agree to the terms in the LICENSE.
        Third-party codes are subject to their respective licenses.</p>

        <p><b>Developed by:</b><br>
        Research Group CAMMA, IHU Strasbourg, University of Strasbourg<br>
        Website: <a href="http://camma.u-strasbg.fr">http://camma.u-strasbg.fr</a></p>
        """

        # Create QLabel with HTML formatted text
        label = QLabel()
        label.setText(text)
        label.setWordWrap(True)

        # Create QScrollArea and set QLabel as its widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        scroll_area.setWidget(label)

        layout.addWidget(scroll_area)
        self.setLayout(layout)

class Help(QWidget):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Formatted text
        text = """
        <h1>Using Endoshare: A Guide</h1>
        <p><b>Settings Window:</b><br>
        In the settings window, you can set your preferred directories:</p>
        <ul>
            <li><b>Local Directory:</b> Must be a folder on your computer. Videos stored here contain the patient’s name/ID as their name. Endoshare organizes these recordings into a dataset, merging videos and blurring out-of-body frames. Additionally, a spreadsheet is saved in this directory, containing associations between patient name/ID and deidentified alphanumeric codes.</li>
            <li><b>Shared Directory:</b> Can be a local folder or a cloud folder used by your team to safely share files with research partners. Files saved here are free of patient data and ready to use.</li>
        </ul>

        <p><b>Home Window:</b><br>
        The home window is the main interface for completing video uploads:</p>
        <ul>
            <li>Enter patient’s name/ID.</li>
            <li>Select the folder containing videos extracted from the Operating Room Information Technology system using the 'Select video folder' button.</li>
            <li>A small window with two columns appears. The left side shows the selected folder and the right side shows the selected videos.</li>
            <li>Double-click to select patient videos and the selected videos will appear on the right side.</li>
            <li>Rearrange their order if necessary via drag and drop. Preview videos for better visualization.</li>
            <li>Click 'Confirm selected videos' to create a new patient file, appearing in red in the 'Ready to Process' widget. Add more patients using 'Add new patient'.</li>
            <li>In case of an error, remove the wrong patient with the specific button<li>
            <li>Click 'Process Files' to start processing. Terminate processing anytime with 'Terminate Processing'.</li>
            <li>Reset to start a new process with new patients.</li>
        </ul>

        <p><b>Guided Operations:</b><br>
        Some buttons are disabled during specific operations to guide users through different steps.</p>

        <p><b>Video Tutorial:</b><br>
        You can find a video tutorial that explains the full procedure down here.</p>
        """

        # Create QLabel with HTML formatted text
        label = QLabel()
        label.setText(text)
        label.setWordWrap(True)

        # Create QScrollArea and set QLabel as its widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")  # Remove border
        scroll_area.setWidget(label)

        layout.addWidget(scroll_area)
        self.setLayout(layout)

if __name__ == "__main__":
    if mp.current_process().name == "MainProcess":
        # Enable Qt’s automatic screen scaling
        QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)
        os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

        # 2) Create the application
        app = QApplication(sys.argv)

        # 3) Load & scale the splash pixmap
        raw_pix = QPixmap(resource_path("icons/splash.png"))
        splash_size = QSize(400, 300)  # or whatever final size you want
        pix = raw_pix.scaled(splash_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 4) Create a frameless, always‑on‑top splash
        splash = QSplashScreen(pix,
            Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )

        # 5) Center it on the primary screen
        screen_geo = app.primaryScreen().availableGeometry()
        splash_geo  = splash.geometry()
        x = screen_geo.left() + (screen_geo.width()  - splash_geo.width()) // 2
        y = screen_geo.top()  + (screen_geo.height() - splash_geo.height())// 2
        splash.move(x, y)

        # 6) Show it and force Qt to draw it immediately
        splash.show()
        app.processEvents()

        # 7) Construct & show your main window
        main_app = MainApp()
        main_app.show()

        # 8) Now that everything is up, close the splash
        splash.finish(main_app)

        # 9) Enter the event loop
        sys.exit(app.exec_())
