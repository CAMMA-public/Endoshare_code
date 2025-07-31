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
from uuid import uuid4
from copy import deepcopy
from pathlib import Path

import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from loguru import logger
from vidgear.gears import WriteGear

from ..processing import deid
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
from PyQt5.QtGui import (
    QIcon,
    QPen,
    QPixmap,
    QFontDatabase,
    QColor,
    QKeySequence,
    QPainter,
)
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
import multiprocessing as mp

from ..utils.resources import (
    resource_path,
    load_icon,
    tinted_icon,
    ICON_COLORS,
    FFMPEG_BIN,
    FFPROBE_BIN,
)
from ..utils.types import ProcessingMode, ProcessingInterrupted
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

