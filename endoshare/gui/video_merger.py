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

