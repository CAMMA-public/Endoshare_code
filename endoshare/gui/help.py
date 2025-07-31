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

