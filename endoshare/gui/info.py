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

