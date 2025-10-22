import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QSizePolicy,
    QTreeView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QFileSystemModel,
    QHeaderView,
)
import platform
import subprocess

from ..utils.resources import load_icon
from loguru import logger

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.mpeg', '.mpg', '.ts', '.m2ts']

class VideoBrowser(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Browser")
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

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

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(avail_group)
        splitter.addWidget(sel_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_selected_button)
        button_layout.addWidget(self.add_all_button)
        button_layout.addWidget(self.preview_button)
        button_layout.addWidget(self.remove_button)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(splitter, 1)
        main_layout.addLayout(button_layout, 0)
        self.setLayout(main_layout)

        self.selected_videos_list.itemClicked.connect(self.on_item_clicked)
        self.selected_videos_list.clicked.connect(self.update_last_clicked_item)
        self.tree_view.doubleClicked.connect(self.add_video_from_tree)
        self.tree_view.clicked.connect(self.update_last_clicked_item)

        self.last_clicked_item = None

    def populate_videos(self, folder):
        model = QFileSystemModel()
        model.setNameFilters([f"*{ext}" for ext in VIDEO_EXTENSIONS])
        model.setNameFilterDisables(False)
        model.setRootPath(folder)
        self.tree_view.setModel(model)
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
                item.setData(Qt.UserRole, file_path)
                self.selected_videos_list.addItem(item)

    def add_single_video(self):
        indexes = self.tree_view.selectionModel().selectedIndexes()
        for idx in indexes:
            if idx.column() == 0:
                self.add_video_from_tree(idx)
                break

    def add_all_videos(self):
        all_videos = []
        root_path = self.tree_view.model().rootPath()
        for root, dirs, files in os.walk(root_path):
            for file in files:
                file_path = os.path.join(root, file)
                if (os.path.splitext(file_path)[1].lower() in VIDEO_EXTENSIONS
                    and not os.path.basename(file_path).startswith('._')
                ):
                    all_videos.append(file_path)
        all_videos.sort()
        self.selected_videos_list.clear()
        for file_path in all_videos:
            file_name = os.path.basename(file_path)
            item = QListWidgetItem(file_name)
            item.setData(Qt.UserRole, file_path)
            self.selected_videos_list.addItem(item)

    def remove_selected_videos(self):
        for item in self.selected_videos_list.selectedItems():
            self.selected_videos_list.takeItem(self.selected_videos_list.row(item))

    def on_item_clicked(self, item):
        self.selected_videos_list.clearSelection()
        item.setSelected(True)

    def update_last_clicked_item(self, index):
        if isinstance(self.sender(), QTreeView):
            self.last_clicked_item = self.tree_view.model().filePath(index)
        elif isinstance(self.sender(), QListWidget):
            item = self.selected_videos_list.currentItem()
            if item:
                self.last_clicked_item = item.data(Qt.UserRole)

    def preview_video(self):
        file_path = self.last_clicked_item
        if file_path and os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in VIDEO_EXTENSIONS:
            logger.info(f"Previewing {file_path} in System Video Player ...")
            if platform.system() == "Darwin":
                subprocess.run(["open", file_path], check=True)
            elif platform.system() == "Windows":
                subprocess.run(["start", file_path], check=True)
            else:
                subprocess.run(["xdg-open", file_path], check=True)

    def get_selected_video_list(self):
        return self.selected_videos_list
