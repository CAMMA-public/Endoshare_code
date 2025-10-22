import os
import json
import shutil

from PyQt5.QtCore import (
    QSize,
    Qt,
    pyqtSignal,
)

from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QSizePolicy,
    QStackedWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QLabel,
    QLineEdit,
    QCheckBox,
    QMessageBox,
)
from .slider import LabeledSlider

from ..utils.resources import (
    resource_path,
    load_icon,
)
from ..utils.types import ProcessingMode

class AppSettings(QWidget):
    mode_changed = pyqtSignal(str)  # emits "Fast" or "Advanced"
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
        fps_slider = LabeledSlider(1, len(fps_vals), labels=list(map(str, fps_vals)))
        fps_slider.sl.setTickInterval(1); fps_slider.sl.setSingleStep(1); fps_slider.sl.setPageStep(1)
        fps_slider.sl.setValue(fps_vals.index(self.controller.runtime_settings["fps"]) + 1)
        fps_slider.sl.valueChanged.connect(
            lambda i: self.controller.runtime_settings.__setitem__("fps", fps_vals[i-1])
        )
        adv_form.addRow("Framerate:", fps_slider)
        # Quality slider
        res_vals = {480:"Low (480p)", 720:"Medium (720p)", 1080:"High (1080p)", -1:"Original"}
        res_slider = LabeledSlider(1, len(res_vals), labels=list(res_vals.values()))
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
            ),
            self.mode_changed.emit("Fast" if idx == 0 else "Advanced")
        ))

        # Replace in your main layout:
        layout.addWidget(directories_box)
        layout.addSpacing(20)
        layout.addWidget(mode_box)
        layout.addStretch(1)

        # initialize from runtime settings (in case loaded earlier)
        rt = self.controller.runtime_settings
        self.local_folder_entry.setText(rt.get("local_folder_path", ""))
        self.shared_folder_entry.setText(rt.get("shared_folder_path", ""))
        self.purge_checkbox.setChecked(not rt.get("purge_after", False))
        self.archive_entry_changed()

    
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
        
        self.controller.runtime_settings["local_folder_path"] = cfg["local_folder_path"]
        self.controller.runtime_settings["shared_folder_path"] = cfg["shared_folder_path"]
        self.controller.runtime_settings["purge_after"] = cfg["purge_after"]

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
            
