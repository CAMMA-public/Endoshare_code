from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QTextBrowser,
    QFrame,
)
from PyQt5.QtCore import Qt

class Info(QWidget):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.project_url = "http://arbua.github.io/endoshare"
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setReadOnly(True)
        self.text_browser.setFrameStyle(0)
        self.text_browser.setAcceptRichText(True)
        self.text_browser.setText(self._generate_html())

        # Blend with parent/app background
        self.text_browser.setStyleSheet("QTextBrowser { background: transparent; }")

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.text_browser)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_area.setFrameShape(QFrame.NoFrame)

        layout.addWidget(scroll_area)
        self.setLayout(layout)
        self.setAttribute(Qt.WA_StyledBackground, True)

    def set_project_url(self, url: str):
        self.project_url = url
        self.text_browser.setText(self._generate_html())

    def _generate_html(self) -> str:
        return f"""
        <style>
          .info-container {{
            font-family: system-ui,-apple-system,BlinkMacSystemFont,sans-serif;
            max-width: 900px;
            margin: 0;
            line-height: 1.45;
            color: #1f2d3d;
          }}
          h1 {{
            font-size: 1.8em;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 4px;
            margin-bottom: 8px;
          }}
          h2 {{
            font-size: 1.4em;
            margin-top: 1.2em;
            margin-bottom: 6px;
          }}
          ul, ol {{
            padding-left: 1.2em;
            margin-top: 4px;
          }}
          .badge {{
            display: inline-block;
            background: #f0f5fb;
            color: #1f2d3d;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: 600;
            margin-left: 4px;
          }}
          .inline-code {{
            background: #f7f9fc;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: Menlo, Consolas, monospace;
            font-size: 0.9em;
          }}
          a {{
            color: #1a73e8;
            text-decoration: none;
          }}
          a:hover {{
            text-decoration: underline;
          }}
          .small {{
            font-size: 0.9em;
            color: #555;
          }}
          .section-box {{
            padding: 10px 14px;
            border-radius: 6px;
            background: #f7f9fc;
            margin-bottom: 10px;
          }}
          .footer {{
            margin-top: 1.5em;
            font-size: 0.85em;
            color: #555;
          }}
        </style>

        <div class="info-container" aria-label="Endoshare Information">
          <h1>What Endoshare Does</h1>

          <div class="section-box">
            <h2>Core Function</h2>
            <p>Endoshare prepares endoscopic and laparoscopic video recordings for use in research, education, and collaboration by automating consolidation and de-identification while preserving clinical utility.</p>
          </div>

          <div class="section-box">
            <h2>Main Capabilities</h2>
            <ul>
              <li><strong>Video Consolidation:</strong> Automatically merges fragmented recordings from operating room systems into a single, coherent file to eliminate manual stitching.</li>
              <li><strong>Visual Privacy Filtering:</strong> Applies a deep learning classifier to detect and blur sensitive visual content (e.g., patient identifiers or out-of-body regions), ensuring privacy without manual intervention.</li>
              <li><strong>Metadata Sanitization:</strong> Strips embedded metadata from video files, removing hidden identifiers and reinforcing de-identification.</li>
              <li><strong>Separation of Raw and Shared Data:</strong> Maintains original/source videos locally (or archived) while producing de-identified outputs safe for sharing with collaborators.</li>
              <li><strong>Audit-ready Logging:</strong> Tracks processing sessions to support traceability and compliance review.</li>
            </ul>
          </div>

          <div class="section-box">
            <h2>How It Fits into Clinical Workflows</h2>
            <p>Clinicians or research staff point Endoshare to source video folders, specify patient identifiers, and confirm selections. The system then merges, anonymizes visually and via metadata, and exports ready-to-use de-identified video packages, isolating sensitive information from sharable outputs.</p>
          </div>

          <div class="section-box">
            <h2>Privacy</h2>
            <p>Endoshare enforces a privacy-first pipeline: outputs are de-identified both visually and in metadata. Archive videos remain local, and they report patient name/ID for traceability. Processing activity is logged to support audit.</p>
            <p class="small">Best practice: avoid concurrent modifications of input directories and retain versioned backups of raw data.</p>
          </div>

          <div class="section-box">
            <h2>Support & Attribution</h2>
            <p>Project website: <a href="{self.project_url}" target="_blank">{self.project_url}</a></p>
            <p class="small">Developed by Research Group CAMMA, IHU Strasbourg, University of Strasbourg.</p>
          </div>

          <div class="section-box">
            <h2>License</h2>
            <p>Available for non-commercial scientific research under <strong>CC BY-NC-SA 4.0</strong>. Use implies agreement with the LICENSE file. Third-party components are governed by their own licenses.</p>
          </div>

          <div class="footer">
            <div>Last updated: August 2025</div>
          </div>
        </div>
        """
