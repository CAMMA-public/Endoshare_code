from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QTextBrowser,
    QFrame,
)
from PyQt5.QtCore import Qt

class Help(QWidget):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # QTextBrowser for rich HTML; inherit background
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setReadOnly(True)
        self.text_browser.setFrameStyle(0)
        self.text_browser.setAcceptRichText(True)
        self.text_browser.setText(self._generate_html())

        # Transparent background so it blends with the app
        self.text_browser.setStyleSheet("QTextBrowser { background: transparent; }")

        # Scroll area wrapper
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.text_browser)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_area.setFrameShape(QFrame.NoFrame)

        layout.addWidget(scroll_area)
        self.setLayout(layout)

        # Ensure the help widget itself can inherit styled background
        self.setAttribute(Qt.WA_StyledBackground, True)

    def set_tutorial_url(self, url: str):
        html = self._generate_html(tutorial_url=url)
        self.text_browser.setText(html)

    def _generate_html(self, tutorial_url: str = "#") -> str:
        return fr"""
        <style>
          .help-container {{
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
          .tip {{
            background: #eef8ff;
            border-left: 4px solid #3182ce;
            padding: 10px 14px;
            margin: 10px 0;
            border-radius: 4px;
            font-size: 0.95em;
          }}
          .video-box {{
            background: #f2f7fb;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            padding: 12px;
            display: flex;
            gap: 12px;
            align-items: center;
          }}
          .video-placeholder {{
            flex: 0 0 120px;
            height: 68px;
            background: #d0e7ff;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            font-size: 1.2em;
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
            padding: 12px 14px;
            border-radius: 6px;
            background: #f7f9fc;
            margin-bottom: 12px;
          }}
          .status-note {{
            font-size: 0.85em;
            color: #2d3748;
            margin-top: 4px;
          }}
        </style>

        <div class="help-container" aria-label="Endoshare Help Guide">
          <h1>Using Endoshare</h1>
          <p>Overview of the core workflows and features. Endoshare merges, anonymizes, and prepares endoscopic videos while enforcing privacy and compliance.</p>

          <div class="section-box">
            <h2>1. Settings</h2>
            <p>Configure output directories, and select runtime parameters. Endoshare separates archive videos from de-identified shared outputs. </p>
            <ul>
              <li><strong>Archive Mode:</strong> This feature, when turned on, allows for the preservation of archive videos while creating de-identified copies for sharing.</li>
            </ul>
            <div class="tip">
              <strong>Tip:</strong> A dictionary matching the de-identified video name and the patient ID/name is always stored locally, to facilitate traceability. When archive mode is enabled, it is preserved in the archive directory, otherwise it is stored in the de-identified output directory.
            </div>
            <ul>
              <li><strong>Archive Repository:</strong> Videos in this directory contain identifiers and are used as a local archive. Out-of-body regions are blurred.</li>
              <li><strong>De-Identified Output:</strong> Fully de-identified outputs are stored in this directory; logs and audit information are written here to support debugging.</li>
            </ul>
          </div>

          <div class="section-box">
            <h2>2. Home Window / Patient Selection</h2>
            <p>Assign a patient identifier, select source video folders, and build the set of clips to process.</p>
            <ol>
              <li><strong>Enter Patient Name/ID:</strong> Provide the identifier for the case.</li>
              <li><strong>Browse Videos:</strong> Navigate to the folder containing raw endoscopic recordings; Select the directory, not the single files.</li>
              <li><strong>Video Selection:</strong> The system scans the directory for compatible video files. 'Available Videos' are listed for selection. Use the dedicated buttons to add them to the 'Patient Videos' list; here videos' order can be adjusted by drag and drop. Reviewing the selected clip can be done using the preview feature.</li>
              <li><strong>Confirm Patient Videos:</strong> After selecting and validating, finalize the selection. The case is added to the queue; color coding reflects its state (e.g., pending in red; completed in green).</li>
              <li><strong>Add New Patient:</strong> Begin another case while preserving existing entries in the queue.</li>
              <li><strong>Ready to Process:</strong> Confirmed patients appear in the ready list with status indicators.</li>
            </ol>
            <div class="status-note">
              The system aggregates and checks video resolutions for consistency; mismatches are flagged to avoid downstream errors.
            </div>
          </div>

          <div class="section-box">
            <h2>3. Processing</h2>
            <p>Processing performs merging, visual de-identification (e.g., out-of-body detection and blurring), and metadata stripping according to the selected mode and parameters.</p>
            <ul>
              <li><strong>Start Processing:</strong> Launches the pipeline. UI controls adapt to reflect active jobs and prevent conflicting actions.</li>
              <li><strong>Mode & Parameters:</strong> The chosen processing mode, frame rate, resolution, and purge policy are respected. These settings influence how merges and anonymization are applied.</li>
              <li><strong>Progress & Logging:</strong> Live status updates are shown. Completion changes the case’s queue color to signal success. Detailed logs, including throughput and timing, are saved for audits.</li>
            </ul>
          </div>

          <div class="section-box">
            <h2>4. Controls & Recovery</h2>
            <ul>
              <li><strong>Terminate Processing:</strong> Allows graceful interruption; protected by confirmation to prevent accidental kills. If the application is closed during active work, the user is prompted to confirm. </li>
              <li><strong>Reset:</strong> Clears current UI state, returning the application to a clean slate.</li>
              <li><strong>Delete Patient Entry:</strong> Safely removes a case from the queue, preserving consistency.</li>
              <li><strong>Error Handling:</strong> Errors surface via dialogs, and the interface resets to let the user correct inputs and retry.</li>
            </ul>
          </div>

          <div class="section-box">
            <h2>5. Guided Operations</h2>
            <p>Interactive elements—buttons, status badges, and tooltips—dynamically reflect workflow prerequisites. Disabled actions include contextual explanations to guide correct sequencing and prevent misuse.</p>
          </div>

          <div class="section-box">
            <h2>7. Safety & Data Handling</h2>
            <p><strong>Privacy-first:</strong> Outputs intended for sharing are de-identified visually and at the metadata level. Source material remains local unless explicitly exported.</p>
            <p><strong>Auditability:</strong> All processing sessions and performance metrics are logged to the shared directory to satisfy compliance requirements.</p>
          </div>

          <div class="small" style="margin-top:1.5em;">
            Last updated: August 2025
          </div>
        </div>
        """
 