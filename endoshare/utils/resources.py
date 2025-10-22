import os
import shutil
import stat
import sys
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QPen, QPixmap, QColor, QPainter
from loguru import logger
import tensorflow as tf

# logging configuration
tf.get_logger().setLevel("INFO")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
LOG_PERSIST = "PERSIST"
logger.level(LOG_PERSIST, no=25)


def _log_uncaught(exctype, value, traceback):
    logger.opt(exception=(exctype, value, traceback)).error("An uncaught exception occurred.")


sys.excepthook = _log_uncaught


import os
import sys
from pathlib import Path
import logging  # for fallback debug if loguru isn't initialized yet

def _search_for_resource(relative: str) -> str:
    """
    If the normal resolution fails, walk upward from package root to
    locate resources/<relative> anywhere in the tree as a fallback.
    """
    # assume this file is .../endoshare/utils/resources.py
    pkg_root = Path(__file__).resolve().parent.parent  # endoshare/
    candidate = pkg_root / "resources" / relative
    if candidate.exists():
        return str(candidate)

    # last ditch: glob in resources directory
    for p in (pkg_root / "resources").rglob(os.path.basename(relative)):
        if relative.replace(os.path.basename(relative), "") in str(p.parent):
            return str(p)
    return ""  # not found

def resource_path(relative_path: str) -> str:
    """Return the absolute path to a resource bundled with the package."""
    # normalize passed-in
    relative_path = os.path.normpath(relative_path).lstrip(os.sep)

    # base for non-frozen (source) case
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "resources"))

    resolved = ""
    if getattr(sys, "frozen", False):
        bundle_dir = os.path.dirname(sys.executable)
        res_root = os.path.abspath(os.path.join(bundle_dir, "..", "Resources"))
        res_nested = os.path.join(res_root, "Resources")
        res_macos = bundle_dir
        for candidate_root in (res_root, res_nested, res_macos):
            candidate = os.path.join(candidate_root, relative_path)
            if os.path.exists(candidate):
                return os.path.normpath(candidate)
        resolved = os.path.join(res_root, relative_path)
    else:
        resolved = os.path.join(base_path, relative_path)

    if os.path.exists(resolved):
        return os.path.normpath(resolved)

    # fallback search (helps if something strange stripped "resources")
    fallback = _search_for_resource(relative_path)
    if fallback and os.path.exists(fallback):
        return os.path.normpath(fallback)

    # last resort: return the originally computed path so caller can fail with clear message
    logging.getLogger(__name__).warning(f"resource_path could not locate '{relative_path}', tried '{resolved}' and fallback '{fallback}'")
    return os.path.normpath(resolved)



ICON_DIR = "icons"


def load_icon(name: str) -> QIcon:
    path = resource_path(os.path.join(ICON_DIR, name))
    return QIcon(path)


def tinted_icon(name: str, size: QSize, hex_color: str) -> QIcon:
    base = load_icon(name).pixmap(size)
    painter = QPainter(base)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(base.rect(), QColor(hex_color))
    pen = QPen(QColor(hex_color))
    pen.setWidth(1)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    inset = pen.width() // 2
    r = base.rect().adjusted(inset, inset, -inset, -inset)
    painter.drawRect(r)
    painter.end()
    return QIcon(base)


ICON_COLORS = {
    "home": "#444446",
    "settings": "#444446",
    "info": "#444446",
    "help": "#444446",
}


def _ffmpeg_path() -> str:
    # 1. Bundled raw binary: Externals/ffmpeg/ffmpeg
    raw_bin = resource_path(os.path.join("Externals", "ffmpeg", "ffmpeg"))
    if os.path.isfile(raw_bin):
        try:
            # ensure exec bit
            st = os.stat(raw_bin)
            os.chmod(raw_bin, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            pass
        if os.access(raw_bin, os.X_OK):
            return raw_bin

    raise RuntimeError("No usable ffmpeg binary found.")


FFMPEG_BIN = _ffmpeg_path()
FFPROBE_BIN = FFMPEG_BIN
