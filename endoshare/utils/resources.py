import os
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


def resource_path(relative_path: str) -> str:
    """Return the absolute path to a resource bundled with the package."""
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "resources"))
    if getattr(sys, "frozen", False):
        bundle_dir = os.path.dirname(sys.executable)
        res_root = os.path.abspath(os.path.join(bundle_dir, "..", "Resources"))
        res_nested = os.path.join(res_root, "Resources")
        res_macos = bundle_dir
        for candidate_root in (res_root, res_nested, res_macos):
            candidate = os.path.join(candidate_root, relative_path)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(res_root, relative_path)
    return os.path.join(base_path, relative_path)


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
    base = resource_path("Externals")
    app_bundle = os.path.join(base, "ffmpeg.app")
    cli_bin = os.path.join(app_bundle, "Contents", "MacOS", "ffmpeg")
    raw_bin = os.path.join(base, "ffmpeg")
    return cli_bin if os.path.isfile(cli_bin) else raw_bin


FFMPEG_BIN = _ffmpeg_path()
FFPROBE_BIN = FFMPEG_BIN
