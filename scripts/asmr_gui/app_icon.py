"""Source-controlled application artwork and stable Windows taskbar identity."""
from __future__ import annotations

from pathlib import Path
import sys

from PyQt6.QtGui import QIcon

WINDOWS_APP_ID = "VoiceTransl.ASMRBatch.GUI"
ASSET_DIRECTORY = Path(__file__).with_name("assets")


def set_windows_app_id() -> bool:
    """Call before QApplication so native windows share the app's own identity."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        return ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID) == 0
    except (AttributeError, OSError):
        return False


def application_icon() -> QIcon:
    """ICO contains native sizes; PNG/SVG remain portable fallback assets."""
    for name in ("voicetransl-asmr.ico", "voicetransl-asmr.png", "voicetransl-asmr.svg"):
        icon = QIcon(str(ASSET_DIRECTORY / name))
        if not icon.isNull():
            return icon
    return QIcon()
