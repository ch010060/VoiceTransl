"""Standalone PyQt6 ASMR workbench; deliberately independent of app.py/PyQt5."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--path")
    parser.add_argument("--state-root")
    args = parser.parse_args(argv)
    from PyQt6.QtWidgets import QApplication
    from scripts.asmr_gui.controller import Controller
    from scripts.asmr_gui.window import BatchWindow
    from scripts.asmr_gui.theme import apply_theme
    from scripts.asmr_gui.app_icon import application_icon, set_windows_app_id
    set_windows_app_id()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("VoiceTransl ASMR Batch")
    app.setOrganizationName("VoiceTransl")
    app.setWindowIcon(application_icon())
    apply_theme(app)
    controller = Controller(state_root=args.state_root)
    if args.root:
        controller.set_root(args.root)
    window = BatchWindow(controller)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
