"""Regenerate PNG and multi-size Windows ICO from the adjacent SVG using Qt.

Run with the existing GUI environment: python scripts/asmr_gui/assets/generate_icons.py
"""
from pathlib import Path
import struct

from PyQt6.QtCore import QBuffer, QIODevice
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtSvg import QSvgRenderer


def generate():
    directory = Path(__file__).resolve().parent
    renderer = QSvgRenderer(str(directory / "voicetransl-asmr.svg"))
    if not renderer.isValid():
        raise ValueError("Invalid application icon SVG")
    frames = []
    for size in (16, 24, 32, 48, 64, 128, 256):
        image = QImage(size, size, QImage.Format.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        if not image.save(buffer, "PNG"):
            raise RuntimeError("Qt could not encode icon PNG")
        frames.append((size, bytes(buffer.data())))
        if size == 256:
            (directory / "voicetransl-asmr.png").write_bytes(frames[-1][1])
    offset = 6 + 16 * len(frames)
    entries = []
    for size, data in frames:
        entries.append(struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset))
        offset += len(data)
    (directory / "voicetransl-asmr.ico").write_bytes(
        struct.pack("<HHH", 0, 1, len(frames)) + b"".join(entries) + b"".join(data for _, data in frames))


if __name__ == "__main__":
    generate()
