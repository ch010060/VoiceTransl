"""Progress painting without a child widget for every catalog Track."""
from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem

from .tree_model import ROLE_NODE, TONE_COLORS


class ProgressDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        node = index.data(ROLE_NODE)
        if not node:
            return super().paint(painter, option, index)
        style_option = QStyleOptionViewItem(option)
        self.initStyleOption(style_option, index)
        style_option.text = ""
        style = option.widget.style() if option.widget else None
        if style:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, style_option, painter, option.widget)
        progress = node["progress"]
        color = QColor(TONE_COLORS.get(node["tone"], TONE_COLORS["muted"]))
        rect = QRectF(option.rect.adjusted(8, 7, -12, -7))
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        mode, ratio = progress["mode"], progress.get("ratio")
        if mode in {"aggregate", "reported", "indeterminate"}:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#263244"))
            painter.drawRoundedRect(rect, 4, 4)
            if ratio is not None:
                fill = QRectF(rect)
                fill.setWidth(rect.width() * min(1.0, max(0.0, ratio)))
                tint = QColor(color)
                tint.setAlpha(85)
                painter.setBrush(tint)
                painter.drawRoundedRect(fill, 4, 4)
            elif mode == "indeterminate":
                # A striped busy band conveys activity, never a guessed ratio.
                painter.setClipRect(rect)
                tint = QColor(color)
                tint.setAlpha(55)
                painter.setPen(QPen(tint, 6))
                x = int(rect.left()) - int(rect.height())
                while x < rect.right():
                    painter.drawLine(x, int(rect.bottom()), x + int(rect.height()), int(rect.top()))
                    x += 16
                painter.setClipping(False)
        painter.setPen(color)
        text_rect = option.rect.adjusted(12, 0, -14, 0)
        caption = painter.fontMetrics().elidedText(progress["caption"], Qt.TextElideMode.ElideRight, text_rect.width())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, caption)
        painter.restore()
