"""Native Qt palette and vector action icons for the ASMR workstation."""
from __future__ import annotations

from functools import lru_cache
from string import Template

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPalette, QPen, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QProxyStyle, QStyle, QStyleFactory

THEME = {
    "window": "#111923", "surface": "#1b2836", "control": "#24364a",
    "table": "#15212d", "alternate": "#1a2a39", "selected": "#254e72",
    "border": "#3d566e", "separator": "#2a4054", "primary": "#2584e8",
    "accent": "#2584e8", "hover": "#304b65", "pressed": "#17344f",
    "disabled": "#202d3b", "success": "#5cdbab", "warning": "#f0bd70",
    "error": "#ff8795", "info": "#85c9ff", "text": "#e7edf5", "muted": "#9db0c6",
}

STYLE = Template("""
QWidget { color:$text; font-family:'Microsoft JhengHei UI','Yu Gothic UI','Segoe UI'; font-size:13px; }
QMainWindow,QDialog { background:$window; }
QWidget#surface { background:$window; border:1px solid $border; border-radius:10px; }
QWidget#titlebar { background:#0e1720; border-bottom:1px solid $separator; }
QLabel { background:transparent; }
QLabel#appTitle { font-weight:600; font-size:14px; }
QLabel#muted { color:$muted; }
QWidget#summary { background:#1b2d3d; border:1px solid $border; border-radius:7px; }
QLabel#summaryState { font-size:14px; font-weight:700; color:$info; }
QLabel#summaryNumbers { font-weight:600; }
QLineEdit,QSpinBox,QComboBox,QPlainTextEdit { background:#0d1823; color:$text;
 border:1px solid $border; border-radius:6px; padding:6px;
 selection-background-color:$selected; selection-color:white; }
QLineEdit:focus,QSpinBox:focus,QComboBox:focus { border-color:$info; }
QComboBox { padding-right:24px; }
QComboBox::drop-down { border:0; width:22px; }
QComboBox QAbstractItemView { background:$control; color:$text; selection-background-color:$selected; }
QPushButton,QToolButton { background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #2c4055,stop:1 #203044);
 border:1px solid #516b84; border-radius:6px; padding:6px 9px; min-height:20px; font-size:12px; }
QPushButton:hover,QToolButton:hover { background:$hover; border-color:$info; }
QPushButton:pressed,QToolButton:pressed { background:$pressed; }
QPushButton#start { background:$primary; border-color:#65b3ff; color:white; font-weight:700; }
QPushButton#start:hover { background:#3697f3; }
QPushButton#stop:enabled { color:#ffd4da; border-color:#a05c67; background:#4a2833; }
QPushButton:disabled,QToolButton:disabled { color:#75889d; background:$disabled; border-color:#34475b; }
QPushButton#windowDot { padding:0; min-height:0; border:0; border-radius:7px; }
QTreeView { background:$table; alternate-background-color:$alternate; color:$text;
 border:1px solid $border; border-radius:7px; selection-background-color:$selected;
 selection-color:white; outline:0; }
QTreeView::item { border-bottom:1px solid $separator; padding:5px 4px; }
QTreeView::item:selected { background:$selected; }
QHeaderView::section { background:#293e53; color:#c4d6e8; padding:8px 6px;
 border:0; border-bottom:1px solid #5a738d; border-right:1px solid #3d566e; font-weight:600; }
QCheckBox { spacing:7px; background:transparent; }
QCheckBox::indicator,QTreeView::indicator { width:17px; height:17px; }
QPlainTextEdit { background:#0b131c; color:#b6c9dc; font-family:'Cascadia Mono','Consolas'; font-size:12px; }
QWidget#logHeader { background:$control; border:1px solid $border; border-radius:6px; }
QWidget#backendbar { background:#0d1721; border-top:1px solid $border; }
QSplitter::handle { background:$separator; height:5px; }
QScrollBar:vertical { background:#101b27; width:12px; margin:2px; }
QScrollBar::handle:vertical { background:#58728e; border-radius:4px; min-height:28px; }
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0; }
QScrollBar:horizontal { background:#101b27; height:12px; }
QScrollBar::handle:horizontal { background:#58728e; border-radius:4px; min-width:28px; }
QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal { width:0; }
QToolTip { color:$text; background:$control; border:1px solid #708da8; padding:5px; }
QTabWidget::pane { border:1px solid $border; border-radius:6px; }
QTabBar::tab { background:$control; border:1px solid $border; padding:8px 16px; }
QTabBar::tab:selected { background:$selected; }
""").substitute(THEME)

_PATHS = {
    "scan": '<circle cx="10" cy="10" r="6"/><path d="M15 15l6 6"/>',
    "play": '<path d="M7 4l13 8-13 8z"/>',
    "stop": '<rect x="5" y="5" width="14" height="14" rx="2"/>',
    "clear": '<path d="M4 6h16M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7m4-7v7"/>',
    "settings": '<path d="M9 3h6l1 3 3 1 2 5-2 5-3 1-1 3H9l-1-3-3-1-2-5 2-5 3-1z"/><circle cx="12" cy="12" r="3"/>',
    "backend": '<path d="M5 3v5a4 4 0 0 0 8 0V3M3 3h4m4 0h4M9 12v3a5 5 0 0 0 10 0v-2"/><circle cx="19" cy="10" r="3"/>',
    "folder": '<path d="M3 7V5h7l2 2h9v13H3z"/><path d="M3 10h18"/>',
    "file": '<path d="M5 2h9l5 5v15H5zM14 2v6h5M8 12h8M8 16h8"/>',
    "display": '<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M12 17v4M8 21h8"/>',
    "audio": '<path d="M5 9h4l5-5v16l-5-5H5zM17 8a7 7 0 0 1 0 8M20 5a11 11 0 0 1 0 14"/>',
    "chevron_down": '<path d="M5 9l7 7 7-7"/>',
    "chevron_right": '<path d="M9 5l7 7-7 7"/>',
}


@lru_cache(maxsize=64)
def action_icon(name: str, color: str = "#9fc8fa") -> QIcon:
    icon = QIcon()
    for mode, tint in ((QIcon.Mode.Normal, color), (QIcon.Mode.Disabled, "#677386")):
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
               f'fill="none" stroke="{tint}" stroke-width="1.7" stroke-linecap="round" '
               f'stroke-linejoin="round">{_PATHS[name]}</svg>')
        for scale in (1, 1.25, 1.5, 2, 3):
            pixmap = QPixmap(round(24 * scale), round(24 * scale))
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            QSvgRenderer(svg.encode()).render(painter)
            painter.end()
            pixmap.setDevicePixelRatio(scale)
            icon.addPixmap(pixmap, mode)
    return icon


class CheckmarkStyle(QProxyStyle):
    """A visible tick for both ordinary and model-view selection controls."""
    def drawPrimitive(self, element, option, painter, widget=None):
        if element not in (QStyle.PrimitiveElement.PE_IndicatorCheckBox,
                           QStyle.PrimitiveElement.PE_IndicatorItemViewItemCheck):
            return super().drawPrimitive(element, option, painter, widget)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(option.rect.adjusted(1, 1, -1, -1))
        checked = bool(option.state & QStyle.StateFlag.State_On)
        partial = bool(option.state & QStyle.StateFlag.State_NoChange)
        painter.setPen(QPen(QColor(THEME["info"] if checked or partial else THEME["muted"]), 1))
        painter.setBrush(QColor(THEME["primary"] if checked or partial else THEME["window"]))
        painter.drawRoundedRect(rect, 3, 3)
        painter.setPen(QPen(QColor("white"), 1.8, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        if checked:
            painter.drawLine(QPointF(rect.left() + rect.width() * .22, rect.top() + rect.height() * .52),
                             QPointF(rect.left() + rect.width() * .43, rect.top() + rect.height() * .73))
            painter.drawLine(QPointF(rect.left() + rect.width() * .43, rect.top() + rect.height() * .73),
                             QPointF(rect.left() + rect.width() * .8, rect.top() + rect.height() * .28))
        elif partial:
            painter.drawLine(QPointF(rect.left() + 4, rect.center().y()),
                             QPointF(rect.right() - 4, rect.center().y()))
        painter.restore()


def apply_theme(app) -> None:
    if not app.property("asmrThemeInstalled"):
        app.setStyle(CheckmarkStyle(QStyleFactory.create("Fusion")))
        app.setProperty("asmrThemeInstalled", True)
    palette = app.palette()
    for role, token in ((QPalette.ColorRole.Window, "window"),
                        (QPalette.ColorRole.Base, "table"),
                        (QPalette.ColorRole.AlternateBase, "alternate"),
                        (QPalette.ColorRole.Text, "text"),
                        (QPalette.ColorRole.WindowText, "text"),
                        (QPalette.ColorRole.Button, "control"),
                        (QPalette.ColorRole.ButtonText, "text"),
                        (QPalette.ColorRole.Highlight, "selected"),
                        (QPalette.ColorRole.HighlightedText, "text")):
        palette.setColor(role, QColor(THEME[token]))
    app.setPalette(palette)
    app.setStyleSheet(STYLE)
