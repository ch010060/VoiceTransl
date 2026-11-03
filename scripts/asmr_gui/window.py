"""Dedicated native ASMR workstation; the controller owns every execution action."""
from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

from PyQt6.QtCore import QEvent, QSettings, QSize, Qt, QUrl, QTimer
from PyQt6.QtGui import QDesktopServices, QPainter
from PyQt6.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QPlainTextEdit, QPushButton, QSizePolicy,
    QSplitter, QTreeView, QVBoxLayout, QWidget,
)

from .delegates import ProgressDelegate
from .app_icon import application_icon
from .settings import SettingsDialog, default_settings
from .theme import THEME, action_icon
from .tree_model import ROLE_NODE, WorkFilterProxyModel, WorkTrackModel

_STATES = {
    "IDLE": "等待掃描", "SCANNING": "掃描作品中", "READY": "準備就緒",
    "PREFLIGHT": "執行前檢查", "STARTING": "啟動工作程序", "RUNNING": "批次執行中",
    "STOPPING": "停止與清理中", "RECONCILING": "核對執行結果",
    "FINISHED": "本次批次已結束", "ERROR": "發生錯誤", "CLEANUP_FAILED": "清理失敗，請重試停止",
}


class ElidingLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)

    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignVCenter,
                         self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, self.width()))


class TitleBar(QWidget):
    """Qt 6.12 system operations preserve native snap, maximize and restore state."""
    def __init__(self, window):
        super().__init__(window)
        self.window_owner = window
        self.setObjectName("titlebar")
        self.setFixedHeight(44)
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 0, 16, 0)
        row.setSpacing(9)
        for name, color, callback in (
            ("關閉", "#ff635b", window.close),
            ("最小化", "#ffbd44", window.showMinimized),
            ("最大化／還原", "#00c84a", window.toggle_maximized),
        ):
            dot = QPushButton()
            dot.setObjectName("windowDot")
            dot.setFixedSize(14, 14)
            dot.setToolTip(name)
            dot.setAccessibleName(name)
            dot.setStyleSheet(f"QPushButton#windowDot {{background:{color};}} QPushButton#windowDot:hover {{border:1px solid white;}}")
            dot.clicked.connect(callback)
            row.addWidget(dot)
        row.addSpacing(8)
        app_icon = QLabel()
        app_icon.setPixmap(application_icon().pixmap(20, 20))
        app_icon.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        row.addWidget(app_icon)
        title = QLabel("VoiceTransl ASMR 批次轉錄")
        title.setObjectName("appTitle")
        title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        row.addWidget(title)
        version = QLabel("v1.0")
        version.setObjectName("muted")
        version.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        row.addWidget(version)
        row.addStretch()
        self.hint = QLabel("原生批次工作台")
        self.hint.setObjectName("muted")
        self.hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        row.addWidget(self.hint)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.window_owner.windowHandle()
            if handle:
                handle.startSystemMove()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.window_owner.toggle_maximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class BatchWindow(QMainWindow):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.preferences = QSettings("VoiceTransl", "ASMRBatchGUI")
        self._close_authorized = False
        self._toolbar_rows = 0
        self._startup_root = str(controller.root or self.preferences.value("last_input_root", "") or "")
        self.setWindowTitle("VoiceTransl ASMR 批次轉錄 v1.0")
        self.setWindowIcon(application_icon())
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.setMinimumSize(980, 600)
        self.resize(1220, 840)
        surface = QWidget()
        surface.setObjectName("surface")
        self.setCentralWidget(surface)
        outer = QVBoxLayout(surface)
        outer.setContentsMargins(5, 5, 5, 5)
        outer.setSpacing(0)
        self.title_bar = TitleBar(self)
        self._title_bar = self.title_bar
        outer.addWidget(self.title_bar)
        body = QWidget()
        content = QVBoxLayout(body)
        content.setContentsMargins(14, 12, 14, 10)
        content.setSpacing(10)
        outer.addWidget(body, 1)
        root_row = QHBoxLayout()
        root_row.setSpacing(10)
        root_row.addWidget(QLabel("資料庫目錄："))
        self.root_edit = QLineEdit(self._startup_root)
        self.root_edit.setPlaceholderText("選擇 ASMR 作品根目錄")
        self.root_edit.setClearButtonEnabled(True)
        self.root_edit.setMinimumWidth(0)
        self.root_edit.editingFinished.connect(self._root_edited)
        root_row.addWidget(self.root_edit, 1)
        self.browse_button = QPushButton()
        self.browse_button.setIcon(action_icon("folder"))
        self.browse_button.setToolTip("選擇資料庫根目錄")
        self.browse_button.setAccessibleName("選擇資料庫根目錄")
        self.browse_button.setFixedWidth(48)
        self.browse_button.clicked.connect(self._browse)
        root_row.addWidget(self.browse_button)
        content.addLayout(root_row)
        self.toolbar = QWidget()
        self.toolbar_layout = QGridLayout(self.toolbar)
        self.toolbar_layout.setContentsMargins(0, 0, 0, 0)
        self.toolbar_layout.setHorizontalSpacing(8)
        self.toolbar_layout.setVerticalSpacing(7)
        self.toolbar_buttons = []
        for attr, text, icon, action in (
            ("scan_button", "掃描作品", "scan", self._scan),
            ("start_button", "開始批次", "play", self._start),
            ("stop_button", "停止所有任務", "stop", controller.stop),
            ("clear_button", "清空清單", "clear", controller.clear),
            ("settings_button", "批次設定", "settings", self._settings),
            ("backend_button", "後端檢查", "backend", controller.check_backends),
            ("output_button", "開啟輸出目錄", "folder", self._open_output),
            ("journal_button", "完整日誌", "file", self._open_journal),
            ("preferences_button", "介面偏好", "display", self._preferences),
        ):
            button = QPushButton(text)
            button.setIcon(action_icon(icon, "#ffffff" if attr == "start_button" else "#9fc8fa"))
            button.setIconSize(QSize(18, 18))
            button.setToolTip(text)
            button.setMinimumWidth(0)
            button.clicked.connect(action)
            setattr(self, attr, button)
            self.toolbar_buttons.append(button)
        self.start_button.setObjectName("start")
        self.stop_button.setObjectName("stop")
        self.journal_button.setToolTip("完整診斷日誌（事件紀錄保留於執行目錄）")
        content.addWidget(self.toolbar)
        summary = QWidget()
        summary.setObjectName("summary")
        summary.setFixedHeight(62)
        summary_row = QHBoxLayout(summary)
        summary_row.setContentsMargins(13, 8, 13, 8)
        summary_row.setSpacing(16)
        summary_icon = QLabel()
        summary_icon.setPixmap(action_icon("file").pixmap(22, 22))
        summary_row.addWidget(summary_icon)
        summary_row.addWidget(QLabel("批次總覽"))
        self.state_label = QLabel()
        self.state_label.setObjectName("summaryState")
        summary_row.addWidget(self.state_label)
        summary_row.addWidget(self._separator())
        self.summary_label = ElidingLabel()
        self.summary_label.setObjectName("summaryNumbers")
        summary_row.addWidget(self.summary_label, 1)
        content.addWidget(summary)
        filters = QHBoxLayout()
        filters.setSpacing(10)
        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("search_input")
        self.search_edit.setPlaceholderText("搜尋作品或 Track（作品標題、檔名、RJ 編號…）")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.addAction(action_icon("scan"), QLineEdit.ActionPosition.LeadingPosition)
        filters.addWidget(self.search_edit, 1)
        filters.addWidget(QLabel("顯示："))
        self.filter_combo = QComboBox()
        self.filter_combo.setMinimumWidth(144)
        for label, code in (("全部作品", "all"), ("已選作品", "selected"),
                            ("執行中", "active"), ("有失敗", "failed"), ("已結束", "finished")):
            self.filter_combo.addItem(label, code)
        filters.addWidget(self.filter_combo)
        self.selection_label = QLabel()
        self.selection_label.setObjectName("muted")
        filters.addWidget(self.selection_label)
        content.addLayout(filters)
        self.tree = QTreeView()
        self.tree.setObjectName("work_tree")
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setAnimated(False)
        self.tree.setSortingEnabled(False)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.setIndentation(23)
        self.tree.setMinimumHeight(140)
        self.tree_model = WorkTrackModel(controller.run_model, self)
        self.proxy = WorkFilterProxyModel(self)
        self.proxy.setSourceModel(self.tree_model)
        self.tree.setModel(self.proxy)
        self._expansion_root = controller.run_model.plan.root_id if controller.run_model.plan else None
        self._expanded_work_ids = set()
        self._restoring_expansion = False
        self.tree.expanded.connect(lambda index: self._remember_expansion(index, True))
        self.tree.collapsed.connect(lambda index: self._remember_expansion(index, False))
        self.tree_model.modelReset.connect(self._restore_work_expansion)
        self.tree.setItemDelegateForColumn(2, ProgressDelegate(self.tree))
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column, width in ((1, 205), (2, 225), (3, 112)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            header.resizeSection(column, width)
        header.setMinimumSectionSize(78)
        self.search_edit.textChanged.connect(self._search_changed)
        self.filter_combo.currentIndexChanged.connect(self._filter_changed)
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.tree)
        self.log_panel = QWidget()
        log_layout = QVBoxLayout(self.log_panel)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(0)
        log_header = QWidget()
        log_header.setObjectName("logHeader")
        log_header.setFixedHeight(37)
        log_row = QHBoxLayout(log_header)
        log_row.setContentsMargins(8, 0, 9, 0)
        self.log_toggle = QPushButton("執行日誌")
        self.log_toggle.setCheckable(True)
        self.log_toggle.setChecked(self.preferences.value("logs/visible", True, type=bool))
        self.log_toggle.setStyleSheet("border:0;background:transparent;text-align:left;")
        self.log_toggle.clicked.connect(self._toggle_logs)
        log_row.addWidget(self.log_toggle)
        log_row.addStretch()
        self.follow_log = QCheckBox("自動捲動")
        self.follow_log.setChecked(True)
        log_row.addWidget(self.follow_log)
        open_log = QPushButton("開啟完整日誌")
        open_log.setToolTip("完整診斷日誌（事件紀錄保留於執行目錄）")
        open_log.setIcon(action_icon("file"))
        open_log.setStyleSheet("padding:3px 8px;min-height:17px;")
        open_log.clicked.connect(self._open_journal)
        log_row.addWidget(open_log)
        log_layout.addWidget(log_header)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1500)
        self.log_view.setMinimumHeight(65)
        log_layout.addWidget(self.log_view, 1)
        self.splitter.addWidget(self.log_panel)
        self.splitter.setStretchFactor(0, 5)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([480, 145])
        content.addWidget(self.splitter, 1)
        footer = QWidget()
        footer.setObjectName("backendbar")
        footer.setFixedHeight(48)
        footer_row = QHBoxLayout(footer)
        footer_row.setContentsMargins(14, 6, 14, 6)
        footer_row.setSpacing(14)
        self.asr_label = ElidingLabel()
        self.translation_label = ElidingLabel()
        footer_row.addWidget(self.asr_label, 1)
        footer_row.addWidget(self._separator())
        footer_row.addWidget(self.translation_label, 1)
        footer_row.addWidget(self._separator())
        self.execution_label = QLabel()
        self.execution_label.setObjectName("muted")
        footer_row.addWidget(self.execution_label)
        outer.addWidget(footer)
        controller.state_changed.connect(self._sync)
        controller.catalog_changed.connect(self._catalog_changed)
        controller.backend_changed.connect(self._backend)
        controller.log_message.connect(self._append_log)
        controller.settings_changed.connect(self._settings_changed)
        controller.close_ready.connect(self._safe_close)
        controller.run_model.changed.connect(self._sync)
        controller.run_model.selection_changed.connect(self._sync)
        self._settings_changed()
        self._sync()
        self._toggle_logs()
        self._layout_toolbar()
        QApplication.instance().installEventFilter(self)
        QTimer.singleShot(0, self._startup_scan)

    def _separator(self):
        label = QLabel()
        label.setFixedWidth(1)
        label.setStyleSheet(f"background:{THEME['border']};")
        return label

    def _layout_toolbar(self):
        required = sum(button.sizeHint().width() for button in self.toolbar_buttons) + 8 * 8
        rows = 2 if required > self.toolbar.width() else 1
        if rows == self._toolbar_rows:
            return
        self._toolbar_rows = rows
        for index in range(10):
            self.toolbar_layout.setColumnStretch(index, 0)
        for button in self.toolbar_buttons:
            self.toolbar_layout.removeWidget(button)
        for index, button in enumerate(self.toolbar_buttons):
            row, column = (0, index) if rows == 1 else ((0, index) if index < 4 else (1, index - 4))
            self.toolbar_layout.addWidget(button, row, column)
        self.toolbar_layout.setColumnStretch(9 if rows == 1 else 5, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "toolbar"):
            self._layout_toolbar()

    def toggle_maximized(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def eventFilter(self, watched, event):
        if watched is self.toolbar and event.type() == QEvent.Type.Resize:
            self._layout_toolbar()
        if (isinstance(watched, QWidget) and watched.window() is self
                and event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton and not self.isMaximized()):
            point = self.mapFromGlobal(event.globalPosition().toPoint())
            edges = Qt.Edge(0)
            if point.x() < 6:
                edges |= Qt.Edge.LeftEdge
            elif point.x() >= self.width() - 6:
                edges |= Qt.Edge.RightEdge
            if point.y() < 6:
                edges |= Qt.Edge.TopEdge
            elif point.y() >= self.height() - 6:
                edges |= Qt.Edge.BottomEdge
            if edges and self.windowHandle():
                return self.windowHandle().startSystemResize(edges)
        return super().eventFilter(watched, event)

    def _startup_scan(self):
        if not self.controller.safe_to_edit or self.controller.run_model.plan is not None:
            return
        root = self._startup_root.strip()
        if root and Path(root).is_dir():
            if self.controller.set_root(root):
                self.preferences.setValue("last_input_root", root)
                self.controller.scan()
        elif root:
            self._append_log("上次資料庫目錄不存在，請重新選擇：" + root)

    def _set_root_and_scan(self, root):
        root = str(root).strip()
        if not self.controller.safe_to_edit:
            self.root_edit.setText(str(self.controller.root or ""))
            return False
        if not root or not Path(root).is_dir():
            self._append_log("資料庫目錄不存在或無效：" + root)
            return False
        changed = Path(root).resolve() != Path(self.controller.root).resolve() if self.controller.root else True
        if not self.controller.set_root(root):
            return False
        self.preferences.setValue("last_input_root", root)
        self.root_edit.setText(root)
        if changed or self.controller.run_model.plan is None:
            return self.controller.scan()
        return True

    def _root_edited(self):
        candidate = self.root_edit.text().strip()
        if not candidate:
            self._append_log("請先選擇有效的 ASMR 作品目錄。")
            return False
        if candidate != self.controller.root:
            return self._set_root_and_scan(candidate)
        return Path(candidate).is_dir()

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "選擇 ASMR 作品根目錄", self.root_edit.text())
        if path:
            self._set_root_and_scan(path)

    def _scan(self):
        candidate = self.root_edit.text().strip()
        if not candidate:
            self._append_log("請先選擇有效的 ASMR 作品目錄。")
        elif candidate != self.controller.root:
            self._set_root_and_scan(candidate)
        else:
            self.controller.scan()

    def _start(self):
        if self._root_edited():
            self.controller.start()

    def _settings(self):
        if not self.controller.safe_to_edit:
            return
        dialog = SettingsDialog(self.controller.settings, self.controller.mode, self.controller.limit, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and self.controller.safe_to_edit:
            values, mode, limit = dialog.values()
            self.controller.apply_settings(values, mode=mode, limit=limit)

    def _settings_changed(self):
        self._footer_settings = default_settings()
        self._footer_settings.update(self.controller.settings)
        self._backend(self.controller.last_backend)
        self._sync()

    def _catalog_changed(self):
        self.root_edit.setText(str(self.controller.root or ""))
        self._sync()

    def _remember_expansion(self, index, expanded):
        if self._restoring_expansion or self.search_edit.text().strip():
            return
        node = index.data(ROLE_NODE)
        if node and node["kind"] == "work":
            if expanded:
                self._expanded_work_ids.add(node["id"])
            else:
                self._expanded_work_ids.discard(node["id"])

    def _restore_work_expansion(self):
        plan = self.controller.run_model.plan
        if plan is None:
            return
        if self._expansion_root != plan.root_id:
            self._expansion_root = plan.root_id
            self._expanded_work_ids.clear()
        self._expanded_work_ids.intersection_update(work.work_id for work in plan.works)
        previous = self._restoring_expansion
        self._restoring_expansion = True
        try:
            query = self.search_edit.text().strip().casefold()
            for row in range(self.proxy.rowCount()):
                index = self.proxy.index(row, 0)
                node = index.data(ROLE_NODE)
                source = self.proxy.mapToSource(index)
                reveal = bool(query) and any(
                    query in str(self.tree_model.index(child, 0, source).data()).casefold()
                    for child in range(self.tree_model.rowCount(source)))
                self.tree.setExpanded(index, node["id"] in self._expanded_work_ids or reveal)
        finally:
            self._restoring_expansion = previous

    def _search_changed(self, query):
        self._restoring_expansion = True
        try:
            self.proxy.set_query(query)
            self._restore_work_expansion()
        finally:
            self._restoring_expansion = False

    def _filter_changed(self):
        self._restoring_expansion = True
        try:
            self.proxy.set_status_filter(self.filter_combo.currentData())
            self._restore_work_expansion()
        finally:
            self._restoring_expansion = False

    def _sync(self, *args):
        state = self.controller.state
        summary = self.controller.run_model.summary()
        selected = summary.get("selected_works", 0)
        total = summary.get("total_works", 0)
        planned = summary.get("planned", 0)
        resolved = summary.get("resolved", 0)
        failed = summary.get("failed", 0)
        settings = self._footer_settings
        mode = "僅轉錄" if self.controller.mode == "transcribe_only" else "完整處理"
        self.execution_label.setText(
            f"設定：最多 {settings['parallel']} 工作 / {settings['transcribe_parallel']} ASR | 模式：{mode}")
        self.execution_label.setToolTip(
            f"設定上限：{settings['parallel']} 個工作、{settings['transcribe_parallel']} 個 ASR\n"
            f"目前執行中：{summary.get('active_works', 0)} 個工作（同步顯示於批次總覽）\n"
            f"模式：{mode}。設定上限不代表實際活躍任務數。")
        self.state_label.setText(_STATES.get(state, state))
        tone = "error" if state in {"ERROR", "CLEANUP_FAILED"} or failed else "info"
        if state == "FINISHED" and not failed:
            tone = "warning" if summary.get("cancelled", 0) or summary.get("not_started", 0) else "success"
        self.state_label.setStyleSheet(f"color:{THEME[tone]};font-weight:700;")
        self.summary_label.setText(
            f"作品 {total}（已選 {selected}）   執行中 {summary.get('active_works', 0)}   │   "
            f"Track 已處理 {resolved} / {planned}   │   "
            f"成功 {summary.get('success', 0)}   失敗 {failed}   "
            f"略過 {summary.get('skipped', 0)}   複製 {summary.get('copied', 0)}   僅轉錄 {summary.get('transcribed', 0)}")
        self.selection_label.setText(f"已選 {selected} / {total}")
        safe = self.controller.safe_to_edit
        for widget in (self.scan_button, self.clear_button, self.settings_button,
                       self.backend_button, self.browse_button, self.root_edit):
            widget.setEnabled(safe)
        self.start_button.setEnabled(safe and selected > 0 and planned > 0)
        self.stop_button.setEnabled(bool(self.controller.busy) or state == "CLEANUP_FAILED")
        self.journal_button.setEnabled(getattr(self.controller, "diagnostic_log_path", None) is not None
                                       or self.controller.last_run_dir is not None)
        self.output_button.setEnabled(bool(self.controller.root))
        self.output_button.setToolTip(f"開啟輸出所在根目錄：{self.controller.root}" if self.controller.root else "開啟輸出目錄")
        self.tree.viewport().update()

    def _backend(self, result):
        result = result or {}
        settings = default_settings()
        settings.update(self.controller.settings)
        asr = result.get("asr", {})
        translation = result.get("translation", {})
        backend = asr.get("backend", settings["transcribe_backend"])
        device = asr.get("device", settings["kotoba_device"] if backend == "kotoba" else "cuda")
        asr_state = self._backend_state(asr, result)
        device_display = str(device).upper()
        asr_suffix = f" · {device_display}" if asr.get("available") and not asr.get("checking") else ""
        self.asr_label.setText(f"ASR · {str(backend).upper()} | {asr_state}{asr_suffix}")
        from scripts.asmr_transcribe_lib.cli import _effective_transcribe_model
        model = _effective_transcribe_model(backend, settings.get("transcribe_model"))
        driver_note = "CUDA 驅動：執行前另行驗證。\n" if asr.get("driver_checked") is False and device == "cuda" else ""
        self.asr_label.setToolTip(
            f"ASR 後端：{backend}\n模型：{self._safe_backend_detail(model)}\n裝置：{device_display}\n"
            f"本機前置資源：{asr_state}\n{self._backend_error(asr, result)}{driver_note}"
            "推論：尚未驗證（不載入模型或執行測試轉錄）。")
        translation_state = "本次不需要" if self.controller.mode == "transcribe_only" else self._backend_state(translation, result, translation=True)
        self.translation_label.setText(f"Sakura LLM | {translation_state}")
        try:
            endpoint = urlsplit(settings["translate_api"])
            host = endpoint.hostname or ""
            host = f"[{host}]" if ":" in host else host
            if endpoint.port is not None:
                host += f":{endpoint.port}"
            # Display only the origin. Paths, queries and userinfo may contain credentials.
            endpoint_display = urlunsplit((endpoint.scheme, host, "", "", ""))
        except (TypeError, ValueError):
            endpoint_display = "無效 endpoint"
        self.translation_label.setToolTip(
            f"Sakura API：{endpoint_display}\n模型：{self._safe_backend_detail(settings['translate_model'])}\n"
            f"API 連線：{self._availability_detail(translation.get('api_reachable'))}\n"
            f"設定模型：{self._availability_detail(translation.get('model_available'))}\n"
            f"狀態：{translation_state}\n{self._backend_error(translation, result)}"
            "推論：尚未驗證（不送出生成請求）。")
        self.asr_label.setStyleSheet(f"color:{THEME['error'] if asr.get('error_code') else THEME['muted']};")
        self.translation_label.setStyleSheet(f"color:{THEME['error'] if translation.get('error_code') and self.controller.mode != 'transcribe_only' else THEME['muted']};")

    def _backend_state(self, stage, result, *, translation=False):
        if stage.get("checking"):
            return "檢查中"
        if translation:
            if stage.get("api_reachable") is True:
                if stage.get("model_available") is False:
                    return "API 在線 · 模型不可用"
                if stage.get("model_available") is True:
                    return "API 在線 · 模型可用"
                return "API 在線 · 模型未確認"
            if stage.get("api_reachable") is False:
                return "無法連線"
        if stage.get("error_code"):
            return "檢查失敗" if translation else "本機不可用"
        if result.get("error_code"):
            return "檢查失敗"
        if "required" in stage and not stage["required"]:
            return "本計畫不需要"
        if stage.get("available"):
            return "API／模型可用" if translation else "本機可用"
        return "尚未驗證"

    @staticmethod
    def _availability_detail(value):
        return "可用" if value is True else "不可用" if value is False else "尚未確認"

    @staticmethod
    def _safe_backend_detail(value):
        text = str(value)
        if "://" in text:
            return "[已隱藏 URL；請於批次設定查看]"
        text = re.sub(r"(?i)(token|password|passwd|secret|api[_-]?key)\s*[:=]\s*\S+", r"\1=[已隱藏]", text)
        text = re.sub(r"(?i)\b(?:sk-|hf_)[A-Za-z0-9_-]+", "[已隱藏]", text)
        return text.replace("\n", " ").replace("\r", " ")[:240]

    @staticmethod
    def _backend_error(stage, result):
        code = str(stage.get("error_code") or result.get("error_code") or "")
        if not code:
            return ""
        # Error codes are safe diagnostics. Never echo raw backend exceptions or bodies.
        code = code if re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", code) else "BACKEND_CHECK_FAILED"
        reasons = {
            "SAKURA_UNREACHABLE": "無法連線至設定的 Sakura API，請檢查服務與連線設定。",
            "SAKURA_MODEL_MISSING": "API 可連線，但設定的模型 ID 不在服務提供的模型清單中。",
            "ASR_EXECUTABLE_MISSING": "找不到本機 XXL 執行檔，請檢查 ASR 設定。",
            "ASR_LOCAL_MODEL_MISSING": "找不到設定的本機 ASR 模型或必要模型檔案。",
            "ASR_LOCAL_MODEL_REQUIRED": "請設定完整的本機 ASR 模型目錄；不會自動下載模型。",
            "ASR_DEPENDENCY_MISSING": "缺少設定後端需要的本機 Python 套件。",
            "ASR_BACKEND_UNSUPPORTED": "不支援設定的 ASR 後端。",
            "ASR_DEVICE_UNSUPPORTED": "不支援設定的 ASR 運算裝置。",
            "ASR_CUDA_UNAVAILABLE": "本機 CUDA 裝置不可用，請檢查驅動程式與 ASR 裝置設定。",
            "ASR_COMPUTE_TYPE_UNSUPPORTED": "不支援設定的 ASR 計算精度。",
            "ASR_DEVICE_COMPUTE_MISMATCH": "設定的 ASR 裝置與計算精度不相容。",
            "ASR_CHECK_FAILED": "無法完成本機 ASR 前置資源檢查。",
            "TRANSLATION_ENDPOINT_INVALID": "Sakura API 位址格式無效，請檢查批次設定。",
            "TRANSLATION_CONNECTION_FAILED": "無法連線至 Sakura API，請檢查服務與連線設定。",
            "TRANSLATION_TIMEOUT": "Sakura API 未在時限內回應；可於服務恢復後重新檢查。",
            "TRANSLATION_HTTP_ERROR": "Sakura API 回傳 HTTP 錯誤，請檢查服務與存取設定。",
            "TRANSLATION_RESPONSE_INVALID": "Sakura API 未回傳有效的模型清單。",
            "TRANSLATION_RESPONSE_TOO_LARGE": "Sakura API 模型清單超過允許的檢查大小。",
            "TRANSLATION_MODEL_UNAVAILABLE": "API 可連線，但設定的模型 ID 不在服務提供的模型清單中。",
            "TRANSLATION_CHECK_FAILED": "無法完成 Sakura API 與模型可用性檢查。",
        }
        reason = reasons.get(code, "請檢查後端服務、本機模型與批次設定；詳細診斷請參閱完整日誌。")
        return f"錯誤：{code}\n{reason}\n"

    def _append_log(self, message):
        scrollbar = self.log_view.verticalScrollBar()
        previous = scrollbar.value()
        self.log_view.appendPlainText(str(message))
        if self.follow_log.isChecked():
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(previous)

    def _toggle_logs(self):
        visible = self.log_toggle.isChecked()
        if not visible and not self.log_view.isHidden():
            sizes = self.splitter.sizes()
            if sum(sizes):
                self._expanded_log_sizes = sizes
        self.log_view.setVisible(visible)
        self.log_toggle.setIcon(action_icon("chevron_down" if visible else "chevron_right"))
        self.log_panel.setMaximumHeight(16777215 if visible else 37)
        self.log_panel.setMinimumHeight(102 if visible else 37)
        if visible and getattr(self, "_expanded_log_sizes", None):
            self.splitter.setSizes(self._expanded_log_sizes)
        self.preferences.setValue("logs/visible", visible)

    def _open_output(self):
        root = self.controller.root
        if root:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(root)))

    def _open_journal(self):
        diagnostic = getattr(self.controller, "diagnostic_log_path", None)
        if diagnostic is not None and Path(diagnostic).is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(diagnostic)))
            return
        directory = self.controller.last_run_dir
        if directory is None:
            self._append_log("目前尚無執行日誌。")
            return
        path = Path(directory) / "events.jsonl"
        if path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            self._append_log("本次執行日誌尚未建立。")

    def _preferences(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("介面偏好")
        dialog.setMinimumWidth(320)
        layout = QVBoxLayout(dialog)
        show_logs = QCheckBox("顯示執行日誌")
        show_logs.setChecked(self.log_toggle.isChecked())
        follow = QCheckBox("自動捲動日誌")
        follow.setChecked(self.follow_log.isChecked())
        layout.addWidget(show_logs)
        layout.addWidget(follow)
        layout.addWidget(QLabel("介面偏好不影響本次執行設定。"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.log_toggle.setChecked(show_logs.isChecked())
            self.follow_log.setChecked(follow.isChecked())
            self._toggle_logs()

    def _safe_close(self):
        # close_ready means native cleanup was verified. Finalize the controller's
        # timer/executor lifecycle before allowing the actual window close.
        if not self.controller.request_close():
            return
        self._close_authorized = True
        self.close()

    def closeEvent(self, event):
        if self._close_authorized or self.controller.request_close():
            QApplication.instance().removeEventFilter(self)
            event.accept()
        else:
            event.ignore()
