"""Four-column Work/Track tree with stable plan identities and view-only filters."""
from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6.QtCore import QAbstractItemModel, QModelIndex, QSortFilterProxyModel, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QIcon, QPixmap, QPainter, QPen

from .run_model import TERMINAL, format_elapsed, status_label

ROLE_NODE = int(Qt.ItemDataRole.UserRole) + 1
HEADERS = ("作品 / Track", "狀態", "進度", "經過時間")
TONE_COLORS = {"muted": "#8c98aa", "active": "#6fa7ff", "success": "#72c895",
               "danger": "#f27e89", "warning": "#e8bb68"}


@dataclass(eq=False)
class _Node:
    kind: str
    id: str
    row: int
    title: str
    parent: _Node | None = None
    children: list[_Node] = field(default_factory=list)
    search_text: str = ""


class WorkTrackModel(QAbstractItemModel):
    """Only Work checkboxes edit execution selection; Track rows are read-only."""

    def __init__(self, run_model, parent=None):
        super().__init__(parent)
        self.run_model = run_model
        self._works = []
        self._nodes = {}
        self._audio_icon = None
        run_model.changed.connect(self._changed)
        self._rebuild()
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._refresh_elapsed)
        self._timer.start()

    def _rebuild(self):
        self.beginResetModel()
        self._works, self._nodes = [], {}
        plan = self.run_model.plan
        if plan is not None:
            for row, work in enumerate(plan.works):
                title = "根目錄音軌（虛擬作品）" if work.kind == "synthetic_root" else work.relative_path
                node = _Node("work", work.work_id, row, title, search_text=title.casefold())
                self._works.append(node)
                self._nodes[node.id] = node
                for track_id in work.track_ids:
                    track = self.run_model.track_index[track_id]
                    title = track.task.audio_path.relative_to(track.task.work_dir).as_posix()
                    child = _Node("track", track_id, len(node.children), title, node, search_text=title.casefold())
                    node.children.append(child)
                    self._nodes[track_id] = child
        self.endResetModel()

    def _changed(self, change):
        if change.get("reset"):
            self._rebuild()
            return
        if change.get("lock"):
            if self._works:
                self.dataChanged.emit(self.index(0, 0), self.index(len(self._works) - 1, 0), [])
            return
        for node_id in (*change.get("work_ids", ()), *change.get("track_ids", ())):
            node = self._nodes.get(node_id)
            if node is not None:
                self.dataChanged.emit(self.createIndex(node.row, 0, node), self.createIndex(node.row, 3, node), [])

    def _refresh_elapsed(self):
        # Only active rows repaint. A 100k-Track catalog is not walked per tick.
        for node_id in self.run_model.active_track_ids | self.run_model.active_work_ids:
            node = self._nodes.get(node_id)
            if node is not None:
                index = self.createIndex(node.row, 3, node)
                self.dataChanged.emit(index, index, [int(Qt.ItemDataRole.DisplayRole), ROLE_NODE])

    def index_for_id(self, node_id, column=0):
        node = self._nodes.get(node_id)
        return self.createIndex(node.row, column, node) if node is not None and 0 <= column < 4 else QModelIndex()

    def index(self, row, column, parent=QModelIndex()):
        if row < 0 or not 0 <= column < 4 or parent.isValid() and parent.column() != 0:
            return QModelIndex()
        nodes = parent.internalPointer().children if parent.isValid() else self._works
        return self.createIndex(row, column, nodes[row]) if row < len(nodes) else QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        parent = index.internalPointer().parent
        return self.createIndex(parent.row, 0, parent) if parent is not None else QModelIndex()

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return len(parent.internalPointer().children) if parent.column() == 0 else 0
        return len(self._works)

    def columnCount(self, parent=QModelIndex()):
        return 4

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole and 0 <= section < 4:
            return HEADERS[section]
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        node = index.internalPointer()
        if node.kind == "work" and index.column() == 0 and node.children and not self.run_model.locked:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if (not index.isValid() or index.column() != 0 or role != Qt.ItemDataRole.CheckStateRole or
                index.internalPointer().kind != "work"):
            return False
        selected = value in (Qt.CheckState.Checked, Qt.CheckState.Checked.value)
        return self.run_model.set_work_selected(index.internalPointer().id, selected)

    def _node_data(self, node):
        if node.kind == "work":
            summary = self.run_model.work_summary(node.id)
            return {"kind": "work", "id": node.id, **summary}
        state = self.run_model.tracks[node.id]
        tone = ("danger" if state.status == "failed" else "warning" if state.status in {"cancelled", "not_started", "interrupted"}
                or state.warnings else "success" if state.status in TERMINAL else "active" if state.status == "running" else "muted")
        if state.status in TERMINAL or state.status == "interrupted":
            progress = {"mode": "terminal", "ratio": None, "caption": status_label(state)}
        elif state.status == "running" and state.stage == "asr" and state.ratio is not None:
            progress = {"mode": "reported", "ratio": state.ratio, "caption": f"ASR {state.ratio:.0%}"}
        elif state.status == "running":
            progress = {"mode": "indeterminate", "ratio": None, "caption": status_label(state)}
        else:
            progress = {"mode": "terminal", "ratio": None, "caption": "—"}
        return {"kind": "track", "id": node.id, "status": state.status, "tone": tone,
                "progress": progress, "elapsed": self.run_model.track_elapsed(node.id)}

    def _track_icon(self):
        if self._audio_icon is None:
            pixmap = QPixmap(16, 16)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor("#8798b2"), 1.5))
            for x, height in ((3, 4), (6, 10), (9, 14), (12, 7)):
                painter.drawLine(x, (16 - height) // 2, x, (16 + height) // 2)
            painter.end()
            self._audio_icon = QIcon(pixmap)
        return self._audio_icon

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        node, column = index.internalPointer(), index.column()
        if role == ROLE_NODE:
            return self._node_data(node)
        if role == Qt.ItemDataRole.CheckStateRole:
            if node.kind == "work" and column == 0:
                return Qt.CheckState.Checked if node.id in self.run_model.selected_work_ids else Qt.CheckState.Unchecked
            return None
        if role == Qt.ItemDataRole.DecorationRole and column == 0 and node.kind == "track":
            return self._track_icon()
        if role == Qt.ItemDataRole.FontRole and node.kind == "work":
            font = QFont()
            font.setBold(True)
            return font
        if role == Qt.ItemDataRole.TextAlignmentRole and column == 3:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role == Qt.ItemDataRole.ForegroundRole and column in (1, 2):
            return QColor(TONE_COLORS[self._node_data(node)["tone"]])
        if role == Qt.ItemDataRole.ToolTipRole:
            if node.kind == "work":
                work = self.run_model.work_index[node.id]
                return f"{node.title}\n{work.kind}\n{node.id}"
            track, state = self.run_model.track_index[node.id], self.run_model.tracks[node.id]
            lines = [node.title, status_label(state), f"計畫動作：{track.task.action}"]
            if track.dependencies:
                lines.append("依賴：" + ", ".join(track.dependencies))
            if track.representative_track_id:
                lines.append("代表音軌：" + track.representative_track_id)
            if state.reason:
                lines.append("原因：" + str(state.reason))
            if state.warnings:
                lines.append("警告：" + ", ".join(state.warnings))
            return "\n".join(lines)
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if column == 0:
            return node.title
        info = self._node_data(node)
        if column == 3:
            return format_elapsed(info["elapsed"])
        if column == 2:
            return info["progress"]["caption"]
        if node.kind == "track":
            state = self.run_model.tracks[node.id]
            return status_label(state) + (" ⚠" if state.warnings else "")
        label = status_label(info["status"])
        if info["failed"]:
            label += f" · {info['failed']} 音軌"
        elif info["tone"] == "warning" and info["finished"]:
            label += " ⚠"
        return label


class WorkFilterProxyModel(QSortFilterProxyModel):
    """Text and status are presentation filters; they never change selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._query, self._status = "", "all"
        self._work_match = {}
        self.setDynamicSortFilter(True)
        self.setRecursiveFilteringEnabled(False)

    def setSourceModel(self, source_model):
        previous = self.sourceModel()
        if previous is not None:
            try:
                previous.modelReset.disconnect(self._reset_matches)
            except TypeError:
                pass
        super().setSourceModel(source_model)
        if source_model is not None:
            source_model.modelReset.connect(self._reset_matches)
        self._reset_matches()

    def _reset_matches(self):
        self._work_match.clear()
        self.invalidateFilter()

    def set_query(self, query):
        self._query = str(query).strip().casefold()
        self._reset_matches()

    def set_status_filter(self, status):
        if status not in {"all", "selected", "active", "failed", "finished"}:
            raise ValueError("unknown Work status filter")
        self._status = status
        self.invalidateFilter()

    def _status_matches(self, work):
        model = self.sourceModel().run_model
        if self._status == "all":
            return True
        if self._status == "selected":
            return work.id in model.selected_work_ids
        summary = model.work_summary(work.id)
        return bool(summary[{"active": "active", "failed": "failed", "finished": "finished"}[self._status]])

    def filterAcceptsRow(self, source_row, source_parent):
        source = self.sourceModel()
        index = source.index(source_row, 0, source_parent)
        if not index.isValid():
            return False
        node = index.internalPointer()
        work = node if node.kind == "work" else node.parent
        if not self._status_matches(work):
            return False
        if not self._query or self._query in work.search_text:
            return True
        if node.kind == "track":
            return self._query in node.search_text
        if node.id not in self._work_match:
            self._work_match[node.id] = any(self._query in child.search_text for child in node.children)
        return self._work_match[node.id]
