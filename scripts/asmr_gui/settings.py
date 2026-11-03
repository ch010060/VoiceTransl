"""Transactional settings draft using the legacy parser as its defaults source."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

_EXCLUDED = {"path", "execute", "dry_run", "limit", "transcribe_only"}


def default_settings(root=".") -> dict:
    from scripts.asmr_transcribe_lib.cli import build_parser
    values = vars(build_parser().parse_args(["--path", str(root)]))
    return {key: value for key, value in values.items() if key not in _EXCLUDED}


class SettingsDialog(QDialog):
    def __init__(self, settings, mode="full", limit=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批次設定")
        self.resize(650, 590)
        self.setMinimumWidth(560)
        self.fields = {}
        layout = QVBoxLayout(self)
        intro = QLabel("設定於按下確定後套用；重設僅變更本視窗草稿。")
        intro.setObjectName("muted")
        layout.addWidget(intro)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        resources = self._tab(tabs, "執行與資源")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("完整處理（轉錄與翻譯）", "full")
        self.mode_combo.addItem("僅轉錄（不需要 Sakura）", "transcribe_only")
        resources.addRow("處理模式", self.mode_combo)
        for key, label in (("parallel", "整體並行數"), ("transcribe_parallel", "ASR 並行數"),
                           ("transcribe_batch_size", "ASR batch size"),
                           ("translate_chunk_size", "Sakura 每次字幕行數")):
            self._spin(resources, key, label)
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(0, 2147483647)
        self.limit_spin.setSpecialValueText("不限制")
        self.limit_spin.setToolTip("仍須通過代表音軌的依賴完整性驗證；不會自動增加上限。")
        resources.addRow("本次 Track 上限", self.limit_spin)
        self._check(resources, "fast_transcribe", "啟用 XXL 快速轉錄預設（sentence／20 秒 chunk）")
        asr = self._tab(tabs, "ASR")
        backend = QComboBox()
        backend.addItem("XXL（本機執行檔）", "xxl")
        backend.addItem("Kotoba（實驗性，僅本機模型）", "kotoba")
        self.fields["transcribe_backend"] = backend
        asr.addRow("ASR 後端", backend)
        for key, label in (("transcribe_model", "模型（留空使用 CLI 預設）"),
                           ("src_lang", "來源語言"),
                           ("transcribe_compute_type", "Compute type"),
                           ("kotoba_device", "Kotoba 裝置")):
            self._text(asr, key, label)
        self._spin(asr, "kotoba_chunk_length", "Kotoba chunk 秒數")
        self._text(asr, "faster_whisper_param", "命令模板（僅供顯示）", readonly=True)
        info = QLabel("模型與服務須預先備妥；後端檢查不執行推論。")
        info.setWordWrap(True)
        info.setObjectName("muted")
        asr.addRow(info)
        translation = self._tab(tabs, "翻譯與提示")
        self._text(translation, "translate_api", "Sakura API endpoint")
        self._text(translation, "translate_model", "Sakura 精確模型 ID")
        self._check(translation, "no_asmr_prompt", "停用內建 ASMR 詞彙提示")
        self.custom_prompt = QCheckBox("使用自訂全域 initial prompt")
        translation.addRow(self.custom_prompt)
        self._text(translation, "initial_prompt", "自訂 initial prompt")
        self.custom_prompt.toggled.connect(self.fields["initial_prompt"].setEnabled)
        hint = QLabel("Sakura 的啟動與關閉由外部管理。僅轉錄模式不要求 Sakura 連線。")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        translation.addRow(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.reset_button = QPushButton("重設草稿")
        buttons.addButton(self.reset_button, QDialogButtonBox.ButtonRole.ResetRole)
        self.reset_button.clicked.connect(self.reset_draft)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        draft = default_settings()
        draft.update({key: value for key, value in settings.items() if key in draft})
        self._load(draft, mode, limit)

    def _tab(self, tabs, title):
        widget = QWidget()
        form = QFormLayout(widget)
        form.setContentsMargins(18, 18, 18, 18)
        form.setVerticalSpacing(14)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        tabs.addTab(widget, title)
        return form

    def _spin(self, form, key, label):
        field = QSpinBox()
        field.setRange(1, 2147483647)
        self.fields[key] = field
        form.addRow(label, field)

    def _text(self, form, key, label, readonly=False):
        field = QLineEdit()
        field.setReadOnly(readonly)
        self.fields[key] = field
        form.addRow(label, field)

    def _check(self, form, key, label):
        field = QCheckBox(label)
        self.fields[key] = field
        form.addRow(field)

    def _load(self, settings, mode, limit):
        for key, field in self.fields.items():
            value = settings[key]
            if isinstance(field, QCheckBox):
                field.setChecked(value)
            elif isinstance(field, QSpinBox):
                field.setValue(value)
            elif isinstance(field, QComboBox):
                field.setCurrentIndex(max(0, field.findData(value)))
            else:
                field.setText("" if value is None else str(value))
        self.custom_prompt.setChecked(settings["initial_prompt"] is not None)
        self.fields["initial_prompt"].setEnabled(self.custom_prompt.isChecked())
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(mode)))
        self.limit_spin.setValue(limit or 0)

    def reset_draft(self):
        self._load(default_settings(), "full", None)

    def values(self):
        values = {}
        for key, field in self.fields.items():
            if isinstance(field, QCheckBox):
                values[key] = field.isChecked()
            elif isinstance(field, QSpinBox):
                values[key] = field.value()
            elif isinstance(field, QComboBox):
                values[key] = field.currentData()
            else:
                values[key] = field.text()
        values["transcribe_model"] = values["transcribe_model"].strip() or None
        if not self.custom_prompt.isChecked():
            values["initial_prompt"] = None
        return values, self.mode_combo.currentData(), self.limit_spin.value() or None
