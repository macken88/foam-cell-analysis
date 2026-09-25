"""キー割り当てを変更するダイアログ。"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .shortcuts import LABELS, ShortcutMap
from .theme import set_style


class KeymapDialog(QDialog):
    """重複確認・入出力を備えたキー割り当て設定画面。"""

    def __init__(self, parent=None, shortcut_map: ShortcutMap | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("キー割り当て")
        self.setMinimumSize(620, 500)
        self.shortcuts = shortcut_map or ShortcutMap()
        layout = QVBoxLayout(self)
        self.table = QTableWidget(len(self.shortcuts.mapping), 2)
        self.table.setHorizontalHeaderLabels(["操作", "キー"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 300)
        self.actions = list(self.shortcuts.mapping)
        for row, action in enumerate(self.actions):
            label = QTableWidgetItem(LABELS.get(action, action))
            label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
            key = QTableWidgetItem(self.shortcuts.display_key(self.shortcuts[action]))
            self.table.setItem(row, 0, label)
            self.table.setItem(row, 1, key)
        layout.addWidget(self.table)
        self.duplicate_label = QLabel("")
        set_style(self.duplicate_label, state="error")
        layout.addWidget(self.duplicate_label)
        row = QHBoxLayout()
        for title, callback in (("書き出し…", self.export), ("読み込み…", self.import_map)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        self.buttons.accepted.connect(self.save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.table.itemChanged.connect(self._key_changed)

    def _key_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 1:
            return
        action = self.actions[item.row()]
        key = QKeySequence(item.text()).toString(QKeySequence.SequenceFormat.PortableText)
        conflicts = self.shortcuts.duplicates(action, key)
        if conflicts:
            titles = "、".join(LABELS.get(name, name) for name in conflicts)
            self.duplicate_label.setText(f"重複: {titles}")
            answer = QMessageBox.question(
                self,
                "キーが重複しています",
                f"{titles} と入れ替えますか？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                old_key = self.shortcuts[action]
                self.shortcuts.assign(action, key, swap=True)
                for row, name in enumerate(self.actions):
                    if name in conflicts:
                        self.table.item(row, 1).setText(old_key)
                self.duplicate_label.clear()
            else:
                item.setText(self.shortcuts[action])
        else:
            self.shortcuts.mapping[action] = key
            self.duplicate_label.clear()

    def save(self) -> None:
        """重複のない設定を保存して閉じる。"""
        duplicates = [
            (key, names)
            for key in self.shortcuts.mapping
            for names in [self.shortcuts.duplicates(key, self.shortcuts[key])]
            if names
        ]
        if duplicates:
            self.duplicate_label.setText("キーの重複を解消してください")
            return
        self.shortcuts.save()
        self.accept()

    def export(self) -> None:
        """キー割り当てを JSON へ書き出す。"""
        path, _ = QFileDialog.getSaveFileName(
            self, "キー割り当てを書き出す", "keymap.json", "JSON (*.json)"
        )
        if path:
            self.shortcuts.export_to(path)

    def import_map(self) -> None:
        """JSON の割り当てを読み込んで一覧に反映する。"""
        path, _ = QFileDialog.getOpenFileName(self, "キー割り当てを読み込む", "", "JSON (*.json)")
        if path:
            try:
                self.shortcuts.import_from(path)
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, "読み込みエラー", str(error))
                return
            for row, action in enumerate(self.actions):
                self.table.item(row, 1).setText(self.shortcuts[action])
