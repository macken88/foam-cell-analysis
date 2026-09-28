"""キー割り当てを変更するダイアログ。"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .shortcuts import LABELS, ShortcutMap
from .theme import set_style

SHORTCUT_CATEGORIES = {
    "usage_": "振り分け",
    "quality_": "品質",
    "class_": "分類",
    "filter_": "絞り込み",
    "previous_image": "画像移動",
    "next_image": "画像移動",
    "display_mode": "表示",
    "zoom_": "表示",
    "fit_view": "表示",
    "help": "ヘルプ",
    "import": "データ準備",
    "auto_triage": "データ準備",
    "finalize": "データ準備",
    "search": "データ準備",
}


def shortcut_category(action: str) -> str:
    """操作 ID から利用者向け分類を返す。"""
    return next(
        (label for prefix, label in SHORTCUT_CATEGORIES.items() if action.startswith(prefix)),
        "共通操作",
    )


class ShortcutKeyEdit(QLineEdit):
    """一打鍵のキーと修飾キーを記録する欄。"""

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            event.accept()
            return
        if event.key() == Qt.Key.Key_Backspace:
            self.clear()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.text():
            super().keyPressEvent(event)
            return
        sequence = ShortcutMap.event_sequence(event)
        key = sequence.toString(QKeySequence.SequenceFormat.PortableText)
        self.setText(ShortcutMap.display_key(key))
        event.accept()


class ShortcutKeyDelegate(QStyledItemDelegate):
    """キー一覧で一打鍵のキー編集欄を作る。"""

    def createEditor(self, parent, option, index):
        return ShortcutKeyEdit(parent)

    def setEditorData(self, editor, index) -> None:
        editor.setText(str(index.data(Qt.ItemDataRole.EditRole) or ""))

    def setModelData(self, editor, model, index) -> None:
        model.setData(index, editor.text(), Qt.ItemDataRole.EditRole)


class KeymapDialog(QDialog):
    """重複確認・入出力を備えたキー割り当て設定画面。"""

    def __init__(self, parent=None, shortcut_map: ShortcutMap | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("キー割り当て")
        self.setMinimumSize(620, 500)
        self.target_shortcuts = shortcut_map
        self.shortcuts = shortcut_map.copy() if shortcut_map else ShortcutMap()
        layout = QVBoxLayout(self)
        self.table = QTableWidget(len(self.shortcuts.mapping), 3)
        self.table.setHorizontalHeaderLabels(["操作", "分類", "キー"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 300)
        self.table.setItemDelegateForColumn(2, ShortcutKeyDelegate(self.table))
        self.actions = list(self.shortcuts.mapping)
        for row, action in enumerate(self.actions):
            label = QTableWidgetItem(LABELS.get(action, action))
            label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
            key = QTableWidgetItem(self.shortcuts.display_key(self.shortcuts[action]))
            category = QTableWidgetItem(shortcut_category(action))
            category.setFlags(category.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, label)
            self.table.setItem(row, 1, category)
            self.table.setItem(row, 2, key)
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
        if item.column() != 2:
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
                        self.table.item(row, 2).setText(self.shortcuts.display_key(old_key))
                self.duplicate_label.clear()
            else:
                item.setText(self.shortcuts.display_key(self.shortcuts[action]))
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
        if self.target_shortcuts:
            self.target_shortcuts.replace(self.shortcuts.mapping)
            self.target_shortcuts.save()
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
                self.table.item(row, 2).setText(self.shortcuts.display_key(self.shortcuts[action]))


class KeymapWindow(QDialog):
    """現在の割り当てを一覧し、変更ダイアログを開く非モーダル画面。"""

    def __init__(self, parent, shortcut_map: ShortcutMap) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("キー割り当て一覧")
        self.resize(640, 560)
        self.target_shortcuts = shortcut_map
        layout = QVBoxLayout(self)
        self.table = QTableWidget(len(shortcut_map.mapping), 3)
        self.table.setHorizontalHeaderLabels(["操作名", "分類", "現在のキー"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.actions = list(shortcut_map.mapping)
        for row, action in enumerate(self.actions):
            values = (
                LABELS.get(action, action),
                shortcut_category(action),
                shortcut_map.display_key(shortcut_map[action]),
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(False)
        layout.addWidget(self.table)
        self.change_button = QPushButton("変更…")
        self.change_button.clicked.connect(self.change)
        layout.addWidget(self.change_button)
        shortcut_map.changed.connect(self.refresh)

    def refresh(self) -> None:
        """共有キー割り当てを一覧へ反映する。"""
        for row, action in enumerate(self.actions):
            self.table.item(row, 2).setText(
                self.target_shortcuts.display_key(self.target_shortcuts[action])
            )

    def change(self) -> None:
        """割り当て編集ダイアログを開く。"""
        dialog = KeymapDialog(self, self.target_shortcuts)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh()


def show_keymap_window(parent, context) -> KeymapWindow:
    """アプリ共通のキー一覧を作成または前面表示する。"""
    window = context.keymap_window
    if window is None:
        window = KeymapWindow(parent, context.shortcuts)
        context.keymap_window = window
    window.show()
    window.raise_()
    window.activateWindow()
    return window
