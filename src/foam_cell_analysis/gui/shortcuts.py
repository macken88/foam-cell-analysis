"""データ準備の既定キーとユーザー設定を管理する。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import QKeyCombination, QObject, Qt, Signal
from PySide6.QtGui import QKeyEvent, QKeySequence

DEFAULT_SHORTCUTS = {
    "usage_train": "Q",
    "usage_val": "W",
    "usage_excluded": "E",
    "usage_unassigned": "R",
    "quality_good": "Z",
    "quality_ok": "X",
    "quality_bad": "C",
    "next_unassigned": "Space",
    "triage_view": "Return",
    "display_mode": "M",
    "previous_image": "Left",
    "next_image": "Right",
    "help": "?",
    "home": "Ctrl+H",
    "import": "Ctrl+I",
    "auto_triage": "Ctrl+D",
    "export_excel": "Ctrl+E",
    "import_excel": "Ctrl+Shift+E",
    "finalize": "Ctrl+Return",
    "undo": "Ctrl+Z",
    "redo": "Ctrl+Y",
    "search": "Ctrl+F",
    "class_1": "1",
    "class_2": "2",
    "class_3": "3",
    "class_4": "4",
    "class_5": "5",
    "class_6": "6",
    "class_7": "7",
    "class_8": "8",
    "class_9": "9",
    "class_dialog": "K",
    "class_clear": "0",
    "select_all": "Ctrl+A",
    "clear_selection": "Esc",
    "previous_unassigned": "Shift+Space",
    "zoom_in": "+",
    "zoom_out": "-",
    "fit_view": "F",
    "filter_all": "Alt+1",
    "filter_unassigned": "Alt+2",
    "filter_train": "Alt+3",
    "filter_val": "Alt+4",
    "filter_excluded": "Alt+5",
    "filter_errors": "Alt+6",
}

LABELS = {
    "usage_train": "用途を学習にする",
    "usage_val": "用途を検証にする",
    "usage_excluded": "用途を不採用にする",
    "usage_unassigned": "用途を未振り分けにする",
    "quality_good": "品質を良にする",
    "quality_ok": "品質を可にする",
    "quality_bad": "品質を不良にする",
    "next_unassigned": "次の未振り分けへ",
    "triage_view": "連続振り分け",
    "display_mode": "表示形式を切り替え",
    "previous_image": "前の画像",
    "next_image": "次の画像",
    "help": "キー一覧",
    "home": "ホームへ戻る",
    "import": "取り込み",
    "auto_triage": "自動振り分け",
    "export_excel": "Excel出力",
    "import_excel": "Excel取込",
    "finalize": "データセット確定",
    "undo": "元に戻す",
    "redo": "やり直す",
    "search": "検索",
    "class_dialog": "分類を選ぶ",
    "class_clear": "分類を未設定にする",
    "select_all": "表示中のすべてを選択",
    "clear_selection": "選択・絞り込みを解除",
    "previous_unassigned": "前の未振り分けへ",
    "zoom_in": "拡大",
    "zoom_out": "縮小",
    "fit_view": "全体表示",
}
for _number in range(1, 10):
    LABELS[f"class_{_number}"] = f"分類 {_number} を設定"
for _number, _name in enumerate(("すべて", "未振り分け", "学習", "検証", "不採用", "エラー"), 1):
    _action = (
        "filter_all",
        "filter_unassigned",
        "filter_train",
        "filter_val",
        "filter_excluded",
        "filter_errors",
    )[_number - 1]
    LABELS[_action] = f"{_name}で絞り込む"

HINT_LABELS = {
    "usage_train": "学習",
    "usage_val": "検証",
    "usage_excluded": "不採用",
    "usage_unassigned": "未振り分け",
    "quality_good": "良",
    "quality_ok": "可",
    "quality_bad": "不良",
    "next_unassigned": "次へ",
    "triage_view": "連続振り分け",
    "display_mode": "表示形式",
    "previous_image": "前の画像",
    "next_image": "次の画像",
    "previous_unassigned": "前へ",
    "class_clear": "分類解除",
    "zoom_in": "拡大",
    "zoom_out": "縮小",
    "fit_view": "全体表示",
    "help": "キー一覧",
}
HINT_LABELS["class_1"] = "分類"


class ShortcutMap(QObject):
    """操作キーの取得・保存・重複検出を行う。"""

    changed = Signal()

    def __init__(self, path: str | Path | None = None, mapping: dict[str, str] | None = None):
        super().__init__()
        self.path = (
            Path(path)
            if path
            else Path(os.getenv("APPDATA", Path.home())) / "foam-cell-analysis" / "keymap.json"
        )
        self.mapping = dict(DEFAULT_SHORTCUTS)
        self.mapping.update(
            {key: self.normalize_key_text(value) for key, value in (mapping or {}).items()}
        )
        self.load()

    def __getitem__(self, action: str) -> str:
        return self.mapping[action]

    def get(self, action: str, default: str = "") -> str:
        return self.mapping.get(action, default)

    @staticmethod
    def display_key(key: str) -> str:
        """画面で使うキー表記へ変換する。"""
        return (
            key.replace("Control+", "Ctrl+")
            .replace("Return", "Enter")
            .replace("Left", "←")
            .replace("Right", "→")
        )

    def duplicates(self, action: str, key: str) -> list[str]:
        """指定キーを使う他の操作を返す。"""
        normalized = self.normalize_key_text(key).casefold().replace(" ", "")
        return [
            name
            for name, value in self.mapping.items()
            if name != action
            and self.normalize_key_text(value).casefold().replace(" ", "") == normalized
        ]

    def assign(self, action: str, key: str, swap: bool = False) -> list[str]:
        """キーを設定し、必要なら重複操作と入れ替える。"""
        key = self.normalize_key_text(key)
        conflicts = self.duplicates(action, key)
        if conflicts and swap:
            old = self.mapping.get(action, "")
            self.mapping[conflicts[0]] = old
        self.mapping[action] = key
        return conflicts

    def copy(self) -> ShortcutMap:
        """現在の割り当てを独立した作業用マップへ複製する。"""
        return ShortcutMap(self.path, self.mapping)

    @staticmethod
    def normalize_sequence(sequence: QKeySequence) -> QKeySequence:
        """記号キーの配列差を吸収した一打鍵のシーケンスを返す。"""
        if sequence.count() != 1:
            return sequence
        combination = sequence[0]
        key = combination.key()
        modifiers = combination.keyboardModifiers()
        aliases = {
            Qt.Key.Key_BraceLeft: Qt.Key.Key_BracketLeft,
            Qt.Key.Key_BraceRight: Qt.Key.Key_BracketRight,
            Qt.Key.Key_Underscore: Qt.Key.Key_Minus,
        }
        if key == Qt.Key.Key_Equal and modifiers & Qt.KeyboardModifier.ShiftModifier:
            key = Qt.Key.Key_Plus
        elif key == Qt.Key.Key_Slash and modifiers & Qt.KeyboardModifier.ShiftModifier:
            key = Qt.Key.Key_Question
        else:
            key = aliases.get(key, key)
        if key in {
            Qt.Key.Key_Plus,
            Qt.Key.Key_Question,
            Qt.Key.Key_BracketLeft,
            Qt.Key.Key_BracketRight,
            Qt.Key.Key_Minus,
        }:
            modifiers &= ~Qt.KeyboardModifier.ShiftModifier
        return QKeySequence(QKeyCombination(modifiers, key))

    @classmethod
    def normalize_key_text(cls, key: str) -> str:
        """記号キーを設定ファイル用の標準表記へ変換する。"""
        sequence = cls.normalize_sequence(QKeySequence(key.replace("Control+", "Ctrl+")))
        normalized = sequence.toString(QKeySequence.SequenceFormat.PortableText)
        return normalized or key

    @classmethod
    def event_sequence(cls, event: QKeyEvent) -> QKeySequence:
        """キーイベントを設定値と比較できる標準シーケンスへ変換する。"""
        return cls.normalize_sequence(QKeySequence(event.keyCombination()))

    def matches(self, action: str, event: QKeyEvent) -> bool:
        """イベントのキーと修飾キーが操作の割り当てに一致するか返す。"""
        return self.event_sequence(event) == self.normalize_sequence(QKeySequence(self[action]))

    def replace(self, mapping: dict[str, str]) -> None:
        """既知の割り当てを置き換え、変更を通知する。"""
        self.mapping = dict(DEFAULT_SHORTCUTS)
        self.mapping.update(
            {
                key: self.normalize_key_text(value)
                for key, value in mapping.items()
                if key in DEFAULT_SHORTCUTS
            }
        )
        self.changed.emit()

    def load(self) -> None:
        """設定ファイルが存在する場合に読み込む。"""
        if self.path.is_file():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.mapping.update(
                    {
                        str(key): self.normalize_key_text(str(value))
                        for key, value in data.items()
                        if key in DEFAULT_SHORTCUTS
                    }
                )

    def save(self) -> None:
        """ユーザー設定へ保存する。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def export_to(self, path: str | Path) -> None:
        """指定先へ割り当てを書き出す。"""
        Path(path).write_text(
            json.dumps(self.mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def import_from(self, path: str | Path) -> None:
        """指定ファイルから既知の操作キーを読み込む。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("キー割り当てファイルの形式が正しくありません")
        self.mapping.update(
            {
                str(key): self.normalize_key_text(str(value))
                for key, value in data.items()
                if key in DEFAULT_SHORTCUTS
            }
        )

    def hint_items(self) -> list[tuple[str, str]]:
        """常時表示する主なキーと操作の組を返す。"""
        order = (
            "usage_train",
            "usage_val",
            "usage_excluded",
            "usage_unassigned",
            "class_1",
            "quality_good",
            "quality_ok",
            "quality_bad",
            "next_unassigned",
            "triage_view",
            "display_mode",
            "help",
        )
        keys = [self.display_key(self.mapping[f"class_{number}"]) for number in range(1, 10)]
        class_keys = "1–9" if keys == [str(number) for number in range(1, 10)] else " / ".join(keys)
        hints = []
        for action in order:
            key = class_keys if action == "class_1" else self.display_key(self.mapping[action])
            label = "分類" if action == "class_1" else HINT_LABELS[action]
            if action == "next_unassigned":
                label = "次の未振り分け"
            hints.append((key, label))
        return hints
