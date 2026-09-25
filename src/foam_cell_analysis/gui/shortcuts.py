"""データ準備の既定キーとユーザー設定を管理する。"""

import json
import os
from pathlib import Path

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
    "toggle_view": "G",
    "help": "?",
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
    "toggle_view": "表とサムネイルを切り替え",
    "help": "キー一覧",
    "import": "取り込み",
    "auto_triage": "自動振り分け",
    "export_excel": "Excel出力",
    "import_excel": "Excel取込",
    "finalize": "データセット確定",
    "undo": "元に戻す",
    "redo": "やり直す",
    "search": "検索",
    "class_dialog": "分類を選ぶ",
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
    "toggle_view": "表 ⇔ サムネイル",
}
HINT_LABELS["class_1"] = "分類"


class ShortcutMap:
    """操作キーの取得・保存・重複検出を行う。"""

    def __init__(self, path: str | Path | None = None, mapping: dict[str, str] | None = None):
        self.path = (
            Path(path)
            if path
            else Path(os.getenv("APPDATA", Path.home())) / "foam-cell-analysis" / "keymap.json"
        )
        self.mapping = dict(DEFAULT_SHORTCUTS)
        self.mapping.update(mapping or {})
        self.load()

    def __getitem__(self, action: str) -> str:
        return self.mapping[action]

    def get(self, action: str, default: str = "") -> str:
        return self.mapping.get(action, default)

    @staticmethod
    def display_key(key: str) -> str:
        """画面で使うキー表記へ変換する。"""
        return key.replace("Control+", "Ctrl+").replace("Return", "Enter")

    def duplicates(self, action: str, key: str) -> list[str]:
        """指定キーを使う他の操作を返す。"""
        normalized = key.casefold().replace(" ", "")
        return [
            name
            for name, value in self.mapping.items()
            if name != action and value.casefold().replace(" ", "") == normalized
        ]

    def assign(self, action: str, key: str, swap: bool = False) -> list[str]:
        """キーを設定し、必要なら重複操作と入れ替える。"""
        conflicts = self.duplicates(action, key)
        if conflicts and swap:
            old = self.mapping.get(action, "")
            self.mapping[conflicts[0]] = old
        self.mapping[action] = key
        return conflicts

    def load(self) -> None:
        """設定ファイルが存在する場合に読み込む。"""
        if self.path.is_file():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.mapping.update(
                    {str(k): str(v) for k, v in data.items() if k in DEFAULT_SHORTCUTS}
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
        self.mapping.update({str(k): str(v) for k, v in data.items() if k in DEFAULT_SHORTCUTS})

    def hint_items(self) -> list[tuple[str, str]]:
        """常時表示するキーと操作の組を返す。"""
        order = (
            "usage_train",
            "usage_val",
            "usage_excluded",
            "usage_unassigned",
            "quality_good",
            "quality_ok",
            "quality_bad",
            "next_unassigned",
            "triage_view",
            "toggle_view",
        )
        keys = [self.display_key(self.mapping[f"class_{number}"]) for number in range(1, 10)]
        class_keys = "1–9" if keys == [str(number) for number in range(1, 10)] else " / ".join(keys)
        hints = [(self.display_key(self.mapping[key]), HINT_LABELS[key]) for key in order]
        hints.insert(4, (class_keys, "分類"))
        return hints
