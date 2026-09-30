"""成果物の整理ダイアログ（比較・評価設計 19.5）。"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ....services.models import ArtifactGroup, PruneResult
from ...labels import format_bytes
from ...theme import numeric_font, set_style
from ...widgets.table import mark_primary

logger = logging.getLogger(__name__)

# 種類（category）ごとの表示名と、選ぶ前に知らせる注意
CLEANUP_CATEGORIES: tuple[tuple[str, str, str], ...] = (
    ("fold_periodic", "fold の定期保存モデル", ""),
    (
        "fold_selected",
        "fold の選択エポックのモデル",
        "削除すると、将来の OOF 再評価ができなくなります",
    ),
    ("final", "最終学習モデル（final.pt）", ""),
    ("temporary", "中断・失敗した試行の一時ファイル", ""),
)
# 初めに選んでおく種類（学習記録の再評価に使わないもの）
DEFAULT_CHECKED = {"fold_periodic", "temporary"}
NO_FILES_TEXT = "削除できるファイルはありません"
# 理由ごとに並べる対象の数（超えた分は「ほか N 件」にまとめる）
_MAX_LISTED_TARGETS = 3


def _target_text(groups: list[ArtifactGroup]) -> str:
    names = [f"{group.experiment_id} 試行 {group.attempt}" for group in groups]
    if len(names) > _MAX_LISTED_TARGETS:
        rest = len(names) - _MAX_LISTED_TARGETS
        names = [*names[:_MAX_LISTED_TARGETS], f"ほか {rest} 件"]
    return "、".join(names)


class ArtifactCleanupDialog(QDialog):
    """選んだ実験の大きなファイルを種類ごとに選んで削除する。"""

    def __init__(self, backend, experiment_ids: list[str], parent=None) -> None:
        super().__init__(parent)
        self.backend = backend
        self.experiment_ids = list(experiment_ids)
        self.result_value: PruneResult | None = None
        self.setWindowTitle("成果物を整理")
        self.setMinimumWidth(720)
        plan = backend.artifact_cleanup_plan(self.experiment_ids)
        layout = QVBoxLayout(self)
        targets = "、".join(self.experiment_ids)
        self.target_label = QLabel(f"対象の実験: {targets}（{len(self.experiment_ids)} 件）")
        self.target_label.setWordWrap(True)
        layout.addWidget(self.target_label)
        intro = QLabel(
            "記録（設定・学習曲線・OOF 評価・ログ）は残り、選んだファイルだけを削除します。"
        )
        intro.setWordWrap(True)
        set_style(intro, role="note")
        layout.addWidget(intro)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setColumnStretch(0, 1)
        for column, text in enumerate(("削除するもの", "ファイル数", "容量")):
            header = QLabel(text)
            set_style(header, role="note")
            alignment = Qt.AlignmentFlag.AlignLeft if column == 0 else Qt.AlignmentFlag.AlignRight
            grid.addWidget(header, 0, column, alignment)
        self.checks: dict[str, QCheckBox] = {}
        self.notes: dict[str, QLabel] = {}
        row = 1
        for category, label, warning in CLEANUP_CATEGORIES:
            groups = [group for group in plan if group.category == category]
            deletable = [group for group in groups if group.deletable]
            blocked = [group for group in groups if not group.deletable]
            check = QCheckBox(label)
            check.setAccessibleName(label)
            n_files = sum(group.n_files for group in deletable)
            size_bytes = sum(group.size_bytes for group in deletable)
            count_label = QLabel(str(n_files) if deletable else "—")
            size_label = QLabel(format_bytes(size_bytes) if deletable else "—")
            for value in (count_label, size_label):
                value.setFont(numeric_font())
            grid.addWidget(check, row, 0)
            grid.addWidget(count_label, row, 1, Qt.AlignmentFlag.AlignRight)
            grid.addWidget(size_label, row, 2, Qt.AlignmentFlag.AlignRight)
            row += 1
            lines = [warning] if warning and deletable else []
            lines.extend(self._blocked_lines(blocked, whole=not deletable))
            if not deletable and not blocked:
                lines.append(NO_FILES_TEXT)
            if deletable:
                check.setChecked(category in DEFAULT_CHECKED)
            else:
                check.setChecked(False)
                check.setEnabled(False)
                check.setToolTip("\n".join(lines))
            if lines:
                note = QLabel("\n".join(lines))
                note.setWordWrap(True)
                note.setIndent(24)
                set_style(note, role="note")
                grid.addWidget(note, row, 0, 1, 3)
                self.notes[category] = note
                row += 1
            check.toggled.connect(self._update_freed)
            self.checks[category] = check
        layout.addLayout(grid)

        self.freed_label = QLabel()
        self.freed_label.setFont(numeric_font())
        layout.addWidget(self.freed_label)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_button = QPushButton("キャンセル")
        self.delete_button = QPushButton("削除する")
        mark_primary(self.delete_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.delete_button)
        layout.addLayout(buttons)
        self.cancel_button.clicked.connect(self.reject)
        self.delete_button.clicked.connect(self._confirm_delete)
        self.freed_bytes = 0
        self._update_freed()

    @staticmethod
    def _blocked_lines(blocked: list[ArtifactGroup], *, whole: bool) -> list[str]:
        """削除しないまとまりを、理由ごとに 1 行ずつ並べる。"""
        by_reason: dict[str, list[ArtifactGroup]] = {}
        for group in blocked:
            by_reason.setdefault(group.reason or "削除できる条件を満たしていません", []).append(
                group
            )
        lines = []
        for reason, groups in by_reason.items():
            if whole and len(by_reason) == 1:
                lines.append(f"削除しません: {reason}")
            else:
                lines.append(
                    f"うち {len(groups)} 件（{_target_text(groups)}）は削除しません: {reason}"
                )
        return lines

    def selected_categories(self) -> list[str]:
        return [
            category
            for category, check in self.checks.items()
            if check.isEnabled() and check.isChecked()
        ]

    def _update_freed(self, _checked: bool = False) -> None:
        """選んだ種類で空く容量を見積もり直す。"""
        categories = self.selected_categories()
        self.freed_bytes = 0
        if categories:
            try:
                self.freed_bytes = self.backend.estimate_freed_bytes(
                    self.experiment_ids, categories
                )
            except (KeyError, ValueError, OSError):
                logger.exception("空く容量を見積もれません")
                self.freed_label.setText("空く容量: 見積もれません")
                self.delete_button.setEnabled(False)
                self.delete_button.setToolTip("容量を見積もれないため削除できません")
                return
        self.freed_label.setText(f"空く容量: {format_bytes(self.freed_bytes)}")
        enabled = bool(categories)
        self.delete_button.setEnabled(enabled)
        self.delete_button.setToolTip("" if enabled else "削除するものを選んでください")

    def _confirm_delete(self) -> None:
        categories = self.selected_categories()
        if not categories:
            return
        answer = QMessageBox.question(
            self,
            "成果物を整理",
            f"{format_bytes(self.freed_bytes)} を削除します。取り消せません。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.result_value = self.backend.prune_artifacts(self.experiment_ids, categories)
        except (KeyError, ValueError, OSError):
            logger.exception("成果物を整理できません")
            QMessageBox.warning(
                self,
                "成果物を整理できません",
                "ファイルを削除できませんでした。実験の状態を確かめてから、もう一度実行してください。",
            )
            return
        self.accept()


def cleanup_status_message(result: PruneResult) -> str:
    """整理の結果をステータスバー用の文にする。"""
    message = f"成果物を整理しました（{format_bytes(result.freed_bytes)} 空きました）"
    if result.skipped:
        message += f"。条件を満たさない {len(result.skipped)} 件は削除していません"
    return message
