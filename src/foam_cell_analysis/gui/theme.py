"""アプリ全体で共有する色、フォント、Qt スタイル。"""

from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QSpinBox


class Color:
    """設計書で定めた意味色。"""

    STAGE = "#EEF1F3"
    SLIDE = "#FFFFFF"
    GRAPHITE = "#1C2630"
    SLATE = "#5E6B76"
    RULE = "#D3D9DE"
    RULE_SOFT = "#E4E8EB"
    SELECTION = "#D9DFE5"
    SELECTION_CHANGED = "#EDE3B8"
    TRAIN = "#2F5F9E"
    TRAIN_BG = "#E3EBF6"
    VAL = "#C2661B"
    VAL_INK = "#A3520F"
    VAL_BG = "#F8E9DC"
    IDLE = "#8A949C"
    IDLE_BG = "#ECEEF0"
    ERROR = "#B42318"
    ERROR_BG = "#FBE9E7"
    OK = "#2A7147"
    OK_BG = "#E4F1E8"
    CHANGED = "#FFF4CC"
    CHANGED_INK = "#8A5D00"
    IMAGE_BG = "#2E3338"
    HEADER_BG = "#F5F7F8"
    HOVER_BG = "#F6F8F9"
    STATUS_BG = "#E2E6E9"
    CONTROL_RULE = "#BCC4CB"
    DISABLED = "#A6AEB5"
    KEY_RULE = "#B3BBC2"
    WHITE = "#FFFFFF"


SERIES = ("#4A3AA7", "#008300", "#E87BA4", "#EDA100")


class _FocusedWheelFilter(QObject):
    """数値欄と選択欄のホイール操作はフォーカス中だけ許可する。"""

    def eventFilter(self, watched, event):
        if isinstance(watched, (QSpinBox, QDoubleSpinBox, QComboBox)):
            focus_widget = QApplication.focusWidget()
            focused = watched.hasFocus() or (
                focus_widget is not None and watched.isAncestorOf(focus_widget)
            )
            if event.type() == QEvent.Type.Wheel and not focused:
                event.ignore()
                return True
        return False


def _family(*candidates: str) -> str:
    """インストール済みフォントから候補順に書体を選ぶ。

    どれもなければ OS の既定の書体を返す（存在しない名前を渡すと
    Qt が無関係な書体で代用するため）。
    """
    families = set(QFontDatabase.families())
    for name in candidates:
        if name in families:
            return name
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()


def body_font(point_size: float = 10) -> QFont:
    """日本語本文用フォントを返す。"""
    font = QFont(_family("BIZ UDPGothic", "Yu Gothic UI", "Meiryo UI"))
    font.setPointSizeF(point_size)
    return font


def numeric_font(point_size: float | None = None) -> QFont:
    """等幅数字用フォントを返す。"""
    font = QFont(_family("Bahnschrift", "Segoe UI"))
    font.setPointSizeF(10 if point_size is None else point_size)
    font.setFeature(QFont.Tag("tnum"), 1)
    return font


def mono_font(point_size: float = 9) -> QFont:
    """設定プレビュー用等幅フォントを返す。"""
    font = QFont(_family("Consolas"))
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPointSizeF(point_size)
    return font


def build_stylesheet() -> str:
    """2.6.4 の共通部品定義からアプリ全体の QSS を作る。"""
    c = Color
    rules = [
        f"QWidget {{ color: {c.GRAPHITE}; }}",
        f"QMainWindow, QDialog {{ background: {c.STAGE}; }}",
        f"QWidget[role='panel'] {{ background: {c.SLIDE}; border: 1px solid {c.RULE}; }}",
        f"QWidget[role='errorBanner'] {{ background: {c.ERROR_BG}; color: {c.ERROR}; "
        f"border: 1px solid {c.ERROR}; }}",
        "QLabel#pageHeading { font-size: 16pt; font-weight: 600; }",
        f"QPushButton {{ background: {c.SLIDE}; color: {c.GRAPHITE}; "
        f"border: 1px solid {c.CONTROL_RULE}; border-radius: 3px; "
        "min-height: 28px; padding: 0 12px; }",
        f"QPushButton:hover {{ background: {c.IDLE_BG}; }}",
        f"QPushButton#trainingPreviewButton:checked {{ background: {c.IDLE_BG}; }}",
        f"QPushButton:disabled {{ color: {c.DISABLED}; border-color: {c.RULE_SOFT}; }}",
        f"QMenu::item:disabled {{ color: {c.DISABLED}; }}",
        f"QPushButton[role='filterToggle']:checked[usage='error'] "
        f"{{ background: {c.ERROR_BG}; color: {c.ERROR}; border-color: {c.ERROR}; }}",
        f"QPushButton[role='filterToggle']:checked[usage='changed'] "
        f"{{ background: {c.CHANGED}; border-color: {c.GRAPHITE}; }}",
        f'QPushButton[primary="true"] {{ background: {c.GRAPHITE}; '
        f"color: {c.WHITE}; border-color: {c.GRAPHITE}; }}",
        f'QPushButton[primary="true"]:disabled {{ background: {c.IDLE}; color: {c.IDLE_BG}; }}',
        f"QTabBar::tab {{ color: {c.SLATE}; border: 0; padding: 6px 12px; "
        "border-bottom: 2px solid transparent; }",
        f"QTabBar::tab:selected {{ color: {c.GRAPHITE}; font-weight: bold; "
        f"border-bottom-color: {c.GRAPHITE}; }}",
        f"QTableView, QTableWidget {{ background: {c.SLIDE}; color: {c.GRAPHITE}; "
        f"border: 1px solid {c.RULE}; gridline-color: transparent; "
        f"selection-background-color: {c.SELECTION}; selection-color: {c.GRAPHITE}; }}",
        f"QTableView::item, QTableWidget::item {{ border: 0; "
        f"border-bottom: 1px solid {c.RULE_SOFT}; padding: 4px 7px; }}",
        f"QTableView::item:selected, QTableWidget::item:selected "
        f"{{ background: {c.SELECTION}; border: 0; }}",
        "QTableView::item:focus, QTableWidget::item:focus { border: 0; outline: 0; }",
        f"QTableView QHeaderView::section, QTableWidget QHeaderView::section "
        f"{{ background: {c.HEADER_BG}; color: {c.SLATE}; border: 0; "
        f"border-bottom: 1px solid {c.RULE}; padding: 4px 7px; font-weight: normal; }}",
        f"QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit "
        f"{{ background: {c.SLIDE}; color: {c.GRAPHITE}; "
        f"border: 1px solid {c.CONTROL_RULE}; border-radius: 3px; padding: 3px 6px; }}",
        "QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, "
        f"QPlainTextEdit:focus, QTextEdit:focus {{ border: 2px solid {c.GRAPHITE}; }}",
        f"QLineEdit:read-only, QSpinBox:read-only, QDoubleSpinBox:read-only "
        f"{{ background: {c.IDLE_BG}; color: {c.SLATE}; }}",
        f"QPlainTextEdit:read-only, QTextEdit:read-only "
        f"{{ background: {c.IDLE_BG}; color: {c.SLATE}; }}",
        f"QWidget[role='readonly'] {{ background: {c.IDLE_BG}; color: {c.SLATE}; padding: 5px; }}",
        f"QComboBox[state='changed'] {{ background: {c.CHANGED}; }}",
        f"QGroupBox {{ border: 0; border-top: 1px solid {c.RULE}; "
        "margin-top: 12px; padding-top: 8px; }",
        "QGroupBox[trainingSection='true'] { margin-top: 22px; padding-top: 8px; }",
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 0; "
        f"padding-right: 6px; color: {c.GRAPHITE}; }}",
        "QGroupBox[trainingSection='true']::title { font-size: 11pt; font-weight: 600; }",
        f"QLabel[role='sectionHeading'] {{ color: {c.GRAPHITE}; "
        f"font-family: '{body_font(12).family()}'; font-size: 12pt; font-weight: 700; }}",
        f"QFrame[role='sectionHeadingRule'] {{ background: {c.RULE}; "
        "max-height: 1px; border: 0; }",
        f"QFrame#trainingSummaryCard {{ background: {c.SLIDE}; border: 1px solid {c.RULE}; "
        f"border-left: 4px solid {c.GRAPHITE}; }}",
        f"QStatusBar {{ background: {c.STATUS_BG}; color: {c.SLATE}; }}",
        f"QFrame#pipelinePanel {{ background: {c.SLIDE}; border: 1px solid {c.RULE}; }}",
        "QFrame#pipelineStage { background: transparent; border: 0; }",
        f"QFrame#pipelineStage:hover {{ background: {c.HOVER_BG}; }}",
        f"QFrame#pipelineStage:focus {{ border: 1px solid {c.GRAPHITE}; }}",
        f"QFrame#homeUserRow {{ background: {c.SLIDE}; border: 1px solid {c.RULE}; }}",
        f"QFrame[role='filterPanel'] {{ background: {c.SLIDE}; border: 1px solid {c.RULE}; }}",
        f"QFrame#homeUserRow:hover {{ background: {c.HOVER_BG}; }}",
        f"QFrame#homeUserRow:focus {{ border: 1px solid {c.GRAPHITE}; }}",
        f"QFrame#recentOperations {{ border: 0; border-top: 1px solid {c.RULE}; }}",
        "QPushButton[role='ghost'] { background: transparent; border: 0; padding: 3px 8px; }",
        f"QPushButton[role='ghost']:hover {{ background: {c.IDLE_BG}; }}",
        "QPushButton[role='segment'] { border-radius: 0; padding: 3px 9px; }",
        f"QPushButton[role='segment']:checked {{ background: {c.GRAPHITE}; "
        f"color: {c.WHITE}; border-color: {c.GRAPHITE}; }}",
        f"QWidget#modeTopbar {{ background: {c.STAGE}; border: 0; "
        f"border-bottom: 1px solid {c.RULE}; }}",
        "QTabWidget#modeTabs { background: transparent; border: 0; }",
        "QTabWidget#modeTabs::pane { border: 0; background: transparent; }",
        "QTabWidget#modeTabs QWidget { background: transparent; }",
        f"QTabBar#modeTabBar {{ background: {c.STAGE}; border: 0; }}",
        f"QListWidget#mainSidebar {{ background: {c.STAGE}; border: 0; }}",
        "QListWidget#mainSidebar::item { padding: 6px 12px; }",
        f"QListWidget#mainSidebar::item:selected "
        f"{{ background: {c.SELECTION}; color: {c.GRAPHITE}; }}",
        f"QLabel[role='note'] {{ color: {c.SLATE}; font-size: 9pt; }}",
        f"QLabel[role='numeric'] {{ font-family: '{_family('Bahnschrift', 'Segoe UI')}'; }}",
        f"QLabel[state='ok'] {{ color: {c.OK}; }}",
        f"QLabel[state='error'] {{ color: {c.ERROR}; }}",
        f"QLabel[state='changed'] {{ background: {c.CHANGED}; }}",
        f"QLabel[state='warning'] {{ color: {c.VAL_INK}; }}",
        f"QLabel[role='keycap'] {{ background: {c.SLIDE}; "
        f"border: 1px solid {c.KEY_RULE}; border-bottom: 2px solid {c.KEY_RULE}; "
        f"border-radius: 3px; padding: 0 5px; "
        f"font-family: '{_family('Bahnschrift', 'Segoe UI')}'; }}",
        f"QLabel[role='imagePlaceholder'] {{ color: {c.SLATE}; background: {c.IMAGE_BG}; }}",
        f"QLabel[role='imageOverlayLabel'] {{ color: {c.WHITE}; background: {c.GRAPHITE}; "
        "padding: 2px 5px; border: 0; font-size: 8pt; }",
        f"QFrame[role='chipSeparator'] {{ background: {c.RULE}; border: 0; }}",
        f"QLabel[usage='train'] {{ background: {c.TRAIN_BG}; color: {c.TRAIN}; "
        "border-radius: 3px; padding: 0 7px; }",
        f"QLabel[usage='val'] {{ background: {c.VAL_BG}; color: {c.VAL_INK}; "
        "border-radius: 3px; padding: 0 7px; }",
        f"QLabel[usage='unassigned'], QLabel[usage='excluded'] "
        f"{{ background: {c.IDLE_BG}; color: {c.SLATE}; "
        "border-radius: 3px; padding: 0 7px; }",
        "QLabel[usage='excluded'], QLabel[state='非採用'] { text-decoration: line-through; }",
        f"QLabel[state='実行中'], QLabel[state='評価中'] "
        f"{{ background: {c.GRAPHITE}; color: {c.WHITE}; "
        "border-radius: 3px; padding: 0 7px; }",
        f"QLabel[state='完了'], QLabel[state='リリース済み'] "
        f"{{ background: {c.OK_BG}; color: {c.OK}; "
        "border-radius: 3px; padding: 0 7px; }",
        f"QLabel[state='下書き'], QLabel[state='中断'], QLabel[state='候補'], "
        f"QLabel[state='非採用'] {{ background: {c.IDLE_BG}; color: {c.SLATE}; "
        "border-radius: 3px; padding: 0 7px; }",
        f"QLabel[state='失敗'] {{ background: {c.ERROR_BG}; color: {c.ERROR}; "
        "border-radius: 3px; padding: 0 7px; }",
        f"QPushButton[role='countChip'] {{ border: 1px solid {c.RULE}; background: {c.SLIDE}; }}",
        f"QPushButton[role='countChip']:hover {{ background: {c.HOVER_BG}; }}",
        f"QPushButton[role='countChip']:checked {{ background: {c.GRAPHITE}; "
        f"color: {c.WHITE}; border-color: {c.GRAPHITE}; }}",
        f"QPushButton[role='countChip'][usage='error']:checked {{ background: {c.ERROR_BG}; "
        f"color: {c.ERROR}; border-color: {c.ERROR}; }}",
        f"QLabel[state='activeChipText'] {{ color: {c.WHITE}; }}",
        f"QPushButton[usage='error'] {{ color: {c.ERROR}; }}",
        f"QPushButton[usage='error'] QLabel {{ color: {c.ERROR}; }}",
        f"QLabel[role='countSwatch'][usage='train'] {{ background: {c.TRAIN}; }}",
        f"QLabel[role='countSwatch'][usage='val'] {{ background: {c.VAL}; }}",
        f"QLabel[role='countSwatch'][usage='unassigned'], "
        f"QLabel[role='countSwatch'][usage='excluded'] {{ background: {c.IDLE}; }}",
        "QPushButton::menu-indicator { image: none; width: 0; }",
    ]
    return "\n".join(rules)


def apply_theme(app: QApplication) -> None:
    """アプリの既定フォントと共通スタイルを設定する。"""
    app.setFont(body_font())
    app.setStyleSheet(build_stylesheet())
    if not hasattr(app, "_focused_wheel_filter"):
        app._focused_wheel_filter = _FocusedWheelFilter(app)
        app.installEventFilter(app._focused_wheel_filter)


def set_style(widget, **props: str | bool) -> None:
    """動的スタイルプロパティを更新して QSS を再適用する。"""
    for name, value in props.items():
        widget.setProperty(name, value)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()
