"""アプリ全体で共有する色、フォント、Qt スタイル。"""

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QFont, QFontDatabase, QValidator
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
    QToolTip,
    QWidget,
)


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


def _input_owner(widget) -> QWidget | None:
    """ホイールで値が変わる入力部品（数値欄・選択欄）を、子部品からたどって返す。"""
    while isinstance(widget, QWidget) and not widget.isWindow():
        if isinstance(widget, QAbstractSpinBox | QComboBox):
            return widget
        widget = widget.parentWidget()
    return None


def _scroll_area_for(widget: QWidget) -> QAbstractScrollArea | None:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QAbstractScrollArea):
            return parent
        if parent.isWindow():
            break
        parent = parent.parentWidget()
    return None


def spin_range_message(box: QAbstractSpinBox) -> str:
    """数値欄に入力できる範囲の説明を返す。"""
    return (
        f"{box.textFromValue(box.minimum())}〜{box.textFromValue(box.maximum())} "
        "の範囲の数値を入力してください"
    )


class InputGuard(QObject):
    """入力部品の共通の振る舞いをアプリ全体で揃えるイベントフィルター。

    - ホイールでは値を変えず、外側のスクロール領域をスクロールする（フォーカス中も同じ）。
    - 数値欄の上下ボタンを表示しない（値はキーボードで入力する）。
    - 数値欄の入力が範囲外・不正な間は赤枠にし、ツールチップで範囲を示す。
    """

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QEvent.Type.Wheel:
            owner = _input_owner(watched)
            if owner is not None:
                area = _scroll_area_for(owner)
                if area is not None:
                    QApplication.sendEvent(area.viewport(), event)
                return True
        elif kind in (QEvent.Type.Polish, QEvent.Type.Show) and isinstance(
            watched, QAbstractSpinBox
        ):
            self._prepare_spin_box(watched)
        elif (
            kind == QEvent.Type.KeyPress
            and isinstance(watched, QLineEdit)
            and isinstance(watched.parentWidget(), QAbstractSpinBox)
        ):
            self._watch_rejected_key(watched.parentWidget(), event)
        return False

    def _prepare_spin_box(self, box: QAbstractSpinBox) -> None:
        if box.buttonSymbols() != QAbstractSpinBox.ButtonSymbols.NoButtons:
            box.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        if box.property("_inputGuarded"):
            return
        box.setProperty("_inputGuarded", True)
        editor = box.lineEdit()
        if editor is not None:
            editor.textChanged.connect(lambda _text, target=box: self._validate(target))
            box.editingFinished.connect(lambda target=box: self._validate(target))

    @staticmethod
    def _validate(box: QAbstractSpinBox) -> None:
        editor = box.lineEdit()
        if editor is None:
            return
        state = box.validate(editor.text(), editor.cursorPosition())
        state = state[0] if isinstance(state, tuple) else state
        invalid = state != QValidator.State.Acceptable
        if invalid == (box.property("state") == "invalid"):
            return
        if invalid:
            box.setProperty("baseToolTip", box.toolTip())
            box.setToolTip(spin_range_message(box))
            set_style(box, state="invalid")
        else:
            box.setToolTip(str(box.property("baseToolTip") or ""))
            box.setProperty("baseToolTip", None)
            set_style(box, state="")

    @staticmethod
    def _watch_rejected_key(box: QAbstractSpinBox, event) -> None:
        text = event.text()
        if not text or not text.isprintable():
            return
        if event.modifiers() & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
        ):
            return
        editor = box.lineEdit()
        before = editor.text()

        def check() -> None:
            try:
                if editor.text() == before:
                    QToolTip.showText(
                        box.mapToGlobal(box.rect().bottomLeft()), spin_range_message(box), box
                    )
            except RuntimeError:
                pass

        QTimer.singleShot(0, check)


def install_input_guard(app: QApplication | None = None) -> None:
    """InputGuard をアプリへ一度だけ取り付ける。"""
    app = app or QApplication.instance()
    if app is None or getattr(app, "_input_guard", None) is not None:
        return
    app._input_guard = InputGuard(app)
    app.installEventFilter(app._input_guard)


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
        f"QSpinBox[state='invalid'], QDoubleSpinBox[state='invalid'], "
        f"QSpinBox[state='invalid']:focus, QDoubleSpinBox[state='invalid']:focus "
        f"{{ border: 2px solid {c.ERROR}; background: {c.ERROR_BG}; }}",
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
        f"QProgressBar#trainingProgressBar {{ background: {c.RULE}; border: 0; "
        "border-radius: 2px; }",
        f"QProgressBar#trainingProgressBar::chunk {{ background: {c.GRAPHITE}; "
        "border-radius: 2px; }",
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
    install_input_guard(app)


def set_style(widget, **props: str | bool) -> None:
    """動的スタイルプロパティを更新して QSS を再適用する。"""
    for name, value in props.items():
        widget.setProperty(name, value)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()
