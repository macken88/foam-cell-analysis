"""画面に出す処理失敗文。例外の詳細は各呼び出し元でログへ残す。"""

import re

PREPARE_FAILED_TEXT = "処理の準備に失敗しました。作業フォルダのパスが長すぎる可能性があります。"
UNEXPECTED_FAILED_TEXT = "処理を完了できませんでした。ログを確認してください。"


def value_error_message(error: BaseException, fallback: str) -> str:
    """既存画面契約どおり ValueError の区切りより前だけを返す。"""
    if not isinstance(error, ValueError):
        return fallback
    return str(error).split(": ", 1)[0].strip() or fallback


def user_failure_message(
    error: BaseException,
    *,
    prepare_message: str = PREPARE_FAILED_TEXT,
    fallback: str = UNEXPECTED_FAILED_TEXT,
    preserve_value_error: bool = False,
) -> str:
    """例外種別ごとの既存方針を保ち、パスなどの詳細を画面から隠す。"""
    if isinstance(error, OSError):
        return prepare_message
    if isinstance(error, ValueError):
        message = value_error_message(error, fallback)
        if preserve_value_error or re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", message):
            return message
        return fallback
    return fallback
