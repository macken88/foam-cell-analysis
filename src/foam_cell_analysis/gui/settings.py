"""アプリの設定（QSettings）の取得を 1 か所にまとめる。"""

import os

from PySide6.QtCore import QSettings

ORGANIZATION = "FoamCellAnalysis"
APPLICATION = "FoamCellAnalysis"
SETTINGS_FILE_ENV = "FOAM_SETTINGS_FILE"


def app_settings() -> QSettings:
    """アプリ共通の設定を返す。

    環境変数 FOAM_SETTINGS_FILE があれば、その INI ファイルを使う
    （テストや調査用スクリプトが利用者の設定を書き換えないようにするため）。
    なければ OS 標準の保存先（Windows ではレジストリ）を使う。
    """
    path = os.environ.get(SETTINGS_FILE_ENV)
    if path:
        return QSettings(path, QSettings.Format.IniFormat)
    return QSettings(ORGANIZATION, APPLICATION)
