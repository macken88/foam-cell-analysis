import sys

from PySide6.QtWidgets import QApplication, QMainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("気泡インスタンスセグメンテーション")
    window.resize(1280, 800)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
