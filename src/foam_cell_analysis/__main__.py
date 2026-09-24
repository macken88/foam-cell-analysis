"""アプリケーションの起動入口。"""

from .app import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
