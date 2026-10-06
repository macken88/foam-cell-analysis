"""依存のないファイルハッシュ関数。"""

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """ファイルの sha256 を 1 MiB ずつ読んで求める。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
