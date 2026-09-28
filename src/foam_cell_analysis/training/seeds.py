"""学習段階ごとの安定した乱数 seed を作る。"""

from __future__ import annotations

import hashlib


def derive(seed: int, *parts: object) -> int:
    """設計書 8.8 の規則で 32 bit seed を導出する。"""
    value = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**32)
