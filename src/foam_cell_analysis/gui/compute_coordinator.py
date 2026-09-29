"""学習と評価の計算処理を 1 件ずつ実行させる排他制御（比較・推論設計 15.1）。"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

OWNER_NAMES = {"training": "学習", "evaluation": "評価"}


@dataclass(eq=False)
class Ticket:
    """占有の要求 1 件。state は waiting / active / released / cancelled。"""

    owner: str
    label: str
    start: Callable[[], None] = field(repr=False)
    number: int = 0
    state: str = "waiting"


class ComputeCoordinator(QObject):
    """計算処理の占有を先着順で 1 件ずつ与える。

    - request: 空いていればすぐ start を呼ぶ。使用中なら待ち行列に入れる。
    - release: 終端処理（status.json の保存）が終わってから呼ぶ。
      ok=False は終端処理の保存に失敗したことを表し、次の待機要求を開始しない
      「停止中」になる。利用者が確認したら unblock() で再開する。
    - 実行中の処理には割り込まない。
    """

    changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._active: Ticket | None = None
        self._waiting: list[Ticket] = []
        self._blocked_error: str | None = None
        self._numbers = itertools.count(1)
        self._dispatching = False

    # ---- 状態 ----

    @property
    def is_busy(self) -> bool:
        """占有中、または保存失敗で停止中なら True。"""
        return self._active is not None or self._blocked_error is not None

    @property
    def active(self) -> Ticket | None:
        """現在占有している要求。"""
        return self._active

    @property
    def active_owner(self) -> str | None:
        return self._active.owner if self._active else None

    @property
    def active_label(self) -> str | None:
        return self._active.label if self._active else None

    @property
    def waiting(self) -> list[Ticket]:
        """待機中の要求（先着順の写し）。"""
        return list(self._waiting)

    @property
    def is_blocked(self) -> bool:
        """終端処理の保存に失敗し、次の要求を開始しない状態か。"""
        return self._blocked_error is not None

    @property
    def blocked_error(self) -> str | None:
        return self._blocked_error

    def wait_message(self, owner: str | None = None) -> str | None:
        """owner の要求が待たされる理由を画面向けの文にする。待たないなら None。"""
        if self._blocked_error is not None:
            return (
                "前の処理の終了状態を保存できなかったため、次の処理を開始していません: "
                f"{self._blocked_error}"
            )
        if self._active is None or self._active.owner == owner:
            return None
        name = OWNER_NAMES.get(self._active.owner, self._active.label)
        return f"{name}の終了を待っています"

    # ---- 操作 ----

    def request(self, owner: str, label: str, start: Callable[[], None]) -> Ticket:
        """占有を求める。空いていればこの呼び出しの中で start を呼ぶ。

        すぐ開始した start が例外を送出したときは占有を返してから例外を送り直す。
        """
        ticket = Ticket(owner, label, start, next(self._numbers))
        self._waiting.append(ticket)
        failure = self._dispatch(raise_for=ticket)
        self.changed.emit()
        if failure is not None:
            raise failure
        return ticket

    def release(self, ticket: Ticket, ok: bool = True, error: str | None = None) -> None:
        """占有を返す。ok=False なら停止中にして次を開始しない。

        待機中の要求を渡した場合は cancel と同じ。返却済みなら何もしない。
        """
        if ticket.state == "waiting":
            self.cancel(ticket)
            return
        if ticket.state != "active" or ticket is not self._active:
            return
        ticket.state = "released"
        self._active = None
        if not ok:
            self._blocked_error = error or "終端状態を保存できませんでした"
        if not self._dispatching:
            self._dispatch()
        self.changed.emit()

    def cancel(self, ticket: Ticket) -> None:
        """待機中の要求を取り消す。実行中の要求には何もしない。"""
        if ticket.state != "waiting":
            return
        ticket.state = "cancelled"
        if ticket in self._waiting:
            self._waiting.remove(ticket)
        self.changed.emit()

    def unblock(self) -> None:
        """保存失敗による停止を解除し、待機中の要求があれば開始する。"""
        if self._blocked_error is None:
            return
        self._blocked_error = None
        if not self._dispatching:
            self._dispatch()
        self.changed.emit()

    def _dispatch(self, raise_for: Ticket | None = None) -> BaseException | None:
        """空いていれば待機中の要求を先頭から開始する。

        start の中で release された場合（準備の失敗など）は次の要求へ進む。
        raise_for の start が送出した例外だけを返し、ほかは記録して続ける。
        """
        if self._dispatching:
            # start の中から呼ばれた。外側のループが続きを処理する
            return None
        failure = None
        self._dispatching = True
        try:
            while self._active is None and self._blocked_error is None and self._waiting:
                ticket = self._waiting.pop(0)
                ticket.state = "active"
                self._active = ticket
                try:
                    ticket.start()
                except Exception as error:
                    if ticket is self._active:
                        ticket.state = "released"
                        self._active = None
                    if ticket is raise_for:
                        failure = error
                    else:
                        logger.exception("%s を開始できませんでした", ticket.label)
        finally:
            self._dispatching = False
        return failure
