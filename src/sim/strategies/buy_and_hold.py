"""比較用: 最初に買って放置（買い持ち）。

毎日「買い」の合図を出し続ける。最初の日の合図で翌取引日に買い、
その後は積立が入るたびに買い増す。売ることはない。
"""

from __future__ import annotations

from .base import Decision, DayContext, Signal, Strategy


class BuyAndHold(Strategy):
    name = "buy_and_hold"
    description = "最初に買って放置（積立分も買い増し）"

    def decide(self, ctx: DayContext) -> Decision:
        return Decision(Signal.BUY)
