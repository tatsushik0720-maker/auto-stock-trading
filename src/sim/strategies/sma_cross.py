"""移動平均戦略: 終値が N日移動平均を上回ったら買い、下回ったら売り。

移動平均(SMA): 直近 N 取引日の終値の平均。値動きをならした「流れ」を表す。
終値がそれより上 = 上昇の流れ、下 = 下落の流れ、とみなす単純なルール。
取引がなかった日は数えない（直近 N 回の取引日の平均）。
"""

from __future__ import annotations

from .base import Decision, DayContext, Signal, Strategy, StrategyError


class SmaCross(Strategy):
    name = "sma_cross"
    description = "終値が移動平均を上回ったら買い、下回ったら売り"

    def __init__(self, window: int = 25):
        if isinstance(window, bool) or not isinstance(window, int) or window < 2:
            raise StrategyError(f"sma_cross: window は 2 以上の整数にしてください（今: {window!r}）")
        self.window = window

    def warmup_days(self) -> int:
        return self.window

    def params(self) -> dict:
        return {"window": self.window}

    def decide(self, ctx: DayContext) -> Decision:
        close = ctx.prices["Close"]
        if len(close) < self.window:
            return Decision(Signal.HOLD, {"sma": None})   # 平均を出すには日数が足りない
        sma = float(close.iloc[-self.window:].mean())
        price = float(close.iloc[-1])
        if price > sma:
            signal = Signal.BUY
        elif price < sma:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD   # ちょうど同じなら様子見
        return Decision(signal, {"sma": sma})
