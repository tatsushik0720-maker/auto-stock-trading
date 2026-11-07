"""間隔あけ移動平均戦略: 買い⇄売りを切り替えたら、最低 cooldown 営業日は次の切り替えをしない。

基本は sma_cross と同じ（終値が N日平均より上なら買い、下なら売り）。
ただし合図が切り替わった日から cooldown 取引日のあいだは、平均の上下が入れ替わっても
前の判断を続ける。短い間隔での売買の繰り返しを止めるためのルール。

数え方: 切り替えた日を 0 日目として、cooldown 日目から次の切り替えができる。
        その銘柄に取引がない日は数えない。
「売買」はあくまで買い⇄売りの切り替えのこと。買いの状態が続くあいだの
積立分の買い増しは止めない（毎月の入金を遊ばせないため）。

記憶を持たない決まりを守るため、最初の日から合図の流れをたどり直して今日の状態を求める。
"""

from __future__ import annotations

import numpy as np

from .base import Decision, DayContext, Signal, Strategy, StrategyError
from .sma_band import rolling_mean


class SmaCooldown(Strategy):
    name = "sma_cooldown"
    description = "移動平均の上下で売買。切り替えたら最低 cooldown 営業日は次の切り替えをしない"

    def __init__(self, window: int = 25, cooldown: int = 20):
        if isinstance(window, bool) or not isinstance(window, int) or window < 2:
            raise StrategyError(
                f"sma_cooldown: window は 2 以上の整数にしてください（今: {window!r}）")
        if isinstance(cooldown, bool) or not isinstance(cooldown, int) or cooldown < 1:
            raise StrategyError(
                f"sma_cooldown: cooldown は 1 以上の整数にしてください（今: {cooldown!r}）")
        self.window = window
        self.cooldown = cooldown

    def warmup_days(self) -> int:
        return self.window

    def params(self) -> dict:
        return {"window": self.window, "cooldown": self.cooldown}

    def decide(self, ctx: DayContext) -> Decision:
        close = ctx.prices["Close"].to_numpy(dtype=float)
        if len(close) < self.window:
            return Decision(Signal.HOLD, {"sma": None})
        sma = rolling_mean(close, self.window)
        state, since = cooldown_state(close, sma, self.cooldown)
        return Decision(state, {"sma": float(sma[-1]), "days_since_switch": since})


def cooldown_state(close: np.ndarray, sma: np.ndarray, cooldown: int) -> tuple[Signal, int | None]:
    """最後の日の状態と、最後に切り替えてからの取引日数。"""
    state = Signal.HOLD
    last_switch: int | None = None
    for i in range(len(close)):
        if np.isnan(sma[i]):
            continue
        raw = Signal.BUY if close[i] > sma[i] else (Signal.SELL if close[i] < sma[i] else None)
        if raw is None or raw is state:
            continue
        if last_switch is not None and i - last_switch < cooldown:
            continue   # まだ間隔があいていない
        state, last_switch = raw, i
    since = None if last_switch is None else len(close) - 1 - last_switch
    return state, since
