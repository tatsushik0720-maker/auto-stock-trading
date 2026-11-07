"""ダマシ対策つき移動平均戦略: 平均より band 上で買い、band 下で売り。

例（window=25, band=0.02）:
  終値が 25日平均×1.02 を上回ったら買い、25日平均×0.98 を下回ったら売り。
  その間（平均の ±2% の中）にいるあいだは、直前の判断をそのまま続ける。
  平均の近くを行ったり来たりするだけで売買を繰り返す「ダマシ」を減らすためのルール。

記憶を持たない決まりを守るため、「直前の判断」はその日までの株価から毎回求め直す
（さかのぼって、最後に帯の外へ出た日が上か下かを見る）。
"""

from __future__ import annotations

import numpy as np

from .base import Decision, DayContext, Signal, Strategy, StrategyError


class SmaBand(Strategy):
    name = "sma_band"
    description = "移動平均より band 上で買い、band 下で売り（帯の中は前の判断のまま）"

    def __init__(self, window: int = 25, band: float = 0.02):
        if isinstance(window, bool) or not isinstance(window, int) or window < 2:
            raise StrategyError(f"sma_band: window は 2 以上の整数にしてください（今: {window!r}）")
        if isinstance(band, bool) or not isinstance(band, (int, float)) or not 0 <= band < 0.5:
            raise StrategyError(
                f"sma_band: band は 0 以上 0.5 未満の割合にしてください（0.02 = 2%）。今: {band!r}")
        self.window = window
        self.band = float(band)

    def warmup_days(self) -> int:
        return self.window

    def params(self) -> dict:
        return {"window": self.window, "band": self.band}

    def decide(self, ctx: DayContext) -> Decision:
        close = ctx.prices["Close"].to_numpy(dtype=float)
        if len(close) < self.window:
            return Decision(Signal.HOLD, {"sma": None})
        state, last_sma = band_states(close, self.window, self.band)
        info = {"sma": last_sma, "upper": last_sma * (1 + self.band),
                "lower": last_sma * (1 - self.band)}
        return Decision(state, info)


def rolling_mean(close: np.ndarray, window: int) -> np.ndarray:
    """直近 window 日の平均（足りない日は NaN）。"""
    out = np.full(len(close), np.nan)
    if len(close) >= window:
        c = np.cumsum(np.insert(close, 0, 0.0))
        out[window - 1:] = (c[window:] - c[:-window]) / window
    return out


def band_states(close: np.ndarray, window: int, band: float) -> tuple[Signal, float]:
    """最後の日の合図と平均。帯の外に最後に出た日が上なら買い、下なら売り。"""
    sma = rolling_mean(close, window)
    with np.errstate(invalid="ignore"):
        above = close > sma * (1 + band)
        below = close < sma * (1 - band)
    outside = np.flatnonzero(above | below)
    if len(outside) == 0:
        return Signal.HOLD, float(sma[-1])     # まだ一度も帯の外に出ていない
    last = outside[-1]
    return (Signal.BUY if above[last] else Signal.SELL), float(sma[-1])
