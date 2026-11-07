"""戦略の共通ルール（基底クラス）と、合図を日ごとに出す仕組み。

新しい戦略の作り方:
    strategies/ フォルダに1ファイル作り、Strategy を継承したクラスに
    name（config.yaml で指定する名前）と decide() を書くだけ。
    登録は registry.py が自動で行う。

用語:
    合図(Signal): 戦略がその日に出す判断。
        BUY  = 買い（持っていたい状態）
        SELL = 売り（持っていたくない状態）
        HOLD = 様子見（データ不足などで判断しない）
    この戦略の合図は「状態」を表す。終値が平均より上にある間は毎日 BUY が出る。
    シミュレータは BUY の間は買える分を買い（積立分も買い増し）、SELL なら全部売る。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, ClassVar, Mapping

import pandas as pd


class Signal(Enum):
    BUY = "買い"
    SELL = "売り"
    HOLD = "様子見"


@dataclass(frozen=True)
class DayContext:
    """戦略が1日分の判断に使える情報。

    prices: その日（当日を含む）までの日足だけ。未来のデータは入っていない。
            その銘柄に取引がなかった日の行は無い（前日の値で埋めない）。
    position: 今の保有株数。シミュレータから渡される（合図の一覧づくりでは None）。
    extras: 追加情報の入れ物。将来ニュース判定のスコアなどをここに入れる。
            例: {"news_score": -0.8}
    """
    symbol: str
    date: pd.Timestamp
    prices: pd.DataFrame
    position: int | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """戦略の判断結果。info には判断の根拠（例: 25日平均の値）を入れて記録に残す。"""
    signal: Signal
    info: Mapping[str, Any] = field(default_factory=dict)


class StrategyError(ValueError):
    """戦略の名前やパラメータに誤りがあるときのエラー。"""


class Strategy(ABC):
    """すべての戦略の親。"""

    name: ClassVar[str]          # config.yaml の strategy.name で指定する名前
    description: ClassVar[str] = ""

    @abstractmethod
    def decide(self, ctx: DayContext) -> Decision:
        """その日の合図を返す。ctx.prices より先（未来）は見えない。"""

    def warmup_days(self) -> int:
        """判断を始めるのに必要な日数（それまでは HOLD）。"""
        return 0


def generate_signals(strategy: Strategy, symbol: str, prices: pd.DataFrame,
                     extras_by_date: Mapping[date, Mapping[str, Any]] | None = None
                     ) -> pd.DataFrame:
    """全期間について1日ずつ戦略に判断させ、日ごとの合図の表を返す。

    各日には「その日までのデータ」だけを渡すので、未来のデータは使われない。
    戻り値の列: Close, signal（Signal）, それに info の中身（例: sma）
    """
    extras_by_date = extras_by_date or {}
    rows = []
    for i, day in enumerate(prices.index):
        ctx = DayContext(symbol=symbol, date=day, prices=prices.iloc[:i + 1],
                         extras=extras_by_date.get(day.date(), {}))
        d = strategy.decide(ctx)
        rows.append({"Close": prices["Close"].iloc[i], "signal": d.signal, **d.info})
    return pd.DataFrame(rows, index=prices.index)


def signal_changes(signals: pd.DataFrame) -> pd.DataFrame:
    """合図が変わった日（様子見→買い、買い→売り など）だけを取り出す。"""
    decided = signals[signals["signal"] != Signal.HOLD]   # 様子見の日は飛ばして比べる
    s = decided["signal"]
    return decided[s.ne(s.shift())]
