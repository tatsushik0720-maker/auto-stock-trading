"""架空売買シミュレータ。実際の注文は一切出さない。

1日の流れ（銘柄ごと。その銘柄に取引がある日だけ動く）:
  1. その月の最初の取引日なら、積立金を入金する
  2. 前の取引日に出た注文を、今日の始値で約定させる
     （翌営業日にその銘柄の行がなければ、次に取引がある日の始値で約定）
       買い: 始値 ×(1+値差率) で、買える分だけ lot_size 単位で買う。端数は現金のまま
       売り: 始値 ×(1-値差率) で、全部売る
  3. 終値で資産を記録する
  4. 終値を見て戦略が合図を出す → 買い・売りなら翌取引日に約定する注文になる

お金のルール:
  - 手数料 = 約定代金 × 手数料率（買い・売りの両方）
  - 値差の負担額 = 口数 × 始値 × 値差率（思った価格より不利に約定した分）
  - 取得価格は総平均法: 買い増しのたびに「買った総額（手数料込み）÷ 口数」で平均する
  - 実現損益 = 売却代金 − 売却手数料 − 平均取得価格 × 売った口数
  - 税金: 同じ年の売却の損益を通算し、利益の合計 × 税率（1円未満切り捨て）だけ払う。
    売却のたびに「その年の通算損益に対する税額」との差額を精算する（源泉徴収ありの
    特定口座と同じ方式）。年末にまとめて計算した場合と、1年の合計額は同じになる。
    税率 0 なら何もしない。
  - 資金配分: 初期資金と積立額を銘柄数で等分し、銘柄ごとに独立して運用する。
    税だけは口座全体で通算する（銘柄Aの損と銘柄Bの益を相殺する）。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .strategies.base import DayContext, Signal, Strategy

EPS = 1e-9  # 浮動小数点の誤差よけ


@dataclass(frozen=True)
class Costs:
    fee_rate: float = 0.0
    slippage_rate: float = 0.0
    tax_rate: float = 0.0


@dataclass
class Trade:
    """売買1回分の記録。"""
    date: pd.Timestamp
    symbol: str
    side: str                 # "買い" / "売り"
    shares: int               # 口数
    open_price: float         # その日の始値
    exec_price: float         # 約定価格（値差込み）
    amount: float             # 約定代金 = 口数 × 約定価格
    fee: float                # 手数料
    slippage_cost: float      # 値差の負担額
    avg_cost: float           # 取引後の平均取得価格（1口あたり、手数料込み）
    realized_pnl: float | None  # 実現損益（売りのみ）
    tax: float                # 税の精算額（+支払い / −還付）
    cash_after: float         # 取引後の現金


@dataclass
class MissedBuy:
    """買いの合図が出ていたのに、現金が足りず1単位も買えなかった日。"""
    date: pd.Timestamp
    symbol: str
    cash: float               # そのときの現金
    lot_cost: float           # 1単位買うのに必要だった金額（値差・手数料込み）


@dataclass
class SimResult:
    daily: pd.DataFrame          # 銘柄ごとの日次記録（列 symbol 付き、索引は日付）
    portfolio: pd.DataFrame      # 全銘柄合計の日次記録（cash, market_value, equity, deposited）
    trades: list[Trade]
    missed_buys: list[MissedBuy]
    tax_by_year: dict[int, float]

    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(t) for t in self.trades],
                            columns=[f for f in Trade.__dataclass_fields__])

    @property
    def total_fee(self) -> float:
        return sum(t.fee for t in self.trades)

    @property
    def total_slippage(self) -> float:
        return sum(t.slippage_cost for t in self.trades)

    @property
    def total_tax(self) -> float:
        return sum(self.tax_by_year.values())

    @property
    def final_equity(self) -> float:
        return float(self.portfolio["equity"].iloc[-1])

    @property
    def total_deposited(self) -> float:
        return float(self.portfolio["deposited"].iloc[-1])


@dataclass
class _Account:
    """銘柄ごとの小さな口座。"""
    symbol: str
    lot: int
    cash: float
    deposited: float
    shares: int = 0
    cost: float = 0.0              # 保有分の取得総額（手数料込み）
    last_month: tuple[int, int] | None = None
    pending: Signal | None = None  # 翌取引日に約定させる注文

    @property
    def avg_cost(self) -> float:
        return self.cost / self.shares if self.shares else 0.0


@dataclass
class _TaxBook:
    """口座全体の、その年の通算損益と払った税。"""
    rate: float
    year: int | None = None
    realized: float = 0.0
    paid: float = 0.0
    by_year: dict[int, float] = field(default_factory=dict)

    def new_day(self, day: pd.Timestamp) -> None:
        if day.year != self.year:
            self.year, self.realized, self.paid = day.year, 0.0, 0.0

    def settle(self, pnl: float) -> float:
        """売却の実現損益を通算し、精算する税額（+支払い/−還付）を返す。"""
        self.realized += pnl
        if self.rate <= 0:
            return 0.0
        due = math.floor(max(0.0, self.realized) * self.rate + EPS)
        delta = due - self.paid
        self.paid = due
        self.by_year[self.year] = due
        return float(delta)


# ============================================================
# 本体
# ============================================================

def run_simulation(prices: Mapping[str, pd.DataFrame], lot_sizes: Mapping[str, int],
                   strategy: Strategy, initial_cash: float, monthly_deposit: float,
                   costs: Costs,
                   extras_by_date: Mapping[date, Mapping[str, Any]] | None = None
                   ) -> SimResult:
    """全銘柄の架空売買を行う。prices は銘柄ごとの日足（行が無い日は取引なし）。"""
    symbols = list(prices)
    if not symbols:
        raise ValueError("銘柄がありません")
    n = len(symbols)
    extras_by_date = extras_by_date or {}
    accounts = {s: _Account(s, int(lot_sizes[s]), initial_cash / n, initial_cash / n)
                for s in symbols}
    position = {s: {d: i for i, d in enumerate(prices[s].index)} for s in symbols}
    all_days = sorted(set().union(*(set(prices[s].index) for s in symbols)))
    start_month = (all_days[0].year, all_days[0].month)
    for acc in accounts.values():
        acc.last_month = start_month  # 積立は開始月の翌月から

    tax = _TaxBook(costs.tax_rate)
    trades: list[Trade] = []
    missed: list[MissedBuy] = []
    records: list[dict] = []

    for day in all_days:
        tax.new_day(day)
        for s in symbols:
            i = position[s].get(day)
            if i is None:
                continue  # この銘柄はこの日取引なし
            df, acc = prices[s], accounts[s]
            bar = df.iloc[i]

            # 1. 積立
            month = (day.year, day.month)
            if month != acc.last_month:
                acc.last_month = month
                acc.cash += monthly_deposit / n
                acc.deposited += monthly_deposit / n

            # 2. 前の取引日の注文を始値で約定
            if acc.pending is Signal.BUY:
                _buy(acc, day, float(bar["Open"]), costs, trades, missed)
            elif acc.pending is Signal.SELL:
                _sell(acc, day, float(bar["Open"]), costs, tax, trades)
            acc.pending = None

            # 3. 終値で記録
            close = float(bar["Close"])
            # 4. 戦略の判断（翌取引日の注文になる）
            ctx = DayContext(symbol=s, date=day, prices=df.iloc[:i + 1], position=acc.shares,
                             extras=extras_by_date.get(day.date(), {}))
            decision = strategy.decide(ctx)
            if decision.signal is Signal.BUY:
                acc.pending = Signal.BUY
            elif decision.signal is Signal.SELL and acc.shares > 0:
                acc.pending = Signal.SELL

            records.append({
                "date": day, "symbol": s, "close": close, "signal": decision.signal.value,
                "cash": acc.cash, "shares": acc.shares, "avg_cost": acc.avg_cost,
                "market_value": acc.shares * close,
                "equity": acc.cash + acc.shares * close, "deposited": acc.deposited,
            })

    daily = pd.DataFrame(records).set_index("date")
    portfolio = _portfolio(daily, symbols, initial_cash / n)
    return SimResult(daily=daily, portfolio=portfolio, trades=trades, missed_buys=missed,
                     tax_by_year=dict(tax.by_year))


def _buy(acc: _Account, day, open_price: float, costs: Costs,
         trades: list[Trade], missed: list[MissedBuy]) -> None:
    price = open_price * (1 + costs.slippage_rate)
    lot_cost = acc.lot * price * (1 + costs.fee_rate)
    lots = math.floor(acc.cash / lot_cost + EPS)
    if lots <= 0:
        if acc.shares == 0:
            missed.append(MissedBuy(day, acc.symbol, acc.cash, lot_cost))
        return
    shares = lots * acc.lot
    amount = shares * price
    fee = amount * costs.fee_rate
    acc.cash -= amount + fee
    acc.shares += shares
    acc.cost += amount + fee  # 総平均法: 手数料も取得価格に含める
    trades.append(Trade(day, acc.symbol, "買い", shares, open_price, price, amount, fee,
                        shares * open_price * costs.slippage_rate, acc.avg_cost, None, 0.0,
                        acc.cash))


def _sell(acc: _Account, day, open_price: float, costs: Costs, tax: _TaxBook,
          trades: list[Trade]) -> None:
    shares = acc.shares
    price = open_price * (1 - costs.slippage_rate)
    amount = shares * price
    fee = amount * costs.fee_rate
    avg = acc.avg_cost
    pnl = amount - fee - avg * shares
    tax_delta = tax.settle(pnl)
    acc.cash += amount - fee - tax_delta
    acc.shares, acc.cost = 0, 0.0
    trades.append(Trade(day, acc.symbol, "売り", shares, open_price, price, amount, fee,
                        shares * open_price * costs.slippage_rate, avg, pnl, tax_delta,
                        acc.cash))


def _portfolio(daily: pd.DataFrame, symbols: list[str], initial_each: float) -> pd.DataFrame:
    """銘柄ごとの記録を合計する。取引のない日は、その銘柄の直前の記録で評価する
    （価格データを埋めるのではなく、資産額の表示のためだけ）。"""
    cols = ["cash", "market_value", "equity", "deposited"]
    start_fill = {"cash": initial_each, "market_value": 0.0,
                  "equity": initial_each, "deposited": initial_each}
    total = None
    for s in symbols:
        part = daily[daily["symbol"] == s][cols]
        total_index = daily.index.unique().sort_values()
        part = part.reindex(total_index).ffill().fillna(start_fill)
        total = part if total is None else total + part
    return total


# ============================================================
# 出力
# ============================================================

TRADE_COLUMNS_JA = {
    "date": "日付", "symbol": "銘柄", "side": "売買", "shares": "口数",
    "open_price": "始値", "exec_price": "約定価格", "amount": "約定代金", "fee": "手数料",
    "slippage_cost": "値差の負担額", "avg_cost": "平均取得価格", "realized_pnl": "実現損益",
    "tax": "税の精算額", "cash_after": "取引後の現金",
}


def write_outputs(result: SimResult, output_dir: Path, prefix: str = "") -> dict[str, Path]:
    """売買記録と日次記録を CSV に出す（Excel で文字化けしないよう BOM 付き UTF-8）。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"trades": output_dir / f"{prefix}trades.csv",
             "daily": output_dir / f"{prefix}daily.csv",
             "portfolio": output_dir / f"{prefix}portfolio.csv"}
    trades = result.trades_df().rename(columns=TRADE_COLUMNS_JA)
    if len(trades):
        trades["日付"] = pd.to_datetime(trades["日付"]).dt.date
    trades.round(4).to_csv(paths["trades"], index=False, encoding="utf-8-sig")
    result.daily.round(4).to_csv(paths["daily"], index_label="date", encoding="utf-8-sig")
    result.portfolio.round(4).to_csv(paths["portfolio"], index_label="date", encoding="utf-8-sig")
    return paths


def summarize_result(result: SimResult) -> str:
    """結果の要約（段階5で詳しい評価を追加する）。"""
    pnl = result.final_equity - result.total_deposited
    trades = result.trades_df()
    n_buy = int((trades["side"] == "買い").sum()) if len(trades) else 0
    n_sell = int((trades["side"] == "売り").sum()) if len(trades) else 0
    lines = [
        f"最終資産: {result.final_equity:,.0f}円  投入総額: {result.total_deposited:,.0f}円"
        f"  損益: {pnl:+,.0f}円（{pnl / result.total_deposited:+.2%}）",
        f"売買回数: 買い {n_buy}回 / 売り {n_sell}回",
        f"手数料の合計: {result.total_fee:,.0f}円",
        f"値差の負担額の合計: {result.total_slippage:,.0f}円"
        f"（投入総額の {result.total_slippage / result.total_deposited:.1%}）",
        f"税の合計: {result.total_tax:,.0f}円",
    ]
    if len(trades) and n_sell:
        realized = trades["realized_pnl"].dropna()
        lines.append(f"実現損益の合計: {realized.sum():+,.0f}円"
                     f"（勝ち {int((realized > 0).sum())}回 / 負け {int((realized <= 0).sum())}回）")
    lines += _missed_summary(result.missed_buys)
    return "\n".join(lines)


def _missed_summary(missed: list[MissedBuy]) -> list[str]:
    if not missed:
        return ["1単位も買えなかった日: なし"]
    lines = ["[警告] 買いの合図が出ていたのに、現金が足りず1単位も買えなかった日があります"]
    df = pd.DataFrame([asdict(m) for m in missed])
    for s, g in df.groupby("symbol"):
        last = g.iloc[-1]
        lines.append(
            f"  {s}: {len(g)}日（{g['date'].min().date()}〜{g['date'].max().date()}）。"
            f"買えずに待機した現金は最大 {g['cash'].max():,.0f}円"
            f"（最後の日: 現金 {last['cash']:,.0f}円 < 1単位 {last['lot_cost']:,.0f}円）")
    return lines
