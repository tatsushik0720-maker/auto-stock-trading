"""評価（evaluate.py）と比較用の「買って放置」のテスト。"""

from types import SimpleNamespace

import pandas as pd
import pytest

from sim.evaluate import (
    comparison_table, compute_metrics, drawdown, exposure, exposure_by_symbol,
    plot_comparison, twr_index, yearly_comparison, yearly_table,
)
from sim.simulator import Costs, run_simulation
from sim.strategies import create_strategy


def pf(dates, equity, deposited):
    return pd.DataFrame({"equity": equity, "deposited": deposited,
                         "cash": equity, "market_value": 0.0},
                        index=pd.DatetimeIndex(pd.to_datetime(dates)))


# ---- 時間加重リターンとドローダウン ---------------------------------

def test_deposits_alone_are_not_profit():
    """値動きがなく入金だけで資産が増えても、指数は 1 のまま（ドローダウン 0）。"""
    p = pf(["2024-01-31", "2024-02-01", "2024-03-01"], [100, 150, 200], [100, 150, 200])
    assert list(twr_index(p)) == pytest.approx([1, 1, 1])
    assert drawdown(twr_index(p)).min() == 0


def test_drawdown_hand_calculated():
    """
    資産 100 → 120 → 90 → 130（入金なし）
    指数 1.0 → 1.2 → 0.9 → 1.3
    ドローダウン 0 → 0 → 0.9÷1.2−1 = −25% → 0
    最高値 01-02、底 01-03、回復 01-04
    """
    p = pf(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
           [100, 120, 90, 130], [100] * 4)
    assert list(twr_index(p)) == pytest.approx([1.0, 1.2, 0.9, 1.3])
    m = compute_metrics(SimpleNamespace(
        portfolio=p, final_equity=130, total_deposited=100, trades=[],
        daily=pd.DataFrame({"shares": [1, 1, 1, 1]}), total_fee=0, total_slippage=0,
        total_tax=0))
    assert m.max_drawdown == pytest.approx(-0.25)
    assert (m.dd_peak, m.dd_trough, m.dd_recovered) == tuple(
        pd.Timestamp(d) for d in ("2024-01-02", "2024-01-03", "2024-01-04"))
    assert m.max_pnl_drop == pytest.approx(-30)    # 損益 +20 → −10
    assert m.pnl == 30 and m.pnl_pct == pytest.approx(0.30)


def test_deposit_does_not_hide_a_loss():
    """
    資産 100 → 80（−20%）→ 130（うち入金 50）
    3日目の増減率 = (130 − 50) ÷ 80 − 1 = 0% → 指数 1.0 → 0.8 → 0.8
    入金で資産額は最高値を超えたが、運用成績は −20% のまま（未回復）。
    """
    p = pf(["2024-01-30", "2024-01-31", "2024-02-01"], [100, 80, 130], [100, 100, 150])
    idx = twr_index(p)
    assert list(idx) == pytest.approx([1.0, 0.8, 0.8])
    assert drawdown(idx).min() == pytest.approx(-0.2)


# ---- 年ごとの損益 --------------------------------------------------

def test_yearly_table_hand_calculated():
    """
    日付        資産  投入総額  損益(資産−投入)
    2023-06-01  100   100        0
    2023-12-29  110   100      +10   → 2023年の損益 +10、リターン +10%
    2024-01-04  165   150      +15   （入金 50）
    2024-12-30  132   150      −18   → 2024年の損益 −18 − (+10) = −28
    2024年のリターン: 01-04 (165−50)÷110 = 1.04545… → 12-30 132÷165 = 0.8
                     → 1.04545 × 0.8 − 1 = −16.36%
    """
    p = pf(["2023-06-01", "2023-12-29", "2024-01-04", "2024-12-30"],
           [100, 110, 165, 132], [100, 100, 150, 150])
    t = yearly_table(SimpleNamespace(portfolio=p))
    assert list(t.index) == [2023, 2024]
    assert list(t["損益"]) == pytest.approx([10, -28])
    assert t["リターン"].iloc[0] == pytest.approx(0.10)
    assert t["リターン"].iloc[1] == pytest.approx(115 / 110 * 0.8 - 1)


def test_yearly_comparison_marks_winner():
    a = SimpleNamespace(portfolio=pf(["2023-01-02", "2023-12-29", "2024-12-30"],
                                     [100, 110, 99], [100] * 3))
    b = SimpleNamespace(portfolio=pf(["2023-01-02", "2023-12-29", "2024-12-30"],
                                     [100, 105, 105], [100] * 3))
    y = yearly_comparison({"戦略": a, "放置": b})
    assert list(y["勝ち"]) == ["戦略", "放置"]


# ---- 保有日の割合 --------------------------------------------------

def test_exposure():
    daily = pd.DataFrame({"symbol": ["A", "A", "A", "A", "B", "B"],
                          "shares": [0, 10, 10, 0, 5, 5]})
    assert exposure(daily) == pytest.approx(4 / 6)
    assert exposure_by_symbol(daily) == pytest.approx({"A": 0.5, "B": 1.0})


# ---- 比較用「最初に買って放置」 --------------------------------------

def bars(closes, start="2024-01-29"):
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes,
                         "Adj Close": closes, "Volume": 1.0}, index=idx)


def test_buy_and_hold_buys_once_then_adds_deposits_and_never_sells():
    """01-29(月) から 6日: 初日の合図 → 01-30 に買い。02-01 の積立分を買い増し。売らない。"""
    prices = {"A": bars([100.0, 100, 100, 100, 120, 80])}
    r = run_simulation(prices, {"A": 1}, create_strategy("buy_and_hold"), 1_000, 500,
                       Costs())
    assert [(t.date.strftime("%m-%d"), t.side, t.shares) for t in r.trades] == \
        [("01-30", "買い", 10), ("02-01", "買い", 5)]
    m = compute_metrics(r)
    assert m.exposure == pytest.approx(5 / 6)   # 初日だけ持っていない
    assert m.final_equity == pytest.approx(15 * 80)


def test_comparison_table_and_chart(tmp_path):
    prices = {"A": bars([100.0, 102, 98, 105, 110, 90, 95, 100, 108, 112] * 3)}
    args = (prices, {"A": 1})
    money = (10_000, 1_000, Costs(slippage_rate=0.001))
    s = run_simulation(*args, create_strategy("sma_cross", {"window": 3}), *money)
    b = run_simulation(*args, create_strategy("buy_and_hold"), *money)
    table = comparison_table({"戦略": s, "放置": b})
    for row in ["最終資産", "損益", "最大ドローダウン", "株を持っていた日の割合", "値差の負担額の合計"]:
        assert row in table.index
    assert list(table.columns) == ["戦略", "放置"]
    path = plot_comparison(s, b, tmp_path / "equity.png", "戦略", "放置")
    assert path.exists() and path.stat().st_size > 10_000
