"""売買の少ない戦略（帯つき・間隔あけ）と、期間を区切った成績のテスト。"""

import pandas as pd
import pytest

from sim.config import ConfigError, parse_config
from sim.evaluate import period_metrics
from sim.simulator import Costs, run_simulation
from sim.strategies import Signal, StrategyError, create_strategy, generate_signals
from sim.strategies.sma_band import SmaBand
from sim.strategies.sma_cooldown import SmaCooldown

from test_strategies import prices_from

B, S, H = Signal.BUY, Signal.SELL, Signal.HOLD


# ---- 帯つき移動平均（window=2, band=10% の手計算） --------------------

def test_sma_band_hand_calculated():
    #        0    1    2     3    4    5    6
    closes = [100, 100, 130, 100, 60, 100, 100]
    # 2日平均: -  100  115  115   80   80  100
    # 上の帯: -  110 126.5 126.5 88   88  110   → 2日目 130>126.5 で買い
    # 下の帯: -   90 103.5 103.5 72   72   90   → 3日目 100 は下の帯を割る→売り、4日目 60<72 売り
    # 5日目 100>88 で買い、6日目 100 は帯の中 → 前の判断（買い）のまま
    sig = generate_signals(SmaBand(window=2, band=0.10), "X", prices_from(closes))
    assert list(sig["signal"]) == [H, H, B, S, S, B, B]


def test_sma_band_stays_inside_band_keeps_previous_decision():
    """平均のすぐ上下を行き来するだけなら、帯なしは毎日切り替わり、帯つきは切り替わらない。"""
    closes = [100] * 10 + [105] + [101, 99] * 10
    plain = generate_signals(create_strategy("sma_cross", {"window": 5}), "X", prices_from(closes))
    band = generate_signals(SmaBand(window=5, band=0.02), "X", prices_from(closes))
    assert (plain["signal"].iloc[12:] != plain["signal"].iloc[12:].shift()).sum() > 10
    assert (band["signal"].iloc[10:] == B).all()


def test_sma_band_zero_band_equals_sma_cross_when_never_equal():
    closes = [100, 103, 99, 104, 98, 97, 105, 110, 90, 95]
    a = generate_signals(SmaBand(window=3, band=0.0), "X", prices_from(closes))
    b = generate_signals(create_strategy("sma_cross", {"window": 3}), "X", prices_from(closes))
    assert list(a["signal"].iloc[3:]) == list(b["signal"].iloc[3:])


@pytest.mark.parametrize("params", [{"band": -0.01}, {"band": 0.5}, {"window": 1}, {"band": "2%"}])
def test_sma_band_rejects_bad_params(params):
    with pytest.raises(StrategyError):
        create_strategy("sma_band", params)


# ---- 間隔あけ移動平均（window=2, cooldown=3 の手計算） ----------------

def test_sma_cooldown_hand_calculated():
    #        0    1    2   3    4    5    6    7
    closes = [100, 110, 100, 110, 100, 90, 120, 100]
    # 帯なしの合図:   -   買  売  買  売  売  買  売
    # 1日目に買いへ切り替え → 4日目まで切り替え禁止（1+3）
    # 4日目: 売りへ切り替え → 7日目から切り替え可
    # 6日目の買いは禁止中で無視、7日目は売りのままなので変化なし
    sig = generate_signals(SmaCooldown(window=2, cooldown=3), "X", prices_from(closes))
    assert list(sig["signal"]) == [H, B, B, B, S, S, S, S]
    assert list(sig["days_since_switch"].iloc[1:5]) == [0, 1, 2, 0]


def test_sma_cooldown_does_not_depend_on_memory():
    """途中までのデータで判断しても、通しで判断しても、その日の合図は同じ（記憶を持たない）。"""
    closes = [100, 110, 100, 110, 100, 90, 120, 100, 130, 80]
    strat = SmaCooldown(window=2, cooldown=3)
    full = generate_signals(strat, "X", prices_from(closes))
    part = generate_signals(SmaCooldown(window=2, cooldown=3), "X", prices_from(closes[:7]))
    assert list(full["signal"].iloc[:7]) == list(part["signal"])


@pytest.mark.parametrize("params", [{"cooldown": 0}, {"cooldown": 2.5}, {"window": True}])
def test_sma_cooldown_rejects_bad_params(params):
    with pytest.raises(StrategyError):
        create_strategy("sma_cooldown", params)


# ---- 期間を区切った成績 ----------------------------------------------

def test_period_metrics_hand_calculated():
    """
    1口単位、初期資金100、積立なし、コストなし、買って放置。
    始値=終値: 1/1 は 100 で合図 → 1/2 の始値 100 で 1口買う
    資産: 1/1 100, 1/2 100, 1/3 120, 1/4 90, 1/5 108
    期間 1/3〜1/5 のリターン = 108 ÷ 100（直前 1/2 の資産） − 1 = +8%
    期間内の最大DD = 90 ÷ 120 − 1 = −25%。売買は期間外なので 0 回
    """
    df = prices_from([100, 100, 120, 90, 108], start="2024-01-01")
    r = run_simulation({"X": df}, {"X": 1}, create_strategy("buy_and_hold"), 100, 0, Costs())
    m = period_metrics(r, "2024-01-03", "2024-01-05")
    assert m.twr_total == pytest.approx(0.08)
    assert m.max_drawdown == pytest.approx(-0.25)
    assert (m.n_trades, m.n_sells, m.slippage) == (0, 0, 0)
    whole = period_metrics(r, "2024-01-01")
    assert whole.n_trades == 1 and whole.twr_total == pytest.approx(0.08)


def test_period_metrics_counts_only_trades_inside():
    df = prices_from([100, 120, 90, 130, 80, 140, 70, 150], start="2024-01-01")
    r = run_simulation({"X": df}, {"X": 1}, create_strategy("sma_cross", {"window": 2}),
                       1000, 0, Costs(slippage_rate=0.01))
    t = r.trades_df()
    m = period_metrics(r, "2024-01-05", "2024-01-10")
    inside = t[pd.to_datetime(t["date"]) >= "2024-01-05"]
    assert m.n_trades == len(inside) and 0 < m.n_trades < len(t)
    assert m.slippage == pytest.approx(inside["slippage_cost"].sum())


# ---- 設定の comparison -----------------------------------------------

BASE = {"symbols": [{"code": "X", "lot_size": 1}], "period": {"start": "2018-01-01"},
        "account": {"initial_cash": 100}, "costs": {}, "strategy": {"name": "sma_cross"}}


def comparison(**over):
    c = {"select_period": {"start": "2018-01-01", "end": "2022-12-31"},
         "check_period": {"start": "2023-01-01", "end": None},
         "candidates": [{"name": "sma_band", "params": {"window": 25, "band": 0.02}}]}
    c.update(over)
    return {**BASE, "comparison": c}


def test_comparison_is_optional_and_parsed():
    assert parse_config(BASE).comparison is None
    cmp = parse_config(comparison()).comparison
    assert str(cmp.select_period.end) == "2022-12-31" and cmp.check_period.end is None
    assert cmp.candidates[0].params == {"window": 25, "band": 0.02}


def test_comparison_periods_must_not_overlap():
    with pytest.raises(ConfigError, match="後にしてください"):
        parse_config(comparison(check_period={"start": "2022-06-01"}))


def test_comparison_rejects_duplicate_candidates():
    c = {"name": "sma_cross", "params": {"window": 25}}
    with pytest.raises(ConfigError, match="2回"):
        parse_config(comparison(candidates=[c, c]))
