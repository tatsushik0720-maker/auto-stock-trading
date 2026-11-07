"""戦略のテスト。偽データで「買いの日・売りの日」が正しく出るかを確かめる。"""

import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from sim.strategies import (
    DayContext, Decision, Signal, Strategy, StrategyError, available_strategies,
    create_strategy, generate_signals, signal_changes,
)
from sim.strategies.sma_cross import SmaCross


def prices_from(closes, start="2024-01-01", skip=()):
    days = pd.bdate_range(start, periods=len(closes) + len(skip))
    days = days[~days.isin(pd.to_datetime(list(skip)))][:len(closes)]
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Adj Close": c,
                         "Volume": 1.0}, index=pd.DatetimeIndex(days, name="Date"))


# ---- 移動平均戦略（手計算できる小さな例: window=3） -------------------

def test_sma_cross_hand_calculated_example():
    #          0    1    2    3    4    5    6    7
    closes = [10, 10, 10, 13, 12, 9, 8, 12]
    # 3日平均:  -    -   10  11  11.67 11.33 9.67 9.67
    # 判断:   様子見 様子見 同値=様子見 13>11 買 12>11.67 買 9<11.33 売 8<9.67 売 12>9.67 買
    sig = generate_signals(SmaCross(window=3), "X", prices_from(closes))
    expected = [Signal.HOLD, Signal.HOLD, Signal.HOLD,
                Signal.BUY, Signal.BUY, Signal.SELL, Signal.SELL, Signal.BUY]
    assert list(sig["signal"]) == expected
    assert sig["sma"].iloc[3] == pytest.approx(11.0)
    assert sig["sma"].iloc[5] == pytest.approx((13 + 12 + 9) / 3)
    assert sig["sma"].iloc[:2].isna().all()

    changes = signal_changes(sig)
    assert list(changes.index.strftime("%m-%d")) == ["01-04", "01-08", "01-10"]
    assert list(changes["signal"]) == [Signal.BUY, Signal.SELL, Signal.BUY]


def test_sma_cross_25_days_uptrend_then_downtrend():
    """25日平均: 上昇中は買い、急落して平均を割った日に売りへ変わる。"""
    closes = list(range(100, 140)) + [120]   # 40日上昇のあと急落
    sig = generate_signals(SmaCross(window=25), "X", prices_from(closes))
    assert (sig["signal"].iloc[:24] == Signal.HOLD).all()        # 25日目までは平均が出せない
    assert (sig["signal"].iloc[24:40] == Signal.BUY).all()
    assert sig["signal"].iloc[40] == Signal.SELL
    assert sig["sma"].iloc[24] == pytest.approx(np.mean(range(100, 125)))


def test_flat_prices_give_no_buy_or_sell():
    sig = generate_signals(SmaCross(window=5), "X", prices_from([100] * 20))
    assert (sig["signal"] == Signal.HOLD).all()
    assert signal_changes(sig).empty


def test_hold_between_same_signals_is_not_a_change():
    """買い → 様子見 → 買い は、合図の変化として数えない。"""
    s = [Signal.BUY, Signal.HOLD, Signal.BUY, Signal.SELL]
    df = pd.DataFrame({"signal": s}, index=pd.bdate_range("2024-01-01", periods=4))
    assert list(signal_changes(df)["signal"]) == [Signal.BUY, Signal.SELL]


def test_missing_day_is_not_counted_in_average():
    """取引のない日（行が無い日）は平均に含めず、直近 N 回の取引日で平均する。"""
    p = prices_from([10, 20, 30, 40], skip=["2024-01-03"])
    sig = generate_signals(SmaCross(window=3), "X", p)
    assert pd.Timestamp("2024-01-03") not in sig.index
    assert sig["sma"].iloc[-1] == pytest.approx((20 + 30 + 40) / 3)


# ---- 未来のデータを使っていないか ----------------------------------

class Spy(Strategy):
    """受け取ったデータの最後の日を記録するだけの戦略。"""
    name = "_spy"

    def __init__(self):
        self.seen = []

    def decide(self, ctx):
        self.seen.append((ctx.date, ctx.prices.index[-1], len(ctx.prices), dict(ctx.extras)))
        return Decision(Signal.HOLD)


def test_strategy_never_sees_future_prices():
    spy = Spy()
    p = prices_from(range(30))
    generate_signals(spy, "X", p)
    for i, (day, last_seen, n, _) in enumerate(spy.seen):
        assert last_seen == day == p.index[i]
        assert n == i + 1


def test_extras_are_passed_per_day():
    """将来のニュース判定用: 日付ごとの追加情報がその日の判断にだけ渡る。"""
    spy = Spy()
    p = prices_from(range(5))
    day2 = p.index[2].date()
    generate_signals(spy, "X", p, extras_by_date={day2: {"news_score": -0.8}})
    assert spy.seen[2][3] == {"news_score": -0.8}
    assert spy.seen[1][3] == {} and spy.seen[3][3] == {}


def test_strategy_can_use_extras_and_position():
    """extras（ニュース等）と保有株数を使う戦略も、今の仕組みのまま書ける。"""
    class NewsAware(Strategy):
        name = "_news"

        def decide(self, ctx: DayContext) -> Decision:
            if ctx.extras.get("news_score", 0) < -0.5 and ctx.position:
                return Decision(Signal.SELL, {"reason": "悪いニュース"})
            return Decision(Signal.HOLD)

    p = prices_from([1, 2])
    ctx = DayContext("X", p.index[-1], p, position=10, extras={"news_score": -0.9})
    assert NewsAware().decide(ctx).signal == Signal.SELL


# ---- 戦略の登録と作成 ----------------------------------------------

def test_sma_cross_is_registered():
    assert "sma_cross" in available_strategies()


def test_create_strategy_from_config_values():
    s = create_strategy("sma_cross", {"window": 25})
    assert isinstance(s, SmaCross) and s.window == 25


@pytest.mark.parametrize("name, params, message", [
    ("no_such", {}, "見つかりません"),
    ("sma_cross", {"windw": 25}, "パラメータ"),       # つづり間違い
    ("sma_cross", {"window": 1}, "2 以上"),
    ("sma_cross", {"window": "25"}, "整数"),
])
def test_create_strategy_errors(name, params, message):
    with pytest.raises(StrategyError, match=message):
        create_strategy(name, params)


def test_new_strategy_file_is_found_automatically(tmp_path, monkeypatch):
    """1戦略1ファイル: フォルダにファイルを置くだけで登録される。"""
    pkg = tmp_path / "my_strategies"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "always_buy.py").write_text(textwrap.dedent("""
        from sim.strategies.base import Decision, Signal, Strategy

        class AlwaysBuy(Strategy):
            name = "always_buy"
            def decide(self, ctx):
                return Decision(Signal.BUY)
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    found = available_strategies("my_strategies")
    assert list(found) == ["always_buy"]
    sys.modules.pop("my_strategies", None)
    sys.modules.pop("my_strategies.always_buy", None)
