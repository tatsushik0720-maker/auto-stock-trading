"""口座の状態の保存と再開のテスト（将来のフォワードテスト用）。

一番大事な性質:
  「全期間を一度に動かした結果」と「途中で保存→読み込み→続きを動かした結果」が完全に一致する。
"""

import json

import numpy as np
import pandas as pd
import pytest

from sim.simulator import Costs, run_simulation
from sim.state import SimState, StateError
from sim.strategies.sma_cross import SmaCross

COSTS = Costs(fee_rate=0.001, slippage_rate=0.001, tax_rate=0.20315)
LOTS = {"A": 10, "B": 1}


def random_prices(seed, start="2023-10-02", days=300, skip=(), drift=0.003):
    """乱高下する偽の日足（年またぎ・月またぎを含む）。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=days)
    idx = idx[~idx.isin(pd.to_datetime(list(skip)))]
    close = 1000 * np.exp(np.cumsum(rng.normal(drift, 0.02, len(idx))))
    open_ = close * np.exp(rng.normal(0, 0.01, len(idx)))
    return pd.DataFrame({"Open": open_, "High": close, "Low": close, "Close": close,
                         "Adj Close": close, "Volume": 1000.0},
                        index=pd.DatetimeIndex(idx, name="Date"))


PRICES = {"A": random_prices(1), "B": random_prices(2, skip=["2024-01-10", "2024-03-15"])}


def simulate(prices, state=None, window=5, costs=COSTS, lots=LOTS):
    return run_simulation(prices, lots, SmaCross(window=window), 500_000, 10_000, costs,
                          state=state)


def upto(prices, last):
    """last の日までのデータだけ（＝その日の夕方に動かした状況）。"""
    return {s: df[df.index <= pd.Timestamp(last)] for s, df in prices.items()}


def assert_same_result(full, parts):
    """一度に動かした結果と、分けて動かした結果が同じか。"""
    assert json.dumps(full.state.to_dict(), sort_keys=True) == \
        json.dumps(parts[-1].state.to_dict(), sort_keys=True)
    full_trades = full.trades_df()
    split_trades = pd.concat([p.trades_df() for p in parts if p.trades], ignore_index=True)
    pd.testing.assert_frame_equal(full_trades, split_trades)
    pd.testing.assert_frame_equal(full.daily, pd.concat([p.daily for p in parts]))
    pd.testing.assert_frame_equal(full.portfolio, pd.concat([p.portfolio for p in parts]),
                                  check_freq=False)


# ---- 再開しても結果が同じ ----------------------------------------

@pytest.mark.parametrize("split", ["2023-11-15", "2023-12-29", "2024-01-04", "2024-01-10",
                                   "2024-06-28"])
def test_resume_gives_identical_result(tmp_path, split):
    """月末・年末・年始・欠けた日・注文が残っている日など、どこで区切っても一致する。"""
    full = simulate(PRICES)
    assert len(full.trades) > 20                      # 売買が十分に起きている
    assert full.total_tax > 0                         # 税の通算も試される
    assert any(t.tax < 0 for t in full.trades)        # 還付も起きている

    first = simulate(upto(PRICES, split))
    first.state.save(tmp_path / "state.json")
    loaded = SimState.load(tmp_path / "state.json")
    second = simulate(PRICES, state=loaded)  # 続きの日だけ処理される
    assert_same_result(full, [first, second])


def test_daily_forward_run_one_day_at_a_time(tmp_path):
    """フォワードテストの想定: 毎日1回、その日までのデータで動かして保存する。"""
    full = simulate(PRICES)
    path = tmp_path / "state.json"
    days = sorted(set(PRICES["A"].index) | set(PRICES["B"].index))
    parts = []
    for i, day in enumerate(days[:120]):   # 最初の120日分を1日ずつ
        state = SimState.load(path) if i else None
        r = simulate(upto(PRICES, day), state=state)
        r.state.save(path)
        parts.append(r)
    parts.append(simulate(PRICES, state=SimState.load(path)))  # 残りはまとめて
    assert_same_result(full, parts)


def test_running_twice_with_same_data_does_nothing_the_second_time():
    r1 = simulate(PRICES)
    r2 = simulate(PRICES, state=r1.state)
    assert r2.trades == [] and r2.daily.empty and r2.portfolio.empty
    assert r2.final_equity == pytest.approx(r1.final_equity)
    assert r2.state.to_dict() == r1.state.to_dict()


def test_given_state_is_not_modified():
    first = simulate(upto(PRICES, "2023-12-29"))
    before = json.dumps(first.state.to_dict(), sort_keys=True)
    simulate(PRICES, state=first.state)
    assert json.dumps(first.state.to_dict(), sort_keys=True) == before


# ---- 状態ファイルの中身 ------------------------------------------

def test_state_file_contents_are_readable(tmp_path):
    r = simulate(upto(PRICES, "2024-02-15"))
    path = tmp_path / "state.json"
    r.state.save(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["last_date"] == "2024-02-15"
    a = d["accounts"]["A"]
    for key in ["cash", "shares", "avg_cost", "cost", "deposited", "pending", "last_date"]:
        assert key in a
    assert a["avg_cost"] == pytest.approx(a["cost"] / a["shares"] if a["shares"] else 0)
    assert d["settings"]["strategy"] == {"name": "sma_cross", "params": {"window": 5}}
    assert d["tax"]["year"] == 2024
    assert d["equity"] == pytest.approx(r.final_equity)


def test_save_keeps_previous_as_backup(tmp_path):
    path = tmp_path / "state.json"
    simulate(upto(PRICES, "2023-12-01")).state.save(path)
    simulate(upto(PRICES, "2023-12-08")).state.save(path)
    assert json.loads(path.read_text())["last_date"] == "2023-12-08"
    assert json.loads((tmp_path / "state.json.bak").read_text())["last_date"] == "2023-12-01"
    assert not (tmp_path / "state.json.tmp").exists()


# ---- 設定が変わっていたら止める ----------------------------------

@pytest.mark.parametrize("change", [
    {"window": 10},                                    # 戦略のパラメータ
    {"costs": Costs(0.0, 0.001, 0.20315)},             # 手数料
    {"lots": {"A": 100, "B": 1}},                      # 売買単位
])
def test_resume_with_different_settings_is_rejected(change):
    first = simulate(upto(PRICES, "2023-12-29"))
    with pytest.raises(StateError, match="設定が違う"):
        simulate(PRICES, state=first.state, **change)


def test_resume_with_different_symbols_is_rejected():
    first = simulate(upto(PRICES, "2023-12-29"))
    with pytest.raises(StateError, match="symbols"):
        run_simulation({"A": PRICES["A"]}, {"A": 10}, SmaCross(5), 500_000, 10_000, COSTS,
                       state=first.state)


def test_missing_or_broken_state_file(tmp_path):
    with pytest.raises(StateError, match="見つかりません"):
        SimState.load(tmp_path / "none.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{ broken")
    with pytest.raises(StateError, match="読めません"):
        SimState.load(bad)
