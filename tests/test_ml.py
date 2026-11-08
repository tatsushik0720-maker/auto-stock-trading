"""機械学習の戦略（ml_logistic）のテスト。特に「未来のデータが学習に混ざらない」こと。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compare_ml import auc  # noqa: E402
from sim.config import ConfigError, parse_config  # noqa: E402
from sim.strategies import Signal, StrategyError, create_strategy, generate_signals  # noqa: E402
from sim.strategies.ml_logistic import (  # noqa: E402
    MlLogistic, features, fit_logistic, labels, retrain_positions, shuffle_labels, training_rows)


def make_prices(close: np.ndarray, start="2015-01-05") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                         "Volume": 1000}, index=idx)


def random_walk(n=1200, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 1000 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))


def mean_reverting(n=1500, seed=1) -> np.ndarray:
    """直近5日に上がった分を、次の5日で戻しやすい値動き（ret5 から予想できるパターン入り）。"""
    rng = np.random.default_rng(seed)
    lr = np.zeros(n)
    for i in range(n):
        past5 = lr[max(0, i - 5):i].sum()
        lr[i] = -0.25 * past5 + rng.normal(0, 0.01)
    return 1000 * np.exp(np.cumsum(lr))


SMALL = {"min_train": 200, "threshold": 0.5}


# ------------------------------------------------------------
# 特徴と答え
# ------------------------------------------------------------

def test_features_use_only_past_closes():
    close = random_walk(300)
    full = features(close)
    for i in (80, 150, 299):
        part = features(close[:i + 1])
        np.testing.assert_allclose(part[-1], full[i], rtol=1e-9)


def test_features_values():
    close = np.arange(1.0, 101.0)
    f = features(close)
    assert f[99, 0] == pytest.approx(100 / 95 - 1)           # ret5
    assert f[99, 1] == pytest.approx(100 / 80 - 1)           # ret20
    assert f[99, 2] == pytest.approx(100 / 40 - 1)           # ret60
    assert f[99, 3] == pytest.approx(100 / np.mean(close[75:]) - 1)   # gap25
    assert np.isnan(f[73, 4]) and not np.isnan(f[74, 4])     # 75日平均は75個目から
    assert np.isnan(f[59, 2]) and not np.isnan(f[60, 2])


def test_labels_are_future_up_and_last_days_unknown():
    close = np.array([10, 11, 9, 12, 12, 13, 8, 9.0])
    y = labels(close, 2)
    assert list(y[:-2]) == [0, 1, 1, 1, 0, 0]               # close[i+2] > close[i]
    assert np.isnan(y[-2:]).all()


def test_training_rows_only_use_labels_known_at_cutoff():
    """学習データの答えは、学び直す日（渡した終値の最後）までに確定したものだけ。"""
    close = random_walk(400)
    for cutoff in (200, 333, 399):
        X, y, rows = training_rows(close[:cutoff + 1], horizon=5)
        assert rows.max() + 5 <= cutoff                     # 答えに使う終値も cutoff 以前
        np.testing.assert_array_equal(y, (close[rows + 5] > close[rows]).astype(float))
        # 未来の終値をめちゃくちゃにしても、学習データは同じ
        wild = close.copy()
        wild[cutoff + 1:] = wild[cutoff + 1:] * 50
        X2, y2, rows2 = training_rows(wild[:cutoff + 1], horizon=5)
        np.testing.assert_array_equal(rows, rows2)
        np.testing.assert_array_equal(y, y2)


# ------------------------------------------------------------
# 学び直す日
# ------------------------------------------------------------

def test_retrain_positions_are_first_trading_day_of_quarter():
    idx = pd.bdate_range("2020-02-20", "2021-01-10")
    pos = retrain_positions(idx, 3)
    assert [d.strftime("%Y-%m-%d") for d in idx[pos]] == \
        ["2020-04-01", "2020-07-01", "2020-10-01", "2021-01-01"]
    assert len(retrain_positions(idx, 1)) == 11


# ------------------------------------------------------------
# 未来のデータが混ざらない（戦略として）
# ------------------------------------------------------------

def test_future_prices_do_not_change_past_decisions():
    """ある日より後の株価をでたらめに書き換えても、その日までの判断と確率は1つも変わらない。
    同じ戦略の入れ物（学んだモデルを取っておく所）を使い回しても変わらないことも見る。"""
    close = random_walk(1000, seed=3)
    prices = make_prices(close)
    shared = MlLogistic(**SMALL)
    base = generate_signals(shared, "X", prices)
    for cut in (450, 700, 900):
        wild = close.copy()
        rng = np.random.default_rng(cut)
        wild[cut + 1:] = wild[cut + 1:] * rng.uniform(0.3, 3.0, len(wild) - cut - 1)
        for strat in (MlLogistic(**SMALL), shared):
            sig = generate_signals(strat, "X", make_prices(wild))
            assert (sig["signal"].iloc[:cut + 1] == base["signal"].iloc[:cut + 1]).all()
            a = sig["prob"].iloc[:cut + 1].astype(float).to_numpy()
            b = base["prob"].iloc[:cut + 1].astype(float).to_numpy()
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
            np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=0)
    assert (base["signal"] != Signal.HOLD).sum() > 300       # ちゃんと判断している


def test_model_learned_at_cutoff_ignores_later_data():
    close = random_walk(900, seed=4)
    prices = make_prices(close)
    r = int(retrain_positions(prices.index, 3)[-2])
    a = MlLogistic(**SMALL).model_at(close[:r + 1], prices.index[r])
    full = MlLogistic(**SMALL).walk_forward(prices)          # 全期間を渡しても
    b_strat = MlLogistic(**SMALL)
    b_strat.walk_forward(make_prices(close[:r + 1]))
    b = b_strat.model_at(close[:r + 1], prices.index[r])
    np.testing.assert_allclose(a.coef, b.coef)
    # 学び直した日の予想は、その日の終値までで学んだモデルのもの
    x = features(close[:r + 1])[-1]
    assert full["prob"].iloc[r] == pytest.approx(float(a.predict_proba(x)[0]))


def test_decide_matches_walk_forward():
    prices = make_prices(random_walk(800, seed=5))
    s = MlLogistic(**SMALL)
    sig = generate_signals(s, "X", prices)
    wf = s.walk_forward(prices, "X")
    p = sig["prob"].astype(float)
    np.testing.assert_allclose(p.dropna(), wf["prob"].dropna(), rtol=1e-9)
    assert p.isna().equals(wf["prob"].isna())


def test_random_walk_is_not_predictable():
    """パターンのない値動きでは、予想は当てずっぽう並みになる（未来が混ざっていれば高く出る）。"""
    aucs = []
    for seed in range(6):
        prices = make_prices(random_walk(1500, seed=10 + seed))
        wf = MlLogistic(**SMALL).walk_forward(prices).dropna()
        aucs.append(auc(wf["prob"].to_numpy(), wf["actual"].to_numpy()))
    assert 0.45 < np.mean(aucs) < 0.55


def test_planted_pattern_is_found_and_shuffle_destroys_it():
    """仕込んだパターンは見つけられる。答えを入れ替えた版では見つからない。"""
    prices = make_prices(mean_reverting())
    real = MlLogistic(**SMALL).walk_forward(prices).dropna()
    real_auc = auc(real["prob"].to_numpy(), real["actual"].to_numpy())
    sh = []
    for seed in range(1, 6):
        wf = MlLogistic(**SMALL, shuffle_seed=seed).walk_forward(prices).dropna()
        sh.append(auc(wf["prob"].to_numpy(), wf["actual"].to_numpy()))
    assert real_auc > 0.65
    assert real_auc > max(sh)
    assert 0.35 < np.mean(sh) < 0.65


# ------------------------------------------------------------
# 答えの入れ替え
# ------------------------------------------------------------

def test_shuffle_keeps_counts_and_is_repeatable():
    y = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0.0] * 10)
    d = pd.Timestamp("2020-04-01")
    a = shuffle_labels(y, 7, d)
    assert a.sum() == y.sum() and not np.array_equal(a, y)
    np.testing.assert_array_equal(a, shuffle_labels(y, 7, d))
    assert not np.array_equal(a, shuffle_labels(y, 8, d))
    assert not np.array_equal(a, shuffle_labels(y, 7, pd.Timestamp("2020-07-01")))


def test_shuffled_version_learns_different_model():
    close = random_walk(700, seed=6)
    prices = make_prices(close)
    r = int(retrain_positions(prices.index, 3)[-1])
    a = MlLogistic(**SMALL).model_at(close[:r + 1], prices.index[r])
    b = MlLogistic(**SMALL, shuffle_seed=1).model_at(close[:r + 1], prices.index[r])
    assert not np.allclose(a.coef, b.coef)


# ------------------------------------------------------------
# モデル・判断・設定
# ------------------------------------------------------------

def test_fit_logistic_recovers_direction():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2))
    y = (rng.random(2000) < 1 / (1 + np.exp(-(2 * X[:, 0] - X[:, 1])))).astype(float)
    m = fit_logistic(X, y, l2=0.1)
    assert m.coef[0] > 1.5 and m.coef[1] < -0.7


def test_threshold_decides_buy_or_sell_and_hold_until_enough_data():
    prices = make_prices(random_walk(800, seed=7))
    lo = generate_signals(MlLogistic(min_train=200, threshold=0.01), "X", prices)
    hi = generate_signals(MlLogistic(min_train=200, threshold=0.99), "X", prices)
    decided = lo["signal"] != Signal.HOLD
    assert (lo.loc[decided, "signal"] == Signal.BUY).all()
    assert (hi.loc[decided, "signal"] == Signal.SELL).all()
    first = decided.idxmax()
    assert first >= prices.index[75 + 5 + 200]               # 学ぶデータがたまるまでは様子見
    assert (lo["signal"].loc[:first].iloc[:-1] == Signal.HOLD).all()


@pytest.mark.parametrize("params", [
    {"threshold": 1.0}, {"threshold": 0}, {"horizon": 0}, {"retrain_months": 5},
    {"min_train": 10}, {"l2": -1}, {"shuffle_seed": "a"},
])
def test_bad_params(params):
    with pytest.raises(StrategyError):
        create_strategy("ml_logistic", params)


BASE = {
    "symbols": [{"code": "1321.T", "lot_size": 10}],
    "period": {"start": "2018-01-01", "end": None},
    "account": {"initial_cash": 500000, "monthly_deposit": 10000},
    "costs": {"fee_rate": 0, "slippage_rate": 0.001, "tax_rate": 0},
    "strategy": {"name": "sma_cross", "params": {"window": 25}},
}


def test_config_ml_comparison():
    cfg = parse_config({**BASE, "ml_comparison": {
        "candidates": [{"name": "ml_logistic", "params": {"threshold": 0.6}}],
        "shuffle_runs": 5}})
    assert cfg.ml_comparison.shuffle_runs == 5
    assert cfg.ml_comparison.candidates[0].params == {"threshold": 0.6}
    assert parse_config(BASE).ml_comparison is None
    with pytest.raises(ConfigError):
        parse_config({**BASE, "ml_comparison": {"candidates": []}})
    with pytest.raises(ConfigError):
        parse_config({**BASE, "ml_comparison": {"candidates": [
            {"name": "ml_logistic", "params": {"shuffle_seed": 1}}]}})
    with pytest.raises(ConfigError):
        parse_config({**BASE, "ml_comparison": {"candidates": [{"name": "ml_logistic"}],
                                                "shuffle_runs": 0}})


def test_auc():
    assert auc(np.array([0.9, 0.8, 0.1, 0.2]), np.array([1, 1, 0, 0])) == 1.0
    assert auc(np.array([0.1, 0.2, 0.9, 0.8]), np.array([1, 1, 0, 0])) == 0.0
    assert auc(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5
