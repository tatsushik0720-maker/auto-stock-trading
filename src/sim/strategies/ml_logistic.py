"""機械学習戦略: ロジスティック回帰で「この先5営業日で上がるか」を予想し、確率が高いときだけ持つ。

特徴（その日の終値までで計算する6つ）:
  ret5 / ret20 / ret60 : 直近5日・20日・60日の値動き（終値の変化率）
  gap25 / gap75        : 25日・75日移動平均からの離れ具合（終値 ÷ 平均 − 1）
  vol20                : 直近20日の値動きの荒さ（日々の対数変化率の標準偏差）
答え:
  その日の終値より、horizon 営業日後（既定5）の終値が高ければ 1、そうでなければ 0

学び直し（ウォークフォワード）:
  retrain_months か月ごと（既定3。1・4・7・10月）の最初の取引日の終値のあとに学び直す。
  学ぶのは「その日の終値までで答えが確定している日」だけ。
  つまり答えに使う horizon 日後の終値も、学び直す日以前のもの。
  学び直した日から次に学び直す日の前日まで、同じモデルで毎日予想する。

合図:
  予想の確率が threshold 以上なら買い（持つ）、未満なら売り。
  学ぶデータが min_train 日分に満たないあいだは様子見。

shuffle_seed を指定すると「答えを入れ替えた版」になる（比べるための対照）。
  学び直しのたびに、答えだけを乱数で並べ替えてから学ぶ。特徴と答えの結びつきが壊れるので、
  本物の版がこれと同じくらいの成績なら、見つけた「パターン」は偶然の範囲と考える。

記憶を持たない決まりについて:
  判断は毎回「その日までの株価」から求め直す。学んだモデルは計算を速くするために
  取っておくが、同じ株価からは同じモデルができるので、取っておかなくても結果は変わらない。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import Decision, DayContext, Signal, Strategy, StrategyError
from .sma_band import rolling_mean

FEATURES = ("ret5", "ret20", "ret60", "gap25", "gap75", "vol20")
FEATURE_DAYS = 75          # 特徴がそろうのに必要な終値の数（75日平均が一番長い）


# ============================================================
# 特徴と答え
# ============================================================

def features(close: np.ndarray) -> np.ndarray:
    """各日の特徴（行=日、列=FEATURES）。計算に足りない日は NaN。
    i 日目の行は close[:i+1] だけから計算する（その日より後の終値は使わない）。"""
    close = np.asarray(close, dtype=float)
    n = len(close)
    out = np.full((n, len(FEATURES)), np.nan)
    for j, k in enumerate((5, 20, 60)):
        if n > k:
            out[k:, j] = close[k:] / close[:-k] - 1
    for j, w in ((3, 25), (4, 75)):
        out[:, j] = close / rolling_mean(close, w) - 1
    if n > 20:
        lr = np.diff(np.log(close))                       # lr[i-1] = i日目の対数変化率
        s = pd.Series(lr).rolling(20).std(ddof=0).to_numpy()
        out[1:, 5] = s
    return out


def labels(close: np.ndarray, horizon: int) -> np.ndarray:
    """答え。i 日目 = close[i+horizon] > close[i] なら 1、そうでなければ 0。
    horizon 日後の終値がまだ無い最後の horizon 日は NaN（答えが確定していない）。"""
    close = np.asarray(close, dtype=float)
    y = np.full(len(close), np.nan)
    if len(close) > horizon:
        y[:-horizon] = (close[horizon:] > close[:-horizon]).astype(float)
    return y


def training_rows(close: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """渡された終値（＝学び直す日までの終値）だけで作る学習データ。
    戻り値: X（特徴）, y（答え）, rows（使った日の番号）。
    答えが確定している日（i + horizon <= 最後の日）で、特徴がそろっている日だけを使う。"""
    X, y = features(close), labels(close, horizon)
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    rows = np.flatnonzero(ok)
    return X[ok], y[ok], rows


# ============================================================
# ロジスティック回帰（numpy だけで書く。L2 で係数を小さめに抑える）
# ============================================================

@dataclass(frozen=True)
class LogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    n_train: int

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = ((np.atleast_2d(X) - self.mean) / self.scale) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> LogisticModel:
    """特徴をそろえて（平均0・標準偏差1）から、ニュートン法で学ぶ。切片には L2 をかけない。"""
    X, y = np.asarray(X, float), np.asarray(y, float)
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    Z = np.column_stack([np.ones(len(X)), (X - mean) / scale])
    w = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], float(l2))
    reg[0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        grad = Z.T @ (p - y) + reg * w
        hess = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return LogisticModel(mean=mean, scale=scale, coef=w[1:], intercept=float(w[0]),
                         n_train=len(X))


# ============================================================
# 学び直す日とウォークフォワード
# ============================================================

def retrain_positions(index: pd.DatetimeIndex, months: int) -> np.ndarray:
    """学び直す日の番号。暦の months か月の区切り（1月起点）が変わった最初の取引日。
    データの最初の日は区切りとして数えない（その前の日がわからないため）。"""
    if len(index) == 0:
        return np.array([], dtype=int)
    block = (np.asarray(index.year) * 12 + np.asarray(index.month) - 1) // months
    return np.flatnonzero(np.r_[False, block[1:] != block[:-1]])


def shuffle_labels(y: np.ndarray, seed: int, cutoff: pd.Timestamp) -> np.ndarray:
    """答えを並べ替える（学び直す日ごとに違う並べ替え。同じ seed・同じ日なら毎回同じ）。"""
    rng = np.random.default_rng([int(seed), int(cutoff.strftime("%Y%m%d"))])
    return rng.permutation(y)


class MlLogistic(Strategy):
    name = "ml_logistic"
    description = "ロジスティック回帰で5営業日後に上がる確率を予想し、threshold 以上のときだけ持つ"

    def __init__(self, threshold: float = 0.55, horizon: int = 5, retrain_months: int = 3,
                 min_train: int = 500, l2: float = 1.0, shuffle_seed: int | None = None):
        def bad(msg):
            raise StrategyError(f"ml_logistic: {msg}")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
                or not 0 < threshold < 1:
            bad(f"threshold は 0 より大きく 1 より小さい確率にしてください（0.55 = 55%）。今: {threshold!r}")
        for key, v, lo in (("horizon", horizon, 1), ("retrain_months", retrain_months, 1),
                           ("min_train", min_train, 50)):
            if isinstance(v, bool) or not isinstance(v, int) or v < lo:
                bad(f"{key} は {lo} 以上の整数にしてください（今: {v!r}）")
        if 12 % retrain_months:
            bad(f"retrain_months は 12 を割り切れる月数（1・2・3・4・6・12）にしてください（今: {retrain_months}）")
        if isinstance(l2, bool) or not isinstance(l2, (int, float)) or l2 < 0:
            bad(f"l2 は 0 以上の数にしてください（今: {l2!r}）")
        if shuffle_seed is not None and (isinstance(shuffle_seed, bool)
                                         or not isinstance(shuffle_seed, int)):
            bad(f"shuffle_seed は整数か null にしてください（今: {shuffle_seed!r}）")
        self.threshold = float(threshold)
        self.horizon = horizon
        self.retrain_months = retrain_months
        self.min_train = min_train
        self.l2 = float(l2)
        self.shuffle_seed = shuffle_seed
        self._models: dict[tuple, LogisticModel | None] = {}

    def warmup_days(self) -> int:
        # 特徴がそろう日数 + 答えの確定待ち + 学ぶ日数 + 次の学び直し日まで（最大でおよそ1区切り）
        return FEATURE_DAYS + self.horizon + self.min_train + 21 * self.retrain_months + 5

    def params(self) -> dict:
        p = {"threshold": self.threshold, "horizon": self.horizon,
             "retrain_months": self.retrain_months, "min_train": self.min_train, "l2": self.l2}
        if self.shuffle_seed is not None:
            p["shuffle_seed"] = self.shuffle_seed
        return p

    # ---- 学ぶ ----
    def model_at(self, close: np.ndarray, cutoff_day: pd.Timestamp, key: tuple = ()) -> LogisticModel | None:
        """close（学び直す日までの終値。それより後は入れない）だけで学ぶ。足りなければ None。"""
        close = np.asarray(close, dtype=float)
        k = (key, cutoff_day, len(close), float(close[-1]), float(close.sum()))
        if k not in self._models:
            X, y, _ = training_rows(close, self.horizon)
            if len(y) < self.min_train:
                self._models[k] = None
            else:
                if self.shuffle_seed is not None:
                    y = shuffle_labels(y, self.shuffle_seed, cutoff_day)
                self._models[k] = fit_logistic(X, y, self.l2)
        return self._models[k]

    def decide(self, ctx: DayContext) -> Decision:
        prices = ctx.prices
        close = prices["Close"].to_numpy(dtype=float)
        pos = retrain_positions(prices.index, self.retrain_months)
        if len(pos) == 0:
            return Decision(Signal.HOLD, {"prob": None})
        r = int(pos[-1])                                   # 今日以前で最後に学び直した日
        model = self.model_at(close[:r + 1], prices.index[r], (ctx.symbol, prices.index[0]))
        if model is None:
            return Decision(Signal.HOLD, {"prob": None})
        x = features(close[-(FEATURE_DAYS + 1):])[-1]      # 今日の特徴（直近の終値だけで足りる）
        if np.isnan(x).any():
            return Decision(Signal.HOLD, {"prob": None})
        p = float(model.predict_proba(x)[0])
        sig = Signal.BUY if p >= self.threshold else Signal.SELL
        return Decision(sig, {"prob": p, "trained": prices.index[r].date().isoformat()})

    # ---- 予想の一覧（成績の分析用。decide と同じモデル・同じ特徴を使う） ----
    def walk_forward(self, prices: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        """全日について、その日の予想確率と、あとで分かった答えを並べる。
        列: prob（その日までのデータで学んだモデルの予想。無い日は NaN）, actual（答え）"""
        close = prices["Close"].to_numpy(dtype=float)
        X = features(close)
        prob = np.full(len(close), np.nan)
        pos = list(retrain_positions(prices.index, self.retrain_months)) + [len(close)]
        for r, nxt in zip(pos[:-1], pos[1:]):
            model = self.model_at(close[:r + 1], prices.index[r], (symbol, prices.index[0]))
            if model is None:
                continue
            seg = X[r:nxt]
            ok = ~np.isnan(seg).any(axis=1)
            prob[r:nxt][ok] = model.predict_proba(seg[ok])
        return pd.DataFrame({"prob": prob, "actual": labels(close, self.horizon)},
                            index=prices.index)
