"""機械学習の戦略の「予想の当たり具合」と「パターンを学べたか」の判定。

compare_ml.py と対戦画面（AIくん・ランダムくん）で同じ物差しを使うためにまとめてある。

AUC: 上がった日に付けた確率が、上がらなかった日に付けた確率より高い組の割合。
     0.5 = 当てずっぽうと同じ、1.0 = 全部当たる。
判定: AIくん（本物）の AUC が、どの期間でも「答えを入れ替えて学ばせた版」の全部を上回り、
      しかも 0.5 より大きいときだけ「パターンを学べた」とする。
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .strategies.ml_logistic import MlLogistic, labels, retrain_positions, top_features

MIN_DAYS = 120    # 答え合わせのできた取引日（約半年）がこれより少ない期間は、判定しない


def auc(prob: np.ndarray, actual: np.ndarray) -> float:
    """上がった日に付けた確率が、上がらなかった日より高い組の割合（同じ値は半分と数える）。"""
    prob, actual = np.asarray(prob, float), np.asarray(actual, float)
    pos, neg = prob[actual == 1], prob[actual == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def frames_of(strat: MlLogistic, prices: Mapping[str, pd.DataFrame]) -> list[pd.DataFrame]:
    """銘柄ごとの予想と答え（列 prob・actual）。"""
    return [strat.walk_forward(df, sym) for sym, df in prices.items()]


def _cut(frames: list[pd.DataFrame], a, b) -> pd.DataFrame:
    parts = [f.loc[pd.Timestamp(a):(pd.Timestamp(b) if b is not None else None)].dropna()
             for f in frames]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts) if parts else pd.DataFrame(columns=["prob", "actual"])


def period_quality(frames: list[pd.DataFrame], periods: Mapping[str, tuple],
                   threshold: float | None = None) -> dict:
    """期間ごとの AUC と、答え合わせのできた日数（全銘柄まとめて）。
    threshold を渡すと、持つと判断した日の上昇日の割合なども出す。"""
    out = {}
    for tag, (a, b) in periods.items():
        f = _cut(frames, a, b)
        p, y = f["prob"].to_numpy(float), f["actual"].to_numpy(float)
        q = {"auc": auc(p, y), "n": int(len(y)), "days": int(f.index.nunique()),
             "base": float(y.mean()) if len(y) else float("nan")}
        if threshold is not None:
            held = p >= threshold
            q["held_up"] = float(y[held].mean()) if held.any() else float("nan")
            q["held_share"] = float(held.mean()) if len(y) else float("nan")
        out[tag] = q
    return out


def prediction_quality(strat: MlLogistic, prices: Mapping[str, pd.DataFrame],
                       periods: Mapping[str, tuple]) -> dict:
    return period_quality(frames_of(strat, prices), periods, strat.threshold)


def shuffled_frames(params: dict, prices: Mapping[str, pd.DataFrame], runs: int
                    ) -> list[list[pd.DataFrame]]:
    """答えを入れ替えた版（乱数の種 1〜runs）の予想と答え。"""
    base = {k: v for k, v in params.items() if k != "shuffle_seed"}
    return [frames_of(MlLogistic(**base, shuffle_seed=seed), prices) for seed in range(1, runs + 1)]


def learned_verdict(ai: dict, shuffled: list[dict], period_labels: Mapping[str, str],
                    random_kun: dict | None = None, min_days: int = MIN_DAYS) -> dict:
    """AIくんがパターンを学べたかの判定（画面の結果に出す形）。

    ai / random_kun / shuffled の各要素は period_quality() の戻り値（期間 → {auc, n}）。
    """
    runs = len(shuffled)
    rows, pending, failed = [], [], []
    for tag, label in period_labels.items():
        q = ai[tag]
        sh = [s[tag]["auc"] for s in shuffled if not np.isnan(s[tag]["auc"])]
        beat = sum(v >= q["auc"] for v in sh) if not np.isnan(q["auc"]) else runs
        days = q.get("days", q["n"])
        rows.append({
            "label": label, "n": q["n"], "days": days, "auc_ai": _r(q["auc"]),
            "auc_random": _r(random_kun[tag]["auc"]) if random_kun else None,
            "auc_shuffled": [_r(min(sh)), _r(float(np.median(sh))), _r(max(sh))] if sh else None,
            "beat": int(beat),
        })
        if days < min_days:
            pending.append((label, days))
        elif beat > 0 or not q["auc"] > 0.5:
            failed.append(rows[-1])
    if pending:
        label, n = pending[0]
        status, title = "pending", "まだ判定できない"
        detail = (f"{label}の答え合わせができた日がまだ {n}営業日です"
                  f"（約半年の {min_days}営業日たまると判定します。予想の答えは5営業日後にわかります）")
    elif not failed:
        status, title = "learned", "パターンを学べた"
        detail = (f"どの期間でも、AIくんの予想は答えを入れ替えて学んだ {runs}通りの全部より当たっていた。"
                  "偶然だけでこうなることは少ない")
    else:
        status, title = "not_learned", "パターンを学べなかった"
        r = failed[0]
        why = (f"当てずっぽう（0.5）以下" if r["auc_ai"] is not None and r["auc_ai"] <= 0.5
               else f"答えを入れ替えて学んだ {runs}通りのうち {r['beat']}通りが、AIくん以上に当たっていた")
        detail = (f"{r['label']}は、AIくんの予想の当たり具合（AUC {r['auc_ai']}）が{why}。"
                  "ランダムに学んだのと差がなく、値動きのパターンを学べたとは言えない")
    return {"status": status, "title": title, "detail": detail, "runs": runs, "periods": rows}


def _r(v) -> float | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 3)


# ============================================================
# 対戦画面用: 毎日の予想確率と、学び直しの記録
# ============================================================

def ml_track(strat: MlLogistic, prices: Mapping[str, pd.DataFrame], dates: pd.DatetimeIndex) -> dict:
    """対戦画面の AIくんのメーターと「学び直し」の表示に使うデータ。

    prob     : {銘柄: 各日の予想確率（無い日は null）}
    retrains : [{i: その日の番号, date: 学び直した日, fresh: その日に学び直したか,
                 top: {銘柄: 重視した特徴の上位3つ}}]
               fresh=False は、表示の初日より前に学んで初日から使っているモデル
    """
    probs, events = {}, {}
    for sym, df in prices.items():
        wf = strat.walk_forward(df, sym)
        probs[sym] = [None if pd.isna(v) else round(float(v), 4) for v in wf["prob"].reindex(dates)]
        close = df["Close"].to_numpy(dtype=float)
        initial = None
        for r in retrain_positions(df.index, strat.retrain_months):
            model = strat.model_at(close[:r + 1], df.index[r], sym, df.index[0])
            if model is None:
                continue
            day = df.index[r]
            if day < dates[0]:
                initial = (day, model)
            elif day <= dates[-1]:
                i = int(dates.get_indexer([day])[0])
                if i >= 0:
                    e = events.setdefault(i, {"i": i, "date": day.date().isoformat(),
                                              "fresh": True, "top": {}})
                    e["top"][sym] = top_features(model)
        if initial is not None:
            e = events.setdefault(0, {"i": 0, "date": initial[0].date().isoformat(),
                                      "fresh": False, "top": {}})
            e["top"].setdefault(sym, top_features(initial[1]))
    return {"threshold": strat.threshold, "horizon": strat.horizon, "prob": probs,
            "retrains": [events[i] for i in sorted(events)]}


def action_label(sym_probs: dict, threshold: float) -> str:
    """ログ用: 銘柄ごとの確率の文字列。例 "1321.T:0.6123 / 1306.T:0.5871" """
    return " / ".join(f"{s}:{p:.4f}" for s, p in sym_probs.items() if p is not None)


def parse_probs(text: str) -> dict[str, float]:
    out = {}
    for part in str(text or "").split(" / "):
        if ":" in part:
            sym, v = part.rsplit(":", 1)
            try:
                out[sym.strip()] = float(v)
            except ValueError:
                pass
    return out


def actual_frames(prob_by_symbol: Mapping[str, pd.Series], prices: Mapping[str, pd.DataFrame],
                  horizon: int) -> list[pd.DataFrame]:
    """記録した予想確率（銘柄ごとの Series）に、株価から答えを付ける。"""
    out = []
    for sym, s in prob_by_symbol.items():
        df = prices.get(sym)
        if df is None:
            continue
        y = pd.Series(labels(df["Close"].to_numpy(dtype=float), horizon), index=df.index)
        out.append(pd.DataFrame({"prob": s, "actual": y.reindex(s.index)}))
    return out
