"""シミュレーション結果を「対戦ゲーム風」の1枚の HTML（サーバー不要）に書き出す。

画面の中身は game_template.html。ここでは結果をデータ（JSON）にして埋め込むだけ。

データの形:
  {
    "version": 1,
    "modes": {
      "replay":  データセット,      # 過去のリプレイ
      "forward": データセット|null  # フォワードテスト（今はまだ無いので null）
    }
  }
  データセット:
    dates      : 取引日の一覧（"2018-01-04" …）
    deposited  : 各日の投入総額（初期資金＋積立。全員同じ）
    initial    : 初期資金
    benchmark  : 放置くんの選手データ
    players    : ほかの選手（1人以上）。画面では benchmark と合わせた全員から、2人を自由に選んで対戦させる
    ml_verdict : AIくんとランダムくんがいるとき、「パターンを学べたか」の判定（ml_eval.learned_verdict）
  選手データ:
    id, label（戦略の説明）, equity（各日の資産、円）, twr（入金の影響を除いた指数）,
    char（キャラ: hold=放置くん / strategy=戦略くん / ai=AIくん / random=ランダムくん）, name（キャラの名前）,
    markers（売買: 日の番号 i・売買・銘柄・口数・種類 entry/exit/topup）,
    summary（最終資産・損益・最大の落ち込み・売買回数・値差の負担額）,
    ml（AIくん・ランダムくんだけ。毎日の予想確率・持つ基準・学び直しの記録。ml_eval.ml_track）
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from .evaluate import compute_metrics, twr_index
from .ml_eval import actual_frames, learned_verdict, parse_probs, period_quality, shuffled_frames
from .simulator import SimResult
from .strategies.ml_logistic import LogisticModel, top_features

TEMPLATE = Path(__file__).with_name("game_template.html")
PLACEHOLDER = "/*__GAME_DATA__*/null"


def strategy_label(name: str, params: dict) -> str:
    """画面や表に出す戦略の名前。"""
    w = params.get("window", 25)
    if name == "sma_cross":
        return f"移動平均{w}日"
    if name == "sma_band":
        return f"移動平均{w}日 ±{params.get('band', 0.02):.0%}帯"
    if name == "sma_cooldown":
        return f"移動平均{w}日 {params.get('cooldown', 20)}日あけ"
    if name == "ml_logistic":
        tag = "・答えをランダムに入れ替え" if params.get("shuffle_seed") is not None else ""
        return f"機械学習 確率{params.get('threshold', 0.55):.0%}以上で持つ{tag}"
    if name == "buy_and_hold":
        return "最初に買って放置"
    return f"{name} {params}" if params else name


CHAR_NAMES = {"hold": "放置くん", "strategy": "戦略くん", "ai": "AIくん", "random": "ランダムくん"}


def char_of(name: str, params: dict | None = None) -> str:
    """戦略の名前とパラメータから、画面のキャラを決める。"""
    params = params or {}
    if name == "buy_and_hold":
        return "hold"
    if name == "ml_logistic":
        return "random" if params.get("shuffle_seed") is not None else "ai"
    return "strategy"


def markers(result: SimResult, dates: pd.DatetimeIndex) -> list[dict]:
    """売買の印。売り=exit、売りのあと（または最初）の買い=entry、それ以外の買い=topup（積立分）。"""
    return markers_from(result.trades_df(), dates)


def markers_from(t: pd.DataFrame, dates: pd.DatetimeIndex) -> list[dict]:
    """売買の表（列 date・side・symbol・shares）から印を作る。"""
    if t.empty:
        return []
    t = t.assign(date=pd.to_datetime(t["date"])).sort_values("date", kind="stable")
    pos = {d: i for i, d in enumerate(dates)}
    holding: dict[str, bool] = {}
    out = []
    for r in t.itertuples():
        if r.side == "売り":
            kind = "exit"
            holding[r.symbol] = False
        else:
            kind = "topup" if holding.get(r.symbol) else "entry"
            holding[r.symbol] = True
        i = pos.get(r.date)
        if i is None:
            continue
        out.append({"i": i, "side": "sell" if r.side == "売り" else "buy",
                    "symbol": r.symbol, "shares": int(r.shares), "kind": kind})
    return out


def player(pid: str, label: str, result: SimResult, dates: pd.DatetimeIndex,
           char: str = "strategy", ml: dict | None = None) -> dict:
    pf = result.portfolio.reindex(dates).ffill()
    m = compute_metrics(result)
    twr = twr_index(result.portfolio).reindex(dates).ffill()
    out = {
        "id": pid,
        "label": label, "char": char, "name": CHAR_NAMES[char],
        "equity": [round(float(v)) for v in pf["equity"]],
        "twr": [round(float(v), 5) for v in twr],
        "markers": markers(result, dates),
        "summary": {
            "final_equity": round(m.final_equity), "pnl": round(m.pnl),
            "pnl_pct": round(m.pnl_pct, 5), "max_drawdown": round(m.max_drawdown, 5),
            "dd_peak": m.dd_peak.strftime("%Y-%m-%d") if m.dd_peak is not None else None,
            "n_trades": m.n_trades, "n_sells": sum(t.side == "売り" for t in result.trades),
            "slippage": round(m.slippage), "fee": round(m.fee), "tax": round(m.tax),
        },
    }
    if ml is not None:
        out["ml"] = ml
    return out


def dataset(benchmark: SimResult, players: dict[str, SimResult], initial_cash: float,
            symbols: list[str], benchmark_label: str = "最初に買って放置",
            chars: dict[str, str] | None = None, ml: dict[str, dict] | None = None,
            ml_verdict: dict | None = None) -> dict:
    """1つのモード（リプレイなど）のデータ。players は {表示名: 結果}。
    chars は {表示名: キャラ}（書かなければ戦略くん）、ml は {表示名: ml_eval.ml_track の結果}。"""
    if not players:
        raise ValueError("戦略くんの候補が1つもありません")
    chars, ml = chars or {}, ml or {}
    dates = benchmark.portfolio.index
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "deposited": [round(float(v)) for v in benchmark.portfolio["deposited"]],
        "initial": round(initial_cash),
        "symbols": symbols,
        "benchmark": player("benchmark", benchmark_label, benchmark, dates, "hold"),
        "players": [player(f"p{k}", name, r, dates, chars.get(name, "strategy"), ml.get(name))
                    for k, (name, r) in enumerate(players.items())],
        "ml_verdict": ml_verdict,
    }


# ============================================================
# フォワードテスト（forward/ の記録から作る）
# ============================================================

WEEKDAYS = "月火水木金土日"
RESULT_TIME = time(18, 45)   # 毎日の実行が終わるおおよその時刻


def next_result_day(start: date, now: datetime) -> date:
    """最初の結果が出る見込みの日（開始日以降の、まだ処理していない平日）。"""
    d = max(start, now.date())
    if d == now.date() and now.time() >= RESULT_TIME:
        d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def forward_dataset(folder: Path, benchmark: str, initial_cash: float, symbols: list[str],
                    start: date, now: datetime, prices: dict[str, pd.DataFrame] | None = None,
                    shuffle_runs: int = 20, shuffled: list | None = None) -> dict:
    """forward/log.csv と trades.csv から、画面の「フォワードテスト」モードのデータを作る。

    AIくん・ランダムくん（機械学習の選手）は、log.csv の予想確率と
    forward/state/<id>.model.json（学んだモデル）から、メーターと学び直しの表示を作る。
    prices を渡すと、AIくんがパターンを学べたかの判定も入れる（答え合わせに株価を使う）。
    記録がまだ0日分なら {"empty": True, "message": ...} を返す。
    """
    meta = {"kind": "forward", "start": start.isoformat(),
            "updated_at": now.isoformat(timespec="minutes")}
    log_path = Path(folder) / "log.csv"
    log = (pd.read_csv(log_path, encoding="utf-8-sig", dtype={"戦略id": str})
           if log_path.exists() else pd.DataFrame())
    if log.empty:
        d = next_result_day(start, now)
        return {**meta, "empty": True,
                "message": f"開始前：最初の結果は{d.month}/{d.day}({WEEKDAYS[d.weekday()]})の夕方"}
    log["日付"] = pd.to_datetime(log["日付"])
    dates = pd.DatetimeIndex(sorted(log["日付"].unique()))
    trades_path = Path(folder) / "trades.csv"
    trades = (pd.read_csv(trades_path, encoding="utf-8-sig", dtype={"戦略id": str})
              if trades_path.exists() else pd.DataFrame(columns=["戦略id"]))

    models = {}
    for sid in log["戦略id"].unique():
        mp = Path(folder) / "state" / f"{sid}.model.json"
        if mp.exists():
            models[sid] = json.loads(mp.read_text(encoding="utf-8"))

    def char(sid: str) -> str:
        if sid == benchmark:
            return "hold"
        if sid in models:
            return char_of(models[sid]["strategy"], models[sid]["params"])
        return "strategy"

    def probs_of(sid: str) -> dict[str, pd.Series]:
        rows = log[log["戦略id"] == sid]
        col = rows["予想確率"].fillna("") if "予想確率" in rows else pd.Series("", index=rows.index)
        by_sym: dict[str, dict] = {}
        for d, text in zip(rows["日付"], col):
            for sym, v in parse_probs(text).items():
                by_sym.setdefault(sym, {})[d] = v
        return {sym: pd.Series(v, dtype=float).reindex(dates) for sym, v in by_sym.items()}

    def ml_of(sid: str) -> dict | None:
        if sid not in models:
            return None
        m = models[sid]
        events: dict[int, dict] = {}
        pos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(dates)}
        for mod in m["models"]:
            i = pos.get(mod.get("first_used") or "")
            if i is None:
                continue
            e = events.setdefault(i, {"i": i, "date": mod["trained"],
                                      "fresh": mod["first_used"] == mod["trained"], "top": {}})
            e["top"][mod["symbol"]] = top_features(LogisticModel.from_dict(mod))
        return {"threshold": m["params"]["threshold"], "horizon": m["params"]["horizon"],
                "prob": {sym: [None if pd.isna(v) else round(float(v), 4) for v in s]
                         for sym, s in probs_of(sid).items()},
                "retrains": [events[i] for i in sorted(events)]}

    def one(sid: str) -> dict:
        rows = log[log["戦略id"] == sid].set_index("日付")
        eq = rows["資産"].astype(float).reindex(dates)
        dep = rows["投入総額"].astype(float).reindex(dates)
        first = eq.first_valid_index()
        label = rows["戦略"].iloc[-1]
        if first != dates[0]:     # 後から足した戦略は、足した日まで初期資金のまま表示する
            label += f"（{first:%m/%d}〜）"
        eq, dep = eq.fillna(initial_cash).ffill(), dep.fillna(initial_cash).ffill()
        pf = pd.DataFrame({"equity": eq, "deposited": dep})
        twr = twr_index(pf)
        dd = twr / twr.cummax() - 1
        trough = dd.idxmin()
        t = trades[trades["戦略id"] == sid]
        tt = pd.DataFrame({"date": pd.to_datetime(t.get("日付", pd.Series(dtype=str))),
                           "side": t.get("売買"), "symbol": t.get("銘柄"),
                           "shares": t.get("口数")})
        c = char(sid)
        out = {
            "id": sid, "label": label, "char": c, "name": CHAR_NAMES[c],
            "equity": [round(float(v)) for v in eq], "twr": [round(float(v), 5) for v in twr],
            "markers": markers_from(tt, dates),
            "summary": {
                "final_equity": round(eq.iloc[-1]), "pnl": round(eq.iloc[-1] - dep.iloc[-1]),
                "pnl_pct": round((eq.iloc[-1] - dep.iloc[-1]) / dep.iloc[-1], 5),
                "max_drawdown": round(float(dd.min()), 5),
                "dd_peak": twr.loc[:trough].idxmax().strftime("%Y-%m-%d") if dd.min() < 0 else None,
                "n_trades": len(t), "n_sells": int((t.get("売買") == "売り").sum()) if len(t) else 0,
                "slippage": round(float(t["値差の負担額"].sum())) if len(t) else 0,
                "fee": round(float(t["手数料"].sum())) if len(t) else 0,
                "tax": round(float(t["税の精算額"].sum())) if len(t) else 0,
            },
        }
        ml = ml_of(sid)
        if ml is not None:
            out["ml"] = ml
        return out

    ids = list(dict.fromkeys(log["戦略id"]))
    if benchmark not in ids:
        raise ValueError(f"forward/log.csv に比べる相手（{benchmark}）の記録がありません")
    bench_rows = log[log["戦略id"] == benchmark].set_index("日付")
    deposited = bench_rows["投入総額"].astype(float).reindex(dates).ffill().fillna(initial_cash)

    verdict = None
    ai = next((i for i in ids if char(i) == "ai"), None)
    rnd = next((i for i in ids if char(i) == "random"), None)
    if prices is not None and ai and rnd:
        params = models[ai]["params"]
        periods = {"fw": (dates[0], None)}

        def quality(sid):
            return period_quality(actual_frames(probs_of(sid), prices, params["horizon"]), periods)

        frames = shuffled if shuffled is not None else shuffled_frames(params, prices, shuffle_runs)
        sh = [period_quality(f, periods) for f in frames]
        verdict = learned_verdict(quality(ai), sh, {"fw": "フォワードテストの期間"}, quality(rnd))
    return {
        **meta,
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "deposited": [round(float(v)) for v in deposited],
        "initial": round(initial_cash), "symbols": symbols,
        "benchmark": one(benchmark),
        "players": [one(i) for i in ids if i != benchmark],
        "ml_verdict": verdict,
    }


def render(replay: dict, forward: dict | None = None) -> str:
    data = {"version": 1, "modes": {"replay": replay, "forward": forward}}
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    text = text.replace("</", "<\\/")   # 埋め込んだ文字で <script> が閉じないように
    html = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise ValueError("game_template.html にデータの差し込み位置がありません")
    return html.replace(PLACEHOLDER, text)


def write_game(path: Path, replay: dict, forward: dict | None = None) -> Path:
    """画面を書き出す。隣にアプリのアイコンと manifest（ホーム画面に追加したときの名前・色）も置く。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = path.with_suffix(".webmanifest")
    html = render(replay, forward).replace(MANIFEST_PLACEHOLDER, manifest.name)
    path.write_text(html, encoding="utf-8")
    write_app_files(path.parent, path.name, manifest.name)
    return path


# ============================================================
# アプリのアイコンと manifest
# ============================================================

ASSETS = Path(__file__).with_name("assets")
ICONS = ["icon-180.png", "icon-192.png", "icon-512.png", "icon-maskable-512.png"]
MANIFEST_PLACEHOLDER = "__MANIFEST_HREF__"
APP_NAME, APP_SHORT = "戦略くん VS 放置くん", "戦略VS放置"


def write_app_files(folder: Path, page: str, manifest_name: str) -> None:
    """アイコン（scripts/make_icons.py で作ったもの）と manifest を folder に置く。
    中身が同じなら書き直さない（毎日の実行でコミットに余計な変更を出さないため）。"""
    for name in ICONS:
        src, dst = ASSETS / name, folder / name
        if not src.exists():
            raise FileNotFoundError(f"アイコンがありません: {src}（python scripts/make_icons.py で作れます）")
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            dst.write_bytes(src.read_bytes())
    manifest = {
        "name": APP_NAME, "short_name": APP_SHORT, "lang": "ja",
        "start_url": f"./{page}", "scope": "./", "display": "standalone",
        "background_color": "#f2f2f7", "theme_color": "#f2f2f7",
        "icons": [
            {"src": "icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
            {"src": "icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
            {"src": "icon-maskable-512.png", "sizes": "512x512", "type": "image/png",
             "purpose": "maskable"},
        ],
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    dst = folder / manifest_name
    if not dst.exists() or dst.read_text(encoding="utf-8") != text:
        dst.write_text(text, encoding="utf-8", newline="\n")
