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
    players    : 戦略くんの候補（1人以上。画面の「相手」で切り替える）
  選手データ:
    id, label（戦略の説明）, equity（各日の資産、円）, twr（入金の影響を除いた指数）,
    markers（売買: 日の番号 i・売買・銘柄・口数・種類 entry/exit/topup）,
    summary（最終資産・損益・最大の落ち込み・売買回数・値差の負担額）
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .evaluate import compute_metrics, twr_index
from .simulator import SimResult

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
    if name == "buy_and_hold":
        return "最初に買って放置"
    return f"{name} {params}" if params else name


def markers(result: SimResult, dates: pd.DatetimeIndex) -> list[dict]:
    """売買の印。売り=exit、売りのあと（または最初）の買い=entry、それ以外の買い=topup（積立分）。"""
    t = result.trades_df()
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


def player(pid: str, label: str, result: SimResult, dates: pd.DatetimeIndex) -> dict:
    pf = result.portfolio.reindex(dates).ffill()
    m = compute_metrics(result)
    twr = twr_index(result.portfolio).reindex(dates).ffill()
    return {
        "id": pid,
        "label": label,
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


def dataset(benchmark: SimResult, players: dict[str, SimResult], initial_cash: float,
            symbols: list[str], benchmark_label: str = "最初に買って放置") -> dict:
    """1つのモード（リプレイなど）のデータ。players は {表示名: 結果}。"""
    if not players:
        raise ValueError("戦略くんの候補が1つもありません")
    dates = benchmark.portfolio.index
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "deposited": [round(float(v)) for v in benchmark.portfolio["deposited"]],
        "initial": round(initial_cash),
        "symbols": symbols,
        "benchmark": player("benchmark", benchmark_label, benchmark, dates),
        "players": [player(f"p{k}", name, r, dates) for k, (name, r) in enumerate(players.items())],
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(replay, forward), encoding="utf-8")
    return path
