"""フォワードテスト: 毎日の実際の値動きで、架空売買を1日ずつ続ける。

1回の実行でやること（戦略ごと）:
  1. forward/state/<id>.json から架空口座の状態を読む（初回は初期資金から）
  2. 前回処理した日より後の取引日だけを処理する
       - 前の取引日に記録した注文を、今日の始値で約定させる
       - 今日の終値で合図を判定し、翌取引日の始値で約定させる注文を記録する
  3. 状態を保存し、記録ファイルに追記する
新しい取引日がなければ（土日祝、または株価の更新がまだ）何も書かずに終わる。

何度実行しても、同じ日を二重に処理しない。毎日1回ずつ動かした結果と、
あとからまとめて動かした結果は同じになる（テスト済み。記録した日時の列だけが違う）。

forward/ の中身:
  state/<id>.json      戦略ごとの架空口座の状態（再開に使う。前回分は .bak）
  state/<id>.model.json 機械学習の選手（AIくん・ランダムくん）が学んだモデル。毎日これを読み込んで予想し、
                       学び直したときだけ書き足す（前回分は .bak）
  log.csv              取引日×戦略ごとの資産・その日の売買・翌取引日の注文・放置との差・
                       予想確率と学び直し（機械学習の選手だけ）
  orders.csv           架空の注文（終値で判定した日に記録）
  trades.csv           約定した架空の売買
  execution_check.csv  約定の想定ずれ（本物のお金に移る前に確かめるためのデータ）
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path

import pandas as pd

from .config import Config, ForwardStrategyConfig
from .data_loader import JST
from .game import strategy_label
from .simulator import Costs, SimResult, TRADE_COLUMNS_JA, run_simulation
from .state import SimState, StateError
from .strategies import Strategy, StrategyError, create_strategy

MARKET_OPEN_JST = time(9, 0)
# 終値では1単位も買えない見込みで注文を記録しなかったのに、始値が下がって買えた約定
NO_ORDER = "注文の記録なし（終値では買えない見込みだった）"
REVISION_TOLERANCE = 0.005     # 処理済みの日の終値が 0.5% 以上変わっていたら止める
WEEKDAYS = "月火水木金土日"

LOG_COLUMNS = ["日付", "戦略id", "戦略", "資産", "投入総額", "損益", "その日の売買",
               "翌取引日の注文", "放置との差", "予想確率", "学び直し", "実行日時"]
FIXED = ("選手の設定は追加した時点で固定です。変えたいときは、config.yaml の forward.strategies に"
         "別の id の選手として足してください")
ORDER_COLUMNS = ["記録日時", "戦略id", "銘柄", "合図の日", "合図の日の終値", "注文",
                 "見込み口数", "約定予定"]
EXEC_COLUMNS = ["約定日", "戦略id", "銘柄", "売買", "合図の日", "合図の日の終値", "約定日の始値",
                "寄りの窓", "想定の約定価格", "終値からの不利なずれ", "想定した値差率",
                "見込み口数", "約定口数", "約定代金", "手数料", "約定後の現金",
                "注文を記録した日時", "間に合ったか"]
TRADE_COLUMNS = ["戦略id"] + list(TRADE_COLUMNS_JA.values())
# 実行した時刻で変わる列（毎日1回とまとめての比較では除く）
TIME_COLUMNS = {"実行日時", "記録日時", "注文を記録した日時", "間に合ったか"}


class ForwardError(RuntimeError):
    """フォワードテストを続けられないときのエラー（原因と対処を文にして持つ）。"""


@dataclass
class StrategyRun:
    spec: ForwardStrategyConfig
    label: str
    result: SimResult | None        # 新しい日がなければ None
    state_path: Path
    strategy: Strategy | None = None

    @property
    def model_path(self) -> Path:
        return self.state_path.with_name(f"{self.spec.id}.model.json")

    @property
    def new_days(self) -> list[pd.Timestamp]:
        return [] if self.result is None else list(self.result.portfolio.index)


@dataclass
class ForwardReport:
    run_at: datetime
    runs: list[StrategyRun]
    benchmark: str
    days: list[pd.Timestamp] = field(default_factory=list)   # 今回処理した取引日
    log: pd.DataFrame | None = None
    orders: pd.DataFrame | None = None
    trades: pd.DataFrame | None = None
    executions: pd.DataFrame | None = None

    @property
    def changed(self) -> bool:
        return bool(self.days)


# ============================================================
# 本体
# ============================================================

def run_forward(cfg: Config, prices: dict[str, pd.DataFrame], now: datetime,
                write: bool = True) -> ForwardReport:
    """全戦略を、前回の続きから処理する。write=False なら何も保存しない（試し実行）。"""
    fwd = cfg.forward
    if fwd is None:
        raise ForwardError("config.yaml に forward の項目がありません")
    now = now.astimezone(JST)
    lots = {s.code: s.lot_size for s in cfg.symbols}
    costs = Costs(cfg.costs.fee_rate, cfg.costs.slippage_rate, cfg.costs.tax_rate)
    missing = [s for s in lots if s not in prices]
    if missing:
        raise ForwardError(f"株価データがありません: {', '.join(missing)}")
    prices = {s: prices[s] for s in lots}

    runs = []
    for spec in fwd.strategies:
        path = fwd.dir / "state" / f"{spec.id}.json"
        state = SimState.load(path) if path.exists() else None
        if state is not None:
            check_revisions(spec.id, state, prices)
        start = spec.start or fwd.start
        if not any((df.index >= pd.Timestamp(start)).any() for df in prices.values()):
            runs.append(StrategyRun(spec, _label(spec), None, path))
            continue
        strategy = create_strategy(spec.name, spec.params)
        model_path = path.with_name(f"{spec.id}.model.json")
        if hasattr(strategy, "import_models") and model_path.exists():
            try:   # 学んだモデルを読み込み、それで予想する（学び直す日だけ新しく学ぶ）
                strategy.import_models(json.loads(model_path.read_text(encoding="utf-8")), spec.id)
            except (ValueError, KeyError, TypeError) as e:
                if isinstance(e, StrategyError):
                    raise ForwardError(str(e)) from None
                raise ForwardError(f"{spec.id}: 学んだモデルのファイルが読めません: {model_path}（{e}）"
                                   "。前回分は .bak に残っています") from None
        try:
            result = run_simulation(prices, lots, strategy, cfg.account.initial_cash,
                                    cfg.account.monthly_deposit, costs, state=state, start=start)
        except StateError as e:
            raise ForwardError(f"{spec.id}: {e}。{FIXED}") from None
        runs.append(StrategyRun(spec, _label(spec), result if len(result.portfolio) else None,
                                path, strategy))

    report = ForwardReport(run_at=now, runs=runs, benchmark=fwd.benchmark)
    report.days = sorted({d for r in runs for d in r.new_days})
    if not report.changed:
        return report

    old_orders = _read_csv(fwd.dir / "orders.csv", ORDER_COLUMNS)
    report.orders = build_orders(runs, cfg, now)
    report.trades = build_trades(runs)
    report.executions = build_executions(runs, prices, cfg, pd.concat(
        [old_orders, report.orders], ignore_index=True))
    report.log = build_log(runs, fwd.benchmark, report.orders, now)
    if write:
        _check_header(fwd.dir / "log.csv", LOG_COLUMNS)
        _append_csv(fwd.dir / "log.csv", report.log)
        _append_csv(fwd.dir / "orders.csv", report.orders)
        _append_csv(fwd.dir / "trades.csv", report.trades)
        _append_csv(fwd.dir / "execution_check.csv", report.executions)
        for r in runs:      # 状態は最後に保存する（途中で失敗したら次回やり直せるように）
            if r.result is not None:
                if hasattr(r.strategy, "export_models"):
                    _save_json(r.model_path, r.strategy.export_models())
                r.result.state.save(r.state_path)
    return report


def check_revisions(sid: str, state: SimState, prices: dict[str, pd.DataFrame]) -> None:
    """処理済みの日の終値が、いまのデータと同じか確かめる。

    株式分割などで Yahoo が過去の価格を調整し直すと、保有口数と価格が合わなくなる。
    そのまま続けると資産が大きく狂うので、止めて知らせる。
    """
    for symbol, acc in state.accounts.items():
        if acc.last_date is None or acc.last_close is None:
            continue
        df = prices.get(symbol)
        if df is None or acc.last_date not in df.index:
            raise ForwardError(
                f"{sid}: {symbol} の処理済みの日 {acc.last_date.date()} が株価データにありません。"
                "データが取り直された可能性があります。data/ の内容を確認してください")
        now_close = float(df.loc[acc.last_date, "Close"])
        if abs(now_close / acc.last_close - 1) > REVISION_TOLERANCE:
            raise ForwardError(
                f"{sid}: {symbol} の {acc.last_date.date()} の終値が、処理したときの "
                f"{acc.last_close:,.1f}円 から {now_close:,.1f}円 に変わっています"
                f"（{now_close / acc.last_close:.3g}倍）。株式分割などで過去の価格が調整された"
                "可能性が高く、このまま続けると保有口数と価格が合わなくなるため止めました。"
                "分割なら、状態ファイルの保有口数と取得価格を直してから再実行してください")


def _label(spec: ForwardStrategyConfig) -> str:
    return strategy_label(spec.name, spec.params)


def _save_json(path: Path, data: dict) -> None:
    """壊れにくい保存: 一時ファイルに書いてから置き換え、前回分は .bak に残す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=1) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    if path.exists():
        os.replace(path, path.with_suffix(path.suffix + ".bak"))
    os.replace(tmp, path)


def _probs_text(result: SimResult, day: pd.Timestamp) -> str:
    """その日の予想確率（機械学習の選手だけ）。例 "1321.T:0.6123 / 1306.T:0.5871" """
    parts = []
    for (d, sym), info in result.decisions.items():
        if d == day and info.get("prob") is not None:
            parts.append(f"{sym}:{info['prob']:.4f}")
    return " / ".join(parts)


def _retrain_text(strategy, day: pd.Timestamp) -> str:
    """その日に学んだモデル（学び直し、または最初の学習）と、重視した特徴の上位3つ。"""
    first_used = getattr(strategy, "first_used", None)
    if not first_used:
        return ""
    from .strategies.ml_logistic import top_features
    iso = day.date().isoformat()
    parts = []
    for (sym, trained), used in sorted(first_used.items()):
        if used != iso:
            continue
        model = strategy.trained.get((sym, trained)) or strategy.saved.get((sym, trained))
        kind = "学び直し" if trained == iso else "最初の学習"
        top = "・".join(f"{t['name']}{'↑' if t['sign'] > 0 else '↓'}" for t in top_features(model))
        parts.append(f"{sym} {kind}（{trained}までのデータ。重視: {top}）")
    return " / ".join(parts)


# ============================================================
# 記録づくり
# ============================================================

def build_orders(runs: list[StrategyRun], cfg: Config, now: datetime) -> pd.DataFrame:
    """終値で判定した架空の注文。買いは「1単位以上買える見込みがある日」だけ記録する
    （買いの状態が続く間は毎日買いの合図が出るが、現金が足りない日は何も起きないため）。"""
    slip, fee = cfg.costs.slippage_rate, cfg.costs.fee_rate
    lots = {s.code: s.lot_size for s in cfg.symbols}
    rows = []
    for r in runs:
        if r.result is None:
            continue
        for day, d in r.result.daily.iterrows():
            lot = lots[d["symbol"]]
            if d["signal"] == "買い":
                unit = lot * d["close"] * (1 + slip) * (1 + fee)
                n = math.floor(d["cash"] / unit + 1e-9) * lot
                if n <= 0:
                    continue
                order = "買い"
            elif d["signal"] == "売り" and d["shares"] > 0:
                n, order = int(d["shares"]), "売り"
            else:
                continue
            rows.append({"記録日時": _ts(now), "戦略id": r.spec.id, "銘柄": d["symbol"],
                         "合図の日": day.date().isoformat(), "合図の日の終値": d["close"],
                         "注文": order, "見込み口数": int(n), "約定予定": "翌取引日の始値"})
    return _by_date(pd.DataFrame(rows, columns=ORDER_COLUMNS), "合図の日")


def build_trades(runs: list[StrategyRun]) -> pd.DataFrame:
    parts = []
    for r in runs:
        if r.result is None or not r.result.trades:
            continue
        t = r.result.trades_df().rename(columns=TRADE_COLUMNS_JA)
        t["日付"] = pd.to_datetime(t["日付"]).dt.date.astype(str)
        t.insert(0, "戦略id", r.spec.id)
        parts.append(t)
    if not parts:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return _by_date(pd.concat(parts, ignore_index=True)[TRADE_COLUMNS], "日付")


def build_executions(runs: list[StrategyRun], prices: dict[str, pd.DataFrame], cfg: Config,
                     orders: pd.DataFrame) -> pd.DataFrame:
    """約定の想定ずれ: 合図の日の終値と、実際に約定させた翌取引日の始値の差などを1件ずつ残す。

    本物のお金に移る前に「終値を見てから翌朝の寄付で注文する」やり方で、
    思った値段・思った口数で買えるか、注文が間に合うかを確かめるためのデータ。
    """
    rows = []
    for r in runs:
        if r.result is None:
            continue
        events = [(t.date, t.symbol, t) for t in r.result.trades] + \
                 [(m.date, m.symbol, m) for m in r.result.missed_buys]
        for day, symbol, ev in sorted(events, key=lambda e: (e[0], e[1])):
            df = prices[symbol]
            pos = df.index.get_loc(day)
            sig_day = df.index[pos - 1] if pos > 0 else None
            sig_close = float(df["Close"].iloc[pos - 1]) if pos > 0 else float("nan")
            open_ = float(df.loc[day, "Open"])
            is_trade = hasattr(ev, "side")
            side = ev.side if is_trade else "買い"
            o = orders[(orders["戦略id"] == r.spec.id) & (orders["銘柄"] == symbol)
                       & (orders["合図の日"].astype(str) == (sig_day.date().isoformat()
                                                         if sig_day is not None else ""))]
            recorded = o["記録日時"].iloc[-1] if len(o) else None
            if not is_trade and not len(o):
                continue   # 買える見込みがなく注文も出していない日（想定どおり）
            if is_trade:
                adverse = (ev.exec_price / sig_close - 1) if side == "買い" else \
                    (1 - ev.exec_price / sig_close)
            else:
                adverse = float("nan")
            rows.append({
                "約定日": day.date().isoformat(), "戦略id": r.spec.id, "銘柄": symbol,
                "売買": side if is_trade else "買い（現金不足で0口）",
                "合図の日": sig_day.date().isoformat() if sig_day is not None else "",
                "合図の日の終値": sig_close, "約定日の始値": open_,
                "寄りの窓": round(open_ / sig_close - 1, 6),
                "想定の約定価格": ev.exec_price if is_trade else float("nan"),
                "終値からの不利なずれ": round(adverse, 6),
                "想定した値差率": cfg.costs.slippage_rate,
                "見込み口数": int(o["見込み口数"].iloc[-1]) if len(o) else None,
                "約定口数": ev.shares if is_trade else 0,
                "約定代金": round(ev.amount, 2) if is_trade else 0.0,
                "手数料": round(ev.fee, 2) if is_trade else 0.0,
                "約定後の現金": round(ev.cash_after if is_trade else ev.cash, 2),
                "注文を記録した日時": recorded or "",
                "間に合ったか": _in_time(recorded, day),
            })
    df = pd.DataFrame(rows, columns=EXEC_COLUMNS)
    df["見込み口数"] = df["見込み口数"].astype("Int64")   # 注文の記録がない約定は空欄
    return _by_date(df, "約定日")


def _in_time(recorded: str | None, day: pd.Timestamp) -> str:
    """注文を記録した時刻が、約定日の取引開始（9:00）より前か。"""
    if not recorded:
        return NO_ORDER
    at = datetime.fromisoformat(recorded)
    open_at = datetime.combine(day.date(), MARKET_OPEN_JST, JST)
    return "間に合う" if at < open_at else "遅れ（本番なら寄付に間に合わない）"


def build_log(runs: list[StrategyRun], benchmark: str, orders: pd.DataFrame,
              now: datetime) -> pd.DataFrame:
    bench = next((r for r in runs if r.spec.id == benchmark), None)
    bench_eq = bench.result.portfolio["equity"] if bench and bench.result is not None else None
    rows = []
    for r in runs:
        if r.result is None:
            continue
        trades = r.result.trades_df()
        for day, p in r.result.portfolio.iterrows():
            t = trades[pd.to_datetime(trades["date"]) == day] if len(trades) else trades
            o = orders[(orders["戦略id"] == r.spec.id) & (orders["合図の日"] == day.date().isoformat())]
            if r.spec.id == benchmark:
                diff = ""
            elif bench_eq is not None and day in bench_eq.index:
                diff = round(p["equity"] - bench_eq.loc[day])
            else:
                diff = ""
            rows.append({
                "日付": day.date().isoformat(), "戦略id": r.spec.id, "戦略": r.label,
                "資産": round(p["equity"]), "投入総額": round(p["deposited"]),
                "損益": round(p["equity"] - p["deposited"]),
                "その日の売買": " / ".join(f"{x.symbol} {x.side} {x.shares}口"
                                       for x in t.itertuples()) or "なし",
                "翌取引日の注文": " / ".join(f"{x.銘柄} {x.注文} {x.見込み口数}口"
                                        for x in o.itertuples()) or "なし",
                "放置との差": diff,
                "予想確率": _probs_text(r.result, day),
                "学び直し": _retrain_text(r.strategy, day),
                "実行日時": _ts(now),
            })
    return _by_date(pd.DataFrame(rows, columns=LOG_COLUMNS), "日付")


# ============================================================
# 結果の文章（通知用）
# ============================================================

def summary_text(report: ForwardReport) -> str:
    today = report.run_at.date()
    head = f"フォワードテスト {today:%m/%d}({WEEKDAYS[today.weekday()]})"
    if not report.changed:
        if today.weekday() >= 5:
            why = "土日のため取引なし"
        else:
            why = ("祝日か、今日の株価がまだ出ていません。"
                   "まだ出ていないだけなら、次の実行でまとめて処理します")
        return f"{head}: 新しい株価がないため何もしませんでした（{why}）"
    log = report.log
    last = report.days[-1]
    day_log = log[log["日付"] == last.date().isoformat()]
    span = (f"{report.days[0]:%m/%d}〜{last:%m/%d} の{len(report.days)}営業日分"
            if len(report.days) > 1 else f"{last:%m/%d}")
    lines = [f"{head} {span}の結果"]
    for row in day_log.itertuples():
        lines.append(f"・{row.戦略}: {row.資産:,}円（損益 {row.損益:+,}円）売買: {row.その日の売買}")
    bench = day_log[day_log["戦略id"] == report.benchmark]
    if len(bench):
        bname = bench["戦略"].iloc[0]
        for row in day_log[day_log["戦略id"] != report.benchmark].itertuples():
            if row.放置との差 == "":
                continue
            d = int(row.放置との差)
            who = row.戦略 if d > 0 else bname
            lines.append(f"差: {who}が {abs(d):,}円リード" if d else f"差: {row.戦略}と{bname}は同じ")
    orders = day_log[day_log["翌取引日の注文"] != "なし"]
    if len(orders):
        lines.append("次の取引日の始値で: " + "、".join(
            f"{r.戦略} {r.翌取引日の注文}" for r in orders.itertuples()))
    late = report.executions[report.executions["間に合ったか"].str.startswith("遅れ")] \
        if report.executions is not None and len(report.executions) else []
    if len(late):
        lines.append(f"注意: 寄付に間に合わない時刻に判定した約定が {len(late)}件あります"
                     "（execution_check.csv）")
    return "\n".join(lines)


# ============================================================
# ファイル
# ============================================================

def _by_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """日付順に並べる（同じ日の中は戦略の順のまま）。毎日1回でもまとめてでも同じ並びになる。"""
    return df.sort_values(col, kind="stable").reset_index(drop=True)


def _ts(now: datetime) -> str:
    return now.astimezone(JST).isoformat(timespec="seconds")


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def _check_header(path: Path, columns: list[str]) -> None:
    """追記する前に、今あるファイルの列が同じか確かめる（ずれたまま書き足さないため）。"""
    if not path.exists():
        return
    with path.open(encoding="utf-8-sig") as f:
        head = f.readline().rstrip("\r\n").split(",")
    if head != columns:
        raise ForwardError(
            f"{path.name} の列が今のプログラムと違います（ファイル: {', '.join(head)}）。"
            "列を足すプログラムの変更があったときは、古い記録を別名に移してから実行してください")


def _append_csv(path: Path, df: pd.DataFrame) -> None:
    """追記（初回は見出しつきで作る）。Excel で文字化けしないよう BOM 付き UTF-8。"""
    if df is None or df.empty:
        if not path.exists() and df is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False, encoding="utf-8-sig")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


# ============================================================
# 対戦画面（forward/game.html）
# ============================================================

def write_game_page(cfg: Config, prices: dict[str, pd.DataFrame], now: datetime,
                    path: Path | None = None) -> Path:
    """「過去のリプレイ」と「フォワードテスト」の両方が入った対戦画面を書き出す。

    過去のリプレイ: フォワードテストと同じ戦略で、config の period（2018年〜）を通しで動かした結果
    フォワードテスト: forward/log.csv と trades.csv の記録（0日分なら「開始前」と表示）
    """
    from .game import char_of, dataset, forward_dataset, write_game
    from .ml_eval import frames_of, learned_verdict, ml_track, period_quality, shuffled_frames

    fwd = cfg.forward
    lots = {s.code: s.lot_size for s in cfg.symbols}
    costs = Costs(cfg.costs.fee_rate, cfg.costs.slippage_rate, cfg.costs.tax_rate)
    prices = {s: prices[s] for s in lots}
    # 売買と入金は period の初日から。それより前のデータは、平均や学習に使うだけ
    start = cfg.period.start
    sims, strats = {}, {}
    for spec in fwd.strategies:
        strats[spec.id] = create_strategy(spec.name, spec.params)
        sims[spec.id] = run_simulation(prices, lots, strats[spec.id], cfg.account.initial_cash,
                                       cfg.account.monthly_deposit, costs, start=start)
    labels = {spec.id: _label(spec) for spec in fwd.strategies}
    chars = {labels[s.id]: char_of(s.name, s.params) for s in fwd.strategies}
    dates = sims[fwd.benchmark].portfolio.index
    ml = {labels[i]: ml_track(st, prices, dates) for i, st in strats.items()
          if hasattr(st, "walk_forward")}

    # AIくんとランダムくんがいれば「パターンを学べたか」を判定する（compare_ml.py と同じ物差し）
    verdict, sh = None, None
    ai = next((s for s in fwd.strategies if char_of(s.name, s.params) == "ai"), None)
    rnd = next((s for s in fwd.strategies if char_of(s.name, s.params) == "random"), None)
    runs = cfg.ml_comparison.shuffle_runs if cfg.ml_comparison else 20
    if ai and rnd:
        if cfg.comparison:
            c = cfg.comparison
            periods = {"sel": (c.select_period.start, c.select_period.end),
                       "chk": (c.check_period.start, c.check_period.end)}
            names = {"sel": "選ぶ期間", "chk": "確かめる期間"}
        else:
            periods, names = {"all": (start, None)}, {"all": "全期間"}
        span = {t: _span(dates, a, b) for t, (a, b) in periods.items()}
        period_labels = {t: f"{names[t]}（{span[t]}）" for t in periods}
        sh_frames = shuffled_frames(strats[ai.id].params(), prices, runs)
        sh = sh_frames
        verdict = learned_verdict(period_quality(frames_of(strats[ai.id], prices), periods),
                                  [period_quality(f, periods) for f in sh_frames], period_labels,
                                  period_quality(frames_of(strats[rnd.id], prices), periods))
    symbols = list(lots)
    replay = dataset(sims[fwd.benchmark],
                     {labels[i]: r for i, r in sims.items() if i != fwd.benchmark},
                     cfg.account.initial_cash, symbols, labels[fwd.benchmark],
                     chars=chars, ml=ml, ml_verdict=verdict)
    forward = forward_dataset(fwd.dir, fwd.benchmark, cfg.account.initial_cash, symbols,
                              fwd.start, now, prices=prices, shuffle_runs=runs, shuffled=sh)
    return write_game(path or fwd.dir / "game.html", replay, forward)


def _span(dates: pd.DatetimeIndex, a, b) -> str:
    inside = dates[(dates >= pd.Timestamp(a)) & ((dates <= pd.Timestamp(b)) if b else True)]
    if not len(inside):
        return "-"
    return f"{inside[0]:%Y/%m/%d}〜{inside[-1]:%Y/%m/%d}"
