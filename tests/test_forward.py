"""フォワードテスト（forward.py）のテスト。

一番大事な性質:
  「毎日1回ずつ動かした結果」と「あとでまとめて動かした結果」が同じになる
  （違ってよいのは、実行した時刻で決まる列だけ）。
"""

import json
from datetime import date, datetime, time, timedelta

import pandas as pd
import pytest

from sim.config import ConfigError, parse_config
from sim.data_loader import JST
from sim.forward import NO_ORDER, TIME_COLUMNS, ForwardError, run_forward, summary_text
from sim.state import SimState

from test_state import random_prices

PRICES = {"A": random_prices(1), "B": random_prices(2, skip=["2024-06-12", "2024-07-03"])}
START, END = date(2024, 6, 3), date(2024, 7, 12)   # 月またぎ（7月の積立）と休場日を含む
FILES = ["log.csv", "orders.csv", "trades.csv", "execution_check.csv"]


def config(tmp_path, **forward):
    fwd = {"start": START.isoformat(), "dir": str(tmp_path), "benchmark": "hold",
           "strategies": [{"id": "sma5", "name": "sma_cross", "params": {"window": 5}},
                          {"id": "hold", "name": "buy_and_hold"}]}
    fwd.update(forward)
    return parse_config({
        "symbols": [{"code": "A", "lot_size": 10}, {"code": "B", "lot_size": 1}],
        "period": {"start": "2023-10-02"},
        "account": {"initial_cash": 500000, "monthly_deposit": 10000},
        "costs": {"fee_rate": 0.001, "slippage_rate": 0.001},
        "strategy": {"name": "sma_cross"},
        "forward": fwd,
    })


def upto(day):
    return {s: df[df.index <= pd.Timestamp(day)] for s, df in PRICES.items()}


def evening(day):
    return datetime.combine(day, time(18, 30), JST)


def read(folder):
    out = {f: pd.read_csv(folder / f, encoding="utf-8-sig", dtype=str, keep_default_na=False)
           for f in FILES}
    out["state"] = {p.name: json.loads(p.read_text(encoding="utf-8"))
                    for p in sorted((folder / "state").glob("*.json"))}
    return out


def without_time(df):
    return df.drop(columns=[c for c in df.columns if c in TIME_COLUMNS]).reset_index(drop=True)


def run_daily(tmp_path):
    cfg = config(tmp_path)
    reports = []
    day = START - timedelta(days=2)   # 開始日より前の土日から毎日動かす
    while day <= END:
        reports.append(run_forward(cfg, upto(day), evening(day)))
        day += timedelta(days=1)
    return reports


def test_daily_runs_equal_one_batch_run(tmp_path):
    daily_dir, batch_dir = tmp_path / "daily", tmp_path / "batch"
    reports = run_daily(daily_dir)
    batch = run_forward(config(batch_dir), upto(END), evening(END))

    a, b = read(daily_dir), read(batch_dir)
    assert a["state"] == b["state"]
    for f in FILES:
        pd.testing.assert_frame_equal(without_time(a[f]), without_time(b[f]), obj=f)

    # 売買も積立も実際に起きている（空の比較になっていない）
    assert len(a["trades.csv"]) > 5 and set(a["trades.csv"]["売買"]) == {"買い", "売り"}
    assert a["state"]["hold.json"]["deposited"] == 510000      # 7月に積立が1回
    # 土日と休場日（両銘柄とも取引がない日）は何もしない
    weekends = [r for r in reports if r.run_at.weekday() >= 5]
    assert weekends and not any(r.changed for r in weekends)
    # 毎日動かせば注文は寄付に間に合う。まとめて動かすと「遅れ」になる
    assert set(a["execution_check.csv"]["間に合ったか"]) <= {"間に合う", NO_ORDER}
    assert "間に合う" in set(a["execution_check.csv"]["間に合ったか"])
    assert "遅れ（本番なら寄付に間に合わない）" in set(b["execution_check.csv"]["間に合ったか"])
    assert batch.changed and len(batch.days) > 25


def test_running_twice_on_the_same_day_does_nothing(tmp_path):
    cfg = config(tmp_path)
    day = date(2024, 6, 5)
    first = run_forward(cfg, upto(day), evening(day))
    before = read(tmp_path)
    second = run_forward(cfg, upto(day), evening(day) + timedelta(hours=1))
    assert first.changed and not second.changed
    after = read(tmp_path)
    for f in FILES:
        pd.testing.assert_frame_equal(before[f], after[f])
    assert "新しい株価がない" in summary_text(second)


def test_log_and_summary(tmp_path):
    cfg = config(tmp_path)
    rep = run_forward(cfg, upto(date(2024, 6, 4)), evening(date(2024, 6, 4)))
    log = rep.log
    assert list(log["日付"].unique()) == ["2024-06-03", "2024-06-04"]
    assert set(log["戦略id"]) == {"sma5", "hold"}
    first = log[log["日付"] == "2024-06-03"]
    assert (first["資産"] == 500000).all()                        # 初日は約定前
    hold = log[(log["日付"] == "2024-06-04") & (log["戦略id"] == "hold")].iloc[0]
    assert "買い" in hold["その日の売買"] and hold["放置との差"] == ""
    sma = log[(log["日付"] == "2024-06-04") & (log["戦略id"] == "sma5")].iloc[0]
    assert sma["放置との差"] == sma["資産"] - hold["資産"]
    text = summary_text(rep)
    assert "06/03〜06/04 の2営業日分" in text and "差: " in text
    assert "最初に買って放置" in text and "移動平均5日" in text


def test_dry_run_writes_nothing(tmp_path):
    rep = run_forward(config(tmp_path), upto(date(2024, 6, 5)), evening(date(2024, 6, 5)),
                      write=False)
    assert rep.changed and not any(tmp_path.iterdir())


def test_stops_when_past_prices_were_revised(tmp_path):
    """分割などで処理済みの日の価格が変わったら、口数が合わなくなるので止める。"""
    cfg = config(tmp_path)
    run_forward(cfg, upto(date(2024, 6, 5)), evening(date(2024, 6, 5)))
    split = {s: df.copy() for s, df in upto(date(2024, 6, 6)).items()}
    split["A"][["Open", "High", "Low", "Close", "Adj Close"]] /= 10
    with pytest.raises(ForwardError, match="分割"):
        run_forward(cfg, split, evening(date(2024, 6, 6)))
    assert SimState.load(tmp_path / "state" / "sma5.json").last_date == pd.Timestamp("2024-06-05")


def test_strategy_added_later_starts_on_its_own_day(tmp_path):
    cfg = config(tmp_path)
    run_forward(cfg, upto(date(2024, 6, 10)), evening(date(2024, 6, 10)))
    later = config(tmp_path, strategies=[
        {"id": "sma5", "name": "sma_cross", "params": {"window": 5}},
        {"id": "hold", "name": "buy_and_hold"},
        {"id": "band", "name": "sma_band", "params": {"window": 5, "band": 0.02},
         "start": "2024-06-11"}])
    rep = run_forward(later, upto(date(2024, 6, 11)), evening(date(2024, 6, 11)))
    assert sorted(rep.log["戦略id"]) == ["band", "hold", "sma5"]
    assert set(rep.log["日付"]) == {"2024-06-11"}


def test_forward_config_checks():
    base = {"symbols": [{"code": "A"}], "period": {"start": "2024-01-01"},
            "account": {"initial_cash": 1}, "costs": {}, "strategy": {"name": "x"}}
    ok = {"start": "2024-06-03", "benchmark": "h",
          "strategies": [{"id": "h", "name": "buy_and_hold"}]}
    assert parse_config({**base, "forward": ok}).forward.benchmark == "h"
    for bad, msg in [({"benchmark": "zzz"}, "benchmark"),
                     ({"strategies": [{"id": "Sma 25", "name": "sma_cross"}]}, "id"),
                     ({"strategies": [{"id": "h", "name": "a"}, {"id": "h", "name": "b"}]}, "重なって"),
                     ({"start": None}, "start")]:
        with pytest.raises(ConfigError, match=msg):
            parse_config({**base, "forward": {**ok, **bad}})


# ---- 対戦画面（forward/game.html） --------------------------------

from sim.forward import write_game_page  # noqa: E402
from sim.game import forward_dataset, next_result_day  # noqa: E402


def game_data(path):
    import re
    html = path.read_text(encoding="utf-8")
    return json.loads(re.search(r"const GAME = (.*?);\n", html).group(1).replace(r"<\/", "</"))


@pytest.mark.parametrize("now, expected", [
    (datetime(2026, 11, 8, 10, 0, tzinfo=JST), date(2026, 11, 9)),    # 日曜 → 月曜
    (datetime(2026, 11, 9, 12, 0, tzinfo=JST), date(2026, 11, 9)),    # 月曜の昼 → その日の夕方
    (datetime(2026, 11, 9, 19, 0, tzinfo=JST), date(2026, 11, 10)),   # 月曜の夜（祝日で無かった）→ 翌日
    (datetime(2026, 11, 13, 20, 0, tzinfo=JST), date(2026, 11, 16)),  # 金曜の夜 → 月曜
])
def test_next_result_day(now, expected):
    assert next_result_day(date(2026, 11, 8), now) == expected


def test_forward_dataset_before_start(tmp_path):
    d = forward_dataset(tmp_path, "hold", 500000, ["A"], date(2026, 11, 8),
                        datetime(2026, 11, 8, 10, 0, tzinfo=JST))
    assert d["empty"] and d["message"] == "開始前：最初の結果は11/9(月)の夕方"


def test_forward_dataset_from_records(tmp_path):
    run_daily(tmp_path)
    d = forward_dataset(tmp_path, "hold", 500000, ["A", "B"], START, evening(END))
    log = pd.read_csv(tmp_path / "log.csv", encoding="utf-8-sig")
    days = sorted(log["日付"].unique())
    assert d["kind"] == "forward" and d["dates"] == days
    hold = log[log["戦略id"] == "hold"]
    assert d["benchmark"]["equity"] == [round(v) for v in hold["資産"]]
    assert d["deposited"][-1] == 510000
    sma = d["players"][0]
    assert sma["id"] == "sma5" and sma["label"] == "移動平均5日"
    trades = pd.read_csv(tmp_path / "trades.csv", encoding="utf-8-sig")
    assert sma["summary"]["n_trades"] == int((trades["戦略id"] == "sma5").sum())
    kinds = {m["kind"] for m in sma["markers"]}
    assert {"entry", "exit"} <= kinds


def test_write_game_page_has_both_modes(tmp_path):
    cfg = config(tmp_path)
    run_forward(cfg, upto(date(2024, 6, 7)), evening(date(2024, 6, 7)))
    page = write_game_page(cfg, upto(date(2024, 6, 7)), evening(date(2024, 6, 7)))
    assert page == tmp_path / "game.html"
    data = game_data(page)
    replay, forward = data["modes"]["replay"], data["modes"]["forward"]
    assert replay["dates"][0] == "2023-10-02"           # 過去のリプレイは config の period から
    assert [p["label"] for p in replay["players"]] == ["移動平均5日"]
    assert forward["dates"] == ["2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06", "2024-06-07"]
