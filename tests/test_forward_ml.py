"""フォワードテストの機械学習の選手（AIくん・ランダムくん）のテスト。

確かめること:
  - 毎日1回ずつ動かしても、まとめて動かしても、学んだモデル・予想確率・記録が同じ
  - 学んだモデルを forward/state/<id>.model.json に保存し、次の日はそれを読み込んで予想する
  - 3か月ごと（7月の最初の取引日）に、その日までのデータだけで学び直す
  - 設定を変えたら止まる（追加した時点で固定）
"""

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from sim.forward import LOG_COLUMNS, ForwardError, run_forward, write_game_page
from sim.game import forward_dataset

from test_forward import END, FILES, START, config, evening, game_data, read, upto, without_time

AI = {"id": "ai", "name": "ml_logistic", "params": {"threshold": 0.5, "min_train": 50}}
RND = {"id": "random", "name": "ml_logistic",
       "params": {"threshold": 0.5, "min_train": 50, "shuffle_seed": 1}}
BASE = [{"id": "sma5", "name": "sma_cross", "params": {"window": 5}},
        {"id": "hold", "name": "buy_and_hold"}]


def ml_config(folder, ai=AI, rnd=RND):
    return config(folder, strategies=BASE + [ai, rnd])


def run_daily(folder, until=END):
    cfg = ml_config(folder)
    day = START - timedelta(days=2)
    while day <= until:
        run_forward(cfg, upto(day), evening(day))
        day += timedelta(days=1)
    return cfg


def models(folder, sid="ai"):
    return json.loads((folder / "state" / f"{sid}.model.json").read_text(encoding="utf-8"))


def test_daily_equals_batch_with_ml_players(tmp_path):
    run_daily(tmp_path / "daily")
    run_forward(ml_config(tmp_path / "batch"), upto(END), evening(END))
    a, b = read(tmp_path / "daily"), read(tmp_path / "batch")
    assert a["state"] == b["state"]          # 学んだモデルのファイル（*.model.json）も含む
    for f in FILES:
        pd.testing.assert_frame_equal(without_time(a[f]), without_time(b[f]), obj=f)
    assert "ai.model.json" in a["state"] and "random.model.json" in a["state"]


def test_log_has_probabilities_and_retrains(tmp_path):
    run_daily(tmp_path)
    log = pd.read_csv(tmp_path / "log.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    assert list(log.columns) == LOG_COLUMNS
    for sid in ("ai", "random"):
        rows = log[log["戦略id"] == sid]
        assert len(rows) == len(log[log["戦略id"] == "hold"])
        assert rows["予想確率"].str.contains("A:0.").all()       # 毎日、銘柄ごとの確率
        retrains = rows[rows["学び直し"] != ""]
        # 初日に「最初の学習」（4月の最初の取引日までのデータ）、7月の最初の取引日に「学び直し」
        assert list(retrains["日付"]) == ["2024-06-03", "2024-07-01"]
        assert "最初の学習（2024-04-01までのデータ" in retrains["学び直し"].iloc[0]
        assert "学び直し（2024-07-01までのデータ" in retrains["学び直し"].iloc[1]
        assert "重視:" in retrains["学び直し"].iloc[1]
    others = log[log["戦略id"].isin(["sma5", "hold"])]
    assert (others["予想確率"] == "").all() and (others["学び直し"] == "").all()
    # 本物とランダムくんは、同じ日でも違う確率を出す（違うモデルを学んでいる）
    p = log.set_index(["日付", "戦略id"])["予想確率"]
    assert (p.xs("ai", level=1) != p.xs("random", level=1)).any()


def test_models_learn_only_from_data_up_to_retrain_day(tmp_path):
    run_daily(tmp_path)
    m = models(tmp_path)
    assert {(x["symbol"], x["trained"], x["first_used"]) for x in m["models"]} == {
        ("A", "2024-04-01", "2024-06-03"), ("B", "2024-04-01", "2024-06-03"),
        ("A", "2024-07-01", "2024-07-01"), ("B", "2024-07-01", "2024-07-01")}
    # 7月1日のモデルは、7月1日までの株価だけで学んだものと同じ（それより後の株価を混ぜても同じ）
    from sim.strategies import create_strategy
    st = create_strategy("ml_logistic", AI["params"])
    for sym in ("A", "B"):
        close = upto(date(2024, 7, 1))[sym]
        fresh = st.model_at(close["Close"].to_numpy(), close.index[-1], sym)
        saved = next(x for x in m["models"] if x["symbol"] == sym and x["trained"] == "2024-07-01")
        assert saved["coef"] == pytest.approx(list(fresh.coef), abs=0)
        assert saved["n_train"] == fresh.n_train


def test_next_day_uses_saved_model(tmp_path):
    """保存したモデルを読み込んで予想している: ファイルの係数を書き換えると、次の日の確率が変わる。"""
    cfg = ml_config(tmp_path)
    run_forward(cfg, upto(date(2024, 6, 3)), evening(date(2024, 6, 3)))
    path = tmp_path / "state" / "ai.model.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for x in data["models"]:
        x["coef"] = [0.0] * len(x["coef"])
        x["intercept"] = 2.0                      # どの日も 1/(1+e^-2) ≒ 0.8808
    path.write_text(json.dumps(data), encoding="utf-8")
    rep = run_forward(cfg, upto(date(2024, 6, 4)), evening(date(2024, 6, 4)))
    row = rep.log[(rep.log["戦略id"] == "ai")].iloc[0]
    assert row["予想確率"] == "A:0.8808 / B:0.8808"
    assert row["学び直し"] == ""                  # 学び直していない（読み込んだだけ）


def test_settings_are_fixed_once_added(tmp_path):
    run_forward(ml_config(tmp_path), upto(date(2024, 6, 3)), evening(date(2024, 6, 3)))
    changed = {**AI, "params": {"threshold": 0.6, "min_train": 50}}
    with pytest.raises(ForwardError, match="追加した時点で固定"):
        run_forward(ml_config(tmp_path, ai=changed), upto(date(2024, 6, 4)),
                    evening(date(2024, 6, 4)))


def test_stops_when_log_columns_differ(tmp_path):
    (tmp_path / "log.csv").write_text("日付,戦略id\n", encoding="utf-8-sig")
    with pytest.raises(ForwardError, match="列が今のプログラムと違います"):
        run_forward(ml_config(tmp_path), upto(date(2024, 6, 3)), evening(date(2024, 6, 3)))


def test_forward_dataset_has_ai_meter_and_verdict(tmp_path):
    run_daily(tmp_path)
    d = forward_dataset(tmp_path, "hold", 500000, ["A", "B"], START, evening(END),
                        prices=upto(END), shuffle_runs=3)
    chars = {p["id"]: p["char"] for p in [d["benchmark"], *d["players"]]}
    assert chars == {"hold": "hold", "sma5": "strategy", "ai": "ai", "random": "random"}
    ai = next(p for p in d["players"] if p["id"] == "ai")
    assert ai["name"] == "AIくん" and ai["ml"]["threshold"] == 0.5
    assert len(ai["ml"]["prob"]["A"]) == len(d["dates"])
    assert all(v is not None and 0 <= v <= 1 for v in ai["ml"]["prob"]["A"])
    ev = ai["ml"]["retrains"]
    assert [(e["i"], e["fresh"]) for e in ev] == [(0, False), (d["dates"].index("2024-07-01"), True)]
    assert len(ev[1]["top"]["A"]) == 3 and {"name", "share", "sign"} <= set(ev[1]["top"]["A"][0])
    assert "ml" not in next(p for p in d["players"] if p["id"] == "sma5")
    v = d["ml_verdict"]      # 6週間分しかないので、まだ判定しない
    assert v["status"] == "pending" and "まだ" in v["detail"] and v["runs"] == 3
    assert v["periods"][0]["days"] == len(d["dates"]) - 5      # 最後の5日は答えがまだない


def test_game_page_has_four_players_with_ai_data(tmp_path):
    cfg = run_daily(tmp_path, until=date(2024, 6, 7))
    data = game_data(write_game_page(cfg, upto(date(2024, 6, 7)), evening(date(2024, 6, 7))))
    for mode in ("replay", "forward"):
        ds = data["modes"][mode]
        roster = [ds["benchmark"], *ds["players"]]
        assert sorted(p["name"] for p in roster) == ["AIくん", "ランダムくん", "戦略くん", "放置くん"]
        ai = next(p for p in roster if p["char"] == "ai")
        assert set(ai["ml"]["prob"]) == {"A", "B"} and ai["ml"]["retrains"]
        assert ds["ml_verdict"]["status"] in {"pending", "learned", "not_learned"}
