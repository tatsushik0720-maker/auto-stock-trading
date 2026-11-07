"""対戦ゲーム風の画面（game.py）と、シミュレータの開始日指定のテスト。"""

import json
import re

import pytest

from sim.game import dataset, markers, render, strategy_label, write_game
from sim.simulator import Costs, run_simulation
from sim.strategies import create_strategy

from test_strategies import prices_from

CLOSES = [100, 120, 90, 130, 80, 140, 70, 150, 160, 170]


def runs(start=None):
    df = prices_from(CLOSES, start="2024-01-01")
    args = ({"X": df}, {"X": 1})
    bench = run_simulation(*args, create_strategy("buy_and_hold"), 1000, 0, Costs(), start=start)
    strat = run_simulation(*args, create_strategy("sma_cross", {"window": 2}), 1000, 0, Costs(),
                           start=start)
    return bench, strat


def embedded(html: str) -> dict:
    return json.loads(re.search(r"const GAME = (.*?);\n", html).group(1).replace(r"<\/", "</"))


def test_markers_kinds():
    """最初の買い=entry、売り=exit、売りのあとの買い=entry。放置は最初の1回だけ entry。"""
    bench, strat = runs()
    kinds = [m["kind"] for m in markers(strat, strat.portfolio.index)]
    assert kinds[0] == "entry" and "exit" in kinds
    after_exit = [kinds[k + 1] for k, x in enumerate(kinds[:-1]) if x == "exit"]
    assert set(after_exit) <= {"entry"}
    assert [m["kind"] for m in markers(bench, bench.portfolio.index)] == ["entry"]


def test_dataset_and_render_round_trip(tmp_path):
    bench, strat = runs()
    ds = dataset(bench, {"移動平均2日": strat}, 1000, ["X"])
    assert len(ds["dates"]) == len(CLOSES)
    p = ds["players"][0]
    assert len(p["equity"]) == len(ds["dates"]) == len(p["twr"])
    assert p["summary"]["final_equity"] == round(strat.final_equity)
    assert p["summary"]["n_trades"] == len(strat.trades)
    path = write_game(tmp_path / "game.html", ds)
    data = embedded(path.read_text(encoding="utf-8"))
    assert data["modes"]["forward"] is None
    assert data["modes"]["replay"]["players"][0]["label"] == "移動平均2日"


def test_render_escapes_script_end():
    bench, strat = runs()
    html = render(dataset(bench, {"</script><b>": strat}, 1000, ["X"]))
    assert html.count("</script>") == 1          # 埋め込んだ名前で script が閉じない
    assert embedded(html)["modes"]["replay"]["players"][0]["label"] == "</script><b>"


def test_dataset_needs_a_player():
    bench, _ = runs()
    with pytest.raises(ValueError):
        dataset(bench, {}, 1000, ["X"])


def test_strategy_labels():
    assert strategy_label("sma_cross", {"window": 75}) == "移動平均75日"
    assert strategy_label("sma_band", {"window": 25, "band": 0.02}) == "移動平均25日 ±2%帯"
    assert strategy_label("sma_cooldown", {"window": 200, "cooldown": 20}) == "移動平均200日 20日あけ"


def test_start_uses_earlier_days_only_for_signals():
    """開始日より前は売買も入金もしない。でも平均の計算にはその前の日も使う。

    日付: 1/1 100, 1/2 120, 1/3 90, 1/4 130, 1/5 80, 1/8 140, 1/9 70 …
    開始日 1/8 の終値 140 は、前の日を含めた2日平均 (80+140)/2=110 より上
    → 開始日にすぐ買いの合図が出て、1/9 の始値で買う（前の日を使えなければ 1/10 になる）
    """
    bench, strat = runs(start="2024-01-08")
    assert strat.portfolio.index[0].strftime("%Y-%m-%d") == "2024-01-08"
    assert strat.portfolio["equity"].iloc[0] == 1000
    assert strat.trades[0].date.strftime("%Y-%m-%d") == "2024-01-09"
    assert strat.trades[0].side == "買い"
    assert bench.trades[0].date.strftime("%Y-%m-%d") == "2024-01-09"
