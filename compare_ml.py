"""機械学習の戦略（ml_logistic）を、放置・25日移動平均と期間を分けて比べる。

使い方:
    python compare_ml.py                   # config.yaml の comparison（期間）と ml_comparison で実行
    python compare_ml.py --shuffle-runs 5  # 答えを入れ替えた版の数だけ変えて試す（速く終わる）

やること:
  1. ml_comparison.candidates（確率の基準違い）と、25日移動平均・最初に買って放置を、
     同じ資金・積立・コストで選ぶ期間の初日から通しで動かす
  2. 選ぶ期間の年率が一番良い機械学習の戦略を選び、確かめる期間でも比べる
  3. 選んだ戦略と同じ設定で「答えをランダムに入れ替えて学ばせた版」を shuffle_runs 通り動かし、
     本物と差があるかを見る。本物が入れ替え版の全部より良くなければ
     「パターンは見つからなかった」と判定する
  4. 予想そのものの当たり具合（AUC）も、本物と入れ替え版で比べる

※実際の注文は一切出しません。
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from compare import BENCH, build_table  # noqa: E402
from sim.config import ConfigError, load_config  # noqa: E402
from sim.data_loader import DataError, load_all, summarize  # noqa: E402
from sim.evaluate import period_metrics  # noqa: E402
from sim.game import dataset, strategy_label as label, write_game  # noqa: E402
from sim.simulator import Costs, run_simulation  # noqa: E402
from sim.strategies import StrategyError, create_strategy  # noqa: E402

SMA = "移動平均25日"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="機械学習の戦略を期間を分けて比べる（注文は出しません）")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--shuffle-runs", type=int, help="答えを入れ替えた版の数（設定より優先）")
    args = p.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if cfg.comparison is None or cfg.ml_comparison is None:
            raise ConfigError("config.yaml に comparison と ml_comparison の項目が要ります")
        ml_cands = {label(c.name, c.params): (c.name, dict(c.params))
                    for c in cfg.ml_comparison.candidates}
        strategies = {n: create_strategy(*v) for n, v in ml_cands.items()}
    except (ConfigError, StrategyError) as e:
        print(f"[設定エラー] {e}", file=sys.stderr)
        return 1
    cmp = cfg.comparison
    runs = args.shuffle_runs or cfg.ml_comparison.shuffle_runs

    warmup = max(s.warmup_days() for s in strategies.values())
    data_start = min(cfg.period.start,
                     cmp.select_period.start - timedelta(days=int(warmup * 1.5) + 30))
    print("=== データ取得と点検 ===")
    try:
        loaded = load_all([s.code for s in cfg.symbols], data_start, cmp.check_period.end,
                          cfg.paths.data_dir)
    except DataError as e:
        print(f"[データエラー] {e}", file=sys.stderr)
        return 1
    for pr in loaded.values():
        print(summarize(pr))
    prices = {s: pr.df for s, pr in loaded.items()}
    late = [s for s, df in prices.items()
            if len(df.loc[:pd.Timestamp(cmp.select_period.start)]) < warmup]
    if late:
        print(f"[注意] {', '.join(late)} は選ぶ期間の初日までに {warmup}日分のデータがなく、"
              "機械学習の戦略は最初しばらく様子見になります（学ぶデータが足りないため）")

    lots = {s.code: s.lot_size for s in cfg.symbols}
    money = (cfg.account.initial_cash, cfg.account.monthly_deposit,
             Costs(cfg.costs.fee_rate, cfg.costs.slippage_rate, cfg.costs.tax_rate))
    start = cmp.select_period.start

    def sim(strat):
        return run_simulation(prices, lots, strat, *money, start=start)

    t0 = time.time()
    print(f"\n=== 架空売買（{cmp.select_period.start}から売買。データは {data_start}〜） ===")
    results = {BENCH: sim(create_strategy("buy_and_hold")),
               SMA: sim(create_strategy("sma_cross", {"window": 25}))}
    for name, strat in strategies.items():
        print(f"  {name} …")
        results[name] = sim(strat)

    sel = {n: period_metrics(r, cmp.select_period.start, cmp.select_period.end)
           for n, r in results.items()}
    chk = {n: period_metrics(r, cmp.check_period.start, cmp.check_period.end)
           for n, r in results.items()}
    best = max(strategies, key=lambda n: sel[n].twr_annual)

    # 答えを入れ替えた版（選んだ戦略と同じ設定で、乱数の種だけ変える）
    print(f"  {best} の答えを入れ替えた版 × {runs}通り …")
    shuffled = {}
    for seed in range(1, runs + 1):
        st = create_strategy(ml_cands[best][0], {**ml_cands[best][1], "shuffle_seed": seed})
        shuffled[seed] = (st, sim(st))
    print(f"  （{time.time() - t0:.0f}秒）")

    table = build_table(sel, chk)
    s0, c0 = sel[BENCH], chk[BENCH]
    periods = (f"選ぶ期間 {s0.start.date()}〜{s0.end.date()} / "
               f"確かめる期間 {c0.start.date()}〜{c0.end.date()}")
    print(f"\n=== 成績（{periods}） ===")
    print("年率 = 入金の影響を除いた時間加重リターンの年率。"
          "売買回数のかっこ内は売りの回数（積立分の買い増しも売買1回と数える）")
    print(table.to_string())

    # 予想の当たり具合（AUC: 0.5 = 当てずっぽうと同じ、1.0 = 全部当たる）
    periods_def = {"選": (cmp.select_period.start, cmp.select_period.end),
                   "確": (cmp.check_period.start, cmp.check_period.end)}
    real_q = prediction_quality(strategies[best], prices, periods_def)
    sh_q = {seed: prediction_quality(st, prices, periods_def) for seed, (st, _) in shuffled.items()}
    sh_sel = {seed: period_metrics(r, *periods_def["選"]) for seed, (_, r) in shuffled.items()}
    sh_chk = {seed: period_metrics(r, *periods_def["確"]) for seed, (_, r) in shuffled.items()}

    print(f"\n=== 本物と、答えを入れ替えた版 {runs}通りの比較（{best}） ===")
    ctable = control_table(sel[best], chk[best], real_q, sh_sel, sh_chk, sh_q)
    print(ctable.to_string())
    print("AUC = 上がる日ほど高い確率を付けられたか（0.5 = 当てずっぽうと同じ）。"
          "上昇日の割合 = 持つと判断した日のうち、5営業日後に上がっていた割合（かっこ内は全日の割合）")

    print("\n=== 判定 ===")
    lines = verdict(best, sel, chk, real_q, sh_chk, sh_q, runs)
    print("\n".join(lines))

    out = cfg.paths.output_dir
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "compare_ml.csv", encoding="utf-8-sig")
    ctable.to_csv(out / "compare_ml_shuffle.csv", encoding="utf-8-sig")
    (out / "compare_ml.txt").write_text(
        f"{periods}\n\n{table.to_string()}\n\n{ctable.to_string()}\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8")
    # 対戦画面: 選んだ機械学習 → ほかの基準 → 25日移動平均 → 入れ替え版（年率が真ん中のもの）
    mid = sorted(sh_chk, key=lambda k: sh_chk[k].twr_annual)[len(sh_chk) // 2]
    players = {best: results[best]}
    players.update({n: results[n] for n in strategies if n != best})
    players[SMA] = results[SMA]
    players[f"{best}（答え入れ替え・{runs}通りの真ん中）"] = shuffled[mid][1]
    game = write_game(out / "game_compare_ml.html",
                      dataset(results[BENCH], players, cfg.account.initial_cash,
                              [s.code for s in cfg.symbols]))
    print(f"\n出力: {out / 'compare_ml.csv'}, {out / 'compare_ml_shuffle.csv'}, "
          f"{out / 'compare_ml.txt'}, {game}")
    return 0


# ============================================================
# 予想の当たり具合
# ============================================================

def auc(prob: np.ndarray, actual: np.ndarray) -> float:
    """上がった日に付けた確率が、上がらなかった日より高い組の割合（同じ値は半分と数える）。"""
    pos, neg = prob[actual == 1], prob[actual == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def prediction_quality(strat, prices: dict[str, pd.DataFrame], periods: dict) -> dict:
    """期間ごとの AUC・持つと判断した日の上昇日の割合・全日の上昇日の割合（全銘柄まとめて）。"""
    frames = [strat.walk_forward(df, sym) for sym, df in prices.items()]
    out = {}
    for tag, (a, b) in periods.items():
        parts = []
        for f in frames:
            f = f.loc[pd.Timestamp(a):(pd.Timestamp(b) if b else None)].dropna()
            parts.append(f)
        f = pd.concat(parts)
        p, y = f["prob"].to_numpy(), f["actual"].to_numpy()
        held = p >= strat.threshold
        out[tag] = {"auc": auc(p, y), "n": len(y), "base": float(y.mean()) if len(y) else np.nan,
                    "held_up": float(y[held].mean()) if held.any() else np.nan,
                    "held_share": float(held.mean()) if len(y) else np.nan}
    return out


def control_table(real_sel, real_chk, real_q, sh_sel, sh_chk, sh_q) -> pd.DataFrame:
    rows = {}
    seeds = list(sh_sel)

    def dist(vals, fmt):
        v = np.array(vals, dtype=float)
        return f"{fmt(np.min(v))} / {fmt(np.median(v))} / {fmt(np.max(v))}"

    pct = lambda v: f"{v:+.1%}"            # noqa: E731
    num = lambda v: f"{v:.3f}"             # noqa: E731
    share = lambda v: f"{v:.1%}"           # noqa: E731
    for tag, rm, sm in (("選", real_sel, sh_sel), ("確", real_chk, sh_chk)):
        rows[f"{tag} 年率"] = {
            "本物": pct(rm.twr_annual),
            "入れ替え版（最小 / 真ん中 / 最大）": dist([sm[s].twr_annual for s in seeds], pct),
            "本物を上回った入れ替え版": f"{sum(sm[s].twr_annual >= rm.twr_annual for s in seeds)}"
                                       f" / {len(seeds)}"}
        q = real_q[tag]
        rows[f"{tag} AUC"] = {
            "本物": num(q["auc"]),
            "入れ替え版（最小 / 真ん中 / 最大）": dist([sh_q[s][tag]["auc"] for s in seeds], num),
            "本物を上回った入れ替え版": f"{sum(sh_q[s][tag]['auc'] >= q['auc'] for s in seeds)}"
                                       f" / {len(seeds)}"}
        rows[f"{tag} 上昇日の割合"] = {
            "本物": f"{share(q['held_up'])}（全日 {share(q['base'])}）",
            "入れ替え版（最小 / 真ん中 / 最大）": dist([sh_q[s][tag]["held_up"] for s in seeds], share),
            "本物を上回った入れ替え版":
                f"{sum(sh_q[s][tag]['held_up'] >= q['held_up'] for s in seeds)} / {len(seeds)}"}
        rows[f"{tag} 持った日の割合"] = {
            "本物": share(q["held_share"]),
            "入れ替え版（最小 / 真ん中 / 最大）": dist([sh_q[s][tag]["held_share"] for s in seeds], share),
            "本物を上回った入れ替え版": "-"}
    return pd.DataFrame(rows).T.rename_axis("項目")


def verdict(best, sel, chk, real_q, sh_chk, sh_q, runs) -> list[str]:
    s, c = sel[best], chk[best]
    lines = [f"選ぶ期間で一番良かった機械学習の戦略: {best}"]
    for tag, m in (("選ぶ期間    ", sel), ("確かめる期間", chk)):
        lines.append(f"  {tag}: 年率 {m[best].twr_annual:+.1%}（放置 {m[BENCH].twr_annual:+.1%}、"
                     f"25日平均 {m[SMA].twr_annual:+.1%}）  最大DD {m[best].max_drawdown:.1%}"
                     f"（放置 {m[BENCH].max_drawdown:.1%}）")
    for rival in (BENCH, SMA):
        won_s = s.twr_annual > sel[rival].twr_annual
        won_c = c.twr_annual > chk[rival].twr_annual
        gap = (chk[rival].twr_annual - c.twr_annual) * 100
        diffs = (f"（年率の差 選ぶ期間 {(s.twr_annual - sel[rival].twr_annual) * 100:+.2f}pt、"
                 f"確かめる期間 {-gap:+.2f}pt）")
        lines.append(f"  対 {rival}{diffs}")
        if won_s and not won_c:
            lines.append(f"  → 選ぶ期間では勝ったが、確かめる期間では {gap:.1f}pt 負けた。"
                         "選ぶ期間の成績は、その期間にたまたま合っていただけの可能性が高い")
        elif won_s and won_c:
            lines.append("  → 両方の期間で勝った")
        elif won_c:
            lines.append("  → 選ぶ期間では負けたが、確かめる期間では勝った")
        else:
            lines.append("  → 両方の期間で負けた")

    # 予想はどちらの期間もウォークフォワード（その時点までのデータで学んだモデル）なので、
    # AUC は両方の期間で見る。年率は基準を選ぶのに使っていない確かめる期間で見る
    beat_ret = sum(sh_chk[k].twr_annual >= c.twr_annual for k in sh_chk)
    beat_auc = {t: sum(sh_q[k][t]["auc"] >= real_q[t]["auc"] for k in sh_q) for t in ("選", "確")}
    lines.append("")
    lines.append(f"答えを入れ替えた版 {runs}通りとの比較:")
    lines.append(f"  確かめる期間の年率 : 本物以上だった入れ替え版 {beat_ret} / {runs}")
    for t, name in (("選", "選ぶ期間のAUC    "), ("確", "確かめる期間のAUC")):
        lines.append(f"  {name}: 本物以上だった入れ替え版 {beat_auc[t]} / {runs}"
                     f"（本物 {real_q[t]['auc']:.3f}、0.5 = 当てずっぽう）")
    if beat_ret == 0 and beat_auc["選"] == 0 and beat_auc["確"] == 0:
        lines.append(f"  → 本物は、どちらの期間の予想の当たり具合でも、確かめる期間の年率でも、"
                     f"入れ替え版の全部を上回った。偶然だけでこうなる確率はおよそ 1/{runs + 1} 以下で、"
                     "値動きの中に使えるパターンがあった可能性がある"
                     "（ただし1つの市場での結果なので、フォワードテストで確かめたい）")
    else:
        lines.append("  → 【パターンは見つからなかった】本物は、答えをでたらめにして学ばせた版と"
                     "はっきりした差がなかった。この特徴とモデルでは、5営業日後に上がるかを"
                     "安定して予想できていない")
        if beat_auc["確"] == 0 and beat_auc["選"] > 0:
            lines.append("    確かめる期間だけは入れ替え版より当たっていたが、選ぶ期間では"
                         f"入れ替え版 {beat_auc['選']}通りに負けていた（AUC {real_q['選']['auc']:.3f}）。"
                         "期間によって当たったり外れたりするものは、パターンとは言えない")
        elif beat_auc["選"] == 0 and beat_auc["確"] > 0:
            lines.append("    選ぶ期間だけは入れ替え版より当たっていたが、確かめる期間では"
                         f"入れ替え版 {beat_auc['確']}通りに負けていた（AUC {real_q['確']['auc']:.3f}）")
    return lines


if __name__ == "__main__":
    sys.exit(main())
