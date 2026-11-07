"""売買の少ない戦略を、期間を分けて比べる。

使い方:
    python compare.py                    # config.yaml の comparison の設定で実行
    python compare.py --fee-rate 0.001   # 手数料だけ変えて試す

やること:
  1. comparison.candidates の戦略と「最初に買って放置」を、同じ資金・積立・コストで
     通しで動かす（途中で口座をリセットしない。実際に続けて運用した場合と同じ）
  2. 成績を「選ぶ期間」と「確かめる期間」に分けて測る
  3. 選ぶ期間の年率リターンが一番良い戦略を選び、確かめる期間でも放置と比べる

平均を出すのに必要な日数分、データは選ぶ期間より前から読み込み、
選ぶ期間の初日にはどの戦略も合図を出せる状態にしておく（200日平均だけ出遅れないように）。
※実際の注文は一切出しません。
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402

from sim.config import ConfigError, load_config  # noqa: E402
from sim.data_loader import DataError, load_all, summarize  # noqa: E402
from sim.evaluate import PeriodMetrics, period_metrics  # noqa: E402
from sim.simulator import Costs, run_simulation  # noqa: E402
from sim.strategies import StrategyError, create_strategy  # noqa: E402

BENCH = "最初に買って放置"


def label(name: str, params: dict) -> str:
    """表に出す戦略の名前。"""
    w = params.get("window", 25)
    if name == "sma_cross":
        return f"移動平均{w}日"
    if name == "sma_band":
        b = params.get("band", 0.02)
        return f"移動平均{w}日 ±{b:.0%}帯"
    if name == "sma_cooldown":
        return f"移動平均{w}日 {params.get('cooldown', 20)}日あけ"
    return f"{name} {params}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="戦略を期間を分けて比べる（注文は出しません）")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--fee-rate", type=float)
    p.add_argument("--tax-rate", type=float)
    args = p.parse_args(argv)
    overrides = {k: v for k, v in (("fee_rate", args.fee_rate), ("tax_rate", args.tax_rate))
                 if v is not None}
    try:
        cfg = load_config(args.config, overrides)
    except ConfigError as e:
        print(f"[設定エラー] {e}", file=sys.stderr)
        return 1
    cmp = cfg.comparison
    if cmp is None:
        print("[設定エラー] config.yaml に comparison の項目がありません", file=sys.stderr)
        return 1

    try:
        strategies = {label(c.name, c.params): create_strategy(c.name, c.params)
                      for c in cmp.candidates}
    except StrategyError as e:
        print(f"[戦略エラー] {e}", file=sys.stderr)
        return 1

    # 平均を出すための助走期間（取引日 → 暦日はおよそ 7/5 倍。祝日の分を足す）
    warmup = max(s.warmup_days() for s in strategies.values())
    data_start = min(cfg.period.start,
                     cmp.select_period.start - timedelta(days=int(warmup * 1.5) + 30))
    data_end = cmp.check_period.end or cfg.period.end
    print("=== データ取得と点検 ===")
    try:
        prices = load_all([s.code for s in cfg.symbols], data_start, data_end,
                          cfg.paths.data_dir)
    except DataError as e:
        print(f"[データエラー] {e}", file=sys.stderr)
        return 1
    for pr in prices.values():
        print(summarize(pr))
    late = [s for s, pr in prices.items()
            if len(pr.df.loc[:pd.Timestamp(cmp.select_period.start)]) < warmup]
    if late:
        print(f"[注意] {', '.join(late)} は選ぶ期間の初日までに {warmup}日分のデータがなく、"
              "長い平均の戦略は最初しばらく合図を出せません")

    sim_args = ({s: pr.df for s, pr in prices.items()},
                {s.code: s.lot_size for s in cfg.symbols})
    money = (cfg.account.initial_cash, cfg.account.monthly_deposit,
             Costs(cfg.costs.fee_rate, cfg.costs.slippage_rate, cfg.costs.tax_rate))
    print(f"\n=== 架空売買（{len(strategies) + 1}通り、データ {data_start}〜） ===")
    results = {BENCH: run_simulation(*sim_args, create_strategy("buy_and_hold"), *money)}
    for name, strat in strategies.items():
        print(f"  {name} …")
        results[name] = run_simulation(*sim_args, strat, *money)

    sel = {n: period_metrics(r, cmp.select_period.start, cmp.select_period.end)
           for n, r in results.items()}
    chk = {n: period_metrics(r, cmp.check_period.start, cmp.check_period.end)
           for n, r in results.items()}

    table = build_table(sel, chk)
    s0, c0 = sel[BENCH], chk[BENCH]
    print(f"\n=== 成績（選ぶ期間 {s0.start.date()}〜{s0.end.date()} / "
          f"確かめる期間 {c0.start.date()}〜{c0.end.date()}） ===")
    print("年率 = 入金の影響を除いた時間加重リターンの年率。"
          "売買回数のかっこ内は売りの回数（積立分の買い増しも売買1回と数える）")
    print(table.to_string())

    print("\n=== 判定 ===")
    print("\n".join(verdict(sel, chk)))

    out = cfg.paths.output_dir
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "compare.csv", encoding="utf-8-sig")
    print(f"\n出力: {out / 'compare.csv'}")
    return 0


def build_table(sel: dict[str, PeriodMetrics], chk: dict[str, PeriodMetrics]) -> pd.DataFrame:
    """全戦略を1つの表に並べる。選ぶ期間の年率が高い順（放置は先頭）。"""
    order = [BENCH] + sorted((n for n in sel if n != BENCH),
                             key=lambda n: sel[n].twr_annual, reverse=True)
    rows = {}
    for n in order:
        row = {}
        for tag, m, b in (("選", sel[n], sel[BENCH]), ("確", chk[n], chk[BENCH])):
            row[f"{tag} 年率"] = f"{m.twr_annual:+.1%}"
            row[f"{tag} 放置との差"] = "-" if n == BENCH else \
                f"{(m.twr_annual - b.twr_annual) * 100:+.1f}pt"
            row[f"{tag} 最大DD"] = f"{m.max_drawdown:.1%}"
            row[f"{tag} 売買回数"] = f"{m.n_trades}({m.n_sells})"
            row[f"{tag} 値差負担"] = f"{m.slippage:,.0f}円"
        rows[n] = row
    return pd.DataFrame(rows).T.rename_axis("戦略")


def verdict(sel: dict[str, PeriodMetrics], chk: dict[str, PeriodMetrics]) -> list[str]:
    cands = [n for n in sel if n != BENCH]
    best = max(cands, key=lambda n: sel[n].twr_annual)
    sb, cb = sel[BENCH], chk[BENCH]
    s, c = sel[best], chk[best]
    rank_chk = sorted(cands, key=lambda n: chk[n].twr_annual, reverse=True).index(best) + 1
    lines = [
        f"選ぶ期間で一番良かった戦略: {best}",
        f"  選ぶ期間    : 年率 {s.twr_annual:+.1%}（放置 {sb.twr_annual:+.1%}）"
        f"  最大DD {s.max_drawdown:.1%}（放置 {sb.max_drawdown:.1%}）",
        f"  確かめる期間: 年率 {c.twr_annual:+.1%}（放置 {cb.twr_annual:+.1%}）"
        f"  最大DD {c.max_drawdown:.1%}（放置 {cb.max_drawdown:.1%}）",
        f"  確かめる期間の順位: {len(cands)}戦略中 {rank_chk}位",
    ]
    if s.twr_annual > sb.twr_annual and c.twr_annual < cb.twr_annual:
        lines.append(f"  → 選ぶ期間では放置に勝ったが、確かめる期間では放置に"
                     f"{(cb.twr_annual - c.twr_annual) * 100:.1f}pt 負けた。"
                     "選ぶ期間の成績は、その期間の値動きにたまたま合っていただけの可能性が高い")
    elif s.twr_annual > sb.twr_annual:
        lines.append("  → 選ぶ期間でも確かめる期間でも放置に勝った")
    elif c.twr_annual < cb.twr_annual:
        lines.append("  → 選ぶ期間でも確かめる期間でも放置に負けた"
                     "（一番良い戦略でも放置を超えられなかった）")
    else:
        lines.append("  → 選ぶ期間では放置に負けたが、確かめる期間では勝った")

    flipped = [n for n in cands if sel[n].twr_annual > sb.twr_annual
               and chk[n].twr_annual < cb.twr_annual]
    if flipped:
        lines.append("選ぶ期間では放置に勝ったのに、確かめる期間で負けた戦略: " + "、".join(flipped))
    both = [n for n in cands if sel[n].twr_annual > sb.twr_annual
            and chk[n].twr_annual > cb.twr_annual]
    lines.append("両方の期間で放置に勝った戦略: " + ("、".join(both) if both else "なし"))
    return lines


if __name__ == "__main__":
    sys.exit(main())
