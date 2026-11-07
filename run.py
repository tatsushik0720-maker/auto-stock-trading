"""架空売買シミュレータの実行入口。

使い方:
    python run.py                     # config.yaml の設定で実行
    python run.py --fee-rate 0.001    # 手数料だけ 0.1% に変えて試す（ファイルは書き換えない）
    python run.py --config 別の設定.yaml

※実際の注文は一切出しません。
※段階2の時点では「設定の表示」と「データ取得・点検」までです。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sim.config import Config, ConfigError, load_config  # noqa: E402
from sim.data_loader import DataError, load_all, summarize  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="株の架空売買シミュレータ（注文は出しません）")
    p.add_argument("--config", default="config.yaml", help="設定ファイルのパス")
    p.add_argument("--fee-rate", type=float, help="手数料率を一時的に上書き（例: 0.001 = 0.1%%）")
    p.add_argument("--tax-rate", type=float, help="税率を一時的に上書き（例: 0.20315）")
    return p.parse_args(argv)


def describe(cfg: Config) -> str:
    """設定内容を人が読める形にする。"""
    end = cfg.period.end or "最新の取引日"
    lines = [
        "=== 設定内容 ===",
        "銘柄: " + ", ".join(f"{s.code}（{s.lot_size}株単位）" for s in cfg.symbols),
        f"期間: {cfg.period.start} 〜 {end}",
        f"初期資金: {cfg.account.initial_cash:,.0f}円"
        f"（1銘柄あたり {cfg.account.initial_cash / len(cfg.symbols):,.0f}円）",
        f"毎月の積立: {cfg.account.monthly_deposit:,.0f}円",
        f"手数料率: {cfg.costs.fee_rate * 100:.3f}%  値差: {cfg.costs.slippage_rate * 100:.3f}%"
        f"  税率: {cfg.costs.tax_rate * 100:.3f}%",
        f"戦略: {cfg.strategy.name} {cfg.strategy.params}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {}
    if args.fee_rate is not None:
        overrides["fee_rate"] = args.fee_rate
    if args.tax_rate is not None:
        overrides["tax_rate"] = args.tax_rate
    try:
        cfg = load_config(args.config, overrides)
    except ConfigError as e:
        print(f"[設定エラー] {e}", file=sys.stderr)
        return 1
    print(describe(cfg))

    print("\n=== データ取得と点検 ===")
    try:
        prices = load_all([s.code for s in cfg.symbols], cfg.period.start, cfg.period.end,
                          cfg.paths.data_dir)
    except DataError as e:
        print(f"[データエラー] {e}", file=sys.stderr)
        return 1
    for p in prices.values():
        print(summarize(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
