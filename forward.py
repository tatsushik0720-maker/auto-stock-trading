"""フォワードテストを1日分進める（毎日1回、平日の夕方に実行する）。

使い方:
    python forward.py              # 株価を取得 → 架空売買を進める → forward/ をコミットして push
    python forward.py --no-push    # コミットと push をしない（手元で試すとき）
    python forward.py --dry-run    # 何も保存しない（結果の文章だけ見る）

最後に、その日の結果を短い文章で表示する（通知に使う）。
失敗したときは「[失敗] 原因」を表示し、終了コード 1 で終わる。
※実際の注文は一切出しません。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sim.config import ConfigError, load_config  # noqa: E402
from sim.data_loader import JST, DataError, load_all  # noqa: E402
from sim.forward import ForwardError, run_forward, summary_text, write_game_page  # noqa: E402
from sim.state import StateError  # noqa: E402
from sim.strategies import StrategyError, create_strategy  # noqa: E402


class GitError(RuntimeError):
    pass


def main(argv: list[str] | None = None) -> int:
    # 予定の実行（スケジュールタスク）から読まれても文字化けしないように
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description="フォワードテストを1日分進める（注文は出しません）")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--no-push", action="store_true", help="コミットと push をしない")
    p.add_argument("--dry-run", action="store_true", help="何も保存しない")
    args = p.parse_args(argv)

    try:
        return run(args)
    except (ConfigError, DataError, StrategyError, StateError, ForwardError, GitError) as e:
        kind = {ConfigError: "設定", DataError: "株価データ", StrategyError: "戦略",
                StateError: "口座の状態ファイル", ForwardError: "フォワードテスト",
                GitError: "コミット・push"}[type(e)]
        print(f"[失敗] フォワードテスト（{kind}のエラー）: {e}")
        return 1
    except Exception as e:  # 想定外のエラーも、原因がわかる形で終える
        import traceback
        traceback.print_exc()
        print(f"[失敗] 想定外のエラー（{type(e).__name__}）: {e}")
        return 1


def run(args) -> int:
    cfg = load_config(args.config)
    fwd = cfg.forward
    if fwd is None:
        raise ConfigError("config.yaml に forward の項目がありません（README の「フォワードテスト」を参照）")
    now = datetime.now(JST)

    # 平均や学習のために、開始日より前のデータも読む。対戦画面の「過去のリプレイ」は
    # バックテストの開始日（period.start）から売買するので、その前の助走期間の分も読む
    warmup = max(create_strategy(s.name, s.params).warmup_days() for s in fwd.strategies)
    data_start = min(fwd.start, cfg.period.start) - timedelta(days=int(warmup * 1.5) + 30)
    prices = load_all([s.code for s in cfg.symbols], data_start, None, cfg.paths.data_dir)
    for pr in prices.values():
        for w in pr.warnings:
            # 開始日より前の日付だけについての注意（古いデータの欠けなど）は、通知に出さない
            days = re.findall(r"\d{4}-\d{2}-\d{2}", w)
            if days and max(days) < fwd.start.isoformat():
                continue
            print(f"[注意] {pr.symbol}: {w}")
        print(f"株価: {pr.symbol} 最新 {pr.df.index[-1].date()}（{pr.source}）")

    report = run_forward(cfg, {s: pr.df for s, pr in prices.items()}, now,
                         write=not args.dry_run)
    text = summary_text(report)

    # 対戦画面（forward/game.html）を作り直す。新しい日がない日は、ファイルがなければ作るだけ
    game = fwd.dir / "game.html"
    if not args.dry_run and (report.changed or not game.exists()):
        write_game_page(cfg, {s: pr.df for s, pr in prices.items()}, now, game)

    pushed = ""
    if report.changed and not args.dry_run and not args.no_push:
        last = report.days[-1].date()
        pushed = commit_and_push(fwd.dir, f"フォワードテスト: {last} の結果")
    print()
    print(text)
    if pushed:
        print(pushed)
    return 0


def git(*args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=120)
    except FileNotFoundError:
        raise GitError("git が見つかりません。git をインストールしてください") from None
    except subprocess.TimeoutExpired:
        raise GitError(f"git {args[0]} が2分たっても終わりませんでした（ネット接続を確認）") from None


def commit_and_push(folder: Path, message: str) -> str:
    """forward/ だけをコミットして push する（ほかの変更は混ぜない）。"""
    rel = str(Path(folder).resolve().relative_to(ROOT))
    r = git("add", "--", rel)
    if r.returncode:
        raise GitError(f"git add に失敗しました: {r.stderr.strip()}")
    if git("diff", "--cached", "--quiet", "--", rel).returncode == 0:
        ahead = git("rev-list", "--count", "@{u}..HEAD").stdout.strip()
        if ahead in ("", "0"):
            return "（forward/ に変更がないため、コミットはしませんでした）"
        # 前回 push に失敗したコミットが残っている → push だけする
    else:
        r = git("commit", "-m", message, "--", rel)
        if r.returncode:
            raise GitError(f"git commit に失敗しました: {(r.stderr or r.stdout).strip()}")
    r = git("push", "origin", "HEAD")
    if r.returncode:
        raise GitError(
            "記録は保存・コミットしましたが、push に失敗しました。次回の実行でまとめて push されます。"
            f"原因: {r.stderr.strip()}")
    return "（forward/ をコミットして GitHub に push しました）"


if __name__ == "__main__":
    sys.exit(main())
