"""成績の評価: 損益・最大ドローダウン・年ごとの損益・保有日数の割合・グラフ。

用語:
  - 損益: 最終資産 − 投入総額（初期資金＋積立の合計）。積立は利益ではないので除く。
  - 時間加重リターン: 入金の影響を除いた「運用の腕前」だけの成績。
      毎日の増減率 = (今日の資産 − 今日の入金) ÷ 前日の資産 − 1 を掛け合わせて作る指数。
  - ドローダウン: その時点までの最高値から何%下がっているか。
      最大ドローダウン = 期間中で一番大きな落ち込み（「最悪どれだけ減ったか」）。
      積立があると入金で資産が増えて落ち込みが小さく見えるため、時間加重の指数で測る。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .simulator import SimResult


@dataclass
class Metrics:
    final_equity: float
    deposited: float
    pnl: float
    pnl_pct: float               # 損益 ÷ 投入総額
    twr_total: float             # 時間加重リターン（期間全体）
    twr_annual: float            # 同、年率
    max_drawdown: float          # 最大ドローダウン（マイナスの割合）
    dd_peak: pd.Timestamp | None
    dd_trough: pd.Timestamp | None
    dd_recovered: pd.Timestamp | None  # 最高値を取り戻した日（未回復なら None）
    max_pnl_drop: float          # 損益（円）の最高値からの最大の落ち込み
    exposure: float              # 株を持っていた日の割合
    n_trades: int
    fee: float
    slippage: float
    tax: float


def twr_index(portfolio: pd.DataFrame) -> pd.Series:
    """入金の影響を除いた資産指数（初日 = 1）。"""
    equity = portfolio["equity"].astype(float)
    flow = portfolio["deposited"].astype(float).diff().fillna(0.0)
    daily = (equity - flow) / equity.shift(1)
    daily.iloc[0] = 1.0
    return daily.cumprod()


def drawdown(index: pd.Series) -> pd.Series:
    """各日のドローダウン（最高値からの下落率、0 以下）。"""
    return index / index.cummax() - 1.0


def exposure(daily: pd.DataFrame) -> float:
    """株を持っていた日の割合（銘柄ごとの日数を合計して計算）。"""
    if daily.empty:
        return 0.0
    return float((daily["shares"] > 0).mean())


def exposure_by_symbol(daily: pd.DataFrame) -> dict[str, float]:
    return {s: float((g["shares"] > 0).mean()) for s, g in daily.groupby("symbol")}


def compute_metrics(result: SimResult) -> Metrics:
    pf = result.portfolio
    idx = twr_index(pf)
    dd = drawdown(idx)
    trough = dd.idxmin() if len(dd) else None
    peak = idx.loc[:trough].idxmax() if trough is not None else None
    recovered = None
    if trough is not None and dd.min() < 0:
        after = idx.loc[trough:]
        back = after[after >= idx.loc[peak]]
        recovered = back.index[0] if len(back) else None
    pnl_series = pf["equity"] - pf["deposited"]
    days = (pf.index[-1] - pf.index[0]).days
    total = float(idx.iloc[-1] - 1)
    annual = float(idx.iloc[-1] ** (365.25 / days) - 1) if days > 0 else 0.0
    return Metrics(
        final_equity=result.final_equity, deposited=result.total_deposited,
        pnl=result.final_equity - result.total_deposited,
        pnl_pct=(result.final_equity - result.total_deposited) / result.total_deposited,
        twr_total=total, twr_annual=annual,
        max_drawdown=float(dd.min()), dd_peak=peak, dd_trough=trough, dd_recovered=recovered,
        max_pnl_drop=float((pnl_series - pnl_series.cummax()).min()),
        exposure=exposure(result.daily), n_trades=len(result.trades),
        fee=result.total_fee, slippage=result.total_slippage, tax=result.total_tax,
    )


def yearly_table(result: SimResult) -> pd.DataFrame:
    """年ごとの損益（円）と時間加重リターン（%）。

    その年の損益 = 年末の（資産 − 投入総額） − 前年末の（資産 − 投入総額）
    """
    pf = result.portfolio
    pnl = (pf["equity"] - pf["deposited"]).groupby(pf.index.year).last()
    idx = twr_index(pf).groupby(pf.index.year).last()
    return pd.DataFrame({
        "損益": pnl.diff().fillna(pnl.iloc[0]),
        "リターン": idx / idx.shift(1).fillna(1.0) - 1,
    }).rename_axis("年")


# ============================================================
# 表
# ============================================================

def comparison_table(results: dict[str, SimResult]) -> pd.DataFrame:
    """戦略と放置などを並べた比較表（文字列に整形済み）。"""
    rows: dict[str, dict[str, str]] = {}
    for name, r in results.items():
        m = compute_metrics(r)
        dd_when = (f"{m.dd_peak.date()}→{m.dd_trough.date()}"
                   if m.dd_peak is not None and m.max_drawdown < 0 else "-")
        rows[name] = {
            "最終資産": f"{m.final_equity:,.0f}円",
            "投入総額": f"{m.deposited:,.0f}円",
            "損益": f"{m.pnl:+,.0f}円",
            "損益率（÷投入総額）": f"{m.pnl_pct:+.1%}",
            "時間加重リターン（年率）": f"{m.twr_annual:+.1%}",
            "最大ドローダウン": f"{m.max_drawdown:.1%}",
            "  その期間": dd_when,
            "  回復した日": str(m.dd_recovered.date()) if m.dd_recovered is not None else "未回復",
            "損益の最大の落ち込み（円）": f"{m.max_pnl_drop:,.0f}円",
            "株を持っていた日の割合": f"{m.exposure:.1%}",
            "売買回数": f"{m.n_trades}回",
            "手数料の合計": f"{m.fee:,.0f}円",
            "値差の負担額の合計": f"{m.slippage:,.0f}円",
            "税の合計": f"{m.tax:,.0f}円",
        }
    return pd.DataFrame(rows)


def yearly_comparison(results: dict[str, SimResult]) -> pd.DataFrame:
    """年ごとの損益を並べ、どちらが勝ったかを付ける（最初の2つを比べる）。"""
    names = list(results)
    tables = {n: yearly_table(r) for n, r in results.items()}
    out = pd.DataFrame(index=tables[names[0]].index)
    for n in names:
        out[f"{n} 損益"] = tables[n]["損益"]
        out[f"{n} リターン"] = tables[n]["リターン"]
    if len(names) >= 2:
        a, b = names[0], names[1]
        out["勝ち"] = [a if x > y else (b if y > x else "引き分け")
                     for x, y in zip(out[f"{a} リターン"], out[f"{b} リターン"])]
    return out


def format_yearly(df: pd.DataFrame) -> str:
    shown = df.copy()
    for c in shown.columns:
        if c.endswith("損益"):
            shown[c] = shown[c].map(lambda v: f"{v:+,.0f}円")
        elif c.endswith("リターン"):
            shown[c] = shown[c].map(lambda v: f"{v:+.1%}")
    return shown.to_string()


# ============================================================
# グラフ
# ============================================================

COLORS = {"strategy": "#2a78d6", "benchmark": "#eb6834", "deposit": "#8a8984"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
JP_FONTS = ["IPAexGothic", "IPAGothic", "IPAPGothic", "Noto Sans CJK JP", "Noto Sans JP",
            "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", "Meiryo", "MS Gothic"]

LABELS = {
    True: {"title": "資産推移", "dd_title": "ドローダウン（入金の影響を除く）",
           "deposit": "投入総額", "yen": "円", "pct": "最高値からの下落"},
    False: {"title": "Equity", "dd_title": "Drawdown (deposits excluded)",
            "deposit": "Deposited", "yen": "JPY", "pct": "From peak"},
}


def _japanese_font() -> str | None:
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in JP_FONTS if f in names), None)


def plot_comparison(strategy: SimResult, benchmark: SimResult, path: Path,
                    strategy_label: str, benchmark_label: str) -> Path:
    """資産推移（戦略・放置・投入総額）とドローダウン推移の2段グラフを PNG で保存する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    font = _japanese_font()
    jp = font is not None
    lab = LABELS[jp]
    if not jp:  # 日本語フォントがなければ英語表記にする（文字化け防止）
        strategy_label, benchmark_label = "Strategy", "Buy & hold"
    rc = {"font.family": font} if jp else {}

    with plt.rc_context(rc):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]},
            facecolor=SURFACE)
        series = [(strategy, strategy_label, COLORS["strategy"]),
                  (benchmark, benchmark_label, COLORS["benchmark"])]
        for r, name, color in series:
            ax1.plot(r.portfolio.index, r.portfolio["equity"], color=color, lw=2, label=name)
        dep = strategy.portfolio["deposited"]
        ax1.plot(dep.index, dep, color=COLORS["deposit"], lw=1.6, ls=(0, (4, 3)),
                 label=lab["deposit"])
        # 右端に最終値を直接書く（凡例と色だけに頼らない）
        for r, name, color in series:
            last = r.portfolio["equity"]
            ax1.annotate(f"{last.iloc[-1] / 10_000:,.0f}万", (last.index[-1], last.iloc[-1]),
                         xytext=(6, 0), textcoords="offset points", va="center",
                         color=INK, fontsize=9)
        ax1.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: f"{v / 10_000:,.0f}万" if jp else f"{v / 1e6:,.1f}M"))
        ax1.set_title(lab["title"], loc="left", color=INK, fontsize=13)
        ax1.legend(loc="upper left", frameon=False, labelcolor=INK)

        for r, name, color in series:
            dd = drawdown(twr_index(r.portfolio)) * 100
            ax2.plot(dd.index, dd, color=color, lw=1.6, label=name)
            m = dd.idxmin()
            ax2.annotate(f"{dd.min():.1f}%", (m, dd.min()), xytext=(4, -10),
                         textcoords="offset points", color=INK, fontsize=9)
        ax2.axhline(0, color=INK_MUTED, lw=0.8)
        ax2.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax2.set_title(lab["dd_title"], loc="left", color=INK, fontsize=11)

        for ax in (ax1, ax2):
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.tick_params(colors=INK_MUTED, labelsize=9)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
        ax1.margins(x=0.01)
        fig.tight_layout()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
    return path
