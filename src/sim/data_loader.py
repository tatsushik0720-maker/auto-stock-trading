"""日足データの取得・保存・点検を行うモジュール。

流れ:
  1. data/ に保存済みのデータ（キャッシュ）を読む
  2. 足りない分だけ yfinance で取得し、保存し直す（再取得を減らす）
  3. ネットにつながらないときは、保存済みデータだけで続ける
  4. Yahoo のデータの誤り（分割直後に価格が 1/10・1/100 になる等）を直す
  5. 最初と最後の日付・行数・欠けている日を点検する

用語:
  - 日足（ひあし）: 1日ごとの 始値(Open)・高値(High)・安値(Low)・終値(Close)・出来高(Volume)
  - 株式分割: 1株を複数株に分けること。価格は分割した分だけ下がる（損得はない）
  - Adj Close（調整後終値）: 分配金（配当）も考慮して過去の価格を調整した終値
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]

JST = timezone(timedelta(hours=9))
MARKET_CLOSE_JST = time(15, 45)   # 東証の終了(15:30)に少し余裕を持たせた時刻
OVERLAP_DAYS = 14                 # 差分取得のとき、保存済みデータと重ねて照合する日数
OVERLAP_TOLERANCE = 0.005         # 照合で「同じ値」とみなす誤差（0.5%）
MAX_NORMAL_GAP_DAYS = 12          # これより長く取引日が空いたら「欠け」とみなす（GW・年末年始は除外）

# fetcher(銘柄, 開始日, 終了日(含む) or None) -> COLUMNS を持つ DataFrame
Fetcher = Callable[[str, date, "date | None"], pd.DataFrame]

log = logging.getLogger(__name__)


class DataError(RuntimeError):
    """データが用意できないときのエラー。"""


@dataclass
class PriceData:
    """1銘柄分の日足と、その取得・点検の記録。"""
    symbol: str
    df: pd.DataFrame                       # 日付が索引、列は COLUMNS
    source: str                            # 取得元の説明（"ネット取得" / "保存済み" など）
    notes: list[str] = field(default_factory=list)     # 直したこと・知らせること
    warnings: list[str] = field(default_factory=list)  # 注意が必要なこと


# ============================================================
# 公開関数
# ============================================================

def load_prices(symbol: str, start: date, end: date | None, data_dir: Path,
                fetcher: Fetcher | None = None,
                now: datetime | None = None) -> PriceData:
    """1銘柄の日足を用意する（保存済みを優先し、足りない分だけ取得）。

    end=None なら最新の取引日まで。
    """
    fetcher = fetcher or fetch_yfinance
    now = (now or datetime.now(JST)).astimezone(JST)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    cache = _read_cache(data_dir, symbol)
    meta = _read_meta(data_dir, symbol)
    notes: list[str] = []
    warnings: list[str] = []

    if cache is not None and _cache_is_enough(cache, meta, start, end, now):
        raw, source = cache, "保存済みデータ（取得不要）"
    else:
        try:
            raw, how = _update_from_network(symbol, start, end, cache, meta, fetcher, now)
            _write_cache(data_dir, symbol, raw, start, now)
            source = f"ネット取得（{how}）"
        except Exception as e:  # ネット不通・Yahoo 側の障害など
            if cache is None or cache.empty:
                raise DataError(
                    f"{symbol}: データを取得できず、保存済みデータもありません。"
                    f"ネット接続と銘柄コードを確認してください（原因: {e}）") from e
            raw, source = cache, "保存済みデータ（ネット取得に失敗したため）"
            warnings.append(
                f"ネットから取得できなかったため、保存済みデータ（最終日 "
                f"{cache.index[-1].date()}）で続けます。原因: {e}")
            if cache.index[0].date() > start + timedelta(days=7):
                warnings.append(f"保存済みデータは {cache.index[0].date()} からで、"
                                f"設定の開始日 {start} より後です")

    df = _clip(raw, start, end)
    df, dropped = _drop_incomplete_rows(df)
    if dropped:
        warnings.append(f"値が欠けた行を {dropped} 行除きました")
    df, no_trade = _drop_no_trade_rows(df)
    if len(no_trade):
        shown = ", ".join(str(d.date()) for d in no_trade[:3])
        notes.append(f"出来高0の行（休場日などに前日の値を入れた行）を {len(no_trade)} 行除きました"
                     f"（{shown} など）。その日は取引なしとして扱います")
    df, repairs, suspicious = repair_unit_errors(df)
    notes.extend(repairs)
    warnings.extend(suspicious)
    warnings.extend(_suspicious_moves(df))
    if df.empty:
        raise DataError(f"{symbol}: 指定期間（{start}〜{end or '最新'}）のデータがありません")
    return PriceData(symbol=symbol, df=df, source=source, notes=notes, warnings=warnings)


def load_all(symbols: list[str], start: date, end: date | None, data_dir: Path,
             fetcher: Fetcher | None = None,
             now: datetime | None = None) -> dict[str, PriceData]:
    """全銘柄を読み込み、銘柄どうしを比べて欠けている日も点検する。"""
    result = {s: load_prices(s, start, end, data_dir, fetcher, now) for s in symbols}
    for symbol, gaps in find_missing_days({s: p.df for s, p in result.items()}).items():
        result[symbol].warnings.extend(gaps)
    return result


def summarize(p: PriceData) -> str:
    """点検結果を人が読める形にする。"""
    df = p.df
    lines = [
        f"■ {p.symbol}  [{p.source}]",
        f"  期間: {df.index[0].date()} 〜 {df.index[-1].date()}  行数: {len(df)}",
        f"  最新の終値: {df['Close'].iloc[-1]:,.1f}円",
    ]
    lines += [f"  [修正] {n}" for n in p.notes]
    lines += [f"  [注意] {w}" for w in p.warnings]
    if not p.notes and not p.warnings:
        lines.append("  欠けている日・異常値: なし")
    return "\n".join(lines)


# ============================================================
# データの点検・修正
# ============================================================

MAX_GLITCH_DAYS = 10  # 桁ずれとみなす連続日数の上限


def repair_unit_errors(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """前後の日と比べて、ほぼ 1/10・1/100（または10倍・100倍）にずれた日だけを直す。

    例: 1→100分割の直後2日だけ、価格が調整されず 1/100 のまま入っていることがある。

    直す条件（両方を満たすときだけ）:
      1. ある日、前日比がほぼ 10 のべき乗倍（1/10、1/100 など）に跳ぶ
      2. その後 MAX_GLITCH_DAYS 日以内に、逆向きに同じ倍率で跳んで元の水準へ戻る
         （途中の日は普通の値動き）
    本物の値動き（-15%、-50% など）は 10 のべき乗ではないので直さない。
    ずれたまま戻らない場合は、本物の変化かもしれないので直さず、警告だけ出す。
    """
    if df.empty:
        return df, [], []
    df = df.copy()
    close = df["Close"].to_numpy(dtype=float)
    n = len(close)
    jump = [1.0] + [_unit_factor(close[i] / close[i - 1]) for i in range(1, n)]

    notes: list[str] = []      # 直したこと
    warnings: list[str] = []   # 直さなかったが怪しいこと
    i = 1
    while i < n:
        f = jump[i]
        if f == 1.0:
            i += 1
            continue
        # i 日目で桁がずれた。元に戻る日（逆向きの跳び）を探す
        end = None
        for j in range(i + 1, min(i + MAX_GLITCH_DAYS, n - 1) + 1):
            if jump[j] == 1.0 / f:
                end = j - 1   # i〜end 日目がずれている
                break
            if jump[j] != 1.0:
                break         # 別の跳びが混ざる → 判断できない
        if end is None:
            warnings.append(f"{df.index[i].date()} に価格がほぼ {_ratio_text(1 / f)} に跳び、"
                         f"元に戻らないため直していません。データを確認してください")
            i += 1
            continue
        rows = df.index[i:end + 1]
        df.loc[rows, PRICE_COLUMNS] = df.loc[rows, PRICE_COLUMNS] * f
        df.loc[rows, "Volume"] = (df.loc[rows, "Volume"] / f).round()
        notes.append(
            f"{rows[0].date()}〜{rows[-1].date()}（{len(rows)}日）の価格が前後の日の約"
            f"{_ratio_text(1 / f)}になっていたため {_ratio_text(f)} に直しました"
            f"（分割の調整漏れとみられます）")
        i = end + 2  # 戻った日の次から再開
    return df, notes, warnings


def _ratio_text(f: float) -> str:
    return f"{f:g}倍" if f >= 1 else f"1/{1 / f:g}"


def find_missing_days(dfs: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """欠けている日を探す。

    ・他の銘柄にはあるのに、その銘柄だけ無い日 → 欠け
    ・取引日の間隔が長すぎる（GW・年末年始より長い）→ 欠けの疑い
    全銘柄に無い平日は、祝日など市場の休みとみなして数えない。
    """
    result: dict[str, list[str]] = {s: [] for s in dfs}
    non_empty = {s: d for s, d in dfs.items() if not d.empty}
    if len(non_empty) >= 2:
        common_start = max(d.index[0] for d in non_empty.values())
        common_end = min(d.index[-1] for d in non_empty.values())
        all_days = sorted(set().union(*(set(d.index) for d in non_empty.values())))
        all_days = [d for d in all_days if common_start <= d <= common_end]
        for s, d in non_empty.items():
            missing = [x for x in all_days if x not in d.index]
            if missing:
                shown = ", ".join(str(x.date()) for x in missing[:5])
                more = f" ほか{len(missing) - 5}日" if len(missing) > 5 else ""
                result[s].append(
                    f"他の銘柄にはある {len(missing)} 日分のデータがありません: {shown}{more}"
                    f"（前日の価格で埋めず、その日は取引なしとして扱います）")
    for s, d in non_empty.items():
        gaps = d.index.to_series().diff().dt.days
        for day, g in gaps[gaps > MAX_NORMAL_GAP_DAYS].items():
            prev = d.index[d.index.get_loc(day) - 1]
            result[s].append(f"{prev.date()} の次が {day.date()} で、{int(g)} 日空いています")
    return result


def _unit_factor(ratio: float) -> float:
    """前日比が 10 のべき乗（1/10000〜10000倍）にほぼ等しければ、戻すための倍率を返す。"""
    if not math.isfinite(ratio) or ratio <= 0:
        return 1.0
    k = round(math.log10(ratio))
    if k == 0 or abs(k) > 4:
        return 1.0
    if abs(math.log10(ratio) - k) < 0.15:  # 10^k の ±約40% 以内
        return 10.0 ** (-k)
    return 1.0


def _suspicious_moves(df: pd.DataFrame, threshold: float = 0.3) -> list[str]:
    """直した後も、1日で30%以上動いた日があれば知らせる。"""
    change = df["Close"].pct_change().abs()
    big = change[change > threshold]
    return [f"{d.date()} に終値が前日から {v:.0%} 動いています（データの誤りの可能性）"
            for d, v in big.items()]


def _drop_no_trade_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """出来高0の行を除く。Yahoo は祝日などに前日の終値で埋めた行を入れることがあるが、
    それは実際の取引ではないので「その日は取引なし」とする（前日の値で埋めない方針）。"""
    no_trade = df["Volume"] == 0
    return df[~no_trade], df.index[no_trade]


def _drop_incomplete_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    bad = df[PRICE_COLUMNS].isna().any(axis=1) | (df["Close"] <= 0)
    return df[~bad], int(bad.sum())


# ============================================================
# 取得と保存
# ============================================================

def fetch_yfinance(symbol: str, start: date, end: date | None) -> pd.DataFrame:
    """yfinance で日足を取得する（end は含む）。取得できなければ例外を出す。"""
    import yfinance as yf  # テストではネットを使わないため、ここで読み込む

    # yfinance は不通時に「possibly delisted（上場廃止？）」と誤解を招く表示をするので抑える。
    # 失敗の原因はこちらで日本語のメッセージにして知らせる。
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    yf_end = (end + timedelta(days=1)) if end else None  # yfinance の end は「含まない」
    df = yf.Ticker(symbol).history(start=start.isoformat(),
                                   end=yf_end.isoformat() if yf_end else None,
                                   interval="1d", auto_adjust=False, actions=False)
    if df is None or df.empty:
        raise DataError("Yahoo から空のデータが返りました（ネット不通か、銘柄コードの誤り）")
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index.tz_localize(None).normalize(), name="Date")
    return df[COLUMNS].astype(float)


def _update_from_network(symbol: str, start: date, end: date | None,
                         cache: pd.DataFrame | None, meta: dict, fetcher: Fetcher,
                         now: datetime) -> tuple[pd.DataFrame, str]:
    """保存済みデータを最新にする。可能なら差分だけ取得する。"""
    cache_start = _meta_start(meta, cache)
    can_append = (cache is not None and not cache.empty
                  and cache_start is not None and cache_start <= start)
    if can_append:
        overlap_from = (cache.index[-1] - timedelta(days=OVERLAP_DAYS)).date()
        new = _drop_unfinished_today(fetcher(symbol, overlap_from, end), now)
        if _overlap_matches(cache, new):
            merged = pd.concat([cache[cache.index < new.index[0]], new])
            added = len(merged) - len(cache)
            return merged, f"差分取得 {added}日分追加"
        # 分割や分配金で過去の価格が調整し直された → 保存済みの範囲も含めて全期間を取り直す
        how = "過去の価格が更新されていたため全期間を再取得"
        start = min(start, cache_start)
    else:
        how = "全期間を取得"
    full = _drop_unfinished_today(fetcher(symbol, start, end), now)
    return full, how


def _overlap_matches(cache: pd.DataFrame, new: pd.DataFrame) -> bool:
    """重なる期間で、保存済みと新しく取得した価格が一致するか。"""
    if new.empty:
        return False
    common = cache.index.intersection(new.index)
    if len(common) == 0:
        return False
    a = cache.loc[common, ["Close", "Adj Close"]]
    b = new.loc[common, ["Close", "Adj Close"]]
    return bool(((a - b).abs() / b.abs()).max().max() <= OVERLAP_TOLERANCE)


def _drop_unfinished_today(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """取引時間中に取得した「今日」の行は終値が確定していないので除く。"""
    today = pd.Timestamp(now.date())
    if now.time() < MARKET_CLOSE_JST and len(df) and df.index[-1] == today:
        return df.iloc[:-1]
    return df


def _cache_is_enough(cache: pd.DataFrame, meta: dict, start: date, end: date | None,
                     now: datetime) -> bool:
    """保存済みデータだけで足りるか（＝ネット取得を省けるか）。"""
    cache_start = _meta_start(meta, cache)
    if cache.empty or cache_start is None or cache_start > start:
        return False
    if end is not None and cache.index[-1].date() >= end:
        return True
    # 「欲しい最後の日」の終値が確定した後に取得済みなら、それ以上新しいデータはない
    # （end が休日だった場合や、end=None で今日すでに取得済みの場合）
    fetched = meta.get("fetched_at")
    if not fetched:
        return False
    fetched_at = datetime.fromisoformat(fetched).astimezone(JST)
    target = min(end, now.date()) if end else now.date()
    cutoff = datetime.combine(target, MARKET_CLOSE_JST, JST)
    if now < cutoff:  # 今日の終値はまだ確定していない → 前日の確定後ならよい
        cutoff -= timedelta(days=1)
    return fetched_at >= cutoff


def _cache_path(data_dir: Path, symbol: str) -> Path:
    return data_dir / f"{symbol}.csv"


def _meta_path(data_dir: Path, symbol: str) -> Path:
    return data_dir / f"{symbol}.meta.json"


def _read_cache(data_dir: Path, symbol: str) -> pd.DataFrame | None:
    path = _cache_path(data_dir, symbol)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
        return df[COLUMNS].astype(float).sort_index()
    except Exception as e:
        log.warning("%s の保存データが読めないため無視します: %s", path, e)
        return None


def _read_meta(data_dir: Path, symbol: str) -> dict:
    path = _meta_path(data_dir, symbol)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _meta_start(meta: dict, cache: pd.DataFrame | None) -> date | None:
    """保存データが「いつから取得したものか」（上場日より前を指定した場合も考慮）。"""
    if "requested_start" in meta:
        return date.fromisoformat(meta["requested_start"])
    if cache is not None and not cache.empty:
        return cache.index[0].date()
    return None


def _write_cache(data_dir: Path, symbol: str, df: pd.DataFrame, start: date,
                 now: datetime) -> None:
    df.to_csv(_cache_path(data_dir, symbol), index_label="Date")
    # 「いつから取得を頼んだか」を記録する。上場日より前を指定した場合も、
    # 次回から無駄に取り直さないようにするため
    old = _read_meta(data_dir, symbol)
    requested = start
    if "requested_start" in old:
        requested = min(start, date.fromisoformat(old["requested_start"]))
    _meta_path(data_dir, symbol).write_text(json.dumps({
        "requested_start": requested.isoformat(),
        "fetched_at": now.isoformat(),
        "rows": len(df),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _clip(df: pd.DataFrame, start: date, end: date | None) -> pd.DataFrame:
    df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df
