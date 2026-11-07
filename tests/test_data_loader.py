"""data_loader.py のテスト。偽の取得関数(FakeFetcher)を使うのでネット接続は不要。"""

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from sim.data_loader import (
    COLUMNS, JST, MAX_GLITCH_DAYS, DataError, find_missing_days, load_all, load_prices, repair_unit_errors,
)

START = date(2024, 1, 1)


def make_prices(start="2024-01-01", end="2024-03-29", base=100.0, skip=()):
    """平日だけの、少しずつ値上がりする偽の日足を作る。"""
    days = pd.bdate_range(start, end)
    days = days[~days.isin(pd.to_datetime(list(skip)))]
    close = base + np.arange(len(days)) * 0.5
    df = pd.DataFrame({"Open": close - 0.2, "High": close + 1, "Low": close - 1,
                       "Close": close, "Adj Close": close * 0.98, "Volume": 1000.0},
                      index=pd.DatetimeIndex(days, name="Date"))
    return df[COLUMNS]


class FakeFetcher:
    """yfinance の代わり。呼ばれた引数を記録し、用意したデータの一部を返す。"""

    def __init__(self, full: pd.DataFrame | None = None, fail: bool = False):
        self.full = full
        self.fail = fail
        self.calls: list[tuple] = []

    def __call__(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        if self.fail:
            raise ConnectionError("ネットにつながりません")
        df = self.full[self.full.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        return df.copy()


def at(y, m, d, hh=18):
    return datetime(y, m, d, hh, 0, tzinfo=JST)


# ---- 取得と保存（キャッシュ） ---------------------------------

def test_first_fetch_saves_cache(tmp_path):
    fetch = FakeFetcher(make_prices())
    p = load_prices("AAA", START, None, tmp_path, fetch, now=at(2024, 3, 29))
    assert len(fetch.calls) == 1
    assert (tmp_path / "AAA.csv").exists()
    assert p.df.index[0] == pd.Timestamp("2024-01-01")
    assert p.df.index[-1] == pd.Timestamp("2024-03-29")
    assert "全期間" in p.source


def test_second_run_same_evening_uses_cache_without_network(tmp_path):
    fetch = FakeFetcher(make_prices())
    load_prices("AAA", START, None, tmp_path, fetch, now=at(2024, 3, 29))
    p = load_prices("AAA", START, None, tmp_path, fetch, now=at(2024, 3, 29, 20))
    assert len(fetch.calls) == 1          # 2回目は取得していない
    assert "取得不要" in p.source


def test_next_day_fetches_only_the_difference(tmp_path):
    full = make_prices(end="2024-04-05")
    load_prices("AAA", START, None, tmp_path, FakeFetcher(full[:"2024-03-29"]),
                now=at(2024, 3, 29))
    fetch = FakeFetcher(full)
    p = load_prices("AAA", START, None, tmp_path, fetch, now=at(2024, 4, 5))
    _, called_start, _ = fetch.calls[0]
    assert called_start > date(2024, 3, 1)   # 先頭からではなく最近の分だけ取得
    assert "差分取得 5日分追加" in p.source
    pd.testing.assert_frame_equal(p.df, full, check_freq=False)


def test_full_refetch_when_past_prices_were_adjusted(tmp_path):
    """分割などで過去の価格が変わっていたら、全期間を取り直す。"""
    old = make_prices(end="2024-03-29")
    load_prices("AAA", START, None, tmp_path, FakeFetcher(old), now=at(2024, 3, 29))
    new = make_prices(end="2024-04-05")
    new[["Open", "High", "Low", "Close", "Adj Close"]] /= 2   # 1→2分割で過去も半分に
    fetch = FakeFetcher(new)
    p = load_prices("AAA", START, None, tmp_path, fetch, now=at(2024, 4, 5))
    assert len(fetch.calls) == 2
    assert fetch.calls[1][1] == START
    assert "全期間を再取得" in p.source
    assert p.df["Close"].iloc[0] == pytest.approx(50.0)


def test_fixed_end_date_covered_by_cache_needs_no_fetch(tmp_path):
    load_prices("AAA", START, None, tmp_path, FakeFetcher(make_prices()), now=at(2024, 3, 29))
    fetch = FakeFetcher(make_prices())
    p = load_prices("AAA", START, date(2024, 2, 29), tmp_path, fetch, now=at(2024, 4, 10))
    assert fetch.calls == []
    assert p.df.index[-1] == pd.Timestamp("2024-02-29")


def test_earlier_start_than_cache_fetches_full(tmp_path):
    full = make_prices(start="2023-10-02")
    load_prices("AAA", START, None, tmp_path, FakeFetcher(full), now=at(2024, 3, 29))
    fetch = FakeFetcher(full)
    p = load_prices("AAA", date(2023, 10, 1), None, tmp_path, fetch, now=at(2024, 4, 1))
    assert fetch.calls[0][1] == date(2023, 10, 1)
    assert p.df.index[0] == pd.Timestamp("2023-10-02")


def test_unfinished_today_row_is_dropped(tmp_path):
    """取引時間中(14時)に取得した今日の行は終値が未確定なので使わない。"""
    fetch = FakeFetcher(make_prices())
    p = load_prices("AAA", START, None, tmp_path, fetch, now=at(2024, 3, 29, 14))
    assert p.df.index[-1] == pd.Timestamp("2024-03-28")


# ---- ネットにつながらないとき ----------------------------------

def test_offline_uses_saved_data_with_warning(tmp_path):
    load_prices("AAA", START, None, tmp_path, FakeFetcher(make_prices()), now=at(2024, 3, 29))
    p = load_prices("AAA", START, None, tmp_path, FakeFetcher(fail=True), now=at(2024, 4, 5))
    assert len(p.df) == len(make_prices())
    assert "失敗" in p.source
    assert any("ネットから取得できなかった" in w for w in p.warnings)


def test_offline_without_saved_data_is_an_error(tmp_path):
    with pytest.raises(DataError, match="保存済みデータもありません"):
        load_prices("AAA", START, None, tmp_path, FakeFetcher(fail=True), now=at(2024, 3, 29))


def test_empty_period_is_an_error(tmp_path):
    with pytest.raises(DataError, match="データがありません"):
        load_prices("AAA", date(2024, 3, 30), date(2024, 3, 31), tmp_path,
                    FakeFetcher(make_prices()), now=at(2024, 4, 1))


# ---- データの誤りの修正 --------------------------------------
# 方針: 「前後の日と比べてほぼ 1/10・1/100 になっている日」だけを直す。
#       本物の大きな値動きは直さない。

PRICES = ["Open", "High", "Low", "Close", "Adj Close"]


def scaled(df, days, factor):
    """指定した日の価格だけ factor 倍にした偽データ。"""
    df = df.copy()
    df.loc[df.index[days], PRICES] *= factor
    return df


def test_repair_split_glitch_1_over_100():
    """1321.T で実際に見つかった誤り: 分割直後の2日だけ価格が 1/100。"""
    df = make_prices()
    bad = df.index[10:12]
    df.loc[bad, PRICES] /= 100
    df.loc[bad, "Volume"] *= 100
    fixed, notes, warnings = repair_unit_errors(df)
    pd.testing.assert_frame_equal(fixed, make_prices())
    assert len(notes) == 1 and "2日" in notes[0] and "1/100" in notes[0]
    assert warnings == []


def test_repair_split_glitch_1_over_10():
    """1306.T で実際に見つかった誤り: 2日だけ価格が 1/10。"""
    df = scaled(make_prices(), slice(30, 32), 0.1)
    fixed, notes, _ = repair_unit_errors(df)
    assert fixed["Close"].to_numpy() == pytest.approx(make_prices()["Close"].to_numpy())
    assert len(notes) == 1 and "10倍" in notes[0]


def test_repair_single_day_10x_spike():
    """逆向き（1日だけ10倍）の桁ずれも直す。"""
    df = scaled(make_prices(), slice(15, 16), 10)
    fixed, notes, _ = repair_unit_errors(df)
    assert fixed["Close"].to_numpy() == pytest.approx(make_prices()["Close"].to_numpy())
    assert "1/10" in notes[0]


def test_repair_handles_normal_moves_inside_the_glitch():
    """ずれている2日の間に普通の値動き(+3%)があっても、倍率どおりに戻す。"""
    df = make_prices()
    df.loc[df.index[21], PRICES] *= 1.03
    expected = df.copy()
    df.loc[df.index[20:22], PRICES] /= 100
    fixed, notes, _ = repair_unit_errors(df)
    assert fixed["Close"].to_numpy() == pytest.approx(expected["Close"].to_numpy())


@pytest.mark.parametrize("factor", [0.85, 0.5, 0.3, 1.8, 3.0])
def test_real_big_moves_are_not_repaired(factor):
    """本物の大きな値動き（-15%、半値、-70%、+80%、3倍）は 10 のべき乗ではないので直さない。"""
    df = scaled(make_prices(), slice(20, None), factor)  # 20日目以降ずっとその水準
    fixed, notes, _ = repair_unit_errors(df)
    assert notes == []
    pd.testing.assert_frame_equal(fixed, df)


def test_big_crash_and_rebound_is_not_repaired():
    """1日 -60% のあと翌日に戻る、のような本物の乱高下も直さない。"""
    df = scaled(make_prices(), slice(20, 21), 0.4)
    fixed, notes, _ = repair_unit_errors(df)
    assert notes == []
    pd.testing.assert_frame_equal(fixed, df)


def test_drop_to_1_over_10_that_never_returns_is_not_repaired():
    """1/10 になったまま戻らない場合は本物の変化かもしれない → 直さず警告だけ。"""
    df = scaled(make_prices(), slice(30, None), 0.1)
    fixed, notes, warnings = repair_unit_errors(df)
    assert notes == []
    pd.testing.assert_frame_equal(fixed, df)
    assert len(warnings) == 1 and "直していません" in warnings[0]


def test_glitch_on_last_day_is_not_repaired():
    """最終日だけ 1/100 → 翌日がないので確かめられない → 直さず警告。"""
    df = make_prices()
    df = scaled(df, slice(len(df) - 1, None), 0.01)
    fixed, notes, warnings = repair_unit_errors(df)
    assert notes == [] and len(warnings) == 1


def test_glitch_longer_than_limit_is_not_repaired():
    """1/10 のずれが MAX_GLITCH_DAYS を超えて続く場合は直さない。"""
    df = scaled(make_prices(), slice(10, 10 + MAX_GLITCH_DAYS + 1), 0.1)
    fixed, notes, warnings = repair_unit_errors(df)
    assert notes == [] and warnings


def test_rows_with_missing_values_are_dropped(tmp_path):
    full = make_prices()
    full.iloc[5, full.columns.get_loc("Close")] = np.nan
    p = load_prices("AAA", START, None, tmp_path, FakeFetcher(full), now=at(2024, 3, 29))
    assert len(p.df) == len(full) - 1
    assert any("1 行除きました" in w for w in p.warnings)


# ---- 欠けている日の点検 --------------------------------------

def test_missing_day_found_by_comparing_symbols():
    a = make_prices()
    b = make_prices(skip=["2024-02-14"])
    gaps = find_missing_days({"A": a, "B": b})
    assert gaps["A"] == []
    assert len(gaps["B"]) == 1 and "2024-02-14" in gaps["B"][0]


def test_holiday_missing_in_all_symbols_is_not_reported():
    a = make_prices(skip=["2024-02-12"])
    b = make_prices(skip=["2024-02-12"])
    assert find_missing_days({"A": a, "B": b}) == {"A": [], "B": []}


def test_long_gap_is_reported():
    a = make_prices(skip=pd.bdate_range("2024-02-05", "2024-02-23").strftime("%Y-%m-%d"))
    gaps = find_missing_days({"A": a})
    assert len(gaps["A"]) == 1 and "日空いています" in gaps["A"][0]


def test_load_all_reports_gaps(tmp_path):
    full_a = make_prices()
    full_b = make_prices(skip=["2024-02-14"])
    fetchers = {"A": FakeFetcher(full_a), "B": FakeFetcher(full_b)}

    def fetch(symbol, start, end):
        return fetchers[symbol](symbol, start, end)

    result = load_all(["A", "B"], START, None, tmp_path, fetch, now=at(2024, 3, 29))
    assert result["A"].warnings == []
    assert any("2024-02-14" in w for w in result["B"].warnings)


def test_missing_day_is_not_filled(tmp_path):
    """欠けている日は前日の価格で埋めず、「その日は取引なし」として行が無いままにする。"""
    fetchers = {"A": FakeFetcher(make_prices()),
                "B": FakeFetcher(make_prices(skip=["2024-02-14"]))}

    def fetch(symbol, start, end):
        return fetchers[symbol](symbol, start, end)

    result = load_all(["A", "B"], START, None, tmp_path, fetch, now=at(2024, 3, 29))
    assert pd.Timestamp("2024-02-14") in result["A"].df.index
    assert pd.Timestamp("2024-02-14") not in result["B"].df.index
    assert len(result["B"].df) == len(result["A"].df) - 1


def test_zero_volume_rows_are_treated_as_no_trading(tmp_path):
    """Yahoo が祝日に入れる「出来高0・前日と同じ値」の行は、取引なしとして除く。"""
    full = make_prices()
    holiday = pd.Timestamp("2024-01-08")
    full.loc[holiday, ["Open", "High", "Low", "Close", "Adj Close"]] = full["Close"].iloc[4]
    full.loc[holiday, "Volume"] = 0
    p = load_prices("AAA", START, None, tmp_path, FakeFetcher(full), now=at(2024, 3, 29))
    assert holiday not in p.df.index
    assert len(p.df) == len(full) - 1
    assert any("出来高0" in n and "2024-01-08" in n for n in p.notes)
