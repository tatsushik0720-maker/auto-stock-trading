"""シミュレータのテスト。小さな偽データで、手計算と1円単位（実際は0.001円単位）で照合する。

台本どおりの合図を出す Scripted 戦略を使い、シミュレータの計算だけを確かめる。
合図は「その日の終値で判断 → 次の取引日の始値で約定」。
"""

import pandas as pd
import pytest

from sim.simulator import Costs, run_simulation, write_outputs
from sim.strategies import Decision, Signal, Strategy

B, S, H = Signal.BUY, Signal.SELL, Signal.HOLD
YEN = 0.001  # 照合の許容誤差（1円よりずっと細かい）


class Scripted(Strategy):
    """日付ごとに決められた合図を出すだけの戦略（テスト用）。"""
    name = "_scripted"

    def __init__(self, script):
        self.script = {pd.Timestamp(k): v for k, v in script.items()}

    def decide(self, ctx):
        return Decision(self.script.get(ctx.date, H))


def bars(rows):
    """[(日付, 始値, 終値), ...] から日足を作る。"""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _, _ in rows], name="Date")
    o = [x for _, x, _ in rows]
    c = [x for _, _, x in rows]
    return pd.DataFrame({"Open": o, "High": c, "Low": c, "Close": c, "Adj Close": c,
                         "Volume": 1000.0}, index=idx)


def run(rows_by_symbol, script, lot=10, cash=10_000, deposit=0, **costs):
    prices = {s: bars(r) for s, r in rows_by_symbol.items()}
    strategy = script if isinstance(script, Strategy) else Scripted(script)
    return run_simulation(prices, {s: lot for s in prices}, strategy, cash, deposit,
                          Costs(**costs))


# ============================================================
# 1. 買って売る: 手数料・値差・端数の現金
# ============================================================

def test_buy_then_sell_with_fee_and_slippage():
    """
    条件: 初期資金 10,000円、売買単位 10口、手数料 0.1%、値差 1%、税 0、積立なし

    日付        始値  終値  合図
    01-15(D1)   100   100   買い
    01-16(D2)   100   105   -      ← D1の買い注文を始値で約定
    01-17(D3)   106   110   売り
    01-18(D4)   120   118   -      ← D3の売り注文を始値で約定
    01-19(D5)   118   119   -

    D2 買い:
      約定価格   = 100 × (1 + 0.01)             = 101
      1単位の費用 = 10口 × 101 × (1 + 0.001)    = 1,011.01
      買える単位  = floor(10,000 ÷ 1,011.01)    = 9単位 → 90口
      約定代金   = 90 × 101                     = 9,090
      手数料     = 9,090 × 0.001                = 9.09
      値差の負担 = 90 × 100 × 0.01              = 90
      現金       = 10,000 − 9,090 − 9.09        = 900.91（端数は現金のまま）
      取得総額   = 9,090 + 9.09 = 9,099.09 → 平均取得価格 9,099.09 ÷ 90 = 101.101
      D2 資産    = 900.91 + 90 × 105            = 10,350.91
    D4 売り:
      約定価格   = 120 × (1 − 0.01)             = 118.8
      約定代金   = 90 × 118.8                   = 10,692
      手数料     = 10,692 × 0.001               = 10.692
      値差の負担 = 90 × 120 × 0.01              = 108
      実現損益   = 10,692 − 10.692 − 9,099.09   = 1,582.218
      現金       = 900.91 + 10,692 − 10.692     = 11,582.218
    合計: 手数料 9.09 + 10.692 = 19.782、値差の負担 90 + 108 = 198
    """
    r = run({"A": [("2024-01-15", 100, 100), ("2024-01-16", 100, 105),
                   ("2024-01-17", 106, 110), ("2024-01-18", 120, 118),
                   ("2024-01-19", 118, 119)]},
            {"2024-01-15": B, "2024-01-17": S}, fee_rate=0.001, slippage_rate=0.01)

    buy, sell = r.trades
    assert (buy.side, buy.date, buy.shares) == ("買い", pd.Timestamp("2024-01-16"), 90)
    assert buy.exec_price == pytest.approx(101, abs=YEN)
    assert buy.amount == pytest.approx(9_090, abs=YEN)
    assert buy.fee == pytest.approx(9.09, abs=YEN)
    assert buy.slippage_cost == pytest.approx(90, abs=YEN)
    assert buy.avg_cost == pytest.approx(101.101, abs=YEN)
    assert buy.cash_after == pytest.approx(900.91, abs=YEN)

    assert (sell.side, sell.date, sell.shares) == ("売り", pd.Timestamp("2024-01-18"), 90)
    assert sell.exec_price == pytest.approx(118.8, abs=YEN)
    assert sell.amount == pytest.approx(10_692, abs=YEN)
    assert sell.fee == pytest.approx(10.692, abs=YEN)
    assert sell.slippage_cost == pytest.approx(108, abs=YEN)
    assert sell.realized_pnl == pytest.approx(1_582.218, abs=YEN)
    assert sell.cash_after == pytest.approx(11_582.218, abs=YEN)

    eq = r.portfolio["equity"]
    assert eq["2024-01-15"] == pytest.approx(10_000, abs=YEN)
    assert eq["2024-01-16"] == pytest.approx(10_350.91, abs=YEN)
    assert eq["2024-01-17"] == pytest.approx(900.91 + 90 * 110, abs=YEN)
    assert eq["2024-01-19"] == pytest.approx(11_582.218, abs=YEN)
    assert r.total_fee == pytest.approx(19.782, abs=YEN)
    assert r.total_slippage == pytest.approx(198, abs=YEN)
    assert r.total_tax == 0 and r.missed_buys == []


# ============================================================
# 2. 積立・買い増し・平均取得価格（総平均法）
# ============================================================

def test_monthly_deposit_and_average_cost():
    """
    条件: 初期資金 1,000円、積立 毎月500円、売買単位 1口、手数料・値差 0、税 20%

    日付        始値  終値  合図
    01-30(D1)   100   100   買い
    01-31(D2)   100   100   買い   ← 100円で 10口（現金 1,000 → 0）。取得総額 1,000
    02-01(D3)    50    50   売り   ← 2月最初の取引日: 積立 +500 → 現金 500
                                     D2の買い（状態が買いなので積立分を買い増し）:
                                     50円で 10口。保有 20口、取得総額 1,500
                                     平均取得価格（総平均法）= 1,500 ÷ 20 = 75
    02-02(D4)    90    90   -      ← 90円で 20口売り: 代金 1,800
                                     実現損益 = 1,800 − 75 × 20 = 300
                                     税 = floor(300 × 0.2) = 60
                                     現金 = 0 + 1,800 − 60 = 1,740
    投入総額 = 1,000 + 500 = 1,500、最終資産 1,740
    """
    r = run({"A": [("2024-01-30", 100, 100), ("2024-01-31", 100, 100),
                   ("2024-02-01", 50, 50), ("2024-02-02", 90, 90)]},
            {"2024-01-30": B, "2024-01-31": B, "2024-02-01": S},
            lot=1, cash=1_000, deposit=500, tax_rate=0.2)

    b1, b2, sell = r.trades
    assert (b1.shares, b1.avg_cost, b1.cash_after) == (10, 100, 0)
    assert (b2.date, b2.shares) == (pd.Timestamp("2024-02-01"), 10)
    assert b2.avg_cost == pytest.approx(75, abs=YEN)
    assert sell.realized_pnl == pytest.approx(300, abs=YEN)
    assert sell.tax == 60
    assert sell.cash_after == pytest.approx(1_740, abs=YEN)
    assert r.total_deposited == pytest.approx(1_500, abs=YEN)
    assert r.final_equity == pytest.approx(1_740, abs=YEN)
    assert r.tax_by_year == {2024: 60}


def test_deposit_starts_next_month_on_first_trading_day():
    """積立は開始月の翌月から、その月の最初の取引日（1日が休みなら次の取引日）に入る。"""
    r = run({"A": [("2024-01-15", 100, 100), ("2024-01-31", 100, 100),
                   ("2024-02-05", 100, 100), ("2024-02-20", 100, 100),
                   ("2024-03-01", 100, 100)]},
            {}, cash=1_000, deposit=300)
    d = r.daily["deposited"]
    assert list(d) == [1_000, 1_000, 1_300, 1_300, 1_600]


# ============================================================
# 3. 税: 同じ年の損と益を相殺、年が変われば別計算
# ============================================================

def test_tax_nets_losses_and_gains_within_a_year():
    """
    条件: 初期資金 1,000円、売買単位 1口、手数料・値差 0、税 20%

    日付          始値  合図   約定（前日の合図）       実現損益  その年の通算  税(通算×20%)  精算
    2023-12-25    100   買い
    2023-12-26    100   売り   買い 10口 @100（1,000）
    2023-12-27     80   買い   売り 10口 @80 → 800       −200       −200          0           0
    2023-12-28     80   売り   買い 10口 @80（800）
    2023-12-29    110   買い   売り 10口 @110 → 1,100    +300       +100         20         +20
    2024-01-04    100   売り   買い 10口 @100（1,000）
    2024-01-05    120   -      売り 10口 @120 → 1,200    +200       +200（新年）  40         +40

    相殺しない場合の2023年の税は 300 × 20% = 60円だが、通算すると 20円になる。
    現金: 1,100 − 20 = 1,080 → 1,000円で買い 80 → 80 + 1,200 − 40 = 1,240
    """
    rows = [("2023-12-25", 100, 100), ("2023-12-26", 100, 100), ("2023-12-27", 80, 80),
            ("2023-12-28", 80, 80), ("2023-12-29", 110, 110), ("2024-01-04", 100, 100),
            ("2024-01-05", 120, 120)]
    script = {"2023-12-25": B, "2023-12-26": S, "2023-12-27": B, "2023-12-28": S,
              "2023-12-29": B, "2024-01-04": S}
    r = run({"A": rows}, script, lot=1, cash=1_000, tax_rate=0.2)

    sells = [t for t in r.trades if t.side == "売り"]
    assert [t.realized_pnl for t in sells] == pytest.approx([-200, 300, 200], abs=YEN)
    assert [t.tax for t in sells] == [0, 20, 40]
    assert r.tax_by_year == {2023: 20, 2024: 40}
    assert r.final_equity == pytest.approx(1_240, abs=YEN)


def test_tax_is_refunded_when_a_later_loss_offsets_an_earlier_gain():
    """
    条件: 初期資金 1,000円、売買単位 1口、手数料・値差 0、税 20%
    D2 買い 10口@100 → D3 売り@150: 損益 +500、通算 +500、税 100（支払い）現金 1,400
    D4 買い 10口@140 → D5 売り@100: 損益 −400、通算 +100、税 20 → 差額 80 を還付
    現金 = 0 + 1,000 + 80 = 1,080。年間の税は 20円（年末にまとめて計算した場合と同じ）
    """
    rows = [("2024-03-04", 100, 100), ("2024-03-05", 100, 100), ("2024-03-06", 150, 150),
            ("2024-03-07", 140, 140), ("2024-03-08", 100, 100)]
    script = {"2024-03-04": B, "2024-03-05": S, "2024-03-06": B, "2024-03-07": S}
    r = run({"A": rows}, script, lot=1, cash=1_000, tax_rate=0.2)
    sells = [t for t in r.trades if t.side == "売り"]
    assert [t.tax for t in sells] == [100, -80]
    assert r.tax_by_year == {2024: 20}
    assert r.final_equity == pytest.approx(1_080, abs=YEN)


def test_tax_is_netted_across_symbols():
    """銘柄Aの損(−200)と銘柄Bの益(+300)を同じ口座として相殺: 税 = 100 × 20% = 20。"""
    a = [("2024-03-04", 100, 100), ("2024-03-05", 100, 100), ("2024-03-06", 80, 80)]
    b = [("2024-03-04", 100, 100), ("2024-03-05", 100, 100), ("2024-03-06", 130, 130)]
    script = {"2024-03-04": B, "2024-03-05": S}
    r = run({"A": a, "B": b}, script, lot=1, cash=2_000, tax_rate=0.2)
    assert r.tax_by_year == {2024: 20}


def test_tax_rate_zero_does_nothing():
    r = run({"A": [("2024-03-04", 100, 100), ("2024-03-05", 100, 100),
                   ("2024-03-06", 150, 150)]},
            {"2024-03-04": B, "2024-03-05": S}, lot=1, cash=1_000, tax_rate=0.0)
    assert r.tax_by_year == {} and r.total_tax == 0
    assert r.final_equity == pytest.approx(1_500, abs=YEN)


# ============================================================
# 4. 取引のない日・買えない日
# ============================================================

def test_order_executes_on_next_day_with_a_row_for_that_symbol():
    """
    銘柄Bは 03-05 の行がない（取引なし）。
    03-04 の買いの合図 → A は 03-05 の始値 101、B は次に取引がある 03-06 の始値 120 で約定。
    合計資産: 03-05 は B の行がないので、B は直前(03-04)の記録で評価する（価格は埋めない）。
    """
    a = [("2024-03-04", 100, 100), ("2024-03-05", 101, 102), ("2024-03-06", 103, 104)]
    b = [("2024-03-04", 100, 100), ("2024-03-06", 120, 125)]
    r = run({"A": a, "B": b}, {"2024-03-04": B}, lot=1, cash=2_000)
    ta = next(t for t in r.trades if t.symbol == "A")
    tb = next(t for t in r.trades if t.symbol == "B")
    assert (ta.date, ta.open_price, ta.shares) == (pd.Timestamp("2024-03-05"), 101, 9)
    assert (tb.date, tb.open_price, tb.shares) == (pd.Timestamp("2024-03-06"), 120, 8)
    assert pd.Timestamp("2024-03-05") not in r.daily[r.daily["symbol"] == "B"].index
    # 03-05 合計 = A(1000−909 + 9×102) + B(03-04 の 1,000)
    assert r.portfolio["equity"]["2024-03-05"] == pytest.approx(91 + 918 + 1_000, abs=YEN)


def test_missed_buy_when_one_lot_is_too_expensive():
    """
    1単位 = 100口 × 100円 = 10,000円 > 現金 5,000円 → 買えない日として記録。
    積立 毎月3,000円で 03-01 に現金 8,000円 → まだ買えない。04-01 に 11,000円 → 買える。
    """
    rows = [("2024-02-27", 100, 100), ("2024-02-28", 100, 100), ("2024-03-01", 100, 100),
            ("2024-04-01", 100, 100), ("2024-04-02", 100, 100)]
    script = {d: B for d, _, _ in rows}
    r = run({"A": rows}, script, lot=100, cash=5_000, deposit=3_000)
    assert [m.date for m in r.missed_buys] == [pd.Timestamp(d) for d in
                                               ("2024-02-28", "2024-03-01")]
    assert [m.cash for m in r.missed_buys] == [5_000, 8_000]
    assert r.missed_buys[0].lot_cost == pytest.approx(10_000, abs=YEN)
    assert r.trades[0].date == pd.Timestamp("2024-04-01") and r.trades[0].shares == 100


def test_sell_signal_without_shares_does_nothing_and_last_day_order_is_not_executed():
    rows = [("2024-03-04", 100, 100), ("2024-03-05", 100, 100)]
    r = run({"A": rows}, {"2024-03-04": S, "2024-03-05": B}, lot=1, cash=1_000)
    assert r.trades == []
    assert r.final_equity == 1_000


# ============================================================
# 5. 出力
# ============================================================

def test_trades_csv_has_requested_columns(tmp_path):
    r = run({"A": [("2024-01-15", 100, 100), ("2024-01-16", 100, 105),
                   ("2024-01-17", 106, 110), ("2024-01-18", 120, 118)]},
            {"2024-01-15": B, "2024-01-17": S}, fee_rate=0.001, slippage_rate=0.01)
    paths = write_outputs(r, tmp_path)
    df = pd.read_csv(paths["trades"], encoding="utf-8-sig")
    for col in ["日付", "売買", "口数", "約定価格", "手数料", "値差の負担額", "実現損益"]:
        assert col in df.columns
    assert list(df["売買"]) == ["買い", "売り"]
    assert df["実現損益"].iloc[1] == pytest.approx(1_582.218, abs=YEN)
    assert paths["daily"].exists() and paths["portfolio"].exists()
