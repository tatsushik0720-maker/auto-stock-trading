"""架空口座の状態と、その保存・読み込み。

将来のフォワードテスト（毎日1回動かして架空口座を続ける）のため、
口座の状態を JSON ファイルに保存し、次回そこから再開できるようにする。

保存する内容:
  - 銘柄ごと: 現金・保有口数・取得総額（→平均取得価格）・入金累計・
               最後に処理した日・その日の終値・積立済みの月・未約定の注文
  - 口座全体: その年の通算損益と支払い済みの税（年内の損益通算を続けるため）
  - 設定の控え: 銘柄と売買単位・初期資金・積立額・コスト・戦略
               （再開時に設定が変わっていたら、混ぜずに止めるため）

売買記録（trades.csv）は状態ファイルには入れない。再開した回の記録は別に出す。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .strategies.base import Signal

STATE_VERSION = 1
EPS = 1e-9


class StateError(ValueError):
    """状態ファイルが読めない、または今の設定と合わないときのエラー。"""


@dataclass
class AccountState:
    """銘柄ごとの小さな口座。"""
    symbol: str
    lot: int
    cash: float
    deposited: float
    shares: int = 0
    cost: float = 0.0                         # 保有分の取得総額（手数料込み）
    last_month: tuple[int, int] | None = None  # 積立を入れた最後の月
    pending: Signal | None = None             # 翌取引日に約定させる注文
    last_date: pd.Timestamp | None = None     # 最後に処理した日
    last_close: float | None = None           # その日の終値（評価用）

    @property
    def avg_cost(self) -> float:
        return self.cost / self.shares if self.shares else 0.0

    @property
    def equity(self) -> float:
        return self.cash + self.shares * (self.last_close or 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "lot": self.lot, "cash": self.cash,
            "deposited": self.deposited, "shares": self.shares, "cost": self.cost,
            "avg_cost": self.avg_cost,   # 読むとき用（読み込み時は cost から計算し直す）
            "last_month": list(self.last_month) if self.last_month else None,
            "pending": self.pending.name if self.pending else None,
            "last_date": self.last_date.date().isoformat() if self.last_date is not None else None,
            "last_close": self.last_close,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AccountState":
        return cls(
            symbol=d["symbol"], lot=int(d["lot"]), cash=float(d["cash"]),
            deposited=float(d["deposited"]), shares=int(d["shares"]), cost=float(d["cost"]),
            last_month=tuple(d["last_month"]) if d.get("last_month") else None,
            pending=Signal[d["pending"]] if d.get("pending") else None,
            last_date=pd.Timestamp(d["last_date"]) if d.get("last_date") else None,
            last_close=d.get("last_close"),
        )


@dataclass
class TaxBook:
    """口座全体の、その年の通算損益と払った税。"""
    rate: float
    year: int | None = None
    realized: float = 0.0
    paid: float = 0.0
    by_year: dict[int, float] = field(default_factory=dict)

    def new_day(self, day: pd.Timestamp) -> None:
        if day.year != self.year:
            self.year, self.realized, self.paid = day.year, 0.0, 0.0

    def settle(self, pnl: float) -> float:
        """売却の実現損益を通算し、精算する税額（+支払い/−還付）を返す。"""
        self.realized += pnl
        if self.rate <= 0:
            return 0.0
        due = float(math.floor(max(0.0, self.realized) * self.rate + EPS))
        delta = due - self.paid
        self.paid = due
        self.by_year[self.year] = due
        return delta

    def to_dict(self) -> dict[str, Any]:
        return {"rate": self.rate, "year": self.year, "realized": self.realized,
                "paid": self.paid, "by_year": {str(k): v for k, v in self.by_year.items()}}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaxBook":
        return cls(rate=float(d["rate"]), year=d.get("year"), realized=float(d["realized"]),
                   paid=float(d["paid"]),
                   by_year={int(k): float(v) for k, v in d.get("by_year", {}).items()})


@dataclass
class SimState:
    """架空口座全体の状態。"""
    settings: dict[str, Any]
    accounts: dict[str, AccountState]
    tax: TaxBook

    @classmethod
    def new(cls, settings: dict[str, Any]) -> "SimState":
        symbols = settings["symbols"]
        each = settings["initial_cash"] / len(symbols)
        accounts = {s: AccountState(s, int(lot), each, each) for s, lot in symbols.items()}
        return cls(settings=settings, accounts=accounts,
                   tax=TaxBook(settings["costs"]["tax_rate"]))

    @property
    def last_date(self) -> pd.Timestamp | None:
        dates = [a.last_date for a in self.accounts.values() if a.last_date is not None]
        return max(dates) if dates else None

    @property
    def equity(self) -> float:
        return sum(a.equity for a in self.accounts.values())

    @property
    def deposited(self) -> float:
        return sum(a.deposited for a in self.accounts.values())

    def check_settings(self, settings: dict[str, Any]) -> None:
        """保存時の設定と今の設定が同じか確かめる。違えば止める。"""
        diffs = [k for k in sorted(set(self.settings) | set(settings))
                 if _normalize(self.settings.get(k)) != _normalize(settings.get(k))]
        if diffs:
            detail = "; ".join(f"{k}: 保存時 {self.settings.get(k)!r} → 今 {settings.get(k)!r}"
                               for k in diffs)
            raise StateError(
                "保存された口座と今の設定が違うため再開できません（" + detail + "）。"
                "設定を戻すか、新しい状態ファイルで始めてください")

    # ---- 保存・読み込み ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "last_date": self.last_date.date().isoformat() if self.last_date is not None else None,
            "equity": self.equity,
            "deposited": self.deposited,
            "settings": self.settings,
            "accounts": {s: a.to_dict() for s, a in self.accounts.items()},
            "tax": self.tax.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SimState":
        if d.get("version") != STATE_VERSION:
            raise StateError(f"状態ファイルの版が違います（{d.get('version')}、対応: {STATE_VERSION}）")
        return cls(settings=d["settings"],
                   accounts={s: AccountState.from_dict(a) for s, a in d["accounts"].items()},
                   tax=TaxBook.from_dict(d["tax"]))

    def save(self, path: str | Path) -> None:
        """壊れにくい保存: 一時ファイルに書いてから置き換え、前回分は .bak に残す。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        if path.exists():
            os.replace(path, path.with_suffix(path.suffix + ".bak"))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "SimState":
        path = Path(path)
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            raise StateError(f"状態ファイルが見つかりません: {path}") from None
        except (ValueError, KeyError, TypeError) as e:
            raise StateError(f"状態ファイルが読めません: {path}（{e}）") from None


def make_settings(lot_sizes: dict[str, int], initial_cash: float, monthly_deposit: float,
                  costs: Any, strategy_name: str, strategy_params: dict[str, Any]
                  ) -> dict[str, Any]:
    """状態ファイルに控えておく設定。"""
    return {
        "symbols": {s: int(l) for s, l in lot_sizes.items()},
        "initial_cash": float(initial_cash),
        "monthly_deposit": float(monthly_deposit),
        "costs": {"fee_rate": costs.fee_rate, "slippage_rate": costs.slippage_rate,
                  "tax_rate": costs.tax_rate},
        "strategy": {"name": strategy_name, "params": dict(strategy_params)},
    }


def _normalize(v: Any) -> Any:
    """JSON を通すと tuple→list などに変わるので、比較の前にそろえる。"""
    return json.loads(json.dumps(v, sort_keys=True, default=str))
