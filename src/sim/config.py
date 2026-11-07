"""config.yaml を読み込み、中身をチェックして使いやすい形にまとめるモジュール。

・設定の誤り（マイナスの資金、存在しない日付など）は、シミュレーションを
  始める前に分かりやすいメッセージで止めます。
・読み込んだ設定は「変更できないデータ」(frozen dataclass) にして、
  途中で誤って書き換わる事故を防ぎます。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """設定ファイルの内容に誤りがあるときに出すエラー。"""


@dataclass(frozen=True)
class SymbolConfig:
    code: str       # 銘柄コード（例: "1321.T"）
    lot_size: int   # 売買の最小単位（株）


@dataclass(frozen=True)
class PeriodConfig:
    start: date
    end: date | None  # None = 最新の取引日まで


@dataclass(frozen=True)
class AccountConfig:
    initial_cash: float
    monthly_deposit: float


@dataclass(frozen=True)
class CostConfig:
    fee_rate: float
    slippage_rate: float
    tax_rate: float


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PathConfig:
    data_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class ComparisonConfig:
    """戦略の比べ方（compare.py 用）。選ぶ期間で一番良い戦略を決め、確かめる期間で試す。"""
    select_period: PeriodConfig
    check_period: PeriodConfig
    candidates: tuple[StrategyConfig, ...]


@dataclass(frozen=True)
class Config:
    symbols: tuple[SymbolConfig, ...]
    period: PeriodConfig
    account: AccountConfig
    costs: CostConfig
    strategy: StrategyConfig
    paths: PathConfig
    comparison: ComparisonConfig | None = None   # 書いていなければ None


# ------------------------------------------------------------
# 公開関数
# ------------------------------------------------------------

def load_config(path: str | Path = "config.yaml",
                overrides: dict[str, Any] | None = None) -> Config:
    """YAMLファイルを読み込み、チェック済みの Config を返す。

    overrides: 一時的に上書きしたい値。例: {"fee_rate": 0.001}
               （設定ファイルを書き換えずに手数料0.1%で試すときに使う）
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = parse_config(raw, base_dir=path.parent)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def parse_config(raw: Any, base_dir: Path = Path(".")) -> Config:
    """読み込んだ辞書をチェックして Config に変換する。"""
    raw = _require_dict(raw, "設定ファイル全体")
    return Config(
        symbols=_parse_symbols(raw.get("symbols")),
        period=_parse_period(raw.get("period")),
        account=_parse_account(raw.get("account")),
        costs=_parse_costs(raw.get("costs")),
        strategy=_parse_strategy(raw.get("strategy")),
        paths=_parse_paths(raw.get("paths", {}), base_dir),
        comparison=_parse_comparison(raw.get("comparison")),
    )


# 上書きを許可する項目と、その所属セクション
_OVERRIDABLE = {
    "fee_rate": "costs",
    "slippage_rate": "costs",
    "tax_rate": "costs",
    "initial_cash": "account",
    "monthly_deposit": "account",
}


def apply_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    """一部の数値だけを差し替えた新しい Config を返す（元の cfg は変わらない）。"""
    for key, value in overrides.items():
        if key not in _OVERRIDABLE:
            raise ConfigError(
                f"上書きできない項目です: {key}（可能: {', '.join(_OVERRIDABLE)}）")
        section = _OVERRIDABLE[key]
        if section == "costs":
            v = _rate(value, key)
            cfg = replace(cfg, costs=replace(cfg.costs, **{key: v}))
        else:
            v = _non_negative(value, key)
            if key == "initial_cash" and v <= 0:
                raise ConfigError("initial_cash は 0 より大きい値にしてください")
            cfg = replace(cfg, account=replace(cfg.account, **{key: v}))
    return cfg


# ------------------------------------------------------------
# 各セクションのチェック
# ------------------------------------------------------------

def _parse_symbols(raw: Any) -> tuple[SymbolConfig, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError("symbols に銘柄を1つ以上書いてください")
    result: list[SymbolConfig] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        where = f"symbols の {i + 1} 番目"
        item = _require_dict(item, where)
        code = item.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ConfigError(f"{where}: code（銘柄コード）を文字列で書いてください")
        code = code.strip()
        if code in seen:
            raise ConfigError(f"銘柄 {code} が2回書かれています")
        seen.add(code)
        lot = item.get("lot_size", 1)
        if isinstance(lot, bool) or not isinstance(lot, int) or lot < 1:
            raise ConfigError(f"{code}: lot_size は 1 以上の整数にしてください（今: {lot!r}）")
        result.append(SymbolConfig(code=code, lot_size=lot))
    return tuple(result)


def _parse_period(raw: Any) -> PeriodConfig:
    raw = _require_dict(raw, "period")
    start = _to_date(raw.get("start"), "period.start")
    if start is None:
        raise ConfigError("period.start（開始日）を書いてください。例: \"2018-01-01\"")
    end = _to_date(raw.get("end"), "period.end")
    if end is not None and end <= start:
        raise ConfigError(f"period.end（{end}）は start（{start}）より後にしてください")
    return PeriodConfig(start=start, end=end)


def _parse_account(raw: Any) -> AccountConfig:
    raw = _require_dict(raw, "account")
    initial = _non_negative(raw.get("initial_cash"), "account.initial_cash")
    if initial <= 0:
        raise ConfigError("account.initial_cash は 0 より大きい値にしてください")
    deposit = _non_negative(raw.get("monthly_deposit", 0), "account.monthly_deposit")
    return AccountConfig(initial_cash=initial, monthly_deposit=deposit)


def _parse_costs(raw: Any) -> CostConfig:
    raw = _require_dict(raw, "costs")
    return CostConfig(
        fee_rate=_rate(raw.get("fee_rate", 0.0), "costs.fee_rate"),
        slippage_rate=_rate(raw.get("slippage_rate", 0.0), "costs.slippage_rate"),
        tax_rate=_rate(raw.get("tax_rate", 0.0), "costs.tax_rate"),
    )


def _parse_strategy(raw: Any) -> StrategyConfig:
    raw = _require_dict(raw, "strategy")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("strategy.name（戦略の名前）を書いてください。例: sma_cross")
    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise ConfigError("strategy.params は「名前: 値」の形で書いてください")
    return StrategyConfig(name=name.strip(), params=dict(params))


def _parse_comparison(raw: Any) -> ComparisonConfig | None:
    if raw is None:
        return None
    raw = _require_dict(raw, "comparison")
    select = _parse_period_named(raw.get("select_period"), "comparison.select_period")
    check = _parse_period_named(raw.get("check_period"), "comparison.check_period")
    if select.end is None:
        raise ConfigError("comparison.select_period.end（選ぶ期間の終わり）を書いてください")
    if check.start <= select.end:
        raise ConfigError(
            f"確かめる期間の開始（{check.start}）は、選ぶ期間の終わり（{select.end}）より"
            "後にしてください（重なると、選ぶときに使った値動きで確かめることになります）")
    items = raw.get("candidates")
    if not isinstance(items, list) or not items:
        raise ConfigError("comparison.candidates に比べる戦略を1つ以上書いてください")
    cands = tuple(_parse_strategy(c) for c in items)
    seen = set()
    for c in cands:
        key = (c.name, tuple(sorted(c.params.items())))
        if key in seen:
            raise ConfigError(f"comparison.candidates に同じ戦略が2回あります: {c.name} {c.params}")
        seen.add(key)
    return ComparisonConfig(select_period=select, check_period=check, candidates=cands)


def _parse_period_named(raw: Any, where: str) -> PeriodConfig:
    try:
        return _parse_period(raw)
    except ConfigError as e:
        raise ConfigError(f"{where}: {e}") from None


def _parse_paths(raw: Any, base_dir: Path) -> PathConfig:
    raw = _require_dict(raw or {}, "paths")
    # 相対パスは config.yaml のある場所を基準にする
    return PathConfig(
        data_dir=(base_dir / raw.get("data_dir", "data")),
        output_dir=(base_dir / raw.get("output_dir", "output")),
    )


# ------------------------------------------------------------
# 小さな補助関数
# ------------------------------------------------------------

def _require_dict(raw: Any, where: str) -> dict:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} の書き方が正しくありません（「名前: 値」の形が必要）")
    return raw


def _to_date(value: Any, where: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):  # YAML は 2018-01-01 を自動で日付にすることがある
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ConfigError(f"{where} の日付が読めません: {value!r}（例: \"2018-01-01\"）") from None


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} は数値で書いてください（今: {value!r}）")
    return float(value)


def _non_negative(value: Any, where: str) -> float:
    v = _number(value, where)
    if v < 0:
        raise ConfigError(f"{where} はマイナスにできません（今: {v}）")
    return v


def _rate(value: Any, where: str) -> float:
    """割合（0.001 = 0.1%）のチェック。うっかり「0.1%」のつもりで 0.1 と書くのも防ぐ。"""
    v = _non_negative(value, where)
    if v >= 1.0:
        raise ConfigError(f"{where} は 0〜1 未満の割合で書いてください（0.001 = 0.1%）。今: {v}")
    if where.endswith(("fee_rate", "slippage_rate")) and v > 0.05:
        raise ConfigError(
            f"{where} が {v * 100:.1f}% と大きすぎます。0.1% なら 0.001 と書きます")
    return v
