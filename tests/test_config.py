"""config.py のテスト。ネット接続は不要。"""

import copy
from datetime import date
from pathlib import Path

import pytest

from sim.config import ConfigError, apply_overrides, load_config, parse_config

ROOT = Path(__file__).resolve().parents[1]

VALID = {
    "symbols": [{"code": "1321.T", "lot_size": 1}, {"code": "1306.T", "lot_size": 10}],
    "period": {"start": "2018-01-01", "end": None},
    "account": {"initial_cash": 500000, "monthly_deposit": 10000},
    "costs": {"fee_rate": 0.0, "slippage_rate": 0.001, "tax_rate": 0.0},
    "strategy": {"name": "sma_cross", "params": {"window": 25}},
}


def make(**changes):
    """VALID をコピーして一部を変えた設定を作る。例: make(account={...})"""
    raw = copy.deepcopy(VALID)
    raw.update(changes)
    return raw


# ---- 正常系 -------------------------------------------------

def test_repository_config_yaml_is_valid():
    """リポジトリ同梱の config.yaml が読めて、指定どおりの初期値になっている。"""
    cfg = load_config(ROOT / "config.yaml")
    assert [s.code for s in cfg.symbols] == ["1321.T", "1306.T"]
    assert cfg.account.initial_cash == 500000
    assert cfg.account.monthly_deposit == 10000
    assert cfg.costs.fee_rate == 0.0
    assert cfg.costs.slippage_rate == 0.001
    assert cfg.costs.tax_rate == 0.0
    assert cfg.period.end is None
    assert cfg.strategy.name == "sma_cross"
    assert cfg.strategy.params == {"window": 25}
    assert cfg.paths.data_dir == ROOT / "data"


def test_parse_valid():
    cfg = parse_config(VALID)
    assert cfg.symbols[1].lot_size == 10
    assert cfg.period.start == date(2018, 1, 1)
    assert cfg.period.end is None


def test_end_date_can_be_fixed():
    cfg = parse_config(make(period={"start": "2018-01-01", "end": "2020-12-31"}))
    assert cfg.period.end == date(2020, 12, 31)


def test_yaml_native_date_is_accepted():
    """YAML で引用符なしの 2018-01-01 は date 型になるが、それも受け付ける。"""
    cfg = parse_config(make(period={"start": date(2018, 1, 1)}))
    assert cfg.period.start == date(2018, 1, 1)


def test_lot_size_defaults_to_1():
    cfg = parse_config(make(symbols=[{"code": "1321.T"}]))
    assert cfg.symbols[0].lot_size == 1


def test_monthly_deposit_can_be_zero():
    cfg = parse_config(make(account={"initial_cash": 500000, "monthly_deposit": 0}))
    assert cfg.account.monthly_deposit == 0


def test_override_fee_rate_does_not_modify_original():
    cfg = parse_config(VALID)
    cfg2 = apply_overrides(cfg, {"fee_rate": 0.001})
    assert cfg2.costs.fee_rate == 0.001
    assert cfg.costs.fee_rate == 0.0          # 元はそのまま
    assert cfg2.costs.slippage_rate == 0.001  # 他の値も保たれる


def test_override_tax_rate_general_account():
    cfg = apply_overrides(parse_config(VALID), {"tax_rate": 0.20315})
    assert cfg.costs.tax_rate == pytest.approx(0.20315)


# ---- 誤りを検出できるか -------------------------------------

@pytest.mark.parametrize("raw, message", [
    (make(symbols=[]), "銘柄を1つ以上"),
    (make(symbols=[{"code": "1321.T"}, {"code": "1321.T"}]), "2回"),
    (make(symbols=[{"code": "1321.T", "lot_size": 0}]), "lot_size"),
    (make(symbols=[{"code": "1321.T", "lot_size": 1.5}]), "lot_size"),
    (make(symbols=[{"lot_size": 1}]), "code"),
    (make(period={"start": "2018-13-01"}), "日付が読めません"),
    (make(period={"end": None}), "開始日"),
    (make(period={"start": "2020-01-01", "end": "2019-01-01"}), "より後"),
    (make(account={"initial_cash": 0}), "initial_cash"),
    (make(account={"initial_cash": "50万"}), "数値"),
    (make(account={"initial_cash": 500000, "monthly_deposit": -1}), "マイナス"),
    (make(costs={"fee_rate": 0.1}), "大きすぎ"),      # 0.1% のつもりで 0.1 と書いた
    (make(costs={"tax_rate": 20.315}), "割合"),       # % のまま書いた
    (make(strategy={"params": {}}), "strategy.name"),
    (make(strategy={"name": "x", "params": [1]}), "params"),
])
def test_invalid_config_is_rejected(raw, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(raw)


def test_unknown_override_is_rejected():
    with pytest.raises(ConfigError, match="上書きできない"):
        apply_overrides(parse_config(VALID), {"symbols": []})


def test_missing_file():
    with pytest.raises(ConfigError, match="見つかりません"):
        load_config("no_such_file.yaml")
