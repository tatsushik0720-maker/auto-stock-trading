"""戦略の名前 → 戦略クラスの対応表。

strategies/ フォルダ内のファイルを自動で読み込み、Strategy を継承したクラスを
name で登録する。新しい戦略はファイルを置くだけで使えるようになる。
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

from . import base
from .base import Strategy, StrategyError

_SKIP = {"base", "registry"}


def available_strategies(package_name: str = __package__) -> dict[str, type[Strategy]]:
    """使える戦略の一覧 {名前: クラス}。package_name はテスト用に差し替えられる。"""
    package = importlib.import_module(package_name)
    found: dict[str, type[Strategy]] = {}
    for mod in pkgutil.iter_modules(package.__path__):
        if mod.name in _SKIP or mod.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package_name}.{mod.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (issubclass(cls, base.Strategy) and cls is not base.Strategy
                    and not inspect.isabstract(cls) and cls.__module__ == module.__name__):
                if cls.name in found:
                    raise StrategyError(f"戦略名 {cls.name} が2つのファイルで使われています")
                found[cls.name] = cls
    return found


def create_strategy(name: str, params: dict[str, Any] | None = None) -> Strategy:
    """名前とパラメータから戦略を作る。"""
    strategies = available_strategies()
    if name not in strategies:
        raise StrategyError(
            f"戦略 {name!r} が見つかりません。使える戦略: {', '.join(sorted(strategies))}")
    cls = strategies[name]
    try:
        return cls(**(params or {}))
    except TypeError as e:
        allowed = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
        raise StrategyError(
            f"{name} のパラメータが正しくありません（使えるもの: {', '.join(allowed)}）: {e}"
        ) from None
