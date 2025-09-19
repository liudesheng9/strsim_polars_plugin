from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from strsim_polars_plugin._utils import LIB

if TYPE_CHECKING:
    from strsim_polars_plugin._typing import IntoExprColumn


def damerau_levenshtein(expr: IntoExprColumn, other: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr, other],
        function_name="damerau_levenshtein",
        is_elementwise=True,
    )


def normalized_damerau_levenshtein(expr: IntoExprColumn, other: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr, other],
        function_name="normalized_damerau_levenshtein",
        is_elementwise=True,
    )


def partial_damerau_levenshtein(expr: IntoExprColumn, other: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr, other],
        function_name="partial_damerau_levenshtein",
        is_elementwise=True,
    )


def partial_normalized_damerau_levenshtein(expr: IntoExprColumn, other: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr, other],
        function_name="partial_normalized_damerau_levenshtein",
        is_elementwise=True,
    )


def geometric_weighted_damerau_levenshtein(expr: IntoExprColumn, other: IntoExprColumn, weighted_geometric_ratio: float) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr, other],
        function_name="geometric_weighted_damerau_levenshtein",
        is_elementwise=True,
        kwargs={"weighted_geometric_ratio": weighted_geometric_ratio},
    )
