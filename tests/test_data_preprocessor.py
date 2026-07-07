# -*- coding: utf-8 -*-
import sys
import os

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from data_preprocessor import COLUMN_NAME_HINTS, resolve_column, robust_numeric_parsing


def test_column_name_hints_content():
    assert COLUMN_NAME_HINTS["date"] == ["date", "dates", "data", "day", "dia"]
    assert COLUMN_NAME_HINTS["channel"] == [
        "channel",
        "product_group",
        "product",
        "media",
        "source",
        "campaign",
        "canal",
        "grupo",
    ]
    assert COLUMN_NAME_HINTS["investment"] == [
        "investment",
        "spend",
        "cost",
        "investimento",
        "revenue",
        "total_revenue",
        "valor",
    ]
    assert COLUMN_NAME_HINTS["trends"] == [
        "searches",
        "trends",
        "opportunities",
        "ad opportunities",
        "volume",
        "generic searches",
    ]
    assert COLUMN_NAME_HINTS["kpi"] == [
        "kpi",
        "sessions",
        "conversions",
        "revenue",
        "conversoes",
        "cliques",
        "clicks",
    ]


def test_resolve_column_exact_match():
    df = pd.DataFrame({"Date": [1], "kpi": [2]})
    assert resolve_column(df, "Date", "date", "coluna de data") == "Date"


def test_resolve_column_tolerant_match():
    df = pd.DataFrame({" Data ": [1]})
    assert resolve_column(df, "data", "date", "coluna de data") == " Data "


def test_resolve_column_hint_fallback_single_candidate(capsys):
    df = pd.DataFrame({"Valor": [1], "Outra": [2]})
    result = resolve_column(df, "investment", "investment", "coluna de investimento")
    assert result == "Valor"
    assert "AVISO" in capsys.readouterr().out


def test_resolve_column_hint_fallback_ambiguous_raises():
    df = pd.DataFrame({"canal": [1], "campaign": [2]})
    with pytest.raises(ValueError, match="canal.*campaign|campaign.*canal"):
        resolve_column(df, "channel", "channel", "coluna de canal")


def test_resolve_column_not_found_raises_with_available_columns():
    df = pd.DataFrame({"foo": [1], "bar": [2]})
    with pytest.raises(ValueError, match=r"foo.*bar|bar.*foo"):
        resolve_column(df, "zzz", "kpi", "coluna de KPI")


def test_robust_numeric_parsing_passthrough_when_already_numeric():
    s = pd.Series([1.0, 2.0, 3.0])
    result = robust_numeric_parsing(s)
    assert result.tolist() == [1.0, 2.0, 3.0]


def test_robust_numeric_parsing_br_format():
    s = pd.Series(["63.115,13", "34.816,26", "102,21", "0,00"])
    result = robust_numeric_parsing(s)
    assert result.tolist() == [63115.13, 34816.26, 102.21, 0.0]


def test_robust_numeric_parsing_us_format():
    s = pd.Series(["63,115.13", "34,816.26", "102.21", "0.00"])
    result = robust_numeric_parsing(s)
    assert result.tolist() == [63115.13, 34816.26, 102.21, 0.0]


def test_robust_numeric_parsing_currency_symbol():
    s = pd.Series(["R$ 63.115,13", "R$ 34.816,26"])
    result = robust_numeric_parsing(s)
    assert result.tolist() == [63115.13, 34816.26]


def test_robust_numeric_parsing_percent_sign():
    s = pd.Series(["12,5%", "8,3%"])
    result = robust_numeric_parsing(s)
    assert result.tolist() == [12.5, 8.3]


def test_robust_numeric_parsing_accounting_negative():
    s = pd.Series(["(1.234,56)", "500,00"])
    result = robust_numeric_parsing(s)
    assert result.tolist() == [-1234.56, 500.0]


def test_robust_numeric_parsing_null_tokens_become_nan():
    s = pd.Series(["-", "N/A", "100,50"])
    result = robust_numeric_parsing(s)
    assert result.isna().tolist() == [True, True, False]
    assert result.iloc[2] == 100.5


def test_robust_numeric_parsing_plain_integers_no_ambiguity_warning(capsys):
    s = pd.Series(["2122", "3438", "2317"])
    result = robust_numeric_parsing(s)
    assert result.tolist() == [2122, 3438, 2317]
    assert "AVISO" not in capsys.readouterr().out


def test_robust_numeric_parsing_ambiguous_defaults_to_us_with_warning(capsys):
    s = pd.Series(["1.234"])
    result = robust_numeric_parsing(s)
    assert result.tolist() == [1.234]
    assert "AVISO" in capsys.readouterr().out


def test_robust_numeric_parsing_unparseable_value_logs_warning(capsys):
    s = pd.Series(["abc", "100,50"])
    result = robust_numeric_parsing(s)
    assert result.iloc[0] != result.iloc[0]  # NaN
    assert result.iloc[1] == 100.5
    assert "AVISO" in capsys.readouterr().out
