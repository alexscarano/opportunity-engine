# -*- coding: utf-8 -*-
import sys
import os

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from data_preprocessor import COLUMN_NAME_HINTS, resolve_column


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
