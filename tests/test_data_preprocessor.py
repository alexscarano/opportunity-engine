# -*- coding: utf-8 -*-
import sys
import os

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from data_preprocessor import (
    COLUMN_NAME_HINTS,
    resolve_column,
    robust_numeric_parsing,
    read_csv_robust,
    load_and_prepare_data,
)


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


def test_read_csv_robust_default_comma(tmp_path):
    path = tmp_path / "plain.csv"
    path.write_text("Date,kpi\n2025-01-01,100\n", encoding="utf-8")
    df = read_csv_robust(str(path))
    assert df.columns.tolist() == ["Date", "kpi"]
    assert df.iloc[0]["kpi"] == 100


def test_read_csv_robust_semicolon_delimiter(tmp_path):
    path = tmp_path / "semicolon.csv"
    path.write_text(
        "dates;product_group;total_revenue\n2025-01-01;GOOGLE;1.234,56\n", encoding="utf-8"
    )
    df = read_csv_robust(str(path))
    assert df.columns.tolist() == ["dates", "product_group", "total_revenue"]
    assert df.iloc[0]["total_revenue"] == "1.234,56"


def test_read_csv_robust_strips_bom(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_text("channel,date,investment\nAWIN,01/01/2026,2122\n", encoding="utf-8-sig")
    df = read_csv_robust(str(path))
    assert df.columns.tolist() == ["channel", "date", "investment"]


def test_read_csv_robust_strips_column_whitespace(tmp_path):
    path = tmp_path / "spaced.csv"
    path.write_text(" Data , Investimento \n2025-01-01,100\n", encoding="utf-8")
    df = read_csv_robust(str(path))
    assert df.columns.tolist() == ["Data", "Investimento"]


def _write_csv(path, content):
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_load_and_prepare_data_handles_br_locale_semicolon_investment_file(tmp_path):
    """Reproduces the original bug report: BR-formatted, semicolon-delimited
    investment CSV must no longer crash pivot_table with a dtype TypeError.
    """
    investment_path = _write_csv(
        tmp_path / "investment.csv",
        "dates;product_group;total_revenue\n"
        "2025-01-01;GOOGLE;1.234,56\n"
        "2025-01-02;GOOGLE;2.345,67\n"
        "2025-01-03;GOOGLE;3.456,78\n",
    )
    performance_path = _write_csv(
        tmp_path / "performance.csv",
        "date,kpi\n2025-01-01,100\n2025-01-02,150\n2025-01-03,200\n",
    )

    config = {
        "investment_file_path": investment_path,
        "performance_file_path": performance_path,
        "performance_kpi_column": "kpi",
        "date_formats": {
            "investment_file": "%Y-%m-%d",
            "performance_file": "%Y-%m-%d",
        },
        "treat_outliers": False,
    }

    kpi_df, daily_investment_df, trends_df, correlation_matrix = load_and_prepare_data(config)

    assert daily_investment_df["investment"].tolist() == [1234.56, 2345.67, 3456.78]
    assert kpi_df["kpi"].tolist() == [100.0, 150.0, 200.0]
    assert (daily_investment_df["Product Group"] == "GOOGLE").all()


def test_load_and_prepare_data_normalizes_channel_case(tmp_path):
    investment_path = _write_csv(
        tmp_path / "investment.csv",
        "dates,product_group,total_revenue\n"
        "2025-01-01,Google Ads,100\n"
        "2025-01-02,GOOGLE ADS,200\n",
    )
    performance_path = _write_csv(
        tmp_path / "performance.csv", "date,kpi\n2025-01-01,10\n2025-01-02,20\n"
    )
    config = {
        "investment_file_path": investment_path,
        "performance_file_path": performance_path,
        "performance_kpi_column": "kpi",
        "date_formats": {
            "investment_file": "%Y-%m-%d",
            "performance_file": "%Y-%m-%d",
        },
        "treat_outliers": False,
    }

    _, daily_investment_df, _, _ = load_and_prepare_data(config)

    assert daily_investment_df["Product Group"].unique().tolist() == ["GOOGLE ADS"]


def test_load_and_prepare_data_raises_clear_error_on_empty_result(tmp_path):
    investment_path = _write_csv(
        tmp_path / "investment.csv",
        "dates,product_group,total_revenue\nnot-a-date,GOOGLE,100\n",
    )
    performance_path = _write_csv(
        tmp_path / "performance.csv", "date,kpi\n2025-01-01,10\n"
    )
    config = {
        "investment_file_path": investment_path,
        "performance_file_path": performance_path,
        "performance_kpi_column": "kpi",
        "date_formats": {
            "investment_file": "%Y-%m-%d",
            "performance_file": "%Y-%m-%d",
        },
        "treat_outliers": False,
    }

    with pytest.raises(Exception, match="arquivo de investimento"):
        load_and_prepare_data(config)


def test_load_and_prepare_data_raises_clear_error_on_empty_kpi_result(tmp_path):
    investment_path = _write_csv(
        tmp_path / "investment.csv",
        "dates,product_group,total_revenue\n2025-01-01,GOOGLE,100\n",
    )
    performance_path = _write_csv(
        tmp_path / "performance.csv", "date,kpi\nnot-a-date,10\n"
    )
    config = {
        "investment_file_path": investment_path,
        "performance_file_path": performance_path,
        "performance_kpi_column": "kpi",
        "date_formats": {
            "investment_file": "%Y-%m-%d",
            "performance_file": "%Y-%m-%d",
        },
        "treat_outliers": False,
    }

    with pytest.raises(Exception, match="arquivo de performance"):
        load_and_prepare_data(config)


def test_load_and_prepare_data_resolves_mismatched_column_name(tmp_path):
    """Investment file uses 'Valor' instead of the configured 'total_revenue' —
    should auto-resolve via COLUMN_NAME_HINTS instead of failing.
    """
    investment_path = _write_csv(
        tmp_path / "investment.csv",
        "dates,product_group,Valor\n2025-01-01,GOOGLE,100\n2025-01-02,GOOGLE,200\n",
    )
    performance_path = _write_csv(
        tmp_path / "performance.csv", "date,kpi\n2025-01-01,10\n2025-01-02,20\n"
    )
    config = {
        "investment_file_path": investment_path,
        "performance_file_path": performance_path,
        "performance_kpi_column": "kpi",
        "date_formats": {
            "investment_file": "%Y-%m-%d",
            "performance_file": "%Y-%m-%d",
        },
        "treat_outliers": False,
    }

    _, daily_investment_df, _, _ = load_and_prepare_data(config)

    assert daily_investment_df["investment"].tolist() == [100.0, 200.0]


def test_load_and_prepare_data_uppercases_other_channel(tmp_path):
    """Pins the Product Group uppercase invariant that saturation_curve.py and
    streamlit_app.py rely on to exclude the 'OTHER' channel (case-sensitive
    `!= "OTHER"` comparisons) — a lowercase/mixed-case 'other' in the source
    CSV must come out as 'OTHER', not slip through unnormalized.
    """
    investment_path = _write_csv(
        tmp_path / "investment.csv",
        "dates,product_group,total_revenue\n"
        "2025-01-01,other,100\n"
        "2025-01-02,Other,200\n",
    )
    performance_path = _write_csv(
        tmp_path / "performance.csv", "date,kpi\n2025-01-01,10\n2025-01-02,20\n"
    )
    config = {
        "investment_file_path": investment_path,
        "performance_file_path": performance_path,
        "performance_kpi_column": "kpi",
        "date_formats": {
            "investment_file": "%Y-%m-%d",
            "performance_file": "%Y-%m-%d",
        },
        "treat_outliers": False,
    }

    _, daily_investment_df, _, _ = load_and_prepare_data(config)

    assert daily_investment_df["Product Group"].tolist() == ["OTHER", "OTHER"]
