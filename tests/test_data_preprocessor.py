# -*- coding: utf-8 -*-
import sys
import os

import logging

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from data_preprocessor import (
    COLUMN_NAME_HINTS,
    resolve_column,
    robust_date_parsing,
    robust_numeric_parsing,
    read_csv_robust,
    load_and_prepare_data,
    guess_date_col,
    guess_channel_col,
    guess_investment_col,
    guess_trends_col,
    guess_kpi_col,
    detect_cadence,
    drop_partial_periods,
    drop_bi_export_footer_rows,
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
        "leads",
        "lead",
        "vendas",
        "sales",
        "sessoes",
        "sessões",
        "pedidos",
        "orders",
        "transacoes",
        "transações",
    ]


def test_resolve_column_exact_match():
    df = pd.DataFrame({"Date": [1], "kpi": [2]})
    assert resolve_column(df, "Date", "date", "coluna de data") == "Date"


def test_resolve_column_tolerant_match():
    df = pd.DataFrame({" Data ": [1]})
    assert resolve_column(df, "data", "date", "coluna de data") == " Data "


def test_resolve_column_hint_fallback_single_candidate(caplog):
    df = pd.DataFrame({"Valor": [1], "Outra": [2]})
    result = resolve_column(df, "investment", "investment", "coluna de investimento")
    assert result == "Valor"
    assert "AVISO" in caplog.text


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


def test_robust_numeric_parsing_real_nan_cells_become_nan():
    """A column read from a CSV with genuinely empty cells (e.g. a blank
    trailing field) contains actual missing values, not the string "-" or
    "N/A" tested above. On pandas' 'str' dtype, `.astype(str)` does NOT
    stringify these missing values to "nan" the way legacy 'object' dtype
    does — they survive as bare floats and must still be handled.
    """
    s = pd.Series(["63.115,13", None, "34.816,26"])
    result = robust_numeric_parsing(s)
    assert result.isna().tolist() == [False, True, False]
    assert result.iloc[0] == 63115.13
    assert result.iloc[2] == 34816.26


def test_robust_numeric_parsing_plain_integers_no_ambiguity_warning(caplog):
    s = pd.Series(["2122", "3438", "2317"])
    result = robust_numeric_parsing(s)
    assert result.tolist() == [2122, 3438, 2317]
    assert "AVISO" not in caplog.text


def test_robust_numeric_parsing_ambiguous_defaults_to_us_with_warning(caplog):
    s = pd.Series(["1.234"])
    result = robust_numeric_parsing(s)
    assert result.tolist() == [1.234]
    assert "AVISO" in caplog.text


def test_robust_numeric_parsing_unparseable_value_logs_warning(caplog):
    s = pd.Series(["abc", "100,50"])
    result = robust_numeric_parsing(s)
    assert result.iloc[0] != result.iloc[0]  # NaN
    assert result.iloc[1] == 100.5
    assert "AVISO" in caplog.text


def test_robust_date_parsing_dayfirst_resolves_without_scrambling():
    """Days > 12 in the column disambiguate %d/%m/%Y unambiguously -- the old
    per-row dateutil fallback would have inverted day/month on whatever rows
    happened to have day <= 12, while leaving day > 12 rows alone. Here every
    row has day > 12, so a scrambled result would show up as a wrong month.
    """
    s = pd.Series(["13/01/2025", "14/02/2025", "15/03/2025", "28/06/2026"])
    result = robust_date_parsing(s)
    assert result.tolist() == [
        pd.Timestamp("2025-01-13"),
        pd.Timestamp("2025-02-14"),
        pd.Timestamp("2025-03-15"),
        pd.Timestamp("2026-06-28"),
    ]


def test_robust_date_parsing_monthfirst_resolves_when_second_token_exceeds_12():
    """Month can never exceed 12, so a value like '01/13/2025' can only be
    valid under %m/%d/%Y (month=01, day=13) -- %d/%m/%Y would require
    month=13, which is invalid and forces that whole-column format to fail.
    This is a genuinely discriminating case for month-first, not a tautology.
    """
    s = pd.Series(["01/13/2025", "02/14/2025", "03/15/2025"])
    result = robust_date_parsing(s)
    assert result.tolist() == [
        pd.Timestamp("2025-01-13"),
        pd.Timestamp("2025-02-14"),
        pd.Timestamp("2025-03-15"),
    ]


def test_robust_date_parsing_ambiguous_column_raises_value_error():
    """Every day-of-month is <= 12, so both %d/%m/%Y and %m/%d/%Y parse 100%
    of the column but disagree on the actual dates -- must raise instead of
    silently picking one.
    """
    s = pd.Series(["01/02/2025", "03/04/2025", "05/06/2025"], name="Date")
    with pytest.raises(ValueError, match="Date"):
        robust_date_parsing(s)


def test_robust_date_parsing_iso_format_regression():
    s = pd.Series(["2025-01-01", "2025-02-15", "2025-03-20"])
    result = robust_date_parsing(s)
    assert result.tolist() == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-02-15"),
        pd.Timestamp("2025-03-20"),
    ]


def test_robust_date_parsing_strips_midnight_time_suffix():
    """Regression: exports (e.g. Excel/BI) often serialize a pure date column
    as a full datetime string with the time fixed at midnight, like
    '2025-01-01 00:00:00'. None of DATE_FORMAT_CANDIDATES include a time
    component and pandas requires an exact format match, so every candidate
    used to fail on 100% of rows -- this was the VW dataset crash.
    """
    s = pd.Series(
        ["2025-01-01 00:00:00", "2025-01-02 00:00:00", "2025-01-03 00:00:00"]
    )
    result = robust_date_parsing(s)
    assert result.tolist() == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
    ]


def test_robust_date_parsing_strips_non_midnight_time_suffix():
    """Only the date matters downstream (daily/weekly aggregation), so a
    non-midnight time component is discarded too, not just '00:00:00'.
    """
    s = pd.Series(["2025-01-01 14:35:20", "2025-01-02 09:00:00"])
    result = robust_date_parsing(s)
    assert result.tolist() == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
    ]


def test_drop_bi_export_footer_rows_strips_total_and_filtros_footer(caplog):
    """Regression: Looker Studio / Google Data Studio exports can append a
    'Total' summary row and a multi-line 'Filtros aplicados: ...' footer
    note after the real data. Neither looks like a date, so they used to
    blow up robust_date_parsing's 100%-coverage check on the whole file.
    """
    caplog.set_level(logging.INFO)
    df = pd.DataFrame(
        {
            "Date": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "Total",
                "Filtros aplicados:\nData é igual a ou está depois de "
                "01/01/2025 e está antes de 01/01/2026\n"
                "NOM_FUNNEL_STAGE é CONVERSION",
            ],
            "kpi": [10, 20, 30, 60, None],
        }
    )
    result = drop_bi_export_footer_rows(df, "Date")
    assert result["Date"].tolist() == ["2025-01-01", "2025-01-02", "2025-01-03"]
    assert "AVISO" in caplog.text


def test_drop_bi_export_footer_rows_leaves_clean_data_untouched():
    df = pd.DataFrame({"Date": ["2025-01-01", "2025-01-02", "2025-01-03"]})
    result = drop_bi_export_footer_rows(df, "Date")
    assert result["Date"].tolist() == df["Date"].tolist()


def test_drop_bi_export_footer_rows_handles_trailing_blank_line(caplog):
    """Regression: a trailing blank line in the CSV (common after a footer
    is edited/removed by hand, or just a stray final newline) becomes an
    all-NaN row. On pandas' newer string dtype, plain astype(str) leaves
    that NaN as a bare float instead of stringifying it to 'nan', so a naive
    value.strip() crashes with \"'float' object has no attribute 'strip'\".
    """
    caplog.set_level(logging.INFO)
    df = pd.DataFrame(
        {
            "Date": ["2025-01-01", "2025-01-02", "2025-01-03", None],
            "kpi": [10, 20, 30, None],
        }
    )
    result = drop_bi_export_footer_rows(df, "Date")
    assert result["Date"].tolist() == ["2025-01-01", "2025-01-02", "2025-01-03"]


def test_drop_bi_export_footer_rows_does_not_touch_mid_file_garbage():
    """Only trailing rows are trimmed -- a bad value in the middle of the
    file is real corruption and must still surface as a parsing error
    downstream, not be silently swallowed here.
    """
    df = pd.DataFrame(
        {"Date": ["2025-01-01", "Total", "2025-01-03"]}
    )
    result = drop_bi_export_footer_rows(df, "Date")
    assert result["Date"].tolist() == ["2025-01-01", "Total", "2025-01-03"]


def test_load_and_prepare_data_survives_looker_studio_footer_rows(tmp_path):
    dates = pd.date_range("2025-01-01", periods=20).astype(str)
    inv_rows = pd.DataFrame(
        {
            "dates": dates,
            "product_group": ["PMAX"] * 20,
            "total_revenue": range(20),
        }
    )
    perf_rows = pd.DataFrame({"date": dates, "kpi": range(20)})

    footer_note = (
        "Filtros aplicados:\nData é igual a ou está depois de 01/01/2025 e "
        "está antes de 01/01/2026\nNOM_FUNNEL_STAGE é CONVERSION"
    )
    inv_footer = pd.DataFrame(
        {"dates": ["Total", footer_note], "product_group": ["", ""], "total_revenue": ["", ""]}
    )
    perf_footer = pd.DataFrame({"date": ["Total", footer_note], "kpi": ["", ""]})

    inv_path = _write_csv(
        tmp_path / "investment.csv",
        pd.concat([inv_rows, inv_footer], ignore_index=True).to_csv(index=False),
    )
    perf_path = _write_csv(
        tmp_path / "performance.csv",
        pd.concat([perf_rows, perf_footer], ignore_index=True).to_csv(index=False),
    )

    config = {
        "investment_file_path": inv_path,
        "performance_file_path": perf_path,
        "performance_kpi_column": "kpi",
        "treat_outliers": False,
    }
    kpi_df, daily_investment_df, _, _ = load_and_prepare_data(config)
    assert len(kpi_df) == 20
    assert len(daily_investment_df) == 20


def test_robust_date_parsing_configured_format_not_fully_covering_falls_through(caplog):
    """Configured date_format is %Y-%m-%d but the actual data is dd/mm/yyyy --
    must not be silently accepted partially, must fall through to
    auto-detection and resolve correctly, and must print a warning.
    """
    s = pd.Series(["15/01/2025", "16/01/2025", "17/01/2025"])
    result = robust_date_parsing(s, date_format="%Y-%m-%d")
    assert result.tolist() == [
        pd.Timestamp("2025-01-15"),
        pd.Timestamp("2025-01-16"),
        pd.Timestamp("2025-01-17"),
    ]
    assert "AVISO" in caplog.text


def test_robust_date_parsing_no_candidate_covers_all_raises_with_examples():
    s = pd.Series(["not-a-date", "also-garbage", "2025-01-01"], name="Date")
    with pytest.raises(ValueError, match="Date"):
        robust_date_parsing(s)


def test_robust_date_parsing_reproduces_weekly_investment_bug_scenario():
    """Regression test for the original bug: a weekly dd/mm/yyyy investment
    column (mostly Sundays, spanning two years) was silently scrambled by
    the old per-row dateutil fallback whenever day <= 12 -- producing 0 NaT
    but a max date pushed forward into the wrong month (2026-12-04 instead
    of the true 2026-06-28). Regular 7-day spacing end-to-end proves no row
    got its day/month inverted.
    """
    dates = pd.date_range(start="2025-01-01", periods=78, freq="7D")
    s = pd.Series(dates.strftime("%d/%m/%Y"))

    result = robust_date_parsing(s)

    assert result.isna().sum() == 0
    assert result.min() == dates.min()
    assert result.max() == dates.max()
    assert result.is_monotonic_increasing
    assert (result.diff().dropna() == pd.Timedelta(days=7)).all()


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
    """An unparseable date value now fails fast inside robust_date_parsing
    itself (ValueError naming the bad value), instead of silently coercing
    to NaT and only surfacing later as a generic 'no rows left' error. The
    earlier, more specific error is strictly more useful, so this test now
    pins that message instead of the downstream empty-dataframe one.
    """
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

    with pytest.raises(Exception, match="not-a-date"):
        load_and_prepare_data(config)


def test_load_and_prepare_data_raises_clear_error_on_empty_kpi_result(tmp_path):
    """See test_load_and_prepare_data_raises_clear_error_on_empty_result above:
    the unparseable date now fails inside robust_date_parsing before the
    empty-dataframe check is ever reached."""
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

    with pytest.raises(Exception, match="not-a-date"):
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


def test_guess_kpi_col_handles_tab_delimited_file(tmp_path):
    """Reproduces a real bug: the Streamlit upload form guessed default
    column names with a plain pd.read_csv (comma-only), so a tab-delimited
    performance file collapsed to a single column whose name was the whole
    raw header line, which then failed resolve_column downstream. The
    guess_* helpers must use read_csv_robust so they detect the real
    delimiter, just like the actual data-loading path does.
    """
    path = _write_csv(tmp_path / "performance.csv", "Data\tVendas via lead\n2025-01-01\t10\n")

    assert guess_date_col(path) == "Data"
    assert guess_kpi_col(path, "Sessions") == "Vendas via lead"


def test_guess_investment_col_handles_semicolon_delimited_file(tmp_path):
    path = _write_csv(
        tmp_path / "investment.csv", "Data;Canal;Valor\n2025-01-01;Google;100\n"
    )

    assert guess_date_col(path) == "Data"
    assert guess_channel_col(path) == "Canal"
    assert guess_investment_col(path) == "Valor"


def test_guess_trends_col_handles_semicolon_delimited_file(tmp_path):
    path = _write_csv(
        tmp_path / "trends.csv", "Day;Ad Opportunities\n2025-01-01;5\n"
    )

    assert guess_date_col(path) == "Day"
    assert guess_trends_col(path) == "Ad Opportunities"


def test_guess_kpi_col_prefers_exact_user_configured_name(tmp_path):
    path = _write_csv(
        tmp_path / "performance.csv", "date,Sessions,Conversions\n2025-01-01,10,2\n"
    )

    assert guess_kpi_col(path, "Conversions") == "Conversions"


def test_guess_helpers_return_defaults_on_empty_or_missing_path():
    assert guess_date_col("") == "date"
    assert guess_channel_col("") == "product_group"
    assert guess_investment_col("") == "total_revenue"
    assert guess_trends_col("") == "Ad Opportunities"
    assert guess_kpi_col("", "Sessions") == "Sessions"


# --- New KPI hints (leads/vendas/sessões/etc.) ---


def test_guess_kpi_col_hint_matches_leads_regression(tmp_path):
    """Reproduces the real exemplo_csv/performance_pmax_semanal.csv shape
    (Data;Canal;Leads) with a user-typed KPI name ('Conversions') that
    doesn't match anything in the file. Before this task, 'leads' wasn't a
    KPI hint, so this fell through to the purely positional fallback
    (df.columns[1]), which picks the text 'Canal' column. With 'leads' now
    a hint, the hint match resolves it directly -- 'Leads' is returned
    without ever reaching the positional/numeric-aware fallback.
    """
    path = _write_csv(
        tmp_path / "performance.csv",
        "Data;Canal;Leads\n01/01/2025;Pmax;1242\n05/01/2025;Pmax;2275\n",
    )
    assert guess_kpi_col(path, "Conversions") == "Leads"


def test_guess_kpi_col_hint_matches_vendas_and_sessoes(tmp_path):
    path_vendas = _write_csv(tmp_path / "vendas.csv", "date,Vendas\n2025-01-01,10\n")
    assert guess_kpi_col(path_vendas, "Conversions") == "Vendas"

    path_sessoes = _write_csv(tmp_path / "sessoes.csv", "date,Sessões\n2025-01-01,10\n")
    assert guess_kpi_col(path_sessoes, "Conversions") == "Sessões"


# --- Numeric-aware last-resort fallback in guess_kpi_col / guess_investment_col ---


def test_guess_kpi_col_numeric_fallback_skips_text_positional_column(tmp_path):
    """Data;Canal;Metric -- 'Metric' isn't a recognized hint, so hint matching
    fails and the function must fall through to the positional guess
    (df.columns[1] == 'Canal', a repeated text value). The numeric-aware
    check must discard 'Canal' (0% numeric) and pick 'Metric' instead.
    """
    path = _write_csv(
        tmp_path / "performance.csv",
        "Data;Canal;Metric\n01/01/2025;Pmax;1242\n05/01/2025;Pmax;2275\n12/01/2025;Pmax;2423\n",
    )
    assert guess_kpi_col(path, "Conversions") == "Metric"


def test_guess_kpi_col_falls_back_to_original_guess_when_nothing_numeric(tmp_path):
    """When NO remaining candidate column looks numeric, the function must
    still return the original positional guess (not crash, not invent a new
    failure mode) -- downstream validation in load_and_prepare_data is
    responsible for catching this with a clear error.
    """
    path = _write_csv(
        tmp_path / "performance.csv",
        "Data;Canal;Regiao\n01/01/2025;Pmax;Sul\n05/01/2025;Pmax;Norte\n",
    )
    assert guess_kpi_col(path, "Conversions") == "Canal"


def test_guess_investment_col_numeric_fallback_skips_text_positional_column(tmp_path):
    path = _write_csv(
        tmp_path / "investment.csv",
        "Data;Metric;Canal\n01/01/2025;1242;Pmax\n05/01/2025;2275;Pmax\n12/01/2025;2423;Pmax\n",
    )
    assert guess_investment_col(path) == "Metric"


def test_guess_investment_col_falls_back_to_original_guess_when_nothing_numeric(tmp_path):
    path = _write_csv(
        tmp_path / "investment.csv",
        "Data;Regiao;Canal\n01/01/2025;Sul;Pmax\n05/01/2025;Norte;Pmax\n",
    )
    assert guess_investment_col(path) == "Canal"


def test_guess_kpi_col_numeric_fallback_logs_aviso_on_override(tmp_path, caplog):
    """When the numeric-aware fallback picks a different column than the
    naive positional guess would have, an AVISO must be printed -- this
    feeds the generated config with no visible user-confirmation step.
    """
    path = _write_csv(
        tmp_path / "performance.csv",
        "Data;Canal;Metric\n01/01/2025;Pmax;1242\n05/01/2025;Pmax;2275\n12/01/2025;Pmax;2423\n",
    )
    guess_kpi_col(path, "Conversions")
    assert "AVISO" in caplog.text


def test_guess_investment_col_numeric_fallback_logs_aviso_on_override(tmp_path, caplog):
    path = _write_csv(
        tmp_path / "investment.csv",
        "Data;Metric;Canal\n01/01/2025;1242;Pmax\n05/01/2025;2275;Pmax\n12/01/2025;2423;Pmax\n",
    )
    guess_investment_col(path)
    assert "AVISO" in caplog.text


def test_guess_kpi_col_numeric_fallback_prefers_cleanest_candidate(tmp_path):
    """With two numeric-looking candidates surviving the filter, the dirtier
    one (higher NaN ratio) appears first in column order but must lose to
    the cleaner one (lower NaN ratio) -- first-match-wins would wrongly
    pick 'ColA' here since it comes before 'ColB'.
    """
    path = _write_csv(
        tmp_path / "performance.csv",
        "Data;Canal;ColA;ColB\n"
        "01/01/2025;Pmax;abc;100\n"
        "05/01/2025;Pmax;200;200\n"
        "12/01/2025;Pmax;300;300\n",
    )
    assert guess_kpi_col(path, "Conversions") == "ColB"


def test_guess_investment_col_numeric_fallback_prefers_cleanest_candidate(tmp_path):
    path = _write_csv(
        tmp_path / "investment.csv",
        "Data;ColA;ColB;Canal\n"
        "01/01/2025;abc;100;Pmax\n"
        "05/01/2025;200;200;Pmax\n"
        "12/01/2025;300;300;Pmax\n",
    )
    assert guess_investment_col(path) == "ColB"


# --- detect_cadence ---


def test_detect_cadence_daily_series_returns_1():
    dates = pd.Series(pd.date_range("2025-01-01", periods=30, freq="1D"))
    assert detect_cadence(dates) == 1


def test_detect_cadence_weekly_series_returns_7():
    dates = pd.Series(pd.date_range("2025-01-01", periods=30, freq="7D"))
    assert detect_cadence(dates) == 7


def test_detect_cadence_monthly_series_snaps_to_30():
    dates = pd.Series(pd.date_range("2025-01-01", periods=12, freq="MS"))
    assert detect_cadence(dates) == 30


def test_detect_cadence_non_canonical_returns_raw_median():
    # Fortnightly (14-day) spacing doesn't fall in any known bucket.
    dates = pd.Series(pd.date_range("2025-01-01", periods=10, freq="14D"))
    assert detect_cadence(dates) == 14


def test_detect_cadence_5_day_median_stays_raw_not_snapped_to_7():
    # Just below the 6-8 weekly snapping window -- must NOT snap to 7.
    dates = pd.Series(pd.date_range("2025-01-01", periods=2, freq="5D"))
    assert detect_cadence(dates) == 5


def test_detect_cadence_9_day_median_stays_raw_not_snapped_to_7():
    # Just above the 6-8 weekly snapping window -- must NOT snap to 7.
    dates = pd.Series(pd.date_range("2025-01-01", periods=2, freq="9D"))
    assert detect_cadence(dates) == 9


def test_detect_cadence_fewer_than_2_unique_dates_returns_1():
    """Documented default: with 0 or 1 unique dates there's no diff() to
    compute a cadence from, so we assume daily (1) as a safe default -- this
    shouldn't happen in practice since load_and_prepare_data only calls
    detect_cadence after validating the dataframes are non-empty."""
    assert detect_cadence(pd.Series([], dtype="datetime64[ns]")) == 1
    assert detect_cadence(pd.Series([pd.Timestamp("2025-01-01")])) == 1
    # Duplicate dates collapse to a single unique date too.
    assert detect_cadence(pd.Series([pd.Timestamp("2025-01-01")] * 5)) == 1


# --- drop_partial_periods ---


def test_drop_partial_periods_catches_mid_series_stub_not_just_edges():
    """Regression for the real exemplo_csv weekly file: a New Year's
    re-anchor splits what should have been one normal week (28/12 -> 04/01,
    7 days) into two short gaps (28/12 -> 01/01 = 4 days, 01/01 -> 04/01 = 3
    days). Only 01/01 has BOTH neighbor-distances short, so it's the one
    genuine stub -- its neighbors (28/12 and 04/01) must be kept.
    """
    dates = pd.to_datetime(
        [
            "2025-12-14",
            "2025-12-21",
            "2025-12-28",
            "2026-01-01",  # stub: 4 days after prev, 3 days before next
            "2026-01-04",
            "2026-01-11",
        ]
    )
    df = pd.DataFrame({"Date": dates, "value": range(len(dates))})

    result = drop_partial_periods(df, "Date", cadence=7)

    assert pd.Timestamp("2026-01-01") not in result["Date"].tolist()
    assert pd.Timestamp("2025-12-28") in result["Date"].tolist()
    assert pd.Timestamp("2026-01-04") in result["Date"].tolist()
    assert len(result) == len(dates) - 1


def test_drop_partial_periods_catches_stub_at_start_of_series():
    """The real exemplo_csv file also has a partial first week (2025-01-01
    is only 4 days before the next date, 2025-01-05) -- the very first row
    has no 'previous' neighbor, so the check falls back to the single
    distance it does have (to the next date)."""
    dates = pd.to_datetime(
        ["2025-01-01", "2025-01-05", "2025-01-12", "2025-01-19"]
    )
    df = pd.DataFrame({"Date": dates})

    result = drop_partial_periods(df, "Date", cadence=7)

    assert pd.Timestamp("2025-01-01") not in result["Date"].tolist()
    assert len(result) == 3


def test_drop_partial_periods_catches_stub_at_end_of_series():
    """A trailing partial period (no 'next' date to compare against) must
    still be caught by falling back to the distance from the previous date."""
    dates = pd.to_datetime(
        ["2025-01-01", "2025-01-08", "2025-01-15", "2025-01-17"]
    )  # last gap is only 2 days, well under 0.6*7=4.2
    df = pd.DataFrame({"Date": dates})

    result = drop_partial_periods(df, "Date", cadence=7)

    assert pd.Timestamp("2025-01-17") not in result["Date"].tolist()
    assert len(result) == 3


def test_drop_partial_periods_clean_series_drops_nothing():
    dates = pd.to_datetime(pd.date_range("2025-01-01", periods=10, freq="7D"))
    df = pd.DataFrame({"Date": dates, "value": range(10)})

    result = drop_partial_periods(df, "Date", cadence=7)

    assert len(result) == 10
    assert result["Date"].tolist() == dates.tolist()


def test_drop_partial_periods_logs_dropped_dates(caplog):
    caplog.set_level(logging.INFO)
    dates = pd.to_datetime(["2025-01-01", "2025-01-08", "2025-01-15", "2025-01-17"])
    df = pd.DataFrame({"Date": dates})

    drop_partial_periods(df, "Date", cadence=7)

    out = caplog.text
    assert "17/01/2025" in out
    assert "1 linha(s)" in out


# --- load_and_prepare_data cadence integration ---


def test_load_and_prepare_data_raises_on_divergent_cadences(tmp_path):
    """Investment file reports daily, performance file reports weekly --
    must raise instead of silently trying to reconcile them."""
    investment_path = _write_csv(
        tmp_path / "investment.csv",
        "dates,product_group,total_revenue\n"
        + "".join(
            f"2025-01-{d:02d},GOOGLE,{100 + d}\n" for d in range(1, 15)
        ),
    )
    performance_path = _write_csv(
        tmp_path / "performance.csv",
        "date,kpi\n2025-01-01,10\n2025-01-08,20\n2025-01-15,30\n2025-01-22,40\n",
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

    with pytest.raises(Exception, match="[Cc]adência"):
        load_and_prepare_data(config)


def test_load_and_prepare_data_populates_config_period_days_for_weekly_data(tmp_path):
    dates = pd.date_range("2025-01-01", periods=8, freq="7D").strftime("%Y-%m-%d")
    investment_path = _write_csv(
        tmp_path / "investment.csv",
        "dates,product_group,total_revenue\n"
        + "".join(f"{d},GOOGLE,{100 + i}\n" for i, d in enumerate(dates)),
    )
    performance_path = _write_csv(
        tmp_path / "performance.csv",
        "date,kpi\n" + "".join(f"{d},{10 + i}\n" for i, d in enumerate(dates)),
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

    kpi_df, daily_investment_df, _, _ = load_and_prepare_data(config)

    assert config["period_days"] == 7
    assert len(kpi_df) == 8
    assert len(daily_investment_df) == 8


# --- Post-load NaN validation (>50%) ---


def test_load_and_prepare_data_kpi_over_50pct_nan_raises_specific_error(tmp_path):
    """The wrong KPI column resolves to mostly text -- must raise a specific
    error naming the column, the file, and example values instead of the
    generic 'Nenhuma linha válida restou'.
    """
    investment_path = _write_csv(
        tmp_path / "investment.csv",
        "dates,product_group,total_revenue\n"
        "2025-01-01,GOOGLE,100\n2025-01-02,GOOGLE,200\n2025-01-03,GOOGLE,300\n",
    )
    performance_path = _write_csv(
        tmp_path / "performance.csv",
        "date,kpi\n2025-01-01,abc\n2025-01-02,xyz\n2025-01-03,10\n",
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

    with pytest.raises(Exception) as exc_info:
        load_and_prepare_data(config)

    message = str(exc_info.value)
    assert "kpi" in message
    assert performance_path in message
    assert "abc" in message and "xyz" in message
    assert "Nenhuma linha válida restou" not in message


def test_load_and_prepare_data_investment_over_50pct_nan_raises_specific_error(tmp_path):
    investment_path = _write_csv(
        tmp_path / "investment.csv",
        "dates,product_group,total_revenue\n"
        "2025-01-01,GOOGLE,abc\n2025-01-02,GOOGLE,xyz\n2025-01-03,GOOGLE,300\n",
    )
    performance_path = _write_csv(
        tmp_path / "performance.csv",
        "date,kpi\n2025-01-01,10\n2025-01-02,20\n2025-01-03,30\n",
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

    with pytest.raises(Exception) as exc_info:
        load_and_prepare_data(config)

    message = str(exc_info.value)
    assert "total_revenue" in message
    assert investment_path in message
    assert "abc" in message and "xyz" in message
    assert "Nenhuma linha válida restou" not in message


# --- Exception chaining (raise ... from e) ---


def test_load_and_prepare_data_file_not_found_preserves_original_cause(tmp_path):
    config = {
        "investment_file_path": str(tmp_path / "missing_investment.csv"),
        "performance_file_path": str(tmp_path / "missing_performance.csv"),
        "performance_kpi_column": "kpi",
        "date_formats": {},
        "treat_outliers": False,
    }

    with pytest.raises(FileNotFoundError) as exc_info:
        load_and_prepare_data(config)

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_load_and_prepare_data_unexpected_error_preserves_original_cause(tmp_path):
    """Uses the same >50%-NaN scenario as above, but asserts specifically on
    exception chaining: the wrapper Exception's __cause__ must be the
    original ValueError, not swallowed.
    """
    investment_path = _write_csv(
        tmp_path / "investment.csv",
        "dates,product_group,total_revenue\n"
        "2025-01-01,GOOGLE,abc\n2025-01-02,GOOGLE,xyz\n2025-01-03,GOOGLE,qux\n",
    )
    performance_path = _write_csv(
        tmp_path / "performance.csv",
        "date,kpi\n2025-01-01,10\n2025-01-02,20\n2025-01-03,30\n",
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

    with pytest.raises(Exception) as exc_info:
        load_and_prepare_data(config)

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, ValueError)


# --- Performance-file duplicate-date aggregation (channel breakdown) ---


def test_load_and_prepare_data_dedups_duplicate_performance_dates_with_warning(tmp_path, caplog):
    """Simulates a per-channel breakdown in the performance file (two rows
    for the same date). Must print an AVISO and sum the KPI values per date
    explicitly, instead of leaving the later merge with investment_pivot to
    silently fan out rows.
    """
    investment_path = _write_csv(
        tmp_path / "investment.csv",
        "dates,product_group,total_revenue\n"
        "2025-01-01,GOOGLE,100\n2025-01-02,GOOGLE,200\n2025-01-03,GOOGLE,300\n",
    )
    performance_path = _write_csv(
        tmp_path / "performance.csv",
        "date,kpi\n"
        "2025-01-01,10\n2025-01-01,5\n"
        "2025-01-02,20\n"
        "2025-01-03,30\n",
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

    kpi_df, daily_investment_df, _, _ = load_and_prepare_data(config)

    assert "AVISO" in caplog.text
    assert len(kpi_df) == 3
    row = kpi_df[kpi_df["Date"] == pd.Timestamp("2025-01-01")]
    assert row["kpi"].iloc[0] == 15.0
    assert len(daily_investment_df) == 3


def test_load_and_prepare_data_no_duplicate_dates_is_noop(tmp_path, caplog):
    investment_path = _write_csv(
        tmp_path / "investment.csv",
        "dates,product_group,total_revenue\n"
        "2025-01-01,GOOGLE,100\n2025-01-02,GOOGLE,200\n2025-01-03,GOOGLE,300\n",
    )
    performance_path = _write_csv(
        tmp_path / "performance.csv",
        "date,kpi\n2025-01-01,10\n2025-01-02,20\n2025-01-03,30\n",
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

    kpi_df, _, _, _ = load_and_prepare_data(config)

    out = caplog.text
    assert "data duplicada" not in out
    assert kpi_df["kpi"].tolist() == [10.0, 20.0, 30.0]
