# Ingestão Robusta de CSVs Diversos — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scripts/data_preprocessor.py::load_and_prepare_data` tolerate the CSV dialect variety real advertiser exports actually have (BR vs US number locale, `;`/tab delimiters, BOM/latin-1 encoding, mismatched column names) instead of crashing deep inside `pivot_table(...).mean()` with an unhelpful `TypeError: dtype 'str' does not support operation 'mean'`.

**Architecture:** Four new pure functions in `scripts/data_preprocessor.py` (`resolve_column`, `robust_numeric_parsing`, `read_csv_robust`, plus the `COLUMN_NAME_HINTS` dict they share), following the exact style `robust_date_parsing` already establishes in that file — plain function, automatic fallback, `print("   - AVISO: ...")` when the non-trivial path is taken. `load_and_prepare_data` is rewired to use them. `scripts/streamlit_app.py`'s five upload-time column-guessing helpers (`get_date_col` etc.) are pointed at the same `COLUMN_NAME_HINTS` dict instead of keeping their own hardcoded copies.

**Tech Stack:** pandas, numpy, stdlib `csv` (`csv.Sniffer`) — no new dependencies. Tests via `pytest` (already used in `tests/`, run with `mise exec -- uv run --with pytest pytest`).

**Reference:** `docs/superpowers/specs/2026-07-07-robust-csv-ingestion-design.md`

---

## Baseline

Before starting, confirm the existing suite passes:

```bash
mise exec -- uv run --with pytest pytest tests/ -q
```

Expected: `70 passed` (no failures). If this doesn't pass, stop and investigate before starting — don't build on a red baseline.

---

### Task 1: `COLUMN_NAME_HINTS` + `resolve_column`

**Files:**
- Modify: `scripts/data_preprocessor.py` (add `import csv` at top; insert new code after line 69, before line 72)
- Create: `tests/test_data_preprocessor.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data_preprocessor.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `mise exec -- uv run --with pytest pytest tests/test_data_preprocessor.py -v`
Expected: `ImportError: cannot import name 'COLUMN_NAME_HINTS'` (module doesn't have it yet)

- [ ] **Step 3: Implement `COLUMN_NAME_HINTS` and `resolve_column`**

In `scripts/data_preprocessor.py`, change the top imports (currently lines 6-7):

```python
import csv
import pandas as pd
import numpy as np
```

Then insert the following after `robust_date_parsing` ends (line 69, right before the blank lines leading into `def load_and_prepare_data(config):` on line 72):

```python
COLUMN_NAME_HINTS = {
    "date": ["date", "dates", "data", "day", "dia"],
    "channel": [
        "channel",
        "product_group",
        "product",
        "media",
        "source",
        "campaign",
        "canal",
        "grupo",
    ],
    "investment": [
        "investment",
        "spend",
        "cost",
        "investimento",
        "revenue",
        "total_revenue",
        "valor",
    ],
    "trends": [
        "searches",
        "trends",
        "opportunities",
        "ad opportunities",
        "volume",
        "generic searches",
    ],
    "kpi": ["kpi", "sessions", "conversions", "revenue", "conversoes", "cliques", "clicks"],
}


def resolve_column(df, configured_name, hint_key, description):
    """
    Resolves a configured column name against a dataframe's actual columns.
    Tries exact match, then whitespace/case-insensitive match, then a
    common-name fallback (only when exactly one candidate matches). Raises
    ValueError with an actionable message if nothing resolves cleanly.
    """
    if configured_name in df.columns:
        return configured_name

    normalized = {col.strip().lower(): col for col in df.columns}
    tolerant_match = normalized.get(configured_name.strip().lower())
    if tolerant_match is not None:
        return tolerant_match

    hints = COLUMN_NAME_HINTS.get(hint_key, [])
    candidates = [col for col in df.columns if col.strip().lower() in hints]
    if len(candidates) == 1:
        print(
            f"   - AVISO: Coluna configurada '{configured_name}' ({description}) não encontrada. "
            f"Usando '{candidates[0]}' (detectada automaticamente pelo nome)."
        )
        return candidates[0]

    available = list(df.columns)
    if len(candidates) > 1:
        raise ValueError(
            f"Coluna configurada '{configured_name}' ({description}) não encontrada, e mais de uma "
            f"coluna parece corresponder: {candidates}. Ajuste 'column_mapping' no config para "
            f"desambiguar. Colunas disponíveis: {available}"
        )

    raise ValueError(
        f"Coluna configurada '{configured_name}' ({description}) não encontrada. "
        f"Colunas disponíveis no arquivo: {available}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `mise exec -- uv run --with pytest pytest tests/test_data_preprocessor.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/data_preprocessor.py tests/test_data_preprocessor.py
git commit -m "feat(data_preprocessor): add resolve_column with tolerant/hint-based column matching"
```

---

### Task 2: `robust_numeric_parsing`

**Files:**
- Modify: `scripts/data_preprocessor.py` (insert after the code added in Task 1)
- Modify: `tests/test_data_preprocessor.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data_preprocessor.py` (add to the existing import line and add new test functions):

```python
from data_preprocessor import COLUMN_NAME_HINTS, resolve_column, robust_numeric_parsing


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
```

Replace the earlier `from data_preprocessor import COLUMN_NAME_HINTS, resolve_column` line at the top of the file with the updated import shown above (adds `robust_numeric_parsing`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `mise exec -- uv run --with pytest pytest tests/test_data_preprocessor.py -v`
Expected: `ImportError: cannot import name 'robust_numeric_parsing'`

- [ ] **Step 3: Implement `robust_numeric_parsing`**

Insert into `scripts/data_preprocessor.py`, right after `resolve_column`:

```python
_NULL_TOKENS = {"", "-", "n/a", "na", "null", "none", "nan"}


def _digit_group_vote(value):
    """
    Votes 'us' (comma=thousands, dot=decimal) or 'br' (dot=thousands,
    comma=decimal) for a single numeric-looking string, or None if the
    value gives no reliable signal on its own.
    """
    has_dot = "." in value
    has_comma = "," in value

    if has_dot and has_comma:
        return "us" if value.rfind(".") > value.rfind(",") else "br"

    if has_dot and value.count(".") > 1:
        return "br"  # repeated separator can only be a thousands grouping
    if has_comma and value.count(",") > 1:
        return "us"

    if has_dot and value.count(".") == 1:
        trailing = len(value) - value.rindex(".") - 1
        return "us" if trailing in (1, 2) else None
    if has_comma and value.count(",") == 1:
        trailing = len(value) - value.rindex(",") - 1
        return "br" if trailing in (1, 2) else None

    return None


def _detect_number_format(values):
    """
    Decides 'us' or 'br' for a whole column by majority vote across
    unambiguous per-value signals. Values with no separator at all give no
    signal and don't affect the decision. Returns (format, confident).
    """
    with_separator = [v for v in values if ("." in v or "," in v)]
    if not with_separator:
        return "us", True  # nothing to disambiguate, format choice is a no-op

    votes = {"us": 0, "br": 0}
    for value in with_separator:
        vote = _digit_group_vote(value)
        if vote is not None:
            votes[vote] += 1

    if votes["br"] > votes["us"]:
        return "br", True
    if votes["us"] > votes["br"]:
        return "us", True
    return "us", False  # tie (including no unambiguous signal at all)


def robust_numeric_parsing(series, column_name="valor"):
    """
    Converts a possibly-messy string series to numeric. Auto-detects BR
    (1.234,56) vs US (1,234.56) locale per column, strips currency
    symbols/percent signs, handles accounting-style negatives ((1.234,56)),
    and treats common null tokens as missing instead of parse failures.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series

    text = series.astype(str).str.strip()
    is_null_token = text.str.lower().isin(_NULL_TOKENS)

    is_paren_negative = text.str.match(r"^\(.*\)$", na=False)
    cleaned = text.str.replace(r"^\((.*)\)$", r"\1", regex=True)
    cleaned = cleaned.str.replace(r"[^\d,.\-]", "", regex=True)

    non_null_values = cleaned[~is_null_token & (cleaned != "")]
    number_format, confident = _detect_number_format(non_null_values.tolist())
    if not confident and len(non_null_values) > 0:
        examples = non_null_values.unique()[:5].tolist()
        print(
            f"   - AVISO: Formato numérico da coluna '{column_name}' ambíguo, assumindo padrão "
            f"US (',' milhar / '.' decimal). Exemplos: {examples}"
        )

    if number_format == "br":
        cleaned = cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        cleaned = cleaned.str.replace(",", "", regex=False)

    cleaned = cleaned.where(~is_null_token, None)
    result = pd.to_numeric(cleaned, errors="coerce")
    result = result.where(~is_paren_negative, -result)

    failed = result.isna() & ~is_null_token & text.notna()
    if failed.sum() > 0:
        examples = series[failed].unique()[:5].tolist()
        print(
            f"   - AVISO: {int(failed.sum())} valor(es) da coluna '{column_name}' não puderam ser "
            f"convertidos para número. Exemplos: {examples}"
        )

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `mise exec -- uv run --with pytest pytest tests/test_data_preprocessor.py -v`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/data_preprocessor.py tests/test_data_preprocessor.py
git commit -m "feat(data_preprocessor): add robust_numeric_parsing with BR/US locale auto-detection"
```

---

### Task 3: `read_csv_robust`

**Files:**
- Modify: `scripts/data_preprocessor.py` (insert after `robust_numeric_parsing`)
- Modify: `tests/test_data_preprocessor.py`

- [ ] **Step 1: Write the failing tests**

Update the import line at the top of `tests/test_data_preprocessor.py` to:

```python
from data_preprocessor import (
    COLUMN_NAME_HINTS,
    resolve_column,
    robust_numeric_parsing,
    read_csv_robust,
)
```

Append these tests:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `mise exec -- uv run --with pytest pytest tests/test_data_preprocessor.py -v`
Expected: `ImportError: cannot import name 'read_csv_robust'`

- [ ] **Step 3: Implement `read_csv_robust`**

Insert into `scripts/data_preprocessor.py`, right after `robust_numeric_parsing`:

```python
def read_csv_robust(path, **kwargs):
    """
    Reads a CSV tolerating BOM/latin-1 encoding and ','/';'/tab
    delimiters, and strips whitespace from header names.
    """
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            sample = f.read(8192)
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            sample = f.read(8192)
        encoding = "latin-1"
        print(f"   - AVISO: Encoding não-UTF-8 detectado em '{path}'. Usando 'latin-1'.")

    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        delimiter = ","

    if delimiter != ",":
        print(f"   - AVISO: Delimitador '{delimiter}' detectado em '{path}' (não é vírgula).")

    df = pd.read_csv(path, encoding=encoding, sep=delimiter, **kwargs)
    df.columns = df.columns.str.strip()
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `mise exec -- uv run --with pytest pytest tests/test_data_preprocessor.py -v`
Expected: 20 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/data_preprocessor.py tests/test_data_preprocessor.py
git commit -m "feat(data_preprocessor): add read_csv_robust with delimiter/encoding auto-detection"
```

---

### Task 4: Wire it all into `load_and_prepare_data`

**Files:**
- Modify: `scripts/data_preprocessor.py:72-269` (the `load_and_prepare_data` function)
- Modify: `tests/test_data_preprocessor.py`

- [ ] **Step 1: Write the failing integration tests**

Append to `tests/test_data_preprocessor.py`:

```python
from data_preprocessor import (
    COLUMN_NAME_HINTS,
    resolve_column,
    robust_numeric_parsing,
    read_csv_robust,
    load_and_prepare_data,
)


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

    with pytest.raises(Exception, match="investment"):
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `mise exec -- uv run --with pytest pytest tests/test_data_preprocessor.py -v`
Expected: the 4 new tests FAIL — `test_load_and_prepare_data_handles_br_locale_semicolon_investment_file` fails with the same `TypeError: dtype 'str' does not support operation 'mean'` from the original bug report (wrapped in the generic `Exception`).

- [ ] **Step 3: Rewire `load_and_prepare_data`**

In `scripts/data_preprocessor.py`, replace the CSV-reading lines (currently 87-88):

```python
        kpi_df = pd.read_csv(config["performance_file_path"], thousands=",")
        daily_investment_df = pd.read_csv(config["investment_file_path"], thousands=",")
```

with:

```python
        kpi_df = read_csv_robust(config["performance_file_path"])
        daily_investment_df = read_csv_robust(config["investment_file_path"])
```

Replace the trends-file block (currently lines 90-123):

```python
        if "generic_trends_file_path" in config and config["generic_trends_file_path"]:
            try:
                trends_df = pd.read_csv(
                    config["generic_trends_file_path"], thousands=","
                )
                trends_df.rename(
                    columns={
                        trends_map.get("date_col", "Start Date"): "Date",
                        trends_map.get(
                            "trends_col", "Ad Opportunities"
                        ): "Generic Searches",
                    },
                    inplace=True,
                )
                trends_df["Date"] = robust_date_parsing(
                    trends_df["Date"],
                    date_format=date_formats.get("generic_trends_file"),
                )
                trends_df.dropna(subset=["Date"], inplace=True)
                trends_df = (
                    trends_df[["Date", "Generic Searches"]]
                    .sort_values(by="Date")
                    .reset_index(drop=True)
                )
            except FileNotFoundError:
                print(
                    "   - WARNING: Generic trends file not found. Continuing without trends data."
                )
                trends_df = pd.DataFrame(columns=["Date", "Generic Searches"])
        else:
            print(
                "   - INFO: No generic trends file path provided. Continuing without trends data."
            )
            trends_df = pd.DataFrame(columns=["Date", "Generic Searches"])
```

with:

```python
        if "generic_trends_file_path" in config and config["generic_trends_file_path"]:
            try:
                trends_df = read_csv_robust(config["generic_trends_file_path"])
                trends_date_col = resolve_column(
                    trends_df,
                    trends_map.get("date_col", "Start Date"),
                    "date",
                    "coluna de data do arquivo de tendências",
                )
                trends_value_col = resolve_column(
                    trends_df,
                    trends_map.get("trends_col", "Ad Opportunities"),
                    "trends",
                    "coluna de valor do arquivo de tendências",
                )
                trends_df.rename(
                    columns={trends_date_col: "Date", trends_value_col: "Generic Searches"},
                    inplace=True,
                )
                trends_df["Generic Searches"] = robust_numeric_parsing(
                    trends_df["Generic Searches"], column_name="Generic Searches"
                )
                trends_df["Date"] = robust_date_parsing(
                    trends_df["Date"],
                    date_format=date_formats.get("generic_trends_file"),
                )
                trends_df.dropna(subset=["Date"], inplace=True)
                trends_df = (
                    trends_df[["Date", "Generic Searches"]]
                    .sort_values(by="Date")
                    .reset_index(drop=True)
                )
            except FileNotFoundError:
                print(
                    "   - WARNING: Generic trends file not found. Continuing without trends data."
                )
                trends_df = pd.DataFrame(columns=["Date", "Generic Searches"])
        else:
            print(
                "   - INFO: No generic trends file path provided. Continuing without trends data."
            )
            trends_df = pd.DataFrame(columns=["Date", "Generic Searches"])
```

Replace the KPI rename + cleaning block (currently lines 125-147):

```python
        # --- Dynamically Rename Columns ---
        user_kpi_col = config.get("performance_kpi_column", "Sessions")
        kpi_df.rename(
            columns={
                perf_map.get("date_col", "date"): "Date",
                perf_map.get("kpi_col", user_kpi_col): "kpi",
            },
            inplace=True,
        )

        # --- Clean percentage/thousands strings and convert to numeric ---
        if kpi_df["kpi"].dtype == "object":
            # Handle potential string formatting (e.g. '1.234,56' or '1,234.56')
            # If thousands=',' was used in read_csv, pandas might have already handled it if it matched.
            # But let's be safe for cases where it's mixed with symbols.
            kpi_df["kpi"] = kpi_df["kpi"].str.replace("%", "", regex=False)
            # If there are still commas and dots, we need to know the locale.
            # Assuming standard numeric if read_csv thousands worked.
            kpi_df["kpi"] = pd.to_numeric(
                kpi_df["kpi"].str.replace(",", "", regex=False), errors="coerce"
            )

        kpi_df["kpi"] = pd.to_numeric(kpi_df["kpi"], errors="coerce")

        daily_investment_df.rename(
            columns={
                inv_map.get("date_col", "dates"): "Date",
                inv_map.get("channel_col", "product_group"): "Product Group",
                inv_map.get("investment_col", "total_revenue"): "investment",
            },
            inplace=True,
        )

        # Standardize product group names by stripping whitespace
        daily_investment_df["Product Group"] = daily_investment_df[
            "Product Group"
        ].str.strip()
```

with:

```python
        # --- Dynamically Rename Columns ---
        user_kpi_col = config.get("performance_kpi_column", "Sessions")
        kpi_date_col = resolve_column(
            kpi_df, perf_map.get("date_col", "date"), "date", "coluna de data do arquivo de performance"
        )
        kpi_value_col = resolve_column(
            kpi_df,
            perf_map.get("kpi_col", user_kpi_col),
            "kpi",
            "coluna de KPI do arquivo de performance",
        )
        kpi_df.rename(columns={kpi_date_col: "Date", kpi_value_col: "kpi"}, inplace=True)
        kpi_df["kpi"] = robust_numeric_parsing(kpi_df["kpi"], column_name="kpi")

        inv_date_col = resolve_column(
            daily_investment_df,
            inv_map.get("date_col", "dates"),
            "date",
            "coluna de data do arquivo de investimento",
        )
        inv_channel_col = resolve_column(
            daily_investment_df,
            inv_map.get("channel_col", "product_group"),
            "channel",
            "coluna de canal do arquivo de investimento",
        )
        inv_value_col = resolve_column(
            daily_investment_df,
            inv_map.get("investment_col", "total_revenue"),
            "investment",
            "coluna de investimento do arquivo de investimento",
        )
        daily_investment_df.rename(
            columns={
                inv_date_col: "Date",
                inv_channel_col: "Product Group",
                inv_value_col: "investment",
            },
            inplace=True,
        )

        # Standardize product group names: strip whitespace and normalize case so
        # the same channel exported with inconsistent casing doesn't fragment
        # into separate pivot_table columns.
        daily_investment_df["Product Group"] = (
            daily_investment_df["Product Group"].str.strip().str.upper()
        )
        daily_investment_df["investment"] = robust_numeric_parsing(
            daily_investment_df["investment"], column_name="investment"
        )
```

Finally, add empty-dataframe guards right after the existing `dropna` calls (currently lines 172-175):

```python
        # --- Data Cleaning & Validation ---
        kpi_df.dropna(subset=["Date", "kpi"], inplace=True)
        daily_investment_df.dropna(
            subset=["Date", "investment", "Product Group"], inplace=True
        )
```

becomes:

```python
        # --- Data Cleaning & Validation ---
        kpi_df.dropna(subset=["Date", "kpi"], inplace=True)
        daily_investment_df.dropna(
            subset=["Date", "investment", "Product Group"], inplace=True
        )

        if kpi_df.empty:
            raise ValueError(
                f"Nenhuma linha válida restou no arquivo de performance "
                f"('{config['performance_file_path']}') após a limpeza. Verifique o formato de "
                f"data e do KPI no arquivo."
            )
        if daily_investment_df.empty:
            raise ValueError(
                f"Nenhuma linha válida restou no arquivo de investimento "
                f"('{config['investment_file_path']}') após a limpeza. Verifique o formato de "
                f"data, canal e investimento no arquivo."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `mise exec -- uv run --with pytest pytest tests/test_data_preprocessor.py -v`
Expected: 24 passed

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `mise exec -- uv run --with pytest pytest tests/ -q`
Expected: `94 passed` (70 baseline + 24 new)

- [ ] **Step 6: Commit**

```bash
git add scripts/data_preprocessor.py tests/test_data_preprocessor.py
git commit -m "fix(data_preprocessor): rewire load_and_prepare_data onto robust CSV/column/number parsing"
```

---

### Task 5: Dedup `streamlit_app.py`'s column-guessing lists onto `COLUMN_NAME_HINTS`

**Files:**
- Modify: `scripts/streamlit_app.py:349` (add import)
- Modify: `scripts/streamlit_app.py:871-963` (5 nested functions)

- [ ] **Step 1: Add the import**

In `scripts/streamlit_app.py`, right after the `dashboard_charts` import block closes and before `import base64` (currently lines 349-351):

```python
)

import base64
```

becomes:

```python
)
from data_preprocessor import COLUMN_NAME_HINTS

import base64
```

- [ ] **Step 2: Replace the hardcoded list in `get_date_col`**

Current (lines 871-881):

```python
                def get_date_col(file_path):
                    if not file_path:
                        return "date"
                    try:
                        df = pd.read_csv(file_path, nrows=0)
                        for col in df.columns:
                            if col.lower() in ["date", "dates", "data", "day", "dia"]:
                                return col
                        return df.columns[0]
                    except:
                        return "date"
```

becomes:

```python
                def get_date_col(file_path):
                    if not file_path:
                        return "date"
                    try:
                        df = pd.read_csv(file_path, nrows=0)
                        for col in df.columns:
                            if col.lower() in COLUMN_NAME_HINTS["date"]:
                                return col
                        return df.columns[0]
                    except:
                        return "date"
```

- [ ] **Step 3: Replace the hardcoded list in `get_channel_col`**

Current (lines 883-902):

```python
                def get_channel_col(file_path):
                    if not file_path:
                        return "product_group"
                    try:
                        df = pd.read_csv(file_path, nrows=0)
                        for col in df.columns:
                            if col.lower() in [
                                "channel",
                                "product_group",
                                "product",
                                "media",
                                "source",
                                "campaign",
                                "canal",
                                "grupo",
                            ]:
                                return col
                        return df.columns[0]
                    except:
                        return "product_group"
```

becomes:

```python
                def get_channel_col(file_path):
                    if not file_path:
                        return "product_group"
                    try:
                        df = pd.read_csv(file_path, nrows=0)
                        for col in df.columns:
                            if col.lower() in COLUMN_NAME_HINTS["channel"]:
                                return col
                        return df.columns[0]
                    except:
                        return "product_group"
```

- [ ] **Step 4: Replace the hardcoded list in `get_investment_col`**

Current (lines 904-922):

```python
                def get_investment_col(file_path):
                    if not file_path:
                        return "total_revenue"
                    try:
                        df = pd.read_csv(file_path, nrows=0)
                        for col in df.columns:
                            if col.lower() in [
                                "investment",
                                "spend",
                                "cost",
                                "investimento",
                                "revenue",
                                "total_revenue",
                                "valor",
                            ]:
                                return col
                        return df.columns[-1]
                    except:
                        return "total_revenue"
```

becomes:

```python
                def get_investment_col(file_path):
                    if not file_path:
                        return "total_revenue"
                    try:
                        df = pd.read_csv(file_path, nrows=0)
                        for col in df.columns:
                            if col.lower() in COLUMN_NAME_HINTS["investment"]:
                                return col
                        return df.columns[-1]
                    except:
                        return "total_revenue"
```

- [ ] **Step 5: Replace the hardcoded list in `get_trends_col`**

Current (lines 924-941):

```python
                def get_trends_col(file_path):
                    if not file_path:
                        return "Ad Opportunities"
                    try:
                        df = pd.read_csv(file_path, nrows=0)
                        for col in df.columns:
                            if col.lower() in [
                                "searches",
                                "trends",
                                "opportunities",
                                "ad opportunities",
                                "volume",
                                "generic searches",
                            ]:
                                return col
                        return df.columns[-1]
                    except:
                        return "Ad Opportunities"
```

becomes:

```python
                def get_trends_col(file_path):
                    if not file_path:
                        return "Ad Opportunities"
                    try:
                        df = pd.read_csv(file_path, nrows=0)
                        for col in df.columns:
                            if col.lower() in COLUMN_NAME_HINTS["trends"]:
                                return col
                        return df.columns[-1]
                    except:
                        return "Ad Opportunities"
```

- [ ] **Step 6: Replace the hardcoded list in `get_kpi_col`**

Current (lines 943-963):

```python
                def get_kpi_col(file_path, user_kpi):
                    if not file_path:
                        return user_kpi
                    try:
                        df = pd.read_csv(file_path, nrows=0)
                        if user_kpi in df.columns:
                            return user_kpi
                        for col in df.columns:
                            if col.lower() in [
                                "kpi",
                                "sessions",
                                "conversions",
                                "revenue",
                                "conversoes",
                                "cliques",
                                "clicks",
                            ]:
                                return col
                        return df.columns[1] if len(df.columns) > 1 else df.columns[0]
                    except:
                        return user_kpi
```

becomes:

```python
                def get_kpi_col(file_path, user_kpi):
                    if not file_path:
                        return user_kpi
                    try:
                        df = pd.read_csv(file_path, nrows=0)
                        if user_kpi in df.columns:
                            return user_kpi
                        for col in df.columns:
                            if col.lower() in COLUMN_NAME_HINTS["kpi"]:
                                return col
                        return df.columns[1] if len(df.columns) > 1 else df.columns[0]
                    except:
                        return user_kpi
```

- [ ] **Step 7: Verify the file still parses correctly**

Run: `mise exec -- uv run python -m py_compile scripts/streamlit_app.py`
Expected: no output, exit code 0 (this only checks syntax — it does not execute Streamlit's runtime, which requires a live `streamlit run` session)

- [ ] **Step 8: Verify no hardcoded duplicates remain**

Run: `mise exec -- uv run python -c "import ast,sys; tree = ast.parse(open('scripts/streamlit_app.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

Also manually confirm (grep) that the 5 literal lists are gone from `streamlit_app.py`:

Run: `grep -n '"product_group",$' scripts/streamlit_app.py`
Expected: no output (the old hardcoded list literal is gone; `COLUMN_NAME_HINTS["channel"]` now lives only in `data_preprocessor.py`)

- [ ] **Step 9: Commit**

```bash
git add scripts/streamlit_app.py
git commit -m "refactor(streamlit_app): reuse COLUMN_NAME_HINTS instead of duplicated column-guess lists"
```

---

### Task 6: Final regression check

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `mise exec -- uv run --with pytest pytest tests/ -q`
Expected: `94 passed` — the original 70 plus the 24 added in Tasks 1-4, with no regressions from the Task 5 refactor.

- [ ] **Step 2: Confirm the original bug scenario is fixed**

Run: `mise exec -- uv run --with pytest pytest tests/test_data_preprocessor.py::test_load_and_prepare_data_handles_br_locale_semicolon_investment_file -v`
Expected: PASS — this is the test that reproduces the exact crash from the original log (`TypeError: dtype 'str' does not support operation 'mean'`).

No commit needed for this task — it's verification only.

---

## Post-plan note

Not addressed by this plan (explicitly out of scope per the spec): true multi-dialect CSVs (mixed locale within one column), files with extra header/footer rows or multiple tables per sheet, and any new external dependency. If real client uploads hit these, treat as a new bug report with real data, not a speculative extension here.
