# -*- coding: utf-8 -*-
"""
This module handles all data loading, validation, cleaning, and pre-processing.
"""

import csv
import pandas as pd
import numpy as np


def treat_outliers(df, column):
    """Identifies and caps outliers in a specified column using the 1.5 * IQR rule."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    return df


def geometric_decay(series, alpha):
    """Applies geometric decay for ad-stock."""
    return series.ewm(alpha=alpha, adjust=False).mean()


def find_best_alpha(investment_series, kpi_series):
    """Finds the best adstock alpha for a single channel."""
    correlations = {}
    for alpha in np.arange(0.1, 1.0, 0.1):
        adstocked_series = geometric_decay(investment_series, alpha)
        correlations[alpha] = adstocked_series.corr(kpi_series)

    best_alpha = max(correlations, key=correlations.get)
    return best_alpha, correlations[best_alpha]


DATE_FORMAT_CANDIDATES = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
]


def robust_date_parsing(series, date_format=None):
    """
    Parses a date column deterministically, testing whole-format candidates
    against the WHOLE column instead of guessing row-by-row (dateutil's
    per-row guessing silently inverts day/month for values where the day
    happens to be <= 12, scrambling an otherwise-uniform column with zero
    NaT to show for it).

    The configured `date_format`, if given, wins immediately when it covers
    100% of the non-null rows. Otherwise every candidate in
    DATE_FORMAT_CANDIDATES is tried; the one format that parses 100% of the
    non-null rows (0 NaT) is used. If two or more candidates tie at 100%
    coverage, the column is genuinely ambiguous (e.g. every day-of-month is
    <= 12) and a ValueError is raised instead of silently picking one. If no
    candidate reaches 100% coverage, a ValueError is raised listing example
    values that failed to parse.
    """
    original_non_nulls = series.notna().sum()
    if original_non_nulls == 0:
        return pd.to_datetime(series, errors="coerce")

    column_name = series.name or "data"

    if date_format:
        parsed = pd.to_datetime(series, format=date_format, errors="coerce")
        if parsed.notna().sum() == original_non_nulls:
            return parsed
        print(
            f"   - AVISO: Formato de data configurado '{date_format}' não cobriu 100% das linhas "
            f"da coluna '{column_name}'. Tentando detecção automática de formato."
        )

    matches = {}
    best_fmt = None
    best_count = -1
    best_parsed = None
    for fmt in DATE_FORMAT_CANDIDATES:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        count = parsed.notna().sum()
        if count == original_non_nulls:
            matches[fmt] = parsed
        if count > best_count:
            best_count = count
            best_fmt = fmt
            best_parsed = parsed

    if len(matches) == 1:
        return next(iter(matches.values()))

    if len(matches) > 1:
        examples = series.dropna().unique()[:3].tolist()
        raise ValueError(
            f"Não foi possível determinar o formato de data da coluna '{column_name}' de forma "
            f"inequívoca: os formatos {list(matches.keys())} parseiam 100% das linhas mas produzem "
            f"resultados divergentes. Exemplos de valores: {examples}. Configure 'date_formats' no "
            f"config explicitamente para esta coluna para desambiguar."
        )

    failing_mask = series.notna() & best_parsed.isna()
    examples = series[failing_mask].unique()[:3].tolist()
    raise ValueError(
        f"Não foi possível parsear a coluna de data '{column_name}': nenhum formato testado "
        f"({DATE_FORMAT_CANDIDATES}) cobriu 100% das linhas. Melhor formato testado: '{best_fmt}' "
        f"({best_count}/{original_non_nulls} linhas). Exemplos de valores que falharam: {examples}. "
        f"Configure 'date_formats' no config explicitamente com o formato correto."
    )


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

    # fillna before astype: on pandas' 'str' dtype, astype(str) alone leaves
    # missing values as bare float NaN instead of stringifying them, which
    # breaks the string ops below. "" is already a recognized null token.
    text = series.fillna("").astype(str).str.strip()
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


def guess_date_col(file_path):
    """Guesses the date column of an uploaded CSV to pre-fill the UI."""
    if not file_path:
        return "date"
    try:
        df = read_csv_robust(file_path, nrows=0)
        for col in df.columns:
            if col.lower() in COLUMN_NAME_HINTS["date"]:
                return col
        return df.columns[0]
    except Exception:
        return "date"


def guess_channel_col(file_path):
    """Guesses the channel column of an uploaded CSV to pre-fill the UI."""
    if not file_path:
        return "product_group"
    try:
        df = read_csv_robust(file_path, nrows=0)
        for col in df.columns:
            if col.lower() in COLUMN_NAME_HINTS["channel"]:
                return col
        return df.columns[0]
    except Exception:
        return "product_group"


def guess_investment_col(file_path):
    """Guesses the investment column of an uploaded CSV to pre-fill the UI."""
    if not file_path:
        return "total_revenue"
    try:
        df = read_csv_robust(file_path, nrows=0)
        for col in df.columns:
            if col.lower() in COLUMN_NAME_HINTS["investment"]:
                return col
        return df.columns[-1]
    except Exception:
        return "total_revenue"


def guess_trends_col(file_path):
    """Guesses the trends column of an uploaded CSV to pre-fill the UI."""
    if not file_path:
        return "Ad Opportunities"
    try:
        df = read_csv_robust(file_path, nrows=0)
        for col in df.columns:
            if col.lower() in COLUMN_NAME_HINTS["trends"]:
                return col
        return df.columns[-1]
    except Exception:
        return "Ad Opportunities"


def guess_kpi_col(file_path, user_kpi):
    """Guesses the KPI column of an uploaded CSV to pre-fill the UI."""
    if not file_path:
        return user_kpi
    try:
        df = read_csv_robust(file_path, nrows=0)
        if user_kpi in df.columns:
            return user_kpi
        for col in df.columns:
            if col.lower() in COLUMN_NAME_HINTS["kpi"]:
                return col
        return df.columns[1] if len(df.columns) > 1 else df.columns[0]
    except Exception:
        return user_kpi


def load_and_prepare_data(config):
    """
    Loads and prepares the KPI, investment, and trends data based on the config.
    """
    print("\n" + "=" * 50 + "\nLoading, Cleaning, and Preparing Data...\n" + "=" * 50)

    try:
        # --- Get Column Mappings from Config ---
        mapping = config.get("column_mapping", {})
        inv_map = mapping.get("investment_file", {})
        perf_map = mapping.get("performance_file", {})
        trends_map = mapping.get("generic_trends_file", {})
        date_formats = config.get("date_formats", {})

        # --- Load Data ---
        kpi_df = read_csv_robust(config["performance_file_path"])
        daily_investment_df = read_csv_robust(config["investment_file_path"])

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
                trends_df = pd.DataFrame(
                    {
                        "Date": pd.Series(dtype="datetime64[ns]"),
                        "Generic Searches": pd.Series(dtype="float64"),
                    }
                )
        else:
            print(
                "   - INFO: No generic trends file path provided. Continuing without trends data."
            )
            trends_df = pd.DataFrame(
                {
                    "Date": pd.Series(dtype="datetime64[ns]"),
                    "Generic Searches": pd.Series(dtype="float64"),
                }
            )

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

        # --- Date Formatting ---
        kpi_df["Date"] = robust_date_parsing(
            kpi_df["Date"], date_format=date_formats.get("performance_file")
        )
        daily_investment_df["Date"] = robust_date_parsing(
            daily_investment_df["Date"], date_format=date_formats.get("investment_file")
        )

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

        # --- Conditional Outlier Treatment ---
        outlier_config = config.get("treat_outliers", False)
        if outlier_config:
            print("   - Applying outlier treatment...")
            if isinstance(outlier_config, list):
                # Treat specific columns listed in config
                for col in outlier_config:
                    # Map 'Sessions' or user column to 'kpi' if that's what was intended
                    target_col = (
                        "kpi" if col == user_kpi_col or col == "Sessions" else col
                    )
                    if target_col in kpi_df.columns:
                        kpi_df = treat_outliers(kpi_df, target_col)
                        print(f"     - Treated outliers in KPI column: '{col}'")
            elif isinstance(outlier_config, bool) and outlier_config:
                # Default to treating the KPI column
                kpi_df = treat_outliers(kpi_df, "kpi")
                print(f"     - Treated outliers in KPI column: 'kpi'")

        # --- Debug: Print Date Ranges ---
        print(
            f"   - KPI Data Date Range: {kpi_df['Date'].min()} to {kpi_df['Date'].max()}"
        )
        print(
            f"   - Investment Data Date Range: {daily_investment_df['Date'].min()} to {daily_investment_df['Date'].max()}"
        )
        # --- End Debug ---

        kpi_df = kpi_df[["Date", "kpi"]].sort_values(by="Date").reset_index(drop=True)
        daily_investment_df = (
            daily_investment_df[["Date", "Product Group", "investment"]]
            .sort_values(by="Date")
            .reset_index(drop=True)
        )

        print("   - Data loaded and columns renamed successfully.")

        kpi_col = "kpi"

        # --- Adstock Transformation ---
        print(
            "   - Checking for negative correlations and applying adstock where needed..."
        )
        investment_pivot = daily_investment_df.pivot_table(
            index="Date", columns="Product Group", values="investment"
        ).fillna(0)
        merged_for_corr = pd.merge(kpi_df, investment_pivot, on="Date", how="inner")

        correlation_matrix = merged_for_corr.corr(numeric_only=True)

        for column in investment_pivot.columns:
            if column in correlation_matrix and correlation_matrix[column][kpi_col] < 0:
                print(
                    f"     - Applying adstock to '{column}' due to negative correlation."
                )
                best_alpha, _ = find_best_alpha(
                    merged_for_corr[column], merged_for_corr[kpi_col]
                )
                daily_investment_df.loc[
                    daily_investment_df["Product Group"] == column, "investment"
                ] = geometric_decay(
                    daily_investment_df.loc[
                        daily_investment_df["Product Group"] == column, "investment"
                    ],
                    best_alpha,
                )

        print("   - Data preparation complete.")

        # --- Final Correlation Matrix (for display) ---
        final_pivot = daily_investment_df.pivot_table(
            index="Date", columns="Product Group", values="investment"
        ).fillna(0)
        final_merged = pd.merge(kpi_df, final_pivot, on="Date", how="inner")
        if not trends_df.empty:
            final_merged = pd.merge(final_merged, trends_df, on="Date", how="left")
        correlation_matrix = final_merged.corr(numeric_only=True)
        print(
            "\n"
            + "=" * 50
            + "\nFinal Correlation Matrix (Post-Processing)\n"
            + "=" * 50
        )
        print(correlation_matrix)

        return kpi_df, daily_investment_df, trends_df, correlation_matrix

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"An input file was not found. Please check your config file paths. Details: {e}"
        )
    except Exception as e:
        raise Exception(f"An unexpected error occurred during data preparation: {e}")
