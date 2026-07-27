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


def detect_cadence(dates):
    """
    Detecta a cadência (intervalo típico em dias) de uma série de datas.

    Pega as datas únicas, ordena, calcula o diff() em dias entre datas
    consecutivas e usa a mediana (robusta a alguns outliers/gaps pontuais).
    Para facilitar o uso pelo resto do pipeline, a mediana é "encaixada"
    (snapped) num valor canônico quando cai dentro de uma faixa de
    tolerância:
      - <= 1.5 dia    -> 1  (diário)
      - 6 a 8 dias    -> 7  (semanal; tolera feriados/re-ancoramentos que
                             deslocam o dia da semana em +-1 dia)
      - 27 a 31 dias  -> 30 (mensal; cobre a variação natural de dias no mês)
    Fora dessas faixas, retorna a mediana bruta arredondada (ex: cadência
    quinzenal ~14 dias não tem faixa própria e sai como valor bruto,
    conforme pedido no plano: "ou valor bruto se não bater em nenhum").

    Com menos de 2 datas únicas não há diff() para calcular; retorna 1
    (assume diário) como default seguro -- na prática não deve acontecer,
    já que load_and_prepare_data só chama isto após validar que os
    dataframes não estão vazios.
    """
    unique_dates = pd.to_datetime(pd.Series(dates)).dropna().unique()
    unique_dates = pd.Series(unique_dates).sort_values().reset_index(drop=True)
    if len(unique_dates) < 2:
        return 1

    median_days = unique_dates.diff().dropna().dt.days.median()

    if median_days <= 1.5:
        return 1
    if 6 <= median_days <= 8:
        return 7
    if 27 <= median_days <= 31:
        return 30
    return int(round(median_days))


def drop_partial_periods(df, date_col, cadence):
    """
    Descarta linhas cuja data representa um período "parcial" (mais curto
    que 0.6x a cadência esperada), em qualquer posição da série -- início,
    meio ou fim, não só nas bordas.

    Uma data no MEIO da série só é considerada parcial se a distância até a
    data anterior E até a próxima forem AMBAS menores que 0.6*cadência.
    Checar só um dos lados não funciona em casos reais: quando um período
    normal é "quebrado" no meio (ex: um re-ancoramento de virada de ano cria
    uma data extra entre duas datas normais), as duas datas vizinhas à data
    extra também ficam com só UM dos lados curto -- mas elas não são
    parciais, é a data espremida entre as duas que é. Exigir os dois lados
    curtos isola corretamente só essa data extra.

    Nas pontas da série (primeira/última data), só existe um vizinho para
    comparar, então a checagem usa só a distância disponível (ex: um
    primeiro período de poucos dias antes do próximo, ou um último período
    parcial sem "próxima" data para comparar).

    Loga quantas e quais datas foram descartadas e retorna o dataframe
    filtrado.

    Cuidado ao reusar com uma `cadence` vinda de fora (não derivada via
    `detect_cadence` da própria série): com exatamente 2 datas únicas, ambas
    viram "borda" e comparam a mesma distância simétrica contra o limiar --
    se essa distância cair abaixo de 0.6*cadence, as DUAS datas são
    descartadas e o resultado fica vazio. Isso não acontece no fluxo atual
    (load_and_prepare_data sempre deriva a cadência da própria série), mas é
    uma armadilha real para quem chamar esta função com cadência externa.
    """
    dates = pd.to_datetime(df[date_col])
    unique_dates = pd.Series(dates.unique()).sort_values().reset_index(drop=True)
    n = len(unique_dates)
    if n < 2:
        return df.reset_index(drop=True)

    threshold = 0.6 * cadence
    stub_dates = []
    for i in range(n):
        date = unique_dates[i]
        dist_prev = (date - unique_dates[i - 1]).days if i > 0 else None
        dist_next = (unique_dates[i + 1] - date).days if i < n - 1 else None
        if dist_prev is not None and dist_next is not None:
            is_stub = dist_prev < threshold and dist_next < threshold
        else:
            lone_dist = dist_prev if dist_prev is not None else dist_next
            is_stub = lone_dist < threshold
        if is_stub:
            stub_dates.append(date)

    if not stub_dates:
        print(
            f"   - Nenhum período parcial encontrado (cadência={cadence} dia(s), "
            f"limiar={threshold:.1f} dia(s))."
        )
        return df.reset_index(drop=True)

    mask = dates.isin(stub_dates)
    dropped_str = ", ".join(
        pd.Timestamp(d).strftime("%d/%m/%Y") for d in sorted(stub_dates)
    )
    print(
        f"   - Períodos parciais descartados: {int(mask.sum())} linha(s) em "
        f"{len(stub_dates)} data(s) (distância < {threshold:.1f} dia(s)): {dropped_str}"
    )
    return df[~mask].reset_index(drop=True)


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
    "kpi": [
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
    ],
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


def _numeric_looking_columns(df, exclude_hints=(), max_nan_ratio=0.9):
    """
    Returns, in column order, the columns of a (small sample) dataframe whose
    values mostly convert to numbers via robust_numeric_parsing -- i.e. no
    more than max_nan_ratio come back NaN. Columns whose name matches
    `exclude_hints` (typically the 'date' hints) are skipped entirely: a date
    string like '01/01/2025' can accidentally look numeric once separators
    are stripped by robust_numeric_parsing, which would otherwise make a date
    column a false-positive "numeric" candidate.
    """
    result = []
    for col in df.columns:
        if col.strip().lower() in exclude_hints:
            continue
        series = df[col]
        if len(series) == 0:
            continue
        parsed = robust_numeric_parsing(series, column_name=col)
        if parsed.isna().mean() <= max_nan_ratio:
            result.append(col)
    return result


def guess_investment_col(file_path):
    """Guesses the investment column of an uploaded CSV to pre-fill the UI."""
    if not file_path:
        return "total_revenue"
    try:
        df = read_csv_robust(file_path, nrows=50)
        for col in df.columns:
            if col.lower() in COLUMN_NAME_HINTS["investment"]:
                return col
        # Last resort: purely positional guess. Before trusting it blindly,
        # check whether it actually looks numeric -- if not, and some other
        # column does, prefer that one instead of a text column.
        fallback = df.columns[-1]
        numeric_cols = _numeric_looking_columns(df, exclude_hints=COLUMN_NAME_HINTS["date"])
        if numeric_cols and fallback not in numeric_cols:
            return numeric_cols[0]
        return fallback
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
        df = read_csv_robust(file_path, nrows=50)
        if user_kpi in df.columns:
            return user_kpi
        for col in df.columns:
            if col.lower() in COLUMN_NAME_HINTS["kpi"]:
                return col
        # Last resort: purely positional guess. Before trusting it blindly,
        # check whether it actually looks numeric -- if not, and some other
        # column does, prefer that one instead of a text column (e.g. a
        # 'Canal'/channel column repeated on every row).
        fallback = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        numeric_cols = _numeric_looking_columns(df, exclude_hints=COLUMN_NAME_HINTS["date"])
        if numeric_cols and fallback not in numeric_cols:
            return numeric_cols[0]
        return fallback
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
        kpi_raw_values = kpi_df["kpi"]
        kpi_df["kpi"] = robust_numeric_parsing(kpi_df["kpi"], column_name="kpi")
        kpi_nan_ratio = kpi_df["kpi"].isna().mean() if len(kpi_df) else 0
        if kpi_nan_ratio > 0.5:
            bad_examples = kpi_raw_values[kpi_df["kpi"].isna()].unique()[:3].tolist()
            raise ValueError(
                f"Coluna de KPI ('{kpi_value_col}') no arquivo de performance "
                f"('{config['performance_file_path']}') não pôde ser convertida para número em "
                f"mais de 50% das linhas ({kpi_nan_ratio:.0%} inválido). Isso geralmente indica "
                f"que a coluna de KPI errada foi identificada. Exemplos de valores encontrados: "
                f"{bad_examples}. Verifique 'performance_kpi_column' / "
                f"'column_mapping.performance_file.kpi_col' no config."
            )

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
        investment_raw_values = daily_investment_df["investment"]
        daily_investment_df["investment"] = robust_numeric_parsing(
            daily_investment_df["investment"], column_name="investment"
        )
        investment_nan_ratio = (
            daily_investment_df["investment"].isna().mean() if len(daily_investment_df) else 0
        )
        if investment_nan_ratio > 0.5:
            bad_examples = (
                investment_raw_values[daily_investment_df["investment"].isna()]
                .unique()[:3]
                .tolist()
            )
            raise ValueError(
                f"Coluna de investimento ('{inv_value_col}') no arquivo de investimento "
                f"('{config['investment_file_path']}') não pôde ser convertida para número em "
                f"mais de 50% das linhas ({investment_nan_ratio:.0%} inválido). Isso geralmente "
                f"indica que a coluna de investimento errada foi identificada. Exemplos de "
                f"valores encontrados: {bad_examples}. Verifique "
                f"'column_mapping.investment_file.investment_col' no config."
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

        # --- Aggregate Performance File Rows With Duplicate Dates ---
        # If the performance file has more than one row per date (most likely
        # a per-channel breakdown -- the file has no dedicated channel_col
        # mapping the way the investment file does, so we can't isolate the
        # exact column, but we can still detect the duplication and act on
        # it), summing explicitly here avoids the later merge(s) with
        # investment_pivot/trends_df on "Date" silently fanning out rows.
        if kpi_df["Date"].duplicated().any():
            n_dupe_rows = int(kpi_df["Date"].duplicated().sum())
            print(
                f"   - AVISO: {n_dupe_rows} linha(s) com data duplicada encontrada(s) no arquivo "
                f"de performance (provável detalhamento por canal). Somando os valores de KPI "
                f"por data antes de prosseguir."
            )
            kpi_df = kpi_df.groupby("Date", as_index=False)["kpi"].sum()

        # --- Cadence Detection & Partial-Period Pruning ---
        inv_cadence = detect_cadence(daily_investment_df["Date"])
        kpi_cadence = detect_cadence(kpi_df["Date"])
        if abs(inv_cadence - kpi_cadence) > 2:
            raise ValueError(
                f"Cadências divergentes entre o arquivo de investimento "
                f"({inv_cadence} dia(s)) e o arquivo de performance ({kpi_cadence} dia(s)). "
                f"Não é possível conciliar automaticamente séries com cadências diferentes -- "
                f"verifique se ambos os arquivos reportam na mesma frequência "
                f"(diária/semanal/mensal)."
            )

        n_inv_before = daily_investment_df["Date"].nunique()
        n_kpi_before = kpi_df["Date"].nunique()
        daily_investment_df = drop_partial_periods(daily_investment_df, "Date", inv_cadence)
        kpi_df = drop_partial_periods(kpi_df, "Date", kpi_cadence)
        n_inv_after = daily_investment_df["Date"].nunique()
        n_kpi_after = kpi_df["Date"].nunique()

        if daily_investment_df.empty or kpi_df.empty:
            raise ValueError(
                "Após descartar períodos parciais, não restaram linhas suficientes para "
                "continuar a análise. Verifique a consistência das datas nos arquivos de "
                "entrada."
            )

        config["period_days"] = inv_cadence
        cadence_label = {1: "diária", 7: "semanal", 30: "mensal"}.get(
            inv_cadence, "não-canônica"
        )
        print(
            f"   - Cadência detectada: {cadence_label} ({inv_cadence} dia(s)), "
            f"{n_inv_after} período(s) de investimento "
            f"({n_inv_before - n_inv_after} parcial(is) descartado(s)), "
            f"{n_kpi_after} período(s) de performance "
            f"({n_kpi_before - n_kpi_after} parcial(is) descartado(s))."
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
        ) from e
    except Exception as e:
        raise Exception(f"An unexpected error occurred during data preparation: {e}") from e
