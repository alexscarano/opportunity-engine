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


def robust_date_parsing(series, date_format=None):
    """
    Attempt to parse dates using the provided format. If it fails or results in many NaTs,
    fall back to pandas automatic date inference.
    """
    original_non_nulls = series.notna().sum()
    if original_non_nulls == 0:
        return pd.to_datetime(series, errors="coerce")

    if date_format:
        parsed = pd.to_datetime(series, format=date_format, errors="coerce")
        if parsed.notna().sum() == original_non_nulls:
            return parsed
        print(
            f"   - WARNING: Configured date format '{date_format}' failed for some rows. Falling back to auto-detection."
        )

    # Fallback to general parsing if specific format fails
    try:
        # Attempt to let pandas infer the format
        parsed = pd.to_datetime(series, errors="coerce")

        # If still failing, try with format='mixed' (Pandas 2.0+)
        if parsed.notna().sum() < original_non_nulls:
            try:
                parsed = pd.to_datetime(series, format="mixed", errors="coerce")
            except:
                pass
        return parsed
    except Exception as e:
        print(f"   - ERROR: Date parsing failed completely: {e}")
        return pd.to_datetime(series, errors="coerce")


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
        kpi_df = pd.read_csv(config["performance_file_path"], thousands=",")
        daily_investment_df = pd.read_csv(config["investment_file_path"], thousands=",")

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
