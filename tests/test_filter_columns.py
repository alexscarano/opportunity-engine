# -*- coding: utf-8 -*-
"""
Testa as colunas calculadas pelos filtros da sidebar de Análise de Elasticidade
e a lógica de filtragem resultante.
"""
import numpy as np
import pandas as pd
import pytest

DAYS_IN_MONTH = 30


def build_df(rows=5):
    """DataFrame sintético com colunas base que o Streamlit recebe do engine."""
    daily_inv = np.linspace(1000, 5000, rows)
    proj_kpis = np.linspace(100, 300, rows)
    inc_kpi = np.maximum(0, proj_kpis - proj_kpis[0])
    inc_inv = np.maximum(0, daily_inv - daily_inv[0])
    proj_rev = proj_kpis * 1.5 * 50  # conversion_rate=1.5, avg_ticket=50
    inc_rev = np.maximum(0, proj_rev - proj_rev[0])

    df = pd.DataFrame(
        {
            "Daily_Investment": daily_inv,
            "Projected_Total_KPIs": proj_kpis,
            "Incremental_KPI": inc_kpi,
            "Incremental_Investment": inc_inv,
            "Projected_Revenue": proj_rev,
            "Incremental_Revenue": inc_rev,
        }
    )
    df["Monthly_Investment"] = df["Daily_Investment"] * DAYS_IN_MONTH
    df["Monthly_KPI"] = df["Projected_Total_KPIs"] * DAYS_IN_MONTH
    return df


def apply_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replica exatamente o bloco de cálculo do streamlit_app.py."""
    df = df.copy()
    df["CPA"] = df["Daily_Investment"] / df["Projected_Total_KPIs"]
    df["CPA"] = df["CPA"].replace([np.inf, -np.inf], float("nan"))

    df["iCPA"] = df["Incremental_Investment"] / df["Incremental_KPI"]
    df["iCPA"] = df["iCPA"].replace([np.inf, -np.inf], float("nan")).fillna(0)

    if "Projected_Revenue" in df.columns:
        df["ROAS"] = (df["Projected_Revenue"] / df["Monthly_Investment"]).replace(
            [np.inf, -np.inf], float("nan")
        )
        df["iROAS"] = (
            df["Incremental_Revenue"] / df["Incremental_Investment"]
        ).replace([np.inf, -np.inf], float("nan")).fillna(0)

    df["Pct_Incrementality"] = np.where(
        df["Projected_Total_KPIs"] > 0,
        df["Incremental_KPI"] / df["Projected_Total_KPIs"] * 100,
        0.0,
    )
    return df


# ---------------------------------------------------------------------------
# Testes de cálculo de colunas
# ---------------------------------------------------------------------------

class TestFilterColumnCalculation:

    def setup_method(self):
        self.df = apply_filter_columns(build_df())

    def test_cpa_is_investment_over_kpi(self):
        expected = self.df["Daily_Investment"] / self.df["Projected_Total_KPIs"]
        pd.testing.assert_series_equal(self.df["CPA"], expected, check_names=False)

    def test_cpa_no_inf_or_negative_inf(self):
        assert not np.isinf(self.df["CPA"].dropna()).any()

    def test_icpa_first_row_is_zero(self):
        # Incremental_Investment[0] == 0, so iCPA must be 0 (not inf/nan)
        assert self.df["iCPA"].iloc[0] == 0.0

    def test_icpa_no_inf(self):
        assert not np.isinf(self.df["iCPA"]).any()

    def test_roas_present_when_projected_revenue_exists(self):
        assert "ROAS" in self.df.columns

    def test_roas_equals_revenue_over_monthly_investment(self):
        expected = self.df["Projected_Revenue"] / self.df["Monthly_Investment"]
        pd.testing.assert_series_equal(self.df["ROAS"], expected, check_names=False)

    def test_iroas_first_row_is_zero(self):
        assert self.df["iROAS"].iloc[0] == 0.0

    def test_iroas_no_inf(self):
        assert not np.isinf(self.df["iROAS"]).any()

    def test_pct_incrementality_range(self):
        assert (self.df["Pct_Incrementality"] >= 0).all()
        assert (self.df["Pct_Incrementality"] <= 100).all()

    def test_pct_incrementality_first_row_is_zero(self):
        # baseline row has Incremental_KPI == 0
        assert self.df["Pct_Incrementality"].iloc[0] == 0.0

    def test_pct_incrementality_last_row_is_positive(self):
        assert self.df["Pct_Incrementality"].iloc[-1] > 0.0


# ---------------------------------------------------------------------------
# Testes de lógica de filtragem
# ---------------------------------------------------------------------------

class TestFilterLogic:

    def setup_method(self):
        self.df = apply_filter_columns(build_df(rows=10))

    def _apply_filters(self, min_budget=None, max_budget=None, target_cpa=None,
                       target_icpa=None, min_roas=None, min_iroas=None,
                       min_kpi=None, min_pct_incrementality=None):
        df = self.df
        lo = min_budget if min_budget is not None else 0
        hi = max_budget if max_budget is not None else float("inf")
        fdf = df[(df["Monthly_Investment"] >= lo) & (df["Monthly_Investment"] <= hi)]
        if target_cpa is not None:
            fdf = fdf[fdf["CPA"] <= target_cpa]
        if target_icpa is not None:
            fdf = fdf[fdf["iCPA"] <= target_icpa]
        if min_roas is not None and "ROAS" in fdf.columns:
            fdf = fdf[fdf["ROAS"] >= min_roas]
        if min_iroas is not None and "iROAS" in fdf.columns:
            fdf = fdf[fdf["iROAS"] >= min_iroas]
        if min_kpi is not None:
            fdf = fdf[fdf["Monthly_KPI"] >= min_kpi]
        if min_pct_incrementality is not None:
            fdf = fdf[fdf["Pct_Incrementality"] >= min_pct_incrementality]
        return fdf

    def test_no_filters_returns_all_rows(self):
        assert len(self._apply_filters()) == len(self.df)

    def test_max_budget_below_minimum_returns_empty(self):
        result = self._apply_filters(max_budget=1.0)  # 1 real < min monthly spend
        assert result.empty

    def test_min_budget_above_maximum_returns_empty(self):
        max_monthly = float(self.df["Monthly_Investment"].max())
        result = self._apply_filters(min_budget=max_monthly + 1)
        assert result.empty

    def test_budget_range_restricts_rows(self):
        mid = float(self.df["Monthly_Investment"].median())
        full = self._apply_filters()
        restricted = self._apply_filters(max_budget=mid)
        assert len(restricted) < len(full)

    def test_very_low_cpa_target_returns_empty(self):
        result = self._apply_filters(target_cpa=0.0)
        assert result.empty

    def test_very_low_icpa_keeps_first_row(self):
        # First row has iCPA == 0, so filter iCPA <= 0 should keep it
        result = self._apply_filters(target_icpa=0.0)
        assert len(result) >= 1
        assert result["iCPA"].iloc[0] == 0.0

    def test_very_high_roas_returns_empty(self):
        max_roas = float(self.df["ROAS"].max()) * 10
        result = self._apply_filters(min_roas=max_roas)
        assert result.empty

    def test_min_roas_zero_returns_all(self):
        result = self._apply_filters(min_roas=0.0)
        assert len(result) == len(self.df)

    def test_min_kpi_zero_returns_all(self):
        result = self._apply_filters(min_kpi=0.0)
        assert len(result) == len(self.df)

    def test_min_kpi_above_max_returns_empty(self):
        max_kpi = float(self.df["Monthly_KPI"].max())
        result = self._apply_filters(min_kpi=max_kpi + 1)
        assert result.empty

    def test_100pct_incrementality_returns_empty(self):
        # No row can have 100% incrementality because baseline row is included
        result = self._apply_filters(min_pct_incrementality=100.0)
        assert result.empty

    def test_zero_pct_incrementality_returns_all(self):
        result = self._apply_filters(min_pct_incrementality=0.0)
        assert len(result) == len(self.df)

    def test_roas_absent_when_no_revenue_column(self):
        """Garante que ROAS e iROAS não são calculados sem Projected_Revenue."""
        df_no_rev = build_df()
        df_no_rev = df_no_rev.drop(
            columns=["Projected_Revenue", "Incremental_Revenue"]
        )
        df_processed = apply_filter_columns(df_no_rev)
        assert "ROAS" not in df_processed.columns
        assert "iROAS" not in df_processed.columns
