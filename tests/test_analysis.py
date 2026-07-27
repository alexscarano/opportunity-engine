# -*- coding: utf-8 -*-
import sys
import os

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from analysis import run_opportunity_projection


def _make_synthetic_data(n_days=150, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    investment = 100 + rng.normal(0, 15, n_days)
    investment = np.clip(investment, 10, None)

    # KPI driven by a baseline + a (roughly) saturating response to investment,
    # so the response model has real signal to fit instead of pure noise.
    kpi = 50 + 0.8 * np.sqrt(investment) * 10 + rng.normal(0, 2, n_days)

    daily_investment_df = pd.DataFrame(
        {
            "Date": dates,
            "Product Group": "TestChannel",
            "investment": investment,
        }
    )
    kpi_df = pd.DataFrame({"Date": dates, "kpi": kpi})
    market_trends_df = pd.DataFrame({"Date": dates, "Generic Searches": 50.0})

    return kpi_df, daily_investment_df, market_trends_df


def test_conversions_mode_with_target_roas_does_not_crash_and_trains_model():
    """
    Regression for the Projected_Revenue KeyError: a config in CONVERSIONS mode
    (non-monetary KPI) with financial_targets.target_roas > 0 used to hit the
    ROAS filter block unconditionally, which reads 'Projected_Revenue' -- a
    column only created in revenue_mode. That raised a KeyError caught by the
    broad except in run_opportunity_projection, which then returned an empty
    model_params dict, later causing a bare KeyError: 'alpha' in
    run_causal_impact_analysis for every event.

    The fix: skip the ROAS filter entirely outside revenue_mode. This test
    asserts the function returns successfully with a populated model_params
    dict containing the 'alpha' key -- not a crash, not an empty dict.
    """
    kpi_df, daily_investment_df, market_trends_df = _make_synthetic_data()

    config = {
        "optimization_target": "CONVERSIONS",
        "kpi_is_monetary": False,
        "financial_targets": {"target_roas": 4.0, "target_iroas": 0},
        "average_ticket": 0,
        "conversion_rate_from_kpi_to_bo": 0,
        "investment_limit_factor": 2.0,
        "country_code": "BR",
    }

    (
        response_curve_df,
        scenarios_df,
        baseline_point,
        max_efficiency_point,
        diminishing_return_point,
        saturation_point,
        strategic_limit_point,
        model_params,
        channel_proportions,
    ) = run_opportunity_projection(
        kpi_df, daily_investment_df, market_trends_df, "TestChannel", config
    )

    assert not response_curve_df.empty
    assert isinstance(model_params, dict)
    assert "alpha" in model_params
    assert "k" in model_params
    assert "s" in model_params
