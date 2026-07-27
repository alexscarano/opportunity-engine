# -*- coding: utf-8 -*-
import sys
import os

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from analysis import (
    run_opportunity_projection,
    create_calendar_features,
    find_events,
    periods_to_days,
    _train_response_model,
)


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


# --- Fase 3: cadence-aware calendar features, event windows, and find_events ---


def test_create_calendar_features_daily_generates_full_feature_set():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({"x": range(30)}, index=dates)

    out = create_calendar_features(df, period_days=1)

    for col in ["dayofweek", "is_weekend", "is_payday_period", "is_holiday", "month"] + [
        f"day_{i}" for i in range(7)
    ]:
        assert col in out.columns, f"expected '{col}' for daily data (period_days=1)"


def test_create_calendar_features_weekly_generates_only_month():
    dates = pd.date_range("2024-01-01", periods=30, freq="7D")
    df = pd.DataFrame({"x": range(30)}, index=dates)

    out = create_calendar_features(df, period_days=7)

    assert "month" in out.columns
    degenerate_cols = ["dayofweek", "is_weekend", "is_payday_period", "is_holiday"] + [
        f"day_{i}" for i in range(7)
    ]
    for col in degenerate_cols:
        assert col not in out.columns, (
            f"'{col}' would be constant/degenerate on weekly-cadence data "
            "and should be skipped"
        )


def test_train_response_model_weekly_data_does_not_crash_without_calendar_dummies():
    """
    Downstream safety check: _train_response_model builds a baseline_features
    list via `col.startswith("day_") ... or col in [...]` over
    model_data_featured.columns -- a list comprehension, so it should degrade
    gracefully (to just an intercept, in the worst case) rather than KeyError
    when the day-of-week/payday/holiday columns are entirely absent (weekly
    cadence, period_days=7).
    """
    dates = pd.date_range("2024-01-01", periods=60, freq="7D")
    rng = np.random.default_rng(1)
    investment = 100 + rng.normal(0, 10, 60)
    investment = np.clip(investment, 10, None)
    kpi = 50 + 0.5 * investment + rng.normal(0, 2, 60)
    model_data = pd.DataFrame({"ChannelA": investment, "kpi": kpi}, index=dates)

    config = {"period_days": 7, "country_code": "BR"}

    result = _train_response_model(model_data, "ChannelA", config)

    model_params = result[6]
    assert isinstance(model_params, dict)
    assert "alpha" in model_params


def test_periods_to_days_floor_applied_when_user_days_yield_too_few_periods():
    # post_event_days=14, period_days=7 -> raw 2 periods, below the min_periods=4
    # floor -- the floor should kick in (period_days*4 = 28 days).
    day_span, periods = periods_to_days(14, 7, 4)
    assert periods == 4
    assert day_span == 28


def test_periods_to_days_floor_not_applied_when_user_days_already_enough():
    # post_event_days=60, period_days=7 -> ceil(60/7)=9 periods, already above
    # the floor of 4 -- the user's own config should be respected as-is.
    day_span, periods = periods_to_days(60, 7, 4)
    assert periods == 9
    assert day_span == 63  # 9 * 7


def test_periods_to_days_daily_cadence_matches_configured_days():
    # period_days=1: periods should equal the configured days exactly (no
    # conversion needed), matching the pre-Fase-3 daily behavior.
    day_span, periods = periods_to_days(14, 1, 8)
    assert periods == 14
    assert day_span == 14


def test_periods_to_days_monthly_cadence_floor():
    # min_pre_period_days=14, period_days=30 -> ceil(14/30)=1 period, below
    # the min_periods=8 floor -- floor applied: 8 periods * 30 days = 240 days.
    day_span, periods = periods_to_days(14, 30, 8)
    assert periods == 8
    assert day_span == 240


def test_find_events_weekly_data_avoids_rebucketing_merge_artifact():
    """
    Regression/behavior test: with native "weekly" reporting dates that have
    small jitter (5-8 days apart, the kind detect_cadence still calls
    "semanal"), two native dates can collide into the SAME to_period("W-MON")
    bucket. Re-bucketing (the pre-Fase-3, period_days=1 default behavior)
    then SUMS those two native periods' investment into one row -- which,
    compared against the per-period historical average, fabricates a
    spurious "+100% change" event even though every native period in this
    series is exactly at the flat baseline (no real change at all).

    period_days=7 should skip that re-bucketing and use the native rows
    directly, correctly reporting NO event for this flat series.
    """
    base = pd.Timestamp("2023-09-05")
    # Indices 3 and 4 (5 and 8 days apart respectively) collide into the same
    # W-MON bucket -- verified via to_period("W-MON") against this exact
    # sequence.
    deltas = [7, 7, 7, 5, 8] + [7] * 15
    dates = [base]
    for d in deltas:
        dates.append(dates[-1] + pd.Timedelta(days=d))
    dates = pd.to_datetime(dates)

    investment = np.full(len(dates), 1000.0)
    df = pd.DataFrame(
        {"Date": dates, "Product Group": "TestChannel", "investment": investment}
    )

    old_events, _, _ = find_events(df.copy(), "TestCo", 1.5, 0.5, 14)
    assert not old_events.empty, (
        "sanity check: the W-MON re-bucketing merge should fabricate a "
        "spurious event under the pre-Fase-3 (period_days=1 default) behavior"
    )

    new_events, _, _ = find_events(df.copy(), "TestCo", 1.5, 0.5, 14, period_days=7)
    assert new_events.empty, (
        "period_days=7 should skip the re-bucketing and correctly find no "
        "event in this flat (no real change) weekly series"
    )


def test_find_events_weekly_data_still_detects_a_real_spike():
    """Companion to the merge-artifact test: period_days=7 must still detect
    a genuine, single-period spike -- the fix should suppress the merge
    artifact, not make find_events blind to real weekly changes."""
    dates = pd.date_range("2024-01-01", periods=20, freq="7D")
    investment = np.full(20, 1000.0)
    investment[10] = 3000.0  # unambiguous +200% spike on a single native week

    df = pd.DataFrame(
        {"Date": dates, "Product Group": "TestChannel", "investment": investment}
    )

    event_map_df, _, _ = find_events(df, "TestCo", 1.5, 0.5, 14, period_days=7)

    assert not event_map_df.empty
    assert event_map_df["percentage_change"].abs().max() > 100
