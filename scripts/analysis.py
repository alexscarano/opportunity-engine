# -*- coding: utf-8 -*-
"""
This module contains all the core data processing and analysis functions,
including event detection, causal impact modeling, and opportunity forecasting.
"""

import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import curve_fit, OptimizeWarning
from datetime import timedelta
import warnings
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import holidays
from scipy.signal import lfilter
import os

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def hill_transform(x, k, s):
    """Applies the Hill function for saturation."""
    epsilon = 1e-9
    return (x**k) / (s**k + x**k + epsilon)


def geometric_decay(series, alpha):
    """Applies geometric decay for ad-stock using lfilter for efficiency."""
    return lfilter([1], [1, -alpha], series)


def create_calendar_features(df, country_code="BR", period_days=1):
    """Creates time-series features from a datetime index.

    Day-of-week/weekend/payday/holiday features only make sense on genuinely
    daily data. On weekly-or-coarser data every row falls on the same day of
    week (e.g. always Sunday), so these would be constant/degenerate columns
    that waste model capacity and could get picked up as spurious
    "predictors" by LassoCV purely by chance. `month` stays useful at any
    granularity coarser than daily, so it's always generated.
    """
    df_featured = df.copy()
    df_featured["month"] = df_featured.index.month

    if period_days == 1:
        df_featured["dayofweek"] = df_featured.index.dayofweek
        df_featured["is_weekend"] = (df_featured.index.dayofweek >= 5).astype(int)
        df_featured["is_payday_period"] = (
            (df_featured.index.day >= 1) & (df_featured.index.day <= 5)
            | (df_featured.index.day >= 15) & (df_featured.index.day <= 20)
        ).astype(int)

        try:
            country_holidays = holidays.CountryHoliday(country_code)
            df_featured["is_holiday"] = df_featured.index.map(
                lambda date: date in country_holidays
            ).astype(int)
        except Exception:
            df_featured["is_holiday"] = 0

        for i in range(7):
            df_featured[f"day_{i}"] = (df_featured["dayofweek"] == i).astype(int)

    return df_featured


def periods_to_days(days_config, period_days, min_periods, label=""):
    """
    Converts a day-based window config (e.g. post_event_days=14) into a
    period-aware day-span and period count, given the detected reporting
    cadence `period_days` (days per period, from Fase 2's cadence detection).

    A day-count window is only meaningful in "rows of data" terms once
    divided by the cadence -- 14 days is 14 rows of daily data but only 2
    rows of weekly data, which is too little for the causal-impact modeling
    downstream. This enforces a floor of `min_periods` reporting periods,
    only overriding the user's configured days when they'd otherwise yield
    FEWER periods than that floor (and logs when the floor kicks in).

    Returns (day_span, periods): `day_span` is the number of days to use in
    date-range arithmetic; `periods` is the resulting period count.
    """
    period_days = period_days if period_days else 1
    raw_periods = days_config / period_days
    ceiled_periods = math.ceil(raw_periods)
    periods = max(min_periods, ceiled_periods)
    if periods > ceiled_periods:
        print(
            f"   - AVISO: janela '{label}' configurada em {days_config} dia(s) "
            f"renderia menos que o mínimo de {min_periods} período(s) de dados "
            f"(cadência detectada: {period_days} dia(s)/período). Mínimo de "
            f"{periods} período(s) ({periods * period_days} dias) aplicado."
        )
    return periods * period_days, periods


def find_events(
    df,
    company_name,
    increase_ratio,
    decrease_ratio,
    post_event_days,
    pre_selection_pool_size=30,
    output_dir=None,
    period_days=1,
):
    """
    Finds all significant investment changes, groups overlapping events into a single
    consolidated event per week, and returns the final map.
    """
    print(
        "\n"
        + "=" * 50
        + "\nStarting Comprehensive Event Detection & Grouping...\n"
        + "=" * 50
    )
    try:
        df = df.rename(columns={"Product Group": "product_group", "Date": "dates"})

        all_individual_events = []
        product_groups = df["product_group"].unique()
        print(
            f"   - Analyzing {len(product_groups)} unique ad products for significant changes."
        )

        for product in product_groups:
            product_df = df[df["product_group"] == product].copy()
            if len(product_df) < 14:
                continue

            product_df["dates"] = pd.to_datetime(product_df["dates"])
            if period_days >= 7:
                # Data already reports at weekly-or-coarser cadence -- re-bucketing
                # into ISO "W-MON" weeks can shift native periods across the wrong
                # boundary (e.g. Sunday-reported weekly rows landing in a different
                # Monday-start week than the one they actually belong to). Use the
                # native per-row dates directly as the per-period series instead.
                product_df["weeks"] = product_df["dates"]
            else:
                product_df["weeks"] = product_df["dates"].dt.to_period(
                    "W-MON"
                ).dt.start_time
            weekly_investment_df = (
                product_df.groupby("weeks")["investment"].sum().reset_index()
            )
            if len(weekly_investment_df) < 3:
                continue

            weekly_investment_df["historical_avg"] = (
                weekly_investment_df["investment"]
                .rolling(window=12, min_periods=1)
                .mean()
                .shift(1)
            )
            weekly_investment_df.dropna(subset=["historical_avg"], inplace=True)
            weekly_investment_df = weekly_investment_df[
                weekly_investment_df["historical_avg"] > 0
            ]
            if weekly_investment_df.empty:
                continue

            weekly_investment_df["change_ratio"] = (
                weekly_investment_df["investment"]
                / weekly_investment_df["historical_avg"]
            )
            weekly_investment_df["percentage_change"] = (
                weekly_investment_df["change_ratio"] - 1
            ) * 100

            significant_changes = weekly_investment_df[
                (weekly_investment_df["change_ratio"] > increase_ratio)
                | (weekly_investment_df["change_ratio"] < decrease_ratio)
            ]

            for _, row in significant_changes.iterrows():
                all_individual_events.append(
                    {
                        "date": row["weeks"],
                        "ad_product": product,
                        "percentage_change": round(row["percentage_change"], 2),
                    }
                )

        if not all_individual_events:
            print("   - No significant individual events found across any ad products.")
            return pd.DataFrame(), None, None

        individual_events_df = pd.DataFrame(all_individual_events)
        individual_events_df["date"] = pd.to_datetime(individual_events_df["date"])
        individual_events_df["event_week"] = (
            individual_events_df["date"].dt.to_period("W").dt.start_time
        )

        # Consolidate events that occur in the same week
        consolidated_events = []
        for week, group in individual_events_df.groupby("event_week"):
            channels = sorted(list(group["ad_product"].unique()))
            consolidated_name = ", ".join(channels)
            max_change = group.loc[group["percentage_change"].abs().idxmax()][
                "percentage_change"
            ]

            consolidated_events.append(
                {
                    "date": week.strftime("%Y-%m-%d"),
                    "ad_product": consolidated_name,
                    "percentage_change": max_change,
                }
            )

        if not consolidated_events:
            return pd.DataFrame(), None, None

        event_map_df = pd.DataFrame(consolidated_events).sort_values(by="date")

        # --- New Code: Save detected events to CSV in the brand's output directory ---
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_filepath = os.path.join(output_dir, "detected_events.csv")
            event_map_df.to_csv(output_filepath, index=False)
            print(f"   - Detected events saved to: {output_filepath}")
        # --- End New Code ---

        return event_map_df, None, None

    except (KeyError, TypeError) as e:
        print(
            f"Error processing data for event detection. Please check your input file columns. Details: {e}"
        )
        return pd.DataFrame(), None, None


def run_causal_impact_analysis(
    kpi_df,
    daily_investment_df,
    market_trends_df,
    performance_df,
    pre_period,
    post_period,
    event_name,
    product_group,
    model_params,
    config,
):
    try:
        data = kpi_df.copy()
        data["Date"] = pd.to_datetime(data["Date"])
        data["kpi"] = pd.to_numeric(data["kpi"], errors="coerce").dropna()
        if data["kpi"].empty:
            raise ValueError("The 'kpi' column contains no valid numeric data.")

        investment_pivot_df = daily_investment_df.pivot_table(
            index="Date", columns="Product Group", values="investment"
        ).fillna(0)

        model_data = pd.merge(
            data[["Date", "kpi"]], investment_pivot_df, on="Date", how="inner"
        )
        model_data = pd.merge(model_data, market_trends_df, on="Date", how="left")

        # --- Start of New Dynamic Code ---

        # Rename columns for consistency
        perf_df_renamed = performance_df.rename(
            columns={
                "date": "Date",
                "leads": "Leads",
                "Impressions": "Impressions",
                "Clicks": "Clicks",
            }
        )

        # Define potential covariates and find which ones actually exist in the data
        possible_covariates = ["Leads", "Clicks", "Impressions"]
        found_covariates = [
            col for col in possible_covariates if col in perf_df_renamed.columns
        ]

        if found_covariates:
            print(
                f"   - Found additional covariates in performance data: {found_covariates}"
            )
            for col in found_covariates:
                if perf_df_renamed[col].dtype == "object":
                    perf_df_renamed[col] = perf_df_renamed[col].str.replace(
                        ".", "", regex=False
                    )
                perf_df_renamed[col] = pd.to_numeric(
                    perf_df_renamed[col], errors="coerce"
                )

            perf_df_renamed["Date"] = pd.to_datetime(
                perf_df_renamed["Date"], errors="coerce"
            )
            perf_df_renamed.dropna(subset=["Date"] + found_covariates, inplace=True)

            # Merge only the found and cleaned covariates
            model_data = pd.merge(
                model_data,
                perf_df_renamed[["Date"] + found_covariates],
                on="Date",
                how="left",
            )
        else:
            print(
                "   - No additional performance covariates found. Proceeding without them."
            )

        # --- End of New Dynamic Code ---

        model_data.set_index("Date", inplace=True)
        model_data.sort_index(inplace=True)
        model_data.fillna(0, inplace=True)

        country_code = config.get("country_code", "BR")
        period_days = config.get("period_days", 1)
        model_data = create_calendar_features(
            model_data, country_code=country_code, period_days=period_days
        )

        event_channels = [ch.strip() for ch in product_group.split(",")]
        event_investment_series = model_data[event_channels].sum(axis=1)

        # Use pre-trained model parameters. An empty model_params means
        # run_opportunity_projection failed to train the response model (see its
        # own except block for the root cause) -- surface that explicitly instead
        # of a bare KeyError: 'alpha'.
        if not model_params or "alpha" not in model_params:
            raise ValueError(
                "modelo de projecao nao treinou (model_params vazio) -- analise "
                "causal abortada. Verifique o log de run_opportunity_projection "
                "para a causa raiz."
            )
        best_alpha = model_params["alpha"]
        best_k = model_params["k"]
        best_s = model_params["s"]
        print(
            f"   - Using pre-trained ad-stock alpha for causal model: {best_alpha:.2f}"
        )
        print(
            f"   - Using pre-trained saturation params for causal model: k={best_k:.2f}, s={best_s:.2f}"
        )

        transformed_features = {}
        for channel in investment_pivot_df.columns:
            if channel in model_data.columns:
                adstocked_channel = geometric_decay(model_data[channel], best_alpha)
                saturated_channel = hill_transform(adstocked_channel, best_k, best_s)
                transformed_features[f"{channel.replace(' ', '_')}_transformed"] = (
                    saturated_channel
                )

        transformed_df = pd.DataFrame(transformed_features, index=model_data.index)
        model_data = pd.concat([model_data, transformed_df], axis=1)

        pre_data_for_model = model_data.loc[pre_period[0] : pre_period[1]].copy()
        post_data = model_data.loc[post_period[0] : post_period[1]].copy()

        min_pre_period_days = config.get("min_pre_period_days", 14)  # Require at least configurable days of pre-period data (default 14)
        # Convert the day-based config into a period-aware row-count floor: with
        # weekly/monthly data, 14 "days" is only 2/0.5 rows -- min_pre_period_days
        # compared directly against a row count conflates "days" with "rows",
        # which is only correct for daily data.
        _, min_pre_periods = periods_to_days(
            min_pre_period_days, period_days, 8, label="min_pre_period_days"
        )
        if (
            pre_data_for_model.empty
            or post_data.empty
            or len(pre_data_for_model) < min_pre_periods
        ):
            print(
                f"   - SKIPPED: Insufficient data for causal impact analysis. Pre-period data points: {len(pre_data_for_model)} (min required: {min_pre_periods} período(s), cadência={period_days} dia(s)/período)."
            )
            return None, None, None, None, None

        print("   - Running automated feature selection for causal model...")
        potential_exog_vars = list(transformed_features.keys()) + [
            col
            for col in model_data.columns
            if "generic" in col.lower()
            or "day_" in col
            or col in ["is_weekend", "is_payday_period", "is_holiday"]
        ]

        # --- Add the dynamically found covariates to the list ---
        if found_covariates:
            potential_exog_vars.extend(found_covariates)

        selector = VarianceThreshold(threshold=0.01)
        selector.fit(pre_data_for_model[potential_exog_vars])
        varianced_features = [
            var
            for i, var in enumerate(potential_exog_vars)
            if selector.get_support()[i]
        ]

        if not varianced_features:
            selected_features = list(transformed_features.keys())
        else:
            # Scale features for LassoCV to handle different scales (e.g. Trends vs transformed investment)
            scaler_lasso = MinMaxScaler()
            X_lasso_scaled = scaler_lasso.fit_transform(
                pre_data_for_model[varianced_features]
            )

            lasso = LassoCV(cv=5, random_state=0, max_iter=10000).fit(
                X_lasso_scaled, pre_data_for_model["kpi"]
            )
            selected_features = [
                var for i, var in enumerate(varianced_features) if lasso.coef_[i] != 0
            ]
            if not selected_features:
                selected_features = list(transformed_features.keys())

        print(f"   - Selected features for causal model: {selected_features}")
        model = sm.tsa.UnobservedComponents(
            pre_data_for_model["kpi"],
            "llevel",
            trend=True,
            seasonal=7,
            exog=pre_data_for_model[selected_features],
        ).fit(disp=False)

        in_sample_preds = model.get_prediction(
            exog=pre_data_for_model[selected_features]
        ).predicted_mean
        mae = np.mean(np.abs(pre_data_for_model["kpi"] - in_sample_preds))
        mape = (
            (mae / pre_data_for_model["kpi"].mean()) * 100
            if pre_data_for_model["kpi"].mean() > 0
            else 0
        )
        ss_res = np.sum((pre_data_for_model["kpi"] - in_sample_preds) ** 2)
        ss_tot = np.sum(
            (pre_data_for_model["kpi"] - np.mean(pre_data_for_model["kpi"])) ** 2
        )
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        print(f"   - Causal model R-squared (in-sample): {r_squared:.4f}")

        accuracy_df = pre_data_for_model.copy()
        accuracy_df["Predicted"] = in_sample_preds
        accuracy_df = accuracy_df[["kpi", "Predicted"]].reset_index().tail(90)

        forecast = model.get_forecast(
            steps=len(post_data), exog=post_data[selected_features]
        )
        predicted_mean = forecast.predicted_mean

        impact_df = post_data[["kpi"]].copy()
        impact_df["predicted"] = predicted_mean.values
        impact_df["impact"] = impact_df["kpi"] - impact_df["predicted"]
        impact_df.fillna(0, inplace=True)

        abs_lift = impact_df["impact"].sum()
        rel_lift = (
            (abs_lift / predicted_mean.sum()) * 100 if predicted_mean.sum() != 0 else 0
        )

        # Significance from the state-space model's OWN forecast uncertainty,
        # not a one-sample t-test on the daily impacts. The daily impacts are
        # autocorrelated forecast residuals whose error grows with the horizon;
        # a t-test treats the ~14 days as i.i.d. samples of a mean, inflating the
        # effective N and understating the p-value (false significance). Here we
        # test the CUMULATIVE impact against the forecast's cumulative standard
        # error (se_mean grows with horizon), which is the counterfactual's real
        # noise band. Cumulative SE ~ sqrt(sum se_t^2) treats the per-step errors
        # as independent -- an approximation, but far sounder than the t-test.
        se = np.asarray(forecast.se_mean, dtype=float)
        se = se[np.isfinite(se)]
        cum_se = float(np.sqrt(np.sum(se**2))) if se.size else 0.0
        if cum_se > 0:
            z_stat = abs_lift / cum_se
            p_value = float(2 * stats.norm.sf(abs(z_stat)))
            impact_ci_lower = abs_lift - 1.96 * cum_se
            impact_ci_upper = abs_lift + 1.96 * cum_se
        else:
            p_value = 1.0
            impact_ci_lower = impact_ci_upper = abs_lift

        event_investment_agg = event_investment_series.reset_index().rename(
            columns={0: "investment"}
        )
        event_investment_agg["Date"] = pd.to_datetime(event_investment_agg["Date"])

        # Calculate historical average investment up to the pre-period
        pre_period_end_date = pd.to_datetime(pre_period[1])
        historical_investment = event_investment_agg[
            event_investment_agg["Date"] <= pre_period_end_date
        ].copy()

        # Weekly aggregation for historical average calculation -- but only when
        # the data is genuinely daily. When it's already weekly-or-coarser
        # (period_days >= 7), re-bucketing via to_period("W") has the same
        # boundary-misalignment risk as in find_events: use the native per-row
        # dates directly so "historical_avg" is a per-native-period average.
        if period_days >= 7:
            historical_investment["weeks"] = historical_investment["Date"]
        else:
            historical_investment["weeks"] = (
                historical_investment["Date"].dt.to_period("W").dt.start_time
            )
        weekly_investment_df = (
            historical_investment.groupby("weeks")["investment"].sum().reset_index()
        )

        # Calculate expanding average and get the last value for the pre-event period
        weekly_investment_df["historical_avg"] = (
            weekly_investment_df["investment"]
            .rolling(window=12, min_periods=1)
            .mean()
            .shift(1)
        )

        if not weekly_investment_df.empty and pd.notna(
            weekly_investment_df["historical_avg"].iloc[-1]
        ):
            # post_data rows are native periods. When historical_avg is a
            # per-week average (period_days=1, re-bucketed above), the post
            # period spans len(post_data)/7 weeks. When historical_avg is
            # already per-native-period (period_days>=7, no re-bucketing),
            # post_data's rows ARE the periods, so no /7 conversion applies.
            periods_in_post = (
                len(post_data) if period_days >= 7 else len(post_data) / 7
            )
            total_investment_pre_period = (
                weekly_investment_df["historical_avg"].iloc[-1] * periods_in_post
            )
        else:
            # Fallback to like-for-like if historical average is not available
            like_for_like_pre_period_end = pd.to_datetime(post_period[0]) - timedelta(
                days=1
            )
            like_for_like_pre_period_start = like_for_like_pre_period_end - timedelta(
                days=len(post_data) - 1
            )
            total_investment_pre_period = event_investment_agg[
                (event_investment_agg["Date"] >= like_for_like_pre_period_start)
                & (event_investment_agg["Date"] <= like_for_like_pre_period_end)
            ]["investment"].sum()

        total_investment_post_period = event_investment_agg[
            event_investment_agg["Date"].isin(post_data.index)
        ]["investment"].sum()
        event_period_avg_investment = event_investment_agg[
            event_investment_agg["Date"].isin(post_data.index)
        ]["investment"].mean()

        investment_change_pct = (
            (
                (total_investment_post_period - total_investment_pre_period)
                / total_investment_pre_period
            )
            * 100
            if total_investment_pre_period > 0
            else 0
        )
        cpa_incremental = (
            total_investment_post_period / abs_lift if abs_lift != 0 else 0
        )

        start_date = pd.to_datetime(post_period[0])
        end_date = pd.to_datetime(post_period[1])
        chart_start = start_date - pd.to_timedelta(start_date.dayofweek + 7, unit="d")
        chart_end = end_date + pd.to_timedelta(6 - end_date.dayofweek + 7, unit="d")

        actuals_for_chart = pd.merge(
            model_data.loc[chart_start:chart_end, ["kpi"]].reset_index(),
            event_investment_agg[
                (event_investment_agg["Date"] >= chart_start)
                & (event_investment_agg["Date"] <= chart_end)
            ],
            on="Date",
            how="left",
        ).fillna(0)
        actuals_for_chart.rename(
            columns={"investment": "Total_Investment"}, inplace=True
        )

        forecast_for_chart = pd.DataFrame(
            {"Date": post_data.index, "Forecasted_KPI": predicted_mean.values}
        )
        line_chart_df = pd.merge(
            actuals_for_chart, forecast_for_chart, on="Date", how="left"
        ).rename(columns={"kpi": "Actual_KPI", "Total_Investment": "Investment"})

        investment_bar_df = pd.DataFrame(
            {
                "Period": ["Pre-Event", "Event"],
                "Investment": [
                    total_investment_pre_period,
                    total_investment_post_period,
                ],
            }
        ).set_index("Period")
        sessions_bar_df = pd.DataFrame(
            {
                "Category": ["Forecasted", "Actual"],
                "kpi": [predicted_mean.sum(), post_data["kpi"].sum()],
            }
        ).set_index("Category")

        results_dict = {
            "start_date": post_period[0],
            "end_date": post_period[1],
            "product_group": product_group,
            "absolute_lift": abs_lift,
            "absolute_lift_ci_lower": impact_ci_lower,
            "absolute_lift_ci_upper": impact_ci_upper,
            "relative_lift_pct": rel_lift,
            "p_value": p_value,
            "investment_change_pct": investment_change_pct,
            "cpa_incremental": cpa_incremental,
            "total_investment_post_period": total_investment_post_period,
            "total_investment_pre_period": total_investment_pre_period,
            "mae": mae,
            "mape": mape,
            "model_r_squared": r_squared,
            "event_period_avg_investment": event_period_avg_investment,
        }

        return (
            results_dict,
            line_chart_df,
            investment_bar_df,
            sessions_bar_df,
            accuracy_df,
        )
    except Exception as e:
        print(f"An unexpected error occurred during causal impact analysis: {e}")
        return None, None, None, None, None


def _train_response_model(model_data, product_group, config):
    """
    Trains the ad-stock and saturation model on INCREMENTAL KPIs by first
    modeling and subtracting the baseline performance.
    """
    print(
        "\n"
        + "=" * 50
        + "\nTraining Response Model for Projection on INCREMENTAL Data...\n"
        + "=" * 50
    )

    event_channels = [ch.strip() for ch in product_group.split(",")]
    valid_event_channels = [ch for ch in event_channels if ch in model_data.columns]
    if not valid_event_channels:
        raise ValueError(
            f"None of the event channels {event_channels} were found in the model data columns."
        )

    # --- 1. Model and Subtract Baseline KPI ---
    print("   - Modeling baseline KPI using non-investment features...")
    country_code = config.get("country_code", "BR")
    period_days = config.get("period_days", 1)
    model_data_featured = create_calendar_features(
        model_data.copy(), country_code=country_code, period_days=period_days
    )

    # day_0 is the reference category and is_weekend is a linear combination of
    # day_5/day_6 -- including all of day_0..day_6 + is_weekend alongside the
    # add_constant() intercept is a perfect dummy-variable trap. Drop both so the
    # design matrix is full rank (fitted values are unchanged; this just removes
    # the singular collinearity that made individual coefficients meaningless).
    baseline_features = [
        col
        for col in model_data_featured.columns
        if (col.startswith("day_") and col != "day_0")
        or col in ["is_payday_period", "is_holiday", "Generic Searches"]
    ]
    X_base = model_data_featured[baseline_features]
    X_base = sm.add_constant(X_base)
    y_base = model_data_featured["kpi"]

    baseline_model = sm.OLS(y_base, X_base).fit()
    baseline_kpi_predictions = baseline_model.predict(X_base)

    # Signed residual, NOT clipped at 0 (see the same fix in
    # elasticity_analysis.run_mmm_engine): the OLS baseline has an intercept and
    # passes through the mean, so the residual is mean ~0. Clipping negatives kept
    # only the positive half of the noise and fed it to the saturation fit,
    # manufacturing an incremental response from noise. Fitting on the signed
    # residual reports the honest response, which can legitimately be near flat.
    model_data_featured["incremental_kpi"] = y_base - baseline_kpi_predictions
    print("   - Baseline subtracted. Modeling response on incremental KPI.")

    # --- 2. Model Investment vs. Incremental KPI ---
    full_period_investment = model_data_featured[valid_event_channels].sum(axis=1)
    incremental_kpi = model_data_featured["incremental_kpi"]

    possible_alphas = np.arange(0.1, 1.0, 0.1)
    correlations = {
        alpha: np.corrcoef(
            geometric_decay(full_period_investment, alpha), incremental_kpi
        )[0, 1]
        for alpha in possible_alphas
    }
    best_alpha = max(correlations, key=correlations.get)
    print(f"   - Best ad-stock alpha (incremental data): {best_alpha:.2f}")

    adstocked_investment = pd.Series(
        geometric_decay(full_period_investment, best_alpha),
        index=model_data_featured.index,
    )

    best_k, best_s = (1, 1)
    try:

        def hill_for_fit(x, k, s):
            # Scale to the max of the INCREMENTAL kpi
            return incremental_kpi.max() * hill_transform(x, k, s)

        median_investment = np.median(adstocked_investment[adstocked_investment > 0])
        if median_investment == 0 or np.isnan(median_investment):
            median_investment = 1

        max_adstocked_investment = adstocked_investment.max()
        # Hill k capped at 1.0 (parity with elasticity_analysis.run_mmm_engine).
        # k > 1 makes the response curve convex before its inflection -- marginal
        # ROI *increasing* with spend -- which is impossible for media. On real
        # data this path fit k=5.46 (accelerating returns), feeding both the
        # opportunity projection and the causal model (which reuses these params).
        # k <= 1 forces diminishing returns from the first dollar.
        bounds = ([0.1, 0], [1.0, max_adstocked_investment * 1.5])

        popt, _ = curve_fit(
            hill_for_fit,
            adstocked_investment,
            incremental_kpi,
            p0=[1.0, median_investment],
            bounds=bounds,
            maxfev=5000,
        )
        best_k, best_s = popt
        print(
            f"   - Best saturation params (incremental data): k={best_k:.2f}, s={best_s:.2f}"
        )
    except (RuntimeError, ValueError) as e:
        print(
            f"   - WARNING: Could not find optimal saturation curve for incremental data. Using defaults. Error: {e}"
        )

    # --- 3. Calculate Final Parameters ---
    # The scaler should now be based on the max of the incremental KPI
    max_kpi_scaler = incremental_kpi.max()

    # Base KPI is the average of the predicted baseline
    base_kpi_avg = baseline_kpi_predictions.mean()

    hist_avg_investment = (
        model_data_featured.iloc[-90:][valid_event_channels].sum(axis=1).mean()
    )

    channel_proportions = {}
    if hist_avg_investment > 0:
        for channel in valid_event_channels:
            channel_avg_investment = model_data_featured.iloc[-90:][channel].mean()
            channel_proportions[channel] = channel_avg_investment / hist_avg_investment
    else:
        equal_share = 1 / len(valid_event_channels) if valid_event_channels else 0
        for channel in valid_event_channels:
            channel_proportions[channel] = equal_share
    print(f"   - Calculated channel investment proportions: {channel_proportions}")

    historical_adstocked_inv = hist_avg_investment / (1 - best_alpha)
    historical_saturated_response = hill_transform(
        historical_adstocked_inv, best_k, best_s
    )

    # Historical KPI is now the baseline + the modeled incremental portion
    hist_avg_kpi = base_kpi_avg + (historical_saturated_response * max_kpi_scaler)

    model_params = {
        "alpha": best_alpha,
        "k": best_k,
        "s": best_s,
        "scaler": max_kpi_scaler,
        "base_kpi": base_kpi_avg,
    }
    print("   - Incremental projection model training complete.")
    return (
        best_alpha,
        best_k,
        best_s,
        max_kpi_scaler,
        hist_avg_investment,
        hist_avg_kpi,
        model_params,
        channel_proportions,
    )


def run_opportunity_projection(
    kpi_df, daily_investment_df, market_trends_df, product_group, config
):
    """
    Trains a new response model on all data and then finds the optimal investment sweet spot.
    """
    print(
        "\n"
        + "=" * 50
        + "\nFinding the Investment Sweet Spot & Projecting Opportunity...\n"
        + "=" * 50
    )
    optimization_target = config.get("optimization_target", "REVENUE").upper()

    try:
        investment_pivot_df = (
            daily_investment_df.copy()
            .pivot_table(index="Date", columns="Product Group", values="investment")
            .fillna(0)
        )
        model_data = pd.merge(
            kpi_df[["Date", "kpi"]], investment_pivot_df, on="Date", how="inner"
        )
        model_data = pd.merge(model_data, market_trends_df, on="Date", how="left")
        model_data.set_index("Date", inplace=True)
        model_data.sort_index(inplace=True)
        model_data.fillna(0, inplace=True)

        (
            best_alpha,
            best_k,
            best_s,
            max_kpi_scaler,
            hist_avg_investment,
            hist_avg_kpi,
            model_params,
            channel_proportions,
        ) = _train_response_model(model_data, product_group, config)

        investment_limit_factor = config.get("investment_limit_factor", 2.0)
        max_hist_inv = daily_investment_df["investment"].max()
        investment_scenarios = np.linspace(
            1, max_hist_inv * investment_limit_factor, 200
        )

        base_kpi = model_params.get("base_kpi", 0)
        projected_kpis = [
            base_kpi
            + (
                (hill_transform((inv / (1 - best_alpha)), best_k, best_s))
                * max_kpi_scaler
            )
            for inv in investment_scenarios
        ]

        response_curve_df = pd.DataFrame(
            {
                "Daily_Investment": investment_scenarios,
                "Projected_Total_KPIs": projected_kpis,
            }
        )

        # Clip Projected_Total_KPIs to prevent negative values
        response_curve_df["Projected_Total_KPIs"] = response_curve_df[
            "Projected_Total_KPIs"
        ].clip(lower=0)

        kpi_is_monetary = config.get("kpi_is_monetary", False)
        # ponytail: when the KPI is already money, revenue = kpi (multiplier=1), no
        # count-to-revenue conversion applies, regardless of optimization_target.
        conversion_rate = 1.0 if kpi_is_monetary else config.get("conversion_rate_from_kpi_to_bo", 0)
        avg_ticket = 1.0 if kpi_is_monetary else config.get("average_ticket", 0)
        revenue_mode = kpi_is_monetary or optimization_target == "REVENUE"
        if kpi_is_monetary and optimization_target == "CONVERSIONS":
            print(
                "   - AVISO (config): optimization_target='CONVERSIONS' foi ignorado "
                "porque kpi_is_monetary=true (o KPI ja e R$), entao o modo receita/ROAS "
                "esta em uso. Se o KPI for uma contagem (unidades, nao R$), desmarque "
                "'KPI ja esta em R$' no setup; senao, defina o alvo como RECEITA."
            )

        baseline_point = {
            "Scenario": "Cenário Atual",
            "Daily_Investment": hist_avg_investment,
            "Projected_Total_KPIs": hist_avg_kpi,
            "Incremental_Investment": 0,
            "Incremental_KPI": 0,
            "Incremental_Revenue": 0,
            "Incremental_ROI": 0,
        }
        if revenue_mode:
            baseline_point["Projected_Revenue"] = (
                hist_avg_kpi * conversion_rate * avg_ticket
            )

        baseline_investment = baseline_point["Daily_Investment"]
        baseline_kpi = baseline_point["Projected_Total_KPIs"]

        response_curve_df["Incremental_Investment"] = (
            response_curve_df["Daily_Investment"] - baseline_investment
        )
        response_curve_df["Incremental_KPI"] = (
            response_curve_df["Projected_Total_KPIs"] - baseline_kpi
        )
        response_curve_df.loc[
            response_curve_df["Incremental_Investment"] < 0, "Incremental_Investment"
        ] = 0
        response_curve_df.loc[
            response_curve_df["Incremental_KPI"] < 0, "Incremental_KPI"
        ] = 0

        if revenue_mode:
            if avg_ticket <= 0:
                raise ValueError(
                    "'average_ticket' must be greater than 0 for a REVENUE-based optimization."
                )
            response_curve_df["Projected_Revenue"] = (
                response_curve_df["Projected_Total_KPIs"] * conversion_rate * avg_ticket
            )
            response_curve_df["Incremental_Revenue"] = response_curve_df[
                "Projected_Revenue"
            ] - (baseline_kpi * conversion_rate * avg_ticket)
            response_curve_df["Incremental_ROI"] = (
                response_curve_df["Incremental_Revenue"]
                / response_curve_df["Incremental_Investment"]
            ).fillna(0)
        else:  # CONVERSIONS mode
            response_curve_df["Incremental_Revenue"] = 0
            response_curve_df["Incremental_ROI"] = 0
            response_curve_df["CPA"] = (
                response_curve_df["Daily_Investment"]
                / response_curve_df["Projected_Total_KPIs"]
            ).fillna(0)
            response_curve_df["iCPA"] = (
                response_curve_df["Incremental_Investment"]
                / response_curve_df["Incremental_KPI"]
            ).fillna(0)

        # Find the point of maximum efficiency based on the highest Incremental ROI
        incremental_curve = response_curve_df[
            response_curve_df["Daily_Investment"] >= baseline_investment
        ].copy()
        if not incremental_curve.empty:
            if revenue_mode:
                # For revenue, max efficiency is the point with the highest ROI
                max_efficiency_index = incremental_curve["Incremental_ROI"].idxmax()
            else:
                # For conversions, max efficiency is the point with the lowest incremental CPA
                # Replace inf with NaN to handle cases where Incremental_KPI is 0
                incremental_curve["iCPA"] = incremental_curve["iCPA"].replace(
                    [np.inf, -np.inf], np.nan
                )
                max_efficiency_index = incremental_curve["iCPA"].idxmin()
        else:
            # Fallback if there's no incremental curve to analyze
            max_efficiency_index = response_curve_df.index[0]

        max_efficiency_point = response_curve_df.loc[max_efficiency_index].to_dict()
        max_efficiency_point["Scenario"] = "Máxima Eficiência"

        first_derivative = np.gradient(
            response_curve_df["Projected_Total_KPIs"],
            response_curve_df["Daily_Investment"],
        )
        second_derivative = np.gradient(first_derivative)
        diminishing_return_index = np.argmin(second_derivative)
        diminishing_return_point = response_curve_df.loc[
            diminishing_return_index
        ].to_dict()
        diminishing_return_point["Scenario"] = "Ponto de Inflexão"

        # --- UNIVERSAL FINANCIAL FILTER LOGIC START ---
        strategic_limit_point = None
        financial_targets = config.get("financial_targets", {})

        # Determine applicable filters
        max_cpa = financial_targets.get("target_cpa", float("inf"))
        max_icpa = financial_targets.get("target_icpa", float("inf"))
        min_roas = financial_targets.get("target_roas", 0)
        min_iroas = financial_targets.get(
            "target_iroas", config.get("minimum_acceptable_iroi", 0)
        )

        # Start with all points higher than baseline
        valid_points_df = response_curve_df[
            response_curve_df["Daily_Investment"] > baseline_investment
        ].copy()

        # Apply Conversion filters (if mode is CONVERSIONS or if filters are explicitly set).
        # Skipped when the KPI is already money -- CPA/iCPA (R$ per R$) is meaningless there.
        if not kpi_is_monetary and (
            optimization_target == "CONVERSIONS"
            or max_cpa != float("inf")
            or max_icpa != float("inf")
        ):
            if "CPA" not in valid_points_df.columns:
                valid_points_df["CPA"] = (
                    valid_points_df["Daily_Investment"]
                    / valid_points_df["Projected_Total_KPIs"]
                ).fillna(0)
            if "iCPA" not in valid_points_df.columns:
                valid_points_df["iCPA"] = (
                    valid_points_df["Incremental_Investment"]
                    / valid_points_df["Incremental_KPI"]
                ).fillna(0)

            if max_cpa != float("inf"):
                valid_points_df = valid_points_df[valid_points_df["CPA"] <= max_cpa]
            if max_icpa != float("inf"):
                valid_points_df = valid_points_df[
                    (valid_points_df["iCPA"] > 0)
                    & (valid_points_df["iCPA"] <= max_icpa)
                ]

        # Apply Revenue filters -- only meaningful in revenue_mode, since ROAS needs
        # Projected_Revenue, which is only computed above when revenue_mode is True.
        # target_roas/target_iroas set while optimizing for CONVERSIONS on a
        # non-monetary KPI are meaningless here; ignore them instead of crashing
        # with a KeyError on the missing Projected_Revenue column.
        if revenue_mode:
            if "ROAS" not in valid_points_df.columns:
                # Total Revenue / Total Investment
                valid_points_df["ROAS"] = (
                    valid_points_df["Projected_Revenue"]
                    / valid_points_df["Daily_Investment"]
                ).fillna(0)

            if min_roas > 0:
                valid_points_df = valid_points_df[valid_points_df["ROAS"] >= min_roas]
            if min_iroas > 0:
                valid_points_df = valid_points_df[
                    valid_points_df["Incremental_ROI"] >= min_iroas
                ]
        elif min_roas > 0 or min_iroas > 0:
            print(
                "   - AVISO (config): target_roas/target_iroas foram ignorados porque "
                "o KPI nao e monetario e optimization_target != 'REVENUE' -- ROAS exige "
                "Projected_Revenue, que so existe em modo receita."
            )

        # Find the Strategic Limit: The maximum investment that satisfies all constraints
        if not valid_points_df.empty:
            strategic_limit_point_idx = valid_points_df["Daily_Investment"].idxmax()
            strategic_limit_point = response_curve_df.loc[
                strategic_limit_point_idx
            ].to_dict()
            strategic_limit_point["Scenario"] = "Limite Estratégico"
        # --- UNIVERSAL FINANCIAL FILTER LOGIC END ---

        initial_marginal_gain = first_derivative[0]
        saturation_threshold = initial_marginal_gain * 0.1
        try:
            saturation_index = np.where(first_derivative < saturation_threshold)[0][0]
            saturation_point = response_curve_df.loc[saturation_index].to_dict()
            saturation_point["Scenario"] = "Ponto de Saturação"
        except IndexError:
            saturation_point = response_curve_df.iloc[-1].to_dict()
            saturation_point["Scenario"] = "Ponto de Saturação (Extrapolado)"

        scenarios_df = pd.DataFrame(
            [
                p
                for p in [
                    baseline_point,
                    max_efficiency_point,
                    strategic_limit_point,
                    diminishing_return_point,
                    saturation_point,
                ]
                if p is not None
            ]
        )
        model_params = {
            "alpha": best_alpha,
            "k": best_k,
            "s": best_s,
            "scaler": max_kpi_scaler,
        }

        return (
            response_curve_df,
            scenarios_df,
            baseline_point,
            max_efficiency_point,
            diminishing_return_point,
            saturation_point,
            strategic_limit_point,
            model_params,
            channel_proportions,
        )

    except Exception as e:
        import traceback

        print(f"An error occurred in the sweet spot calculation: {e}")
        print(
            f"   - Config causadora da falha: optimization_target={optimization_target!r}, "
            f"financial_targets={config.get('financial_targets', {})}"
        )
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame(), {}, {}, {}, {}, {}, {}, {}


def get_kpi_for_investment(investment, model):
    """Calculates the projected KPI for a single channel given an investment and a trained model."""
    adstocked_inv = investment / (1 - model["alpha"])
    saturated_response = hill_transform(adstocked_inv, model["k"], model["s"])
    return model.get("base_kpi", 0) + (saturated_response * model["scaler"])


def find_optimal_historical_mix(kpi_df, daily_investment_df):
    """
    Analyzes historical data to find the investment mix from the most efficient periods.
    """
    print("   - - Analyzing historical data to find the optimal investment mix...")

    # Pivot investment data to have channels as columns
    investment_pivot = daily_investment_df.pivot_table(
        index="Date", columns="Product Group", values="investment"
    ).fillna(0)

    # Merge with KPI data
    merged_df = pd.merge(kpi_df, investment_pivot, on="Date", how="inner")
    merged_df.set_index("Date", inplace=True)

    # Resample to weekly frequency
    weekly_df = merged_df.resample("W-MON").sum()

    # Calculate total investment and a rolling efficiency ratio to find sustained high-performance periods
    weekly_df["Total_Investment"] = weekly_df[investment_pivot.columns].sum(axis=1)

    # Use a 4-week rolling window to smooth out anomalies and capture sustained impact
    rolling_kpi = weekly_df["kpi"].rolling(window=4, min_periods=1).sum()
    rolling_investment = (
        weekly_df["Total_Investment"].rolling(window=4, min_periods=1).sum()
    )

    weekly_df["Efficiency_Ratio"] = (rolling_kpi / rolling_investment).fillna(0)

    # Filter out weeks with no investment
    weekly_df = weekly_df[weekly_df["Total_Investment"] > 0]

    if weekly_df.empty:
        print("   - No historical investment data to analyze for optimal mix.")
        return None

    # Identify the top 10% most efficient weeks
    top_quantile = weekly_df["Efficiency_Ratio"].quantile(0.9)
    top_weeks = weekly_df[weekly_df["Efficiency_Ratio"] >= top_quantile]

    if top_weeks.empty:
        print("   - Could not identify top efficiency weeks.")
        return None

    # Calculate the investment share for each channel during these top weeks
    top_weeks_investment = top_weeks[investment_pivot.columns]
    optimal_mix = top_weeks_investment.sum() / top_weeks_investment.sum().sum()

    print(f"   - Found optimal historical mix from top {len(top_weeks)} weeks.")
    return optimal_mix.to_dict()
