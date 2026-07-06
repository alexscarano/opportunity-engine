import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
)

from dashboard_charts import (
    build_icpa_curve,
    build_revenue_roi_curve,
    build_channel_mix_evolution,
    build_channel_saturation_comparison,
    build_events_overview,
    build_accuracy_chart,
    build_causal_line_chart,
    build_investment_bar_chart,
    build_sessions_bar_chart,
    build_response_curve_individual,
    find_saturation_point,
)


class TestBuildIcpaCurve(unittest.TestCase):

    def setUp(self):
        self.df_plot = pd.DataFrame({
            "Monthly_Investment": [0, 1000, 2000, 3000],
            "iCPA": [np.inf, 5.0, 6.0, 8.0],
        })
        self.optimal_point = pd.Series({"Monthly_Investment": 2000, "iCPA": 6.0})
        self.saturation_point = pd.Series({"Monthly_Investment": 3000, "iCPA": 8.0})

    def test_filters_non_finite_and_plots_line(self):
        fig = build_icpa_curve(
            self.df_plot, self.optimal_point, self.saturation_point, kpi_name="Vendas"
        )
        self.assertEqual(fig.data[0].name, "iCPA Marginal")
        self.assertEqual(list(fig.data[0].x), [1000, 2000, 3000])
        self.assertEqual(list(fig.data[0].y), [5.0, 6.0, 8.0])
        self.assertEqual(fig.layout.yaxis.title.text, "iCPA Marginal (Vendas)")

    def test_marks_optimal_and_saturation_points(self):
        fig = build_icpa_curve(
            self.df_plot, self.optimal_point, self.saturation_point
        )
        names = [trace.name for trace in fig.data]
        self.assertIn("Ponto Escolhido (Ótimo)", names)
        self.assertIn("Cenário de Saturação", names)
        optimal_trace = fig.data[names.index("Ponto Escolhido (Ótimo)")]
        self.assertEqual(list(optimal_trace.x), [2000])
        self.assertEqual(list(optimal_trace.y), [6.0])

    def test_adds_target_reference_lines_only_when_provided(self):
        fig_no_targets = build_icpa_curve(
            self.df_plot, self.optimal_point, self.saturation_point
        )
        self.assertEqual(len(fig_no_targets.layout.shapes), 0)

        fig_with_targets = build_icpa_curve(
            self.df_plot,
            self.optimal_point,
            self.saturation_point,
            target_cpa=7.0,
            target_icpa=4.0,
        )
        self.assertEqual(len(fig_with_targets.layout.shapes), 2)
        y_values = sorted(shape.y0 for shape in fig_with_targets.layout.shapes)
        self.assertEqual(y_values, [4.0, 7.0])


class TestBuildRevenueRoiCurve(unittest.TestCase):

    def setUp(self):
        self.df_plot = pd.DataFrame({
            "Monthly_Investment": [1000, 2000, 3000],
            "Projected_Revenue": [100, 150, 200],
            "Incremental_ROI": [np.inf, 2.5, 3.0],
        })

    def test_revenue_trace_scales_daily_to_monthly(self):
        fig = build_revenue_roi_curve(self.df_plot, kpi_name="Vendas")
        self.assertEqual(fig.data[0].name, "Receita Projetada (Mensal)")
        self.assertEqual(list(fig.data[0].y), [3000, 4500, 6000])

    def test_roi_trace_filters_non_finite_and_uses_secondary_axis(self):
        fig = build_revenue_roi_curve(self.df_plot)
        self.assertEqual(fig.data[1].name, "ROI Incremental")
        self.assertEqual(fig.data[1].yaxis, "y2")
        self.assertEqual(list(fig.data[1].x), [2000, 3000])
        self.assertEqual(list(fig.data[1].y), [2.5, 3.0])
        self.assertEqual(fig.layout.yaxis2.overlaying, "y")
        self.assertEqual(fig.layout.yaxis2.side, "right")


class TestBuildChannelMixEvolution(unittest.TestCase):

    def test_stacks_shares_ordered_by_average_share_descending(self):
        df_plot = pd.DataFrame({
            "Monthly_Investment": [1000, 2000],
            "Spend_GOOGLE_Strategic": [700, 1600],
            "Spend_META_Strategic": [300, 400],
        })
        fig = build_channel_mix_evolution(
            df_plot, baseline_monthly_inv=1000, optimal_monthly_inv=2000
        )
        names = [trace.name for trace in fig.data]
        self.assertEqual(names, ["GOOGLE", "META"])
        self.assertEqual(list(fig.data[0].y), [70.0, 80.0])
        self.assertEqual(list(fig.data[1].y), [30.0, 20.0])
        self.assertEqual(fig.data[0].stackgroup, "one")

    def test_adds_baseline_and_optimal_vlines(self):
        df_plot = pd.DataFrame({
            "Monthly_Investment": [1000, 2000],
            "Spend_GOOGLE_Strategic": [1000, 2000],
        })
        fig = build_channel_mix_evolution(
            df_plot, baseline_monthly_inv=1000, optimal_monthly_inv=2000
        )
        self.assertEqual(len(fig.layout.shapes), 2)

    def test_folds_extra_channels_into_outros_beyond_top_n(self):
        df_plot = pd.DataFrame({
            "Monthly_Investment": [1000],
            "Spend_A_Strategic": [500],
            "Spend_B_Strategic": [200],
            "Spend_C_Strategic": [100],
            "Spend_D_Strategic": [80],
            "Spend_E_Strategic": [60],
            "Spend_F_Strategic": [40],
            "Spend_G_Strategic": [15],
            "Spend_H_Strategic": [5],
        })
        fig = build_channel_mix_evolution(df_plot)
        names = [trace.name for trace in fig.data]
        self.assertEqual(names, ["A", "B", "C", "D", "E", "F", "Outros"])
        outros_trace = fig.data[names.index("Outros")]
        self.assertAlmostEqual(outros_trace.y[0], 2.0)

    def test_uses_log_x_axis_and_drops_zero_investment_point(self):
        df_plot = pd.DataFrame({
            "Monthly_Investment": [0, 1000, 2000],
            "Spend_GOOGLE_Strategic": [0, 700, 1600],
            "Spend_META_Strategic": [0, 300, 400],
        })
        fig = build_channel_mix_evolution(df_plot)
        self.assertEqual(fig.layout.xaxis.type, "log")
        self.assertEqual(list(fig.data[0].x), [1000, 2000])


class TestBuildChannelSaturationComparison(unittest.TestCase):

    def test_normalizes_each_channel_to_its_own_0_100_range(self):
        individual_df = pd.DataFrame({
            "Channel": ["GOOGLE", "GOOGLE", "META", "META"],
            "Channel_Spend": [0, 100, 0, 50],
            "Projected_Total_KPIs": [10, 20, 5, 5],
        })
        fig = build_channel_saturation_comparison(individual_df)

        names = [trace.name for trace in fig.data]
        self.assertEqual(names, ["GOOGLE", "META"])

        google_trace = fig.data[names.index("GOOGLE")]
        self.assertEqual(list(google_trace.x), [0, 100])
        self.assertEqual(list(google_trace.y), [0.0, 100.0])

        # META has a flat curve (min == max): normalization must not divide by zero
        meta_trace = fig.data[names.index("META")]
        self.assertEqual(list(meta_trace.y), [0.0, 0.0])


class TestBuildEventsOverview(unittest.TestCase):

    def test_colors_by_direction_and_fades_discarded_events(self):
        events_df = pd.DataFrame({
            "date": ["2025-01-06", "2025-01-13"],
            "ad_product": ["AWIN, BING", "GOOGLE"],
            "percentage_change": [150.0, -80.0],
        })
        validated_keys = {("AWIN_BING", "2025-01-06")}

        fig = build_events_overview(events_df, validated_keys)

        self.assertEqual(list(fig.data[0].x), list(pd.to_datetime(events_df["date"])))
        self.assertEqual(list(fig.data[0].y), [150.0, -80.0])
        self.assertEqual(tuple(fig.data[0].marker.color), ("#2ca02c", "#d62728"))
        self.assertEqual(tuple(fig.data[0].marker.opacity), (1.0, 0.35))


class TestBuildAccuracyChart(unittest.TestCase):

    def setUp(self):
        self.accuracy_df = pd.DataFrame({
            "Date": pd.date_range(start="2026-01-01", periods=3),
            "kpi": [10, 20, 15],
            "Predicted": [12, 18, 14],
            "mae": [1.5, 1.5, 1.5],
        })

    def test_plots_real_and_predicted_lines(self):
        fig = build_accuracy_chart(self.accuracy_df, kpi_name="Vendas")
        names = [trace.name for trace in fig.data]
        self.assertIn("Vendas Real", names)
        self.assertIn("Vendas Previsto (In-Sample)", names)

        real_trace = fig.data[names.index("Vendas Real")]
        self.assertEqual(list(real_trace.y), [10, 20, 15])
        pred_trace = fig.data[names.index("Vendas Previsto (In-Sample)")]
        self.assertEqual(list(pred_trace.y), [12, 18, 14])

    def test_shows_mae_annotation(self):
        fig = build_accuracy_chart(self.accuracy_df)
        self.assertEqual(len(fig.layout.annotations), 1)
        self.assertIn("1.50", fig.layout.annotations[0].text)


class TestBuildCausalLineChart(unittest.TestCase):

    def setUp(self):
        self.line_df = pd.DataFrame({
            "Date": pd.date_range(start="2026-01-01", periods=3),
            "Actual_KPI": [10, 20, 15],
            "Forecasted_KPI": [12, 18, 14],
            "Investment": [100, 200, 150],
        })

    def test_plots_kpi_lines_on_primary_axis(self):
        fig = build_causal_line_chart(self.line_df, kpi_name="Vendas")
        names = [trace.name for trace in fig.data]
        self.assertIn("Vendas Real", names)
        self.assertIn("Vendas Previsto", names)
        real_trace = fig.data[names.index("Vendas Real")]
        self.assertEqual(list(real_trace.y), [10, 20, 15])

    def test_plots_investment_bars_on_secondary_axis(self):
        fig = build_causal_line_chart(self.line_df)
        names = [trace.name for trace in fig.data]
        inv_trace = fig.data[names.index("Investimento")]
        self.assertEqual(inv_trace.type, "bar")
        self.assertEqual(inv_trace.yaxis, "y2")
        self.assertEqual(list(inv_trace.y), [100, 200, 150])
        self.assertEqual(fig.layout.yaxis2.overlaying, "y")
        self.assertEqual(fig.layout.yaxis2.side, "right")


class TestBuildInvestmentBarChart(unittest.TestCase):

    def test_renames_periods_and_colors_bars(self):
        inv_bar_df = pd.DataFrame({"Investment": [1000, 2000]}, index=["Pre-Event", "Event"])
        inv_bar_df.index.name = "Period"

        fig = build_investment_bar_chart(inv_bar_df)
        self.assertEqual(list(fig.data[0].x), ["Pré-Evento", "Evento"])
        self.assertEqual(list(fig.data[0].y), [1000, 2000])
        self.assertEqual(list(fig.data[0].marker.color), ["gray", "green"])


class TestBuildSessionsBarChart(unittest.TestCase):

    def test_renames_categories_and_colors_bars(self):
        sessions_bar_df = pd.DataFrame({"kpi": [100, 150]}, index=["Forecasted", "Actual"])
        sessions_bar_df.index.name = "Category"

        fig = build_sessions_bar_chart(sessions_bar_df, kpi_name="Cliques")
        self.assertEqual(list(fig.data[0].x), ["Cliques Previsto", "Cliques Real"])
        self.assertEqual(list(fig.data[0].y), [100, 150])
        self.assertEqual(list(fig.data[0].marker.color), ["red", "black"])


class TestBuildResponseCurveIndividual(unittest.TestCase):

    def setUp(self):
        self.channel_df = pd.DataFrame({
            "Channel": ["AWIN"] * 3,
            "Channel_Spend": [0, 2500, 5000],
            "Projected_Total_KPIs": [30800, 30800, 30800],
            "Historical_Avg": [4977.94] * 3,
            "Recommended": [0.0] * 3,
        })

    def test_plots_curve_and_historical_avg_line(self):
        fig = build_response_curve_individual(self.channel_df, "AWIN")
        self.assertEqual(list(fig.data[0].y), [30800, 30800, 30800])
        self.assertEqual(len(fig.layout.shapes), 2)  # historical avg + recommended

    def test_skips_recommended_line_when_nan(self):
        df = self.channel_df.copy()
        df["Recommended"] = np.nan
        fig = build_response_curve_individual(df, "AWIN")
        self.assertEqual(len(fig.layout.shapes), 1)  # only historical avg


class TestFindSaturationPoint(unittest.TestCase):

    def test_skips_transient_dip_and_finds_start_of_sustained_flat_region(self):
        # Marginal gain (per R$10 of investment) is 100, 90, then dips to 5 at
        # investment=30 (a transient blip) before recovering to 80, 70, 60, and
        # only truly flattens (3, 2, 1) from investment=70 onward.
        df = pd.DataFrame({
            "Daily_Investment": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            "Projected_Total_KPIs": [0, 1000, 1900, 1950, 2750, 3450, 4050, 4080, 4100, 4110],
        })
        optimal_point = df.iloc[-1]

        result = find_saturation_point(df, optimal_point)

        # A naive "first dip below threshold" scan would wrongly stop at 30;
        # the real, sustained flattening only starts at 70.
        self.assertEqual(result["Daily_Investment"], 70)

    def test_falls_back_to_optimal_point_when_curve_never_rises(self):
        df = pd.DataFrame({
            "Daily_Investment": [0, 10, 20],
            "Projected_Total_KPIs": [100, 90, 80],
        })
        optimal_point = df.iloc[-1]

        result = find_saturation_point(df, optimal_point)

        self.assertEqual(result["Daily_Investment"], optimal_point["Daily_Investment"])

    def test_falls_back_to_optimal_point_when_curve_is_too_short(self):
        df = pd.DataFrame({
            "Daily_Investment": [0],
            "Projected_Total_KPIs": [100],
        })
        optimal_point = df.iloc[-1]

        result = find_saturation_point(df, optimal_point)

        self.assertEqual(result["Daily_Investment"], optimal_point["Daily_Investment"])


if __name__ == "__main__":
    unittest.main()
