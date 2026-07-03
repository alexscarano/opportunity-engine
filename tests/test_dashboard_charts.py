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


if __name__ == "__main__":
    unittest.main()
