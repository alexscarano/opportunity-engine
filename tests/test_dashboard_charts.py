import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
)

from dashboard_charts import build_icpa_curve


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


if __name__ == "__main__":
    unittest.main()
