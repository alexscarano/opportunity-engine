import matplotlib

matplotlib.use("Agg")
import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import tempfile
import pandas as pd

# Ensure scripts directory is in path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
)

from presentation import (
    save_accuracy_plot,
    save_line_chart_plot,
    save_investment_bar_plot,
    save_sessions_bar_plot,
    save_causal_chart_data,
)


class TestPresentationChartsTranslation(unittest.TestCase):
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_save_accuracy_plot_translation(self, mock_close, mock_savefig):
        results_data = {"mae": 12.34}
        accuracy_df = pd.DataFrame(
            {
                "Date": pd.date_range(start="2026-01-01", periods=5),
                "kpi": [10, 20, 15, 30, 25],
                "Predicted": [12, 18, 14, 28, 27],
            }
        )

        with (
            patch("matplotlib.axes.Axes.plot") as mock_plot,
            patch("matplotlib.axes.Axes.set_title") as mock_set_title,
            patch("matplotlib.axes.Axes.text") as mock_text,
        ):
            save_accuracy_plot(
                results_data, accuracy_df, "dummy_path.png", kpi_name="Vendas"
            )

            # Verify plot calls have Portuguese labels
            first_call_kwargs = mock_plot.call_args_list[0][1]
            second_call_kwargs = mock_plot.call_args_list[1][1]

            self.assertEqual(first_call_kwargs["label"], "Vendas Real")
            self.assertEqual(second_call_kwargs["label"], "Vendas Previsto (In-Sample)")

            # Verify title is translated
            mock_set_title.assert_called_once()
            title_str = mock_set_title.call_args[0][0]
            self.assertIn("Acurácia do Modelo", title_str)
            self.assertIn("Real vs. Previsto", title_str)

            # Verify MAE text
            text_str = mock_text.call_args[0][2]
            self.assertIn("MAE (últimos 90 dias)", text_str)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_save_line_chart_plot_translation(self, mock_close, mock_savefig):
        line_df = pd.DataFrame(
            {
                "Date": pd.date_range(start="2026-01-01", periods=5),
                "Actual_KPI": [10, 20, 15, 30, 25],
                "Forecasted_KPI": [12, 18, 14, 28, 27],
                "Investment": [100, 200, 150, 300, 250],
            }
        )

        with (
            patch("matplotlib.axes.Axes.plot") as mock_plot,
            patch("matplotlib.axes.Axes.set_title") as mock_set_title,
            patch("matplotlib.axes.Axes.set_xlabel") as mock_set_xlabel,
            patch("matplotlib.axes.Axes.set_ylabel") as mock_set_ylabel,
            patch("matplotlib.axes.Axes.bar") as mock_bar,
        ):
            save_line_chart_plot(line_df, "dummy_path.png", kpi_name="Vendas")

            # Verify labels
            first_plot_kwargs = mock_plot.call_args_list[0][1]
            second_plot_kwargs = mock_plot.call_args_list[1][1]
            self.assertEqual(first_plot_kwargs["label"], "Vendas Real")
            self.assertEqual(second_plot_kwargs["label"], "Vendas Previsto")

            # Xlabel should be Data
            mock_set_xlabel.assert_any_call("Data", fontsize=14)

            # Ylabel on primary axis is kpi_name ("Vendas")
            mock_set_ylabel.assert_any_call("Vendas", fontsize=14)

            # Bar chart label should be Investimento
            bar_kwargs = mock_bar.call_args[1]
            self.assertEqual(bar_kwargs["label"], "Investimento")

            # Title should be translated
            mock_set_title.assert_called_once()
            title_str = mock_set_title.call_args[0][0]
            self.assertEqual(title_str, "Análise de Causal Impact: Real vs. Previsto")

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_save_investment_bar_plot_translation(self, mock_close, mock_savefig):
        inv_bar_df = pd.DataFrame(
            {"Investment": [1000, 2000]}, index=["Pre-Event", "Event"]
        )
        inv_bar_df.index.name = "Period"

        original_rename = pd.DataFrame.rename
        rename_calls = []

        def spy_rename(df_self, *args, **kwargs):
            rename_calls.append((args, kwargs))
            return original_rename(df_self, *args, **kwargs)

        with (
            patch.object(pd.DataFrame, "rename", autospec=True, side_effect=spy_rename),
            patch("matplotlib.axes.Axes.set_title") as mock_set_title,
            patch("matplotlib.axes.Axes.set_ylabel") as mock_set_ylabel,
        ):
            save_investment_bar_plot(inv_bar_df, "dummy_path.png")

            # Check rename was called with correct mapping
            self.assertTrue(
                any(
                    kwargs.get("index")
                    == {"Pre-Event": "Pré-Evento", "Event": "Evento"}
                    for args, kwargs in rename_calls
                )
            )

            # Title
            mock_set_title.assert_called_once_with(
                "Investimento: Pré-Evento vs. Evento", fontsize=16
            )
            # Ylabel
            mock_set_ylabel.assert_called_once_with("Investimento Total", fontsize=12)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_save_sessions_bar_plot_translation(self, mock_close, mock_savefig):
        sessions_bar_df = pd.DataFrame(
            {"kpi": [100, 150]}, index=["Forecasted", "Actual"]
        )
        sessions_bar_df.index.name = "Category"

        original_rename = pd.DataFrame.rename
        rename_calls = []

        def spy_rename(df_self, *args, **kwargs):
            rename_calls.append((args, kwargs))
            return original_rename(df_self, *args, **kwargs)

        with (
            patch.object(pd.DataFrame, "rename", autospec=True, side_effect=spy_rename),
            patch("matplotlib.axes.Axes.set_title") as mock_set_title,
            patch("matplotlib.axes.Axes.set_ylabel") as mock_set_ylabel,
        ):
            save_sessions_bar_plot(
                sessions_bar_df, "dummy_path.png", kpi_name="Cliques"
            )

            # Check rename was called with correct mapping
            self.assertTrue(
                any(
                    kwargs.get("index")
                    == {"Forecasted": "Cliques Previsto", "Actual": "Cliques Real"}
                    for args, kwargs in rename_calls
                )
            )

            # Title
            mock_set_title.assert_called_once_with(
                "Cliques Real vs. Previsto", fontsize=16
            )
            # Ylabel
            mock_set_ylabel.assert_called_once_with("Total de Cliques", fontsize=12)


class TestSaveCausalChartData(unittest.TestCase):
    def test_persists_all_four_dataframes_with_mae_column(self):
        line_df = pd.DataFrame(
            {
                "Date": pd.date_range(start="2026-01-01", periods=3),
                "Actual_KPI": [10, 20, 15],
                "Forecasted_KPI": [12, 18, 14],
                "Investment": [100, 200, 150],
            }
        )
        inv_bar_df = pd.DataFrame({"Investment": [1000, 2000]}, index=["Pre-Event", "Event"])
        inv_bar_df.index.name = "Period"
        sessions_bar_df = pd.DataFrame({"kpi": [50, 60]}, index=["Forecasted", "Actual"])
        sessions_bar_df.index.name = "Category"
        accuracy_df = pd.DataFrame(
            {
                "Date": pd.date_range(start="2026-01-01", periods=3),
                "kpi": [10, 20, 15],
                "Predicted": [12, 18, 14],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_causal_chart_data(
                tmp_dir, line_df, inv_bar_df, sessions_bar_df, accuracy_df, mae=12.34
            )

            line_out = pd.read_csv(os.path.join(tmp_dir, "line_chart_data.csv"))
            self.assertEqual(list(line_out["Actual_KPI"]), [10, 20, 15])
            self.assertEqual(list(line_out["Investment"]), [100, 200, 150])

            inv_out = pd.read_csv(os.path.join(tmp_dir, "investment_data.csv"), index_col=0)
            self.assertEqual(list(inv_out.index), ["Pre-Event", "Event"])
            self.assertEqual(list(inv_out["Investment"]), [1000, 2000])

            sessions_out = pd.read_csv(os.path.join(tmp_dir, "sessions_data.csv"), index_col=0)
            self.assertEqual(list(sessions_out.index), ["Forecasted", "Actual"])
            self.assertEqual(list(sessions_out["kpi"]), [50, 60])

            accuracy_out = pd.read_csv(os.path.join(tmp_dir, "accuracy_data.csv"))
            self.assertEqual(list(accuracy_out["Predicted"]), [12, 18, 14])
            self.assertEqual(list(accuracy_out["mae"]), [12.34, 12.34, 12.34])


if __name__ == "__main__":
    unittest.main()
