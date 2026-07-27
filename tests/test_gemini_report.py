import sys
from unittest.mock import MagicMock

# Mock 'db' module before importing streamlit_app to avoid database connections
mock_db = MagicMock()
mock_db.get_user_projects.return_value = []
mock_db.get_user_api_key.return_value = ""
mock_db.get_user_projects_mapping.return_value = {}
sys.modules['db'] = mock_db

# Mock Streamlit session state before importing streamlit_app
import streamlit as st
st.session_state["user_id"] = 1
st.session_state["username"] = "test_user"
st.session_state["active_config_path"] = "dummy_config_path"
st.session_state["show_run_success_balloons"] = False

# Ensure scripts directory is in path
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
)

# Now we can safely import _load_event_narrative
from streamlit_app import _load_event_narrative
from gemini_report import generate_global_gemini_report

import unittest
import tempfile
import json
from unittest.mock import MagicMock


class TestEventNarrativeLoading(unittest.TestCase):
    def test_load_from_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            narrative = {
                "report_title": "Custom Test Title",
                "executive_verdict": "Success!",
                "detailed_analysis": "Detailed information.",
                "value_delivered": {
                    "narrative": "Causal narrative.",
                    "methodology_narrative": "Methodology explanation."
                },
                "next_steps": [
                    {"step": "Step 1", "description": "Do this"}
                ],
                "metrics": {
                    "incremental_investment_str": "R$ 10.000",
                    "business_impact_label": "Cliques",
                    "business_impact_value": "5.000",
                    "efficiency_label": "CPA Incremental",
                    "efficiency_value": "R$ 2,00"
                }
            }
            
            # Save to json file
            json_path = os.path.join(tmp_dir, "gemini_report_event_123.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(narrative, f)
                
            res = _load_event_narrative(tmp_dir)
            self.assertIsNotNone(res)
            self.assertEqual(res["report_title"], "Custom Test Title")
            self.assertEqual(res["executive_verdict"], "Success!")
            self.assertEqual(res["metrics"]["business_impact_label"], "Cliques")
            self.assertEqual(res["metrics"]["efficiency_value"], "R$ 2,00")

    def test_load_from_html_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            html_content = """
            <!DOCTYPE html>
            <html>
            <head><title>HTML Parsed Title</title></head>
            <body>
                <h2>Veredito Executivo</h2>
                <p>HTML Verdict text here.</p>
                
                <h2>Análise Aprofundada e Eficiência</h2>
                <p>HTML Detailed Analysis text here.</p>
                
                <h2>O Impacto Causal e Metodologia</h2>
                <p>Causal impact description.</p>
                
                <h3>A Metodologia Opcional</h3>
                <p>Methodology description.</p>
                
                <h2>Próximos Passos Estratégicos</h2>
                <ul>
                    <li><strong>Step A:</strong> Do A.</li>
                    <li><strong>Step B:</strong> Do B.</li>
                </ul>
                
                <ul>
                    <li><strong>Investimento Incremental:</strong> R$ 50.000</li>
                    <li><strong>Lift Mensurável (Conversões):</strong> 1.500</li>
                    <li><strong>ROI Incremental:</strong> 3.50x</li>
                    <li><strong>R-squared (R²):</strong> 0.85</li>
                    <li><strong>P-value (Significância do Lift):</strong> 0.0020</li>
                    <li><strong>Mean Absolute Percentage Error (MAPE):</strong> 12.50%</li>
                </ul>
            </body>
            </html>
            """
            
            # Save to html file
            html_path = os.path.join(tmp_dir, "gemini_report_event_123.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
                
            res = _load_event_narrative(tmp_dir)
            self.assertIsNotNone(res)
            self.assertEqual(res["report_title"], "HTML Parsed Title")
            self.assertEqual(res["executive_verdict"], "HTML Verdict text here.")
            self.assertEqual(res["detailed_analysis"], "HTML Detailed Analysis text here.")
            self.assertEqual(res["value_delivered"]["narrative"], "Causal impact description.")
            self.assertEqual(res["value_delivered"]["methodology_narrative"], "Methodology description.")
            self.assertEqual(len(res["next_steps"]), 2)
            self.assertEqual(res["next_steps"][0]["step"], "Step A")
            self.assertEqual(res["next_steps"][0]["description"], "Do A.")
            
            # Metrics
            self.assertEqual(res["metrics"]["incremental_investment_str"], "R$ 50.000")
            self.assertEqual(res["metrics"]["business_impact_label"], "Conversões")
            self.assertEqual(res["metrics"]["business_impact_value"], "1.500")
            self.assertEqual(res["metrics"]["efficiency_label"], "ROI Incremental")
            self.assertEqual(res["metrics"]["efficiency_value"], "3.50x")
            self.assertEqual(res["metrics"]["r_squared"], 0.85)
            self.assertEqual(res["metrics"]["p_value"], 0.0020)
            self.assertEqual(res["metrics"]["mape"], 12.50)

    def test_load_from_markdown_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            md_content = """# MD Parsed Title

## Veredito Executivo
**MD Verdict text here.**

- **Investimento Incremental:** R$ 25.000
- **Receita Incremental:** R$ 100.000
- **ROI Incremental:** 3.00x

## Análise Aprofundada
MD Detailed Analysis text here.

## O Impacto Causal e Valor Entregue
MD Causal impact description.

## Próximos Passos Estratégicos
### Step X
Do X.

### Step Y
Do Y.
"""
            
            md_path = os.path.join(tmp_dir, "RECOMMENDATIONS.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
                
            res = _load_event_narrative(tmp_dir)
            self.assertIsNotNone(res)
            self.assertEqual(res["report_title"], "MD Parsed Title")
            self.assertEqual(res["executive_verdict"], "MD Verdict text here.")
            self.assertEqual(res["detailed_analysis"], "MD Detailed Analysis text here.")
            self.assertEqual(res["value_delivered"]["narrative"], "MD Causal impact description.")
            self.assertEqual(len(res["next_steps"]), 2)
            self.assertEqual(res["next_steps"][0]["step"], "Step X")
            self.assertEqual(res["next_steps"][0]["description"], "Do X.")
            
            # Metrics
            self.assertEqual(res["metrics"]["incremental_investment_str"], "R$ 25.000")
            self.assertEqual(res["metrics"]["business_impact_label"], "Receita Incremental")
            self.assertEqual(res["metrics"]["business_impact_value"], "R$ 100.000")
            self.assertEqual(res["metrics"]["efficiency_label"], "ROI Incremental")
            self.assertEqual(res["metrics"]["efficiency_value"], "3.00x")


class TestGlobalReportMonthlyExtrapolationFactor(unittest.TestCase):
    """Fase 3: generate_global_gemini_report's summary table must scale
    per-period Daily_Investment/Projected_Total_KPIs by 30/period_days, not a
    flat 30 -- the flat factor overstates monthly figures by ~30/period_days x
    for weekly/monthly cadence data."""

    def _run(self, config_extra, tmp_dir):
        advertiser = "TestAdv"
        output_dir = os.path.join(tmp_dir, advertiser, "global_saturation_analysis")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "SATURATION_CURVE.md"), "w", encoding="utf-8") as f:
            f.write("# dummy\n")

        config = {
            "output_directory": tmp_dir,
            "advertiser_name": advertiser,
            "average_ticket": 0,
            "conversion_rate_from_kpi_to_bo": 0,
            "primary_business_metric_name": "Leads",
        }
        config.update(config_extra)

        fake_client = MagicMock()
        fake_client.model_name = "fake-model"
        fake_response = MagicMock()
        fake_response.text = json.dumps({
            "report_title": "T",
            "executive_summary": "S",
            "analysis_of_scenarios": {"introduction": "I", "scenario_table": []},
            "strategic_recommendations": [{"recommendation": "R1"}, {"recommendation": "R2"}],
        })
        fake_client.generate_content.return_value = fake_response

        kpi_projections = {
            "current": {"Daily_Investment": 1000.0, "Projected_Total_KPIs": 500.0, "Incremental_KPI": 0},
            "optimized": {"Daily_Investment": 1200.0, "Projected_Total_KPIs": 550.0, "Incremental_KPI": 50.0},
            "strategic": {"Daily_Investment": 1500.0, "Projected_Total_KPIs": 600.0, "Incremental_KPI": 100.0},
            "reallocation": {"Daily_Investment": 1000.0, "Projected_Total_KPIs": 520.0, "Incremental_KPI": 20.0},
        }

        generate_global_gemini_report(
            fake_client, config, scenarios=[], total_investment=1000.0,
            kpi_projections=kpi_projections,
        )

        with open(os.path.join(output_dir, "global_report.html"), "r", encoding="utf-8") as f:
            return f.read()

    def test_period_days_7_scales_by_30_over_7_not_flat_30(self):
        with tempfile.TemporaryDirectory() as tmp:
            html_out = self._run({"period_days": 7}, tmp)
        # Daily_Investment=1000, monthly_factor=30/7=4.2857 -> ~R$4.3k, NOT R$30.0k.
        self.assertIn("R$ 4.3k", html_out)
        self.assertNotIn("R$ 30.0k", html_out)

    def test_period_days_1_matches_old_flat_30_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            html_out = self._run({"period_days": 1}, tmp)
        self.assertIn("R$ 30.0k", html_out)

    def test_missing_period_days_defaults_to_flat_30_for_backward_compat(self):
        with tempfile.TemporaryDirectory() as tmp:
            html_out = self._run({}, tmp)
        self.assertIn("R$ 30.0k", html_out)


if __name__ == "__main__":
    unittest.main()
