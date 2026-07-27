import sys
import os
import json
import unittest
from unittest.mock import MagicMock

import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
)

from google_api import suggest_form_fields


def _sample_df():
    return pd.DataFrame(
        {
            "Data": ["2026-01-05", "2026-01-12", "2026-01-19"],
            "Canal": ["Pmax", "Pmax", "Pmax"],
            "Leads": [120, 135, 98],
        }
    )


def _mock_model(response_text):
    model = MagicMock()
    response = MagicMock()
    response.text = response_text
    model.generate_content.return_value = response
    return model


class TestSuggestFormFields(unittest.TestCase):
    def test_valid_response_returns_dict(self):
        model = _mock_model(
            json.dumps(
                {
                    "kpi_column": "Leads",
                    "kpi_is_monetary": False,
                    "optimization_target": "CONVERSIONS",
                }
            )
        )
        result = suggest_form_fields(model, _sample_df())
        self.assertEqual(
            result,
            {
                "kpi_column": "Leads",
                "kpi_is_monetary": False,
                "optimization_target": "CONVERSIONS",
            },
        )

    def test_response_with_json_fences_is_parsed(self):
        model = _mock_model(
            "```json\n"
            + json.dumps(
                {
                    "kpi_column": "Leads",
                    "kpi_is_monetary": False,
                    "optimization_target": "CONVERSIONS",
                }
            )
            + "\n```"
        )
        result = suggest_form_fields(model, _sample_df())
        self.assertEqual(result["kpi_column"], "Leads")

    def test_kpi_column_not_in_dataframe_raises(self):
        model = _mock_model(
            json.dumps(
                {
                    "kpi_column": "Coluna_Inexistente",
                    "kpi_is_monetary": False,
                    "optimization_target": "CONVERSIONS",
                }
            )
        )
        with self.assertRaises(ValueError):
            suggest_form_fields(model, _sample_df())

    def test_invalid_optimization_target_raises(self):
        model = _mock_model(
            json.dumps(
                {
                    "kpi_column": "Leads",
                    "kpi_is_monetary": False,
                    "optimization_target": "LUCRO",
                }
            )
        )
        with self.assertRaises(ValueError):
            suggest_form_fields(model, _sample_df())

    def test_malformed_json_raises_value_error(self):
        model = _mock_model("isto não é json")
        with self.assertRaises(ValueError):
            suggest_form_fields(model, _sample_df())


if __name__ == "__main__":
    unittest.main()
