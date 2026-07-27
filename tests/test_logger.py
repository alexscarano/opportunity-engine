import unittest
import logging
import sys
import io
import os
import tempfile
from pathlib import Path

# Ensure the root directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logger as logger_module
from logger import translate_line, LogContext, setup_logging


class TestLogger(unittest.TestCase):
    def test_translate_line_individual_curve_plot(self):
        line = "Individual curve plot saved for GOOGLE DISPLAY/VIDEO: C:\\Users\\floppydisk\\Documents\\Trabalho\\opportunity-engine\\outputs/user_1/Meu_Projeto_Dynamic_dynamic\\global_saturation_analysis\\individual_response_curve_GOOGLE_DISPLAY_VIDEO.png"
        translated = translate_line(line)
        self.assertEqual(
            translated, "   - Gráfico de saturação gerado para: GOOGLE DISPLAY/VIDEO"
        )

    def test_translate_line_suppressed_patterns(self):
        self.assertIsNone(
            translate_line("==================================================")
        )
        self.assertIsNone(translate_line("Generating Global Gemini Report..."))
        self.assertIsNone(
            translate_line(
                "Assembling Gemini HTML report to 'C:\\path\\to\\report.html'..."
            )
        )
        self.assertIsNone(
            translate_line("Generating Markdown report to 'C:\\path\\to\\report.md'...")
        )
        self.assertIsNone(translate_line("Markdown report generated successfully."))

    def test_translate_line_gemini_global_reports(self):
        line1 = "Global narrative generated and parsed successfully."
        self.assertEqual(
            translate_line(line1), "   - Recomendações geradas com sucesso pela IA."
        )

        line2 = "Global Gemini HTML report saved successfully to: C:\\Users\\floppydisk\\Documents\\Trabalho\\opportunity-engine\\outputs/user_1/Meu_Projeto_Dynamic_dynamic\\global_saturation_analysis\\global_report.html"
        self.assertEqual(
            translate_line(line2),
            "   - Relatório estratégico global em HTML gerado com sucesso.",
        )

        line3 = "Gemini HTML report saved successfully."
        self.assertEqual(
            translate_line(line3),
            "   - Relatório de recomendações em HTML gerado com sucesso.",
        )

    def test_translate_line_additional_patterns(self):
        self.assertIsNone(
            translate_line("--------------------------------------------------")
        )
        self.assertIsNone(
            translate_line(
                "   - No additional performance covariates found. Proceeding without them."
            )
        )
        self.assertEqual(
            translate_line(
                "   - Running automated feature selection for causal model..."
            ),
            "   - Analisando variáveis de contexto para o modelo causal...",
        )
        # Python Warnings & Code lines
        self.assertIsNone(
            translate_line(
                "C:\\path\\site-packages\\some_lib.py:12: SpecificationWarning: Some warning"
            )
        )
        self.assertIsNone(translate_line('  warn("Some warning message")'))
        self.assertIsNone(translate_line("  self._init_dates(dates, freq)"))
        self.assertIsNone(translate_line("  trend = spsolve(...)"))
        self.assertIsNone(translate_line("  var *= np.divide(...)"))
        self.assertIsNone(translate_line("  return get_prediction_index(...)"))

    def test_event_counter_and_threshold_warning(self):
        counted = translate_line("▶ Analyzing Event [12/43]: PMAX on 2026-01-12")
        self.assertIn("[12/43]", counted)
        self.assertIn("'PMAX'", counted)
        self.assertIn("2026-01-12", counted)

        # Legacy line without the counter must still translate.
        self.assertIn(
            "'PMAX'", translate_line("▶ Analyzing Event: PMAX on 2026-01-12")
        )

        warning = translate_line(
            "   - WARNING: threshold flagged 43 of 78 periods for 'PMAX' (55%)."
        )
        self.assertIn("43 de 78", warning)
        self.assertIn("'PMAX'", warning)
        self.assertIn("55%", warning)

class TestLoggingSetup(unittest.TestCase):
    """Handler wiring: console gets pt-BR, files get the technical record."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._real_log_dir = logger_module.LOG_DIR
        logger_module.LOG_DIR = Path(self._tmp.name)
        self.console = io.StringIO()

    def tearDown(self):
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()
        logger_module.LOG_DIR = self._real_log_dir
        self._tmp.cleanup()

    def test_console_translates_and_files_keep_detail(self):
        setup_logging("ns", stream=self.console)
        log = logging.getLogger("scripts.saturation_curve")

        log.debug("internal detail that must not reach the user")
        log.info("Generating Global Gemini Report...")  # suppressed by the ruleset
        log.info("Individual curve plot saved for META: C:\\path\\to\\image.png")
        log.warning("   - WARNING: threshold flagged 43 of 78 periods for 'PMAX' (55%).")

        console = self.console.getvalue()
        self.assertNotIn("internal detail", console)
        self.assertNotIn("Generating Global Gemini Report", console)
        self.assertIn("   - Gráfico de saturação gerado para: META\n", console)
        self.assertIn("43 de 78", console)

        full = (Path(self._tmp.name) / "ns.log").read_text(encoding="utf-8")
        self.assertIn("internal detail", full)  # DEBUG lands in the full log
        self.assertIn("Generating Global Gemini Report", full)  # nothing is suppressed
        self.assertIn("scripts.saturation_curve", full)  # module name is recorded

        errors = (Path(self._tmp.name) / "ns.errors.log").read_text(encoding="utf-8")
        self.assertIn("threshold flagged", errors)
        self.assertNotIn("internal detail", errors)  # WARNING+ only

    def test_log_context_captures_stray_prints(self):
        context = LogContext("ns")
        context.stdout = self.console

        with context:
            print("==================================================")
            print("Individual curve plot saved for META: C:\\path\\to\\image.png")

        console = self.console.getvalue()
        self.assertNotIn("==================================================", console)
        self.assertIn("   - Gráfico de saturação gerado para: META\n", console)

    def test_log_context_records_unhandled_exception(self):
        with self.assertRaises(ValueError):
            with LogContext("ns"):
                raise ValueError("boom")

        errors = (Path(self._tmp.name) / "ns.errors.log").read_text(encoding="utf-8")
        self.assertIn("boom", errors)
        self.assertIn("Traceback", errors)


if __name__ == "__main__":
    unittest.main()
