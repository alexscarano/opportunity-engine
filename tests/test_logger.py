import unittest
import sys
import io
import os

# Ensure the root directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import translate_line, LogContext

class TestLogger(unittest.TestCase):
    def test_translate_line_individual_curve_plot(self):
        line = "Individual curve plot saved for GOOGLE DISPLAY/VIDEO: C:\\Users\\floppydisk\\Documents\\Trabalho\\opportunity-engine\\outputs/user_1/Meu_Projeto_Dynamic_dynamic\\global_saturation_analysis\\individual_response_curve_GOOGLE_DISPLAY_VIDEO.png"
        translated = translate_line(line)
        self.assertEqual(translated, "   - Gráfico de saturação gerado para: GOOGLE DISPLAY/VIDEO")

    def test_translate_line_suppressed_patterns(self):
        self.assertIsNone(translate_line("=================================================="))
        self.assertIsNone(translate_line("Generating Global Gemini Report..."))
        self.assertIsNone(translate_line("Assembling Gemini HTML report to 'C:\\path\\to\\report.html'..."))
        self.assertIsNone(translate_line("Generating Markdown report to 'C:\\path\\to\\report.md'..."))
        self.assertIsNone(translate_line("Markdown report generated successfully."))

    def test_translate_line_gemini_global_reports(self):
        line1 = "Global narrative generated and parsed successfully."
        self.assertEqual(translate_line(line1), "   - Recomendações geradas com sucesso pela IA.")

        line2 = "Global Gemini HTML report saved successfully to: C:\\Users\\floppydisk\\Documents\\Trabalho\\opportunity-engine\\outputs/user_1/Meu_Projeto_Dynamic_dynamic\\global_saturation_analysis\\global_report.html"
        self.assertEqual(translate_line(line2), "   - Relatório estratégico global em HTML gerado com sucesso.")

        line3 = "Gemini HTML report saved successfully."
        self.assertEqual(translate_line(line3), "   - Relatório de recomendações em HTML gerado com sucesso.")

    def test_log_context_suppression(self):
        # We redirect sys.stdout inside LogContext, but we want to capture what LogContext writes to its parent stdout.
        # We can mock the stdout property of LogContext.
        captured_output = io.StringIO()
        
        context = LogContext("test_namespace")
        context.stdout = captured_output
        
        with context:
            print("==================================================")
            print("Generating Global Gemini Report...")
            print("Individual curve plot saved for META: C:\\path\\to\\image.png")
            print("Global narrative generated and parsed successfully.")
            
        output = captured_output.getvalue()
        # The suppressed lines (===, Generating..., etc.) should not be in the output
        self.assertNotIn("==================================================", output)
        self.assertNotIn("Generating Global Gemini Report", output)
        self.assertIn("   - Gráfico de saturação gerado para: META\n", output)
        self.assertIn("   - Recomendações geradas com sucesso pela IA.\n", output)

if __name__ == "__main__":
    unittest.main()
