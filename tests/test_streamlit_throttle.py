import ast
import os
import unittest

STREAMLIT_APP_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scripts", "streamlit_app.py")
)


def _load_should_render():
    """scripts/streamlit_app.py is a monolithic Streamlit script: importing it
    normally runs the whole app top-to-bottom and requires a live
    session_state/DB/cookie context (it raises immediately otherwise). To
    unit-test the pure throttle decision logic without dragging in that
    machinery, extract just the `should_render` function definition via AST
    and exec it in isolation. This still tests the real production source,
    it just skips executing the rest of the script.
    """
    with open(STREAMLIT_APP_PATH, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=STREAMLIT_APP_PATH)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "should_render":
            module = ast.Module(body=[node], type_ignores=[])
            namespace = {}
            exec(compile(module, STREAMLIT_APP_PATH, "exec"), namespace)
            return namespace["should_render"]

    raise AssertionError("should_render not found in scripts/streamlit_app.py")


should_render = _load_should_render()


class TestShouldRender(unittest.TestCase):
    def test_no_render_when_neither_threshold_met(self):
        self.assertFalse(
            should_render(lines_since_last_render=5, seconds_since_last_render=0.1)
        )

    def test_render_when_line_threshold_met(self):
        self.assertTrue(
            should_render(lines_since_last_render=25, seconds_since_last_render=0.0)
        )

    def test_render_when_time_threshold_met(self):
        self.assertTrue(
            should_render(lines_since_last_render=1, seconds_since_last_render=0.4)
        )

    def test_no_render_just_under_both_thresholds(self):
        self.assertFalse(
            should_render(lines_since_last_render=24, seconds_since_last_render=0.39)
        )

    def test_custom_thresholds(self):
        self.assertTrue(should_render(3, 0.05, min_lines=3, min_seconds=10))
        self.assertFalse(should_render(2, 0.05, min_lines=3, min_seconds=10))


if __name__ == "__main__":
    unittest.main()
