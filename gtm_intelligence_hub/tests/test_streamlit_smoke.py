"""Offline Streamlit smoke tests for the two entry screens."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")

class TestStreamlitSmoke(unittest.TestCase):
    def test_command_center(self):
        app = AppTest.from_file(APP, default_timeout=30).run()
        self.assertEqual(len(app.exception), 0, [str(x.message) for x in app.exception])
    def test_creation_intelligence_demo(self):
        app = AppTest.from_file(APP, default_timeout=30).run()
        app.sidebar.radio[0].set_value("Creation Intelligence").run()
        self.assertEqual(len(app.exception), 0, [str(x.message) for x in app.exception])

    def test_pipeline_generation_demo(self):
        app = AppTest.from_file(APP, default_timeout=30).run()
        app.sidebar.radio[0].set_value("Pipeline Generation").run()
        self.assertEqual(len(app.exception), 0, [str(x.message) for x in app.exception])

if __name__ == "__main__":
    unittest.main()
