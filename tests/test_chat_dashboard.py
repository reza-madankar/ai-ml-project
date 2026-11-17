import os
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from retrieval import ROOT


class ChatDashboardTests(unittest.TestCase):
    def test_missing_google_key_shows_setup_instructions(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}), \
             patch("dotenv.load_dotenv", return_value=False):
            app = AppTest.from_file(str(ROOT / "chat_dashboard.py")).run()
        self.assertFalse(app.exception)
        self.assertIn("GOOGLE_API_KEY", app.error[0].value)

    def test_google_client_and_successful_chat(self):
        from types import SimpleNamespace
        chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello!"))])
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}), \
             patch("openai.OpenAI") as client:
            client.return_value.chat.completions.create.return_value = iter([chunk])
            app = AppTest.from_file(str(ROOT / "chat_dashboard.py")).run()
            app.chat_input[0].set_value("Hi").run()
        self.assertFalse(app.exception)
        self.assertEqual(client.call_args.kwargs["base_url"], "https://generativelanguage.googleapis.com/v1beta/openai/")
        self.assertEqual(app.session_state["messages"][-1]["content"], "Hello!")

    def test_failed_request_is_not_saved_as_conversation(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}), \
             patch("openai.OpenAI") as client:
            client.return_value.chat.completions.create.side_effect = RuntimeError("unavailable")
            app = AppTest.from_file(str(ROOT / "chat_dashboard.py")).run()
            app.chat_input[0].set_value("Hi").run()
        self.assertFalse(app.exception)
        self.assertIn("unavailable", app.error[0].value)
        self.assertEqual(app.session_state["messages"], [])


if __name__ == "__main__":
    unittest.main()
