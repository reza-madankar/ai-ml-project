import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from retrieval import ROOT


def click(app, label):
    next(button for button in app.button if button.label == label).click()
    return app.run()


class DashboardTests(unittest.TestCase):
    def test_full_run_persistence_and_no_context_generation_guard(self):
        chunks = [{"passage_id": "123", "chunk_number": 0, "text": "RET is associated with Hirschsprung disease.",
                   "score": 1.0, "bm25_rank": 1, "embedding_rank": 1}]
        generated = {"answer": "RET is associated with the disease [S1].", "model": "openai/gpt-4o-mini",
                     "seconds": 0.1, "usage": {"total_tokens": 100}, "finish_reason": "stop"}
        with patch("hybrid_retrieval.hybrid_search", return_value=chunks) as retrieve, \
             patch("rag.generate_answer", return_value=generated) as generate:
            app = AppTest.from_file(str(ROOT / "retrieval_dashboard.py"), default_timeout=30).run()
            next(area for area in app.text_area if area.label == "Your query").set_value("Hirschsprung genes?")
            click(app, "Run full RAG")
            self.assertFalse(app.exception)
            trace = app.session_state["rag_run"]
            self.assertEqual(trace["generation"], generated)
            self.assertEqual(generate.call_args.args[0], trace["payload"])
            app.run()
            generate.assert_called_once()
            retrieve.assert_called_once()
            self.assertEqual(app.session_state["rag_run"]["id"], trace["id"])
            # A fresh query with no matches must not generate or show the previous answer.
            retrieve.return_value = []
            next(area for area in app.text_area if area.label == "Your query").set_value("unmatchable query")
            click(app, "Run full RAG")
            self.assertFalse(app.exception)
            generate.assert_called_once()
            self.assertNotIn("generation", app.session_state["rag_run"])

    def test_generation_failure_preserves_retrieval_and_prompt(self):
        chunks = [{"passage_id": "123", "chunk_number": 0, "text": "Source evidence.",
                   "score": 1.0, "bm25_rank": 1, "embedding_rank": 1}]
        with patch("hybrid_retrieval.hybrid_search", return_value=chunks), \
             patch("rag.generate_answer", side_effect=RuntimeError("Generation temporarily unavailable")):
            app = AppTest.from_file(str(ROOT / "retrieval_dashboard.py"), default_timeout=30).run()
            next(area for area in app.text_area if area.label == "Your query").set_value("Question?")
            click(app, "Run full RAG")
            self.assertFalse(app.exception)
            run = app.session_state["rag_run"]
            self.assertEqual(len(run["retrieved_chunks"]), 1)
            self.assertIn("Source evidence.", run["payload"]["messages"][1]["content"])
            self.assertIn("generation_error", run)


if __name__ == "__main__":
    unittest.main()
