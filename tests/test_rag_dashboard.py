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
        with patch("retrieval.search", return_value=chunks) as retrieve, \
             patch("rag.generate_answer", return_value=generated) as generate:
            app = AppTest.from_file(str(ROOT / "retrieval_dashboard.py"), default_timeout=30).run()
            next(box for box in app.selectbox if box.label == "Retrieval method").set_value("BM25").run()
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
            next(box for box in app.selectbox if box.label == "Retrieval method").set_value("BM25").run()
            next(area for area in app.text_area if area.label == "Your query").set_value("unmatchable query")
            click(app, "Run full RAG")
            self.assertFalse(app.exception)
            generate.assert_called_once()
            self.assertNotIn("generation", app.session_state["rag_run"])

    def test_ir_rag_comparison_uses_same_bm25_results(self):
        chunks = [{"passage_id": "123", "chunk_number": 0, "text": "RET is associated with Hirschsprung disease.", "score": 1.0}]
        with patch("retrieval.build_index", return_value=object()), \
             patch("retrieval.search", return_value=chunks) as retrieve, \
             patch("ir_rag_comparison.answer_from_context", return_value="RET is associated [S1].") as generate:
            app = AppTest.from_file(str(ROOT / "retrieval_dashboard.py"), default_timeout=30).run()
            next(field for field in app.text_input if field.label == "Question for IR and RAG").set_value("Which genes?")
            click(app, "Run selected system")
            self.assertFalse(app.exception)
            retrieve.assert_called_once()
            self.assertEqual(generate.call_args.args[0], "Which genes?")
            self.assertIn(chunks[0]["text"], generate.call_args.args[1]["text"])
            self.assertEqual(app.session_state["comparison_run"]["answer"], "RET is associated [S1].")
            retrieve.return_value = []
            click(app, "Run selected system")
            generate.assert_called_once()
            self.assertFalse(app.session_state["comparison_run"]["context"]["sources"])

    def test_comparison_modes_control_generation_and_display(self):
        chunks = [{"passage_id": "123", "chunk_number": 0, "text": "Source evidence.", "score": 1.0}]
        with patch("retrieval.build_index", return_value=object()), \
             patch("retrieval.search", return_value=chunks), \
             patch("ir_rag_comparison.answer_from_context", return_value="Answer [S1].") as generate:
            app = AppTest.from_file(str(ROOT / "retrieval_dashboard.py"), default_timeout=30).run()
            next(radio for radio in app.radio if radio.label == "Choose a system").set_value("IR only").run()
            next(field for field in app.text_input if field.label == "Question for IR and RAG").set_value("Question?")
            click(app, "Run selected system")
            self.assertFalse(app.exception)
            generate.assert_not_called()
            self.assertEqual(app.session_state["comparison_run"]["mode"], "IR only")
            next(radio for radio in app.radio if radio.label == "Choose a system").set_value("RAG only").run()
            click(app, "Run selected system")
            self.assertFalse(app.exception)
            generate.assert_called_once()
            self.assertEqual(app.session_state["comparison_run"]["mode"], "RAG only")
            self.assertFalse(any(item.value == "### IR · Ranked passages" for item in app.markdown))

    def test_generation_failure_preserves_retrieval_and_prompt(self):
        chunks = [{"passage_id": "123", "chunk_number": 0, "text": "Source evidence.",
                   "score": 1.0, "bm25_rank": 1, "embedding_rank": 1}]
        with patch("retrieval.search", return_value=chunks), \
             patch("rag.generate_answer", side_effect=RuntimeError("Generation temporarily unavailable")):
            app = AppTest.from_file(str(ROOT / "retrieval_dashboard.py"), default_timeout=30).run()
            next(box for box in app.selectbox if box.label == "Retrieval method").set_value("BM25").run()
            next(area for area in app.text_area if area.label == "Your query").set_value("Question?")
            click(app, "Run full RAG")
            self.assertFalse(app.exception)
            run = app.session_state["rag_run"]
            self.assertEqual(len(run["retrieved_chunks"]), 1)
            self.assertIn("Source evidence.", run["payload"]["messages"][1]["content"])
            self.assertIn("generation_error", run)


if __name__ == "__main__":
    unittest.main()
