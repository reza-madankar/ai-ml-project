import unittest
from unittest.mock import Mock, patch

from rag import LLM_MODEL, assemble_context, citation_status, generate_answer, make_payload


class RagTests(unittest.TestCase):
    def test_context_budget_source_identity_and_exact_prompt(self):
        results = [{"passage_id": "123", "chunk_number": 0, "text": "a b c", "score": 1},
                   {"passage_id": "123", "chunk_number": 0, "text": "a b c", "score": 1},
                   {"passage_id": "456", "chunk_number": 1, "text": "d e f g", "score": .5},
                   {"passage_id": "789", "chunk_number": 0, "text": "h i", "score": .2}]
        context = assemble_context(results, 5)
        self.assertEqual(context["words"], 5)
        self.assertEqual([source["label"] for source in context["sources"]], ["S1", "S2"])
        self.assertEqual(context["sources"][1]["excerpt"], "d e")
        self.assertTrue(context["sources"][1]["truncated"])
        self.assertIn("[S2] Passage 456 | Chunk 2", context["text"])
        payload = make_payload("question?", context)
        self.assertEqual(payload["model"], LLM_MODEL)
        self.assertIn(context["text"], payload["messages"][1]["content"])
        self.assertIn("question?", payload["messages"][1]["content"])

    def test_citation_labels_and_empty_context(self):
        self.assertEqual(assemble_context([])["sources"], [])
        status = citation_status("Claim [S1] and [S9]", [{"label": "S1"}])
        self.assertEqual(status["unknown"], ["S9"])
        self.assertEqual(status["used"], ["S1"])
        with self.assertRaises(ValueError):
            assemble_context([], 0)

    def test_generation_sends_displayed_payload_and_keeps_errors_safe(self):
        payload = make_payload("Question?", assemble_context([]))
        response = Mock(ok=True)
        response.json.return_value = {"choices": [{"message": {"content": "Evidence [S1]"},
                                                    "finish_reason": "stop"}],
                                     "usage": {"total_tokens": 100}}
        with patch.dict("os.environ", {"OPENAI_ROUTER_KEY": "test-secret"}), \
             patch("rag.requests.post", return_value=response) as post:
            result = generate_answer(payload)
            self.assertEqual(post.call_args.kwargs["json"], payload)
            self.assertEqual(result["answer"], "Evidence [S1]")
            response.ok = False
            response.status_code = 401
            with self.assertRaises(RuntimeError) as error:
                generate_answer(payload)
            self.assertNotIn("test-secret", str(error.exception))
            self.assertIn("401", str(error.exception))


if __name__ == "__main__":
    unittest.main()
