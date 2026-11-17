import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vector_retrieval import build_vector_index, embed, normalize, vector_search, MODEL


class VectorRetrievalTests(unittest.TestCase):
    def test_faiss_ranking_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "passages.parquet"
            pq.write_table(pa.table({"id": [1, 2, 3],
                                    "passage": ["RET Hirschsprung disease", "insulin diabetes", "nan"]}), corpus)
            def embeddings(texts, input_type, model):
                return normalize(np.array([[1, 0] if "RET" in text else [0, 1]
                                           for text in texts], dtype="float32"))
            def interrupted(texts, input_type, model):
                if "insulin" in texts[0]:
                    raise RuntimeError("interrupted")
                return embeddings(texts, input_type, model)
            with patch("vector_retrieval.embed", side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    build_vector_index(corpus=corpus, index_dir=root, batch_size=1)
            with patch("vector_retrieval.embed", side_effect=embeddings) as mocked:
                store = build_vector_index(corpus=corpus, index_dir=root, batch_size=1)
                self.assertEqual(mocked.call_count, 1)  # First batch was resumed.
                results = vector_search(store, "RET", 5)
                self.assertEqual(len(results), 2)  # 'nan' was excluded.
                self.assertEqual(results[0]["passage_id"], "1")
                self.assertAlmostEqual(results[0]["score"], 1)
                mocked.reset_mock()
                self.assertEqual(build_vector_index(corpus=corpus, index_dir=root), store)
                mocked.assert_not_called()
                self.assertEqual(vector_search(store, "  "), [])

    def test_response_order_and_perplexity_payload(self):
        response = Mock(status_code=200, ok=True)
        response.json.return_value = {"data": [{"index": 1, "embedding": [0, 2]},
                                                {"index": 0, "embedding": [3, 0]}]}
        with patch.dict("os.environ", {"OPENAI_ROUTER_KEY": "test-key"}), \
             patch("vector_retrieval.requests.post", return_value=response) as post:
            vectors = embed(["a", "b"], "search_document")
            np.testing.assert_allclose(vectors, [[1, 0], [0, 1]])
            self.assertEqual(post.call_args.kwargs["json"]["model"], MODEL)
            self.assertNotIn("input_type", post.call_args.kwargs["json"])

    def test_invalid_vectors(self):
        for vectors in ([[0, 0]], [[float("nan"), 1]], [1, 2]):
            with self.assertRaises(ValueError):
                normalize(vectors)


if __name__ == "__main__":
    unittest.main()
