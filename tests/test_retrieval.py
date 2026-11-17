import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from retrieval import build_index, chunk_text, index_stats, search


class RetrievalTests(unittest.TestCase):
    def test_chunk_boundaries_and_validation(self):
        self.assertEqual(list(chunk_text("a b c d e f g", 4, 1)),
                         [(0, "a b c d"), (1, "d e f g")])
        self.assertEqual(list(chunk_text("", 4, 1)), [])
        with self.assertRaises(ValueError):
            list(chunk_text("text", 4, 4))

    def test_ranking_sources_cache_and_literal_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.parquet"
            pq.write_table(pa.table({
                "id": [101, 202, 303, 404],
                "passage": ["Hirschsprung disease involves RET genes.",
                            "Diabetes treatment includes insulin therapy.",
                            "Genes regulate cellular development.", "nan"],
            }), corpus)
            index = build_index(corpus, 20, 2, root)
            self.assertEqual(index_stats(index)["chunks"], 3)
            results = search(index, "Hirschsprung RET", 2)
            self.assertEqual(results[0]["passage_id"], "101")
            self.assertIn("RET", results[0]["text"])
            self.assertGreater(results[0]["score"], 0)
            self.assertEqual(search(index, "zzzzunknown"), [])
            self.assertEqual(search(index, "nan"), [])
            self.assertEqual(search(index, "!!!"), [])
            self.assertTrue(search(index, '"RET" OR (disease) - : *'))
            modified = index.stat().st_mtime_ns
            self.assertEqual(build_index(corpus, 20, 2, root), index)
            self.assertEqual(index.stat().st_mtime_ns, modified)
            self.assertNotEqual(build_index(corpus, 10, 2, root), index)


if __name__ == "__main__":
    unittest.main()
