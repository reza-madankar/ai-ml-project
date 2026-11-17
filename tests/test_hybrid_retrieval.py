import math
import unittest

from hybrid_retrieval import fuse_rankings
from tune_hybrid import passage_metrics, select_alpha, split_questions


def chunk(passage, number=0, chunk_id=0):
    return {"passage_id": str(passage), "chunk_number": number,
            "chunk_id": chunk_id, "text": f"Passage {passage}"}


class HybridTests(unittest.TestCase):
    def test_fusion_uses_source_identity_and_endpoints(self):
        bm25 = [chunk(1, chunk_id=101), chunk(2)]
        semantic = [chunk(2), chunk(1, chunk_id=0), chunk(3)]
        blended = fuse_rankings(bm25, semantic, 0.75, 10)
        self.assertEqual(len(blended), 3)
        self.assertEqual(blended[0]["passage_id"], "1")
        self.assertAlmostEqual(blended[0]["score"], 0.75 / 61 + 0.25 / 62)
        self.assertEqual(blended[0]["bm25_rank"], 1)
        self.assertEqual(blended[0]["embedding_rank"], 2)
        self.assertEqual([r["passage_id"] for r in fuse_rankings(bm25, semantic, 1, 10)], ["1", "2"])
        self.assertEqual([r["passage_id"] for r in fuse_rankings(bm25, semantic, 0, 10)], ["2", "1", "3"])

    def test_passage_metrics_deduplicate_and_retain_missing_labels(self):
        ranking = [chunk(1), chunk(1, 1), chunk(2), chunk(3)]
        result = passage_metrics(ranking, {"1", "3", "999"})
        self.assertAlmostEqual(result["recall@10"], 2 / 3)
        self.assertEqual(result["mrr@10"], 1)
        self.assertAlmostEqual(result["ndcg@10"], 1.5 / (1 + 1 / math.log2(3) + 0.5))

    def test_split_groups_duplicate_queries_and_selection_uses_dev_scores(self):
        rows = [{"question": f"query {i}"} for i in range(20)] + [{"question": " QUERY 0 "}]
        splits = split_questions(rows)
        self.assertEqual(splits[0], splits[-1])
        self.assertIn("held_out", splits)
        self.assertEqual(splits, split_questions(rows))
        self.assertEqual(select_alpha([{"alpha": 0.0, "ndcg@10": 0.3},
                                       {"alpha": 0.6, "ndcg@10": 0.6},
                                       {"alpha": 1.0, "ndcg@10": 0.5}]), 0.6)


if __name__ == "__main__":
    unittest.main()
