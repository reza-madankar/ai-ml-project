"""Weighted reciprocal rank fusion of BM25 and embedding chunk rankings."""

import json
from pathlib import Path

from retrieval import ROOT, index_stats, search
from vector_retrieval import vector_search

TUNING_REPORT = ROOT / "reports/hybrid-tuning/summary.json"
RRF_K = 60
CANDIDATE_K = 100


def fuse_rankings(bm25, semantic, alpha, top_k=5, rrf_k=RRF_K):
    """alpha weights BM25; 1-alpha weights embeddings. Merge by source chunk.

    Ranks avoid adding incompatible raw BM25 and cosine-similarity scores.
    A missing candidate contributes zero for that branch.
    """
    if not 0 <= alpha <= 1 or top_k < 1 or rrf_k < 1:
        raise ValueError("Require 0 <= alpha <= 1, top_k >= 1, rrf_k >= 1")
    merged = {}
    for branch, weight, results in [("bm25", alpha, bm25), ("embedding", 1-alpha, semantic)]:
        seen = set()
        for rank, result in enumerate(results, 1):
            key = (str(result["passage_id"]), int(result["chunk_number"]))
            if key in seen:
                continue
            seen.add(key)
            if key not in merged:
                merged[key] = {**result, "score": 0.0, "bm25_rank": None, "embedding_rank": None}
            merged[key][f"{branch}_rank"] = rank
            merged[key]["score"] += weight / (rrf_k + rank)
    # An endpoint weight must return only the active branch's candidates.
    ranked = [result for result in merged.values() if result["score"] > 0]
    ranked.sort(key=lambda row: (-row["score"], str(row["passage_id"]), row["chunk_number"]))
    return ranked[:top_k]


def load_tuned_alpha(store, report=TUNING_REPORT):
    path = Path(report)
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    settings = json.loads((Path(store) / "ready.json").read_text())
    for name in ["model", "sha256", "chunk_size", "overlap"]:
        if result["settings"][name] != settings[name]:
            return None
    if (result["settings"].get("bm25_version") != 2
            or result["settings"].get("candidate_k") != CANDIDATE_K
            or result["settings"].get("rrf_k") != RRF_K):
        return None
    return result["selected_alpha"]


def hybrid_search(bm25_index, store, query, alpha, top_k=5, candidate_k=CANDIDATE_K):
    if not 1 <= top_k <= candidate_k <= 100:
        raise ValueError("Require 1 <= top_k <= candidate_k <= 100")
    bm25_settings = index_stats(bm25_index)
    vector_settings = json.loads((Path(store) / "ready.json").read_text())
    if any(bm25_settings[name] != vector_settings[name]
           for name in ["sha256", "chunk_size", "overlap"]):
        raise ValueError("BM25 and FAISS must use the same corpus and chunk settings")
    if bm25_settings["version"] != 2 or vector_settings["limit"] is not None:
        raise ValueError("Hybrid search requires the full, cleaned corpus in both indexes")
    return fuse_rankings(search(bm25_index, query, candidate_k),
                         vector_search(store, query, candidate_k), alpha, top_k)
