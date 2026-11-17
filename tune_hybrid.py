"""Tune hybrid weights on development queries; evaluate once on held-out queries."""

import ast
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import pyarrow.parquet as pq

from hybrid_retrieval import CANDIDATE_K, RRF_K, TUNING_REPORT, fuse_rankings
from retrieval import CORPUS, INDEX_DIR, ROOT, build_index, search
from vector_retrieval import build_vector_index, embed, normalize

QUESTIONS = ROOT / "data/rag-mini-bioasq/data/test.parquet/part.0.parquet"


def passage_metrics(results, relevant, k=10):
    """Binary relevance, unique source passages; unavailable labels remain in denominator."""
    passages = list(dict.fromkeys(str(row["passage_id"]) for row in results))[:k]
    hits = [int(passage in relevant) for passage in passages]
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return {"ndcg@10": dcg / ideal if ideal else 0.0,
            "recall@10": sum(hits) / len(relevant) if relevant else 0.0,
            "mrr@10": next((1 / (rank + 1) for rank, hit in enumerate(hits) if hit), 0.0)}


def split_questions(rows, seed=42):
    # Keep duplicate question text in one split, preventing exact-query leakage.
    groups = sorted(set(" ".join(row["question"].lower().split()) for row in rows))
    random.Random(seed).shuffle(groups)
    development = set(groups[:int(len(groups) * 0.8)])
    return ["development" if " ".join(row["question"].lower().split()) in development
            else "held_out" for row in rows]


def mean_metrics(values):
    return {name: sum(row[name] for row in values) / len(values) for name in values[0]}


def select_alpha(scores):
    # Resolve exact metric ties towards equal weights, without looking at held-out results.
    return max(scores, key=lambda row: (row["ndcg@10"], -abs(row["alpha"] - 0.5)))["alpha"]


def main():
    store = build_vector_index()
    metadata = json.loads((store / "ready.json").read_text())
    chunks = json.loads((store / "chunks.json").read_text())
    available = {row["passage_id"] for row in chunks}
    bm25_index = build_index()
    rows = pq.read_table(QUESTIONS).to_pylist()
    labels = [set(str(value) for value in ast.literal_eval(row["relevant_passage_ids"])) for row in rows]
    if any(not label for label in labels):
        raise ValueError("Empty relevance labels require an explicit evaluation policy")
    splits = split_questions(rows)
    settings = {name: metadata[name] for name in ["model", "sha256", "chunk_size", "overlap", "limit"]}
    settings.update(bm25_version=2, candidate_k=CANDIDATE_K, rrf_k=RRF_K,
                    questions_sha256=hashlib.sha256(QUESTIONS.read_bytes()).hexdigest())
    cache_key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:20]
    cache = INDEX_DIR / f"hybrid-eval-{cache_key}"
    cache.mkdir(exist_ok=True)
    batch_size = 32

    def query_batch(start):
        path = cache / f"queries-{start:05d}.npy"
        if path.exists():
            return normalize(np.load(path, allow_pickle=False))
        vectors = embed([row["question"] for row in rows[start:start + batch_size]],
                        "search_query", metadata["model"])
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, vectors, allow_pickle=False)
        temporary.replace(path)
        return vectors

    print(f"Embedding {len(rows):,} ground-truth queries (cached, batches of {batch_size})", flush=True)
    all_vectors = []
    starts = list(range(0, len(rows), batch_size))
    with ThreadPoolExecutor(max_workers=4) as executor:
        for group in range(0, len(starts), 4):
            futures = [executor.submit(query_batch, start) for start in starts[group:group + 4]]
            for future in futures:
                all_vectors.append(future.result())
            print(f"Query embeddings {min((group + 4) * batch_size, len(rows)):,}/{len(rows):,}", flush=True)
    candidates_path = cache / "candidates.json"
    if candidates_path.exists():
        candidates = json.loads(candidates_path.read_text())
    else:
        vectors = np.concatenate(all_vectors)
        corpus_vectors = np.load(store / "vectors.npy", mmap_mode="r", allow_pickle=False)
        if vectors.shape != (len(rows), corpus_vectors.shape[1]):
            raise ValueError("Query embedding shape does not match corpus index")
        print("Searching NumPy vectors for all queries…", flush=True)
        positions = []
        for start in range(0, len(vectors), 64):
            scores = vectors[start:start + 64] @ corpus_vectors.T
            positions.extend(np.argsort(-scores, axis=1, kind="stable")[:, :CANDIDATE_K])

        def rank_query(number):
            bm25 = search(bm25_index, rows[number]["question"], CANDIDATE_K)
            slim = lambda result: {"passage_id": result["passage_id"], "chunk_number": result["chunk_number"]}
            return {"bm25": [slim(result) for result in bm25],
                    "embedding": [slim(chunks[int(position)]) for position in positions[number] if position >= 0]}

        candidates = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for number, result in enumerate(executor.map(rank_query, range(len(rows))), 1):
                candidates.append(result)
                if number % 250 == 0:
                    print(f"BM25 rankings {number:,}/{len(rows):,}", flush=True)
        temporary = candidates_path.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(candidates))
        temporary.replace(candidates_path)

    dev_ids = [i for i, split in enumerate(splits) if split == "development"]
    test_ids = [i for i, split in enumerate(splits) if split == "held_out"]
    grid = []
    for alpha in np.linspace(0, 1, 21):
        values = []
        for i in dev_ids:
            candidate = candidates[i]
            ranking = fuse_rankings(candidate["bm25"], candidate["embedding"], float(alpha), 200)
            values.append(passage_metrics(ranking, labels[i]))
        result = {"alpha": float(alpha), **mean_metrics(values)}
        grid.append(result)
        print(f"Development alpha={alpha:.2f}: nDCG@10={result['ndcg@10']:.4f}", flush=True)
    selected = select_alpha(grid)
    held_out = {}
    per_query = []
    for strategy, alpha in [("bm25", 1.0), ("embedding", 0.0), ("hybrid", selected)]:
        values = []
        for i in test_ids:
            candidate = candidates[i]
            metrics = passage_metrics(fuse_rankings(candidate["bm25"], candidate["embedding"], alpha, 200), labels[i])
            values.append(metrics)
            per_query.append({"question_id": rows[i]["id"], "strategy": strategy,
                              "relevant_passages": len(labels[i]),
                              "available_relevant_passages": len(labels[i] & available), **metrics})
        held_out[strategy] = mean_metrics(values)

    # Paired resampling of held-out per-query nDCG differences; no weight selection here.
    intervals = {}
    rng = np.random.default_rng(42)
    hybrid_scores = np.array([r["ndcg@10"] for r in per_query if r["strategy"] == "hybrid"])
    for strategy in ["bm25", "embedding"]:
        baseline = np.array([r["ndcg@10"] for r in per_query if r["strategy"] == strategy])
        differences = hybrid_scores - baseline
        boot = np.array([rng.choice(differences, size=len(differences), replace=True).mean() for _ in range(2000)])
        intervals[strategy] = {"mean_difference": float(differences.mean()),
                               "bootstrap_95_ci": np.percentile(boot, [2.5, 97.5]).tolist()}
    coverage = {}
    for name, ids in [("all", list(range(len(rows)))), ("development", dev_ids), ("held_out", test_ids)]:
        coverage[name] = {"questions": len(ids),
                         "questions_with_available_relevant_passages": sum(bool(labels[i] & available) for i in ids),
                         "relevant_labels": sum(len(labels[i]) for i in ids),
                         "available_relevant_labels": sum(len(labels[i] & available) for i in ids)}
    report = {"settings": settings, "selected_alpha": selected,
              "embedding_weight": 1-selected, "selection_metric": "development passage nDCG@10",
              "formula": "alpha / (60 + BM25_rank) + (1-alpha) / (60 + embedding_rank)",
              "seed": 42, "split": "80/20 by normalized question text; duplicate texts grouped",
              "ground_truth_policy": "Original test split repartitioned for tuning/evaluation. Binary passage labels; deduplicate source IDs before top-10. Unavailable labels retained in denominators.",
              "coverage": coverage, "development_grid": grid, "held_out": held_out,
              "held_out_ndcg_differences": intervals,
              "split_question_ids": {"development": [rows[i]["id"] for i in dev_ids],
                                     "held_out": [rows[i]["id"] for i in test_ids]}}
    TUNING_REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = TUNING_REPORT.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(TUNING_REPORT)
    for filename, data in [("development_grid.csv", grid), ("held_out_per_query.csv", per_query)]:
        with (TUNING_REPORT.parent / filename).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps({"selected_alpha": selected, "held_out": held_out, "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
