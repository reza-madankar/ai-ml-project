"""OpenRouter embeddings and a resumable, local FAISS cosine-similarity store."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import faiss
import numpy as np
import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv

from retrieval import CORPUS, INDEX_DIR, ROOT, chunk_text

MODEL = "perplexity/pplx-embed-v1-0.6b"


def embed(texts, input_type, model=MODEL):
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAI_ROUTER_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_ROUTER_KEY in the project's .env file.")
    payload = {"model": model, "input": texts, "encoding_format": "float"}
    # Perplexity uses the same unprefixed text encoding for documents and queries.
    if not model.startswith("perplexity/"):
        payload["input_type"] = input_type
    for attempt in range(4):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={"Authorization": f"Bearer {key}"},
                json=payload, timeout=(15, 120),
            )
        except requests.RequestException:
            if attempt == 3:
                raise RuntimeError("OpenRouter connection failed; retry to resume indexing.") from None
            time.sleep(2 ** attempt)
            continue
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            try:
                delay = float(response.headers.get("Retry-After", 2 ** attempt))
            except ValueError:
                delay = 2 ** attempt
            time.sleep(min(max(delay, 0), 30))
            continue
        if not response.ok:
            # Avoid displaying provider bodies or request headers containing secrets.
            raise RuntimeError(f"OpenRouter embeddings returned HTTP {response.status_code}; "
                               "check model availability, credentials, or account limits. "
                               "Completed batches are saved; rerun to resume.")
        try:
            data = sorted(response.json()["data"], key=lambda row: row["index"])
            if [row["index"] for row in data] != list(range(len(texts))):
                raise ValueError("Embedding count/order mismatch")
            vectors = np.asarray([row["embedding"] for row in data], dtype="float32")
            return normalize(vectors)
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("OpenRouter returned invalid embedding data.") from None


def normalize(vectors):
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    if vectors.ndim != 2 or not vectors.shape[1] or not np.isfinite(vectors).all():
        raise ValueError("Expected a finite two-dimensional embedding matrix")
    if (np.linalg.norm(vectors, axis=1) == 0).any():
        raise ValueError("Zero-length embeddings cannot be used for cosine similarity")
    faiss.normalize_L2(vectors)
    return vectors


def prepare_store(corpus=CORPUS, chunk_size=500, overlap=75,
                  index_dir=INDEX_DIR, model=MODEL, limit=None):
    list(chunk_text("", chunk_size, overlap))
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    settings = {"version": 1, "model": model, "chunk_size": chunk_size,
                "overlap": overlap, "limit": limit, "skip_literal_nan": True,
                "document_input_type": "search_document", "query_input_type": "search_query",
                "sha256": hashlib.sha256(Path(corpus).read_bytes()).hexdigest()}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:20]
    store = Path(index_dir) / f"faiss-{key}"
    store.mkdir(parents=True, exist_ok=True)
    return store, settings


def collect_chunks(corpus, settings):
    chunks = []
    for batch in pq.ParquetFile(corpus).iter_batches(columns=["id", "passage"]):
        for passage in batch.to_pylist():
            text = passage["passage"]
            if not isinstance(text, str) or not text.strip() or text.strip().lower() == "nan":
                continue
            for number, chunk in chunk_text(text, settings["chunk_size"], settings["overlap"]):
                chunks.append(dict(chunk_id=len(chunks), passage_id=str(passage["id"]),
                                   chunk_number=number, text=chunk))
                if settings["limit"] and len(chunks) >= settings["limit"]:
                    return chunks
    return chunks


def build_vector_index(corpus=CORPUS, chunk_size=500, overlap=75, index_dir=INDEX_DIR,
                       model=MODEL, batch_size=32, limit=None, progress=None):
    store, _ = prepare_store(corpus, chunk_size, overlap, index_dir, model, limit)
    with (store / "build.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This index is already being built by another process.") from None
        return _build_vector_index(corpus, chunk_size, overlap, index_dir,
                                   model, batch_size, limit, progress)


def _build_vector_index(corpus=CORPUS, chunk_size=500, overlap=75, index_dir=INDEX_DIR,
                        model=MODEL, batch_size=32, limit=None, progress=None):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    store, settings = prepare_store(corpus, chunk_size, overlap, index_dir, model, limit)
    # Separate checkpoints when batch size changes; the final index is independent of it.
    if (store / "ready.json").exists():
        return store
    chunks = collect_chunks(corpus, settings)
    if not chunks:
        raise ValueError("No usable passages in corpus")
    batches = store / f"batches-{batch_size}"
    batches.mkdir(exist_ok=True)
    def load_batch(start):
        end = min(start + batch_size, len(chunks))
        checkpoint = batches / f"{start:08d}.npy"
        if checkpoint.exists():
            vectors = normalize(np.load(checkpoint, allow_pickle=False))
        else:
            vectors = embed([row["text"] for row in chunks[start:end]], "search_document", model)
            temporary = checkpoint.with_suffix(".tmp.npy")
            np.save(temporary, vectors, allow_pickle=False)
            temporary.replace(checkpoint)
        if len(vectors) != end - start:
            raise ValueError("Checkpoint row count mismatch")
        return end, vectors

    index = None
    # Bound concurrent requests and queued work; checkpoints remain usable on error.
    starts = list(range(0, len(chunks), batch_size))
    with ThreadPoolExecutor(max_workers=4) as executor:
        for group in range(0, len(starts), 4):
            futures = [executor.submit(load_batch, start) for start in starts[group:group + 4]]
            for future in futures:
                end, vectors = future.result()
                if index is None:
                    index = faiss.IndexFlatIP(vectors.shape[1])
                if vectors.shape[1] != index.d:
                    raise ValueError("Embedding dimensions changed; cannot combine batches")
                index.add(vectors)
                if progress:
                    progress(end, len(chunks))
    faiss.write_index(index, str(store / "index.tmp.faiss"))
    (store / "index.tmp.faiss").replace(store / "index.faiss")
    (store / "chunks.json").write_text(json.dumps(chunks))
    metadata = {**settings, "chunks": len(chunks), "dimensions": index.d}
    (store / "ready.tmp.json").write_text(json.dumps(metadata, indent=2))
    (store / "ready.tmp.json").replace(store / "ready.json")
    return store


def vector_search(store, query, top_k=5):
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    if not query.strip():
        return []
    store = Path(store)
    metadata = json.loads((store / "ready.json").read_text())
    chunks = json.loads((store / "chunks.json").read_text())
    index = faiss.read_index(str(store / "index.faiss"))
    vector = embed([query], "search_query", metadata["model"])
    if vector.shape[1] != index.d:
        raise ValueError("Query embedding dimensions do not match index")
    scores, positions = index.search(vector, min(top_k, index.ntotal))
    return [{**chunks[int(position)], "score": float(score)}
            for score, position in zip(scores[0], positions[0]) if position >= 0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Only index this many chunks for a smoke test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query", help="Search after building")
    args = parser.parse_args()
    try:
        store = build_vector_index(limit=args.limit, batch_size=args.batch_size,
                                   progress=lambda done, total: print(f"Embedded {done:,}/{total:,}", flush=True))
        print(f"FAISS store: {store}")
        if args.query:
            print(json.dumps(vector_search(store, args.query), indent=2))
    except (RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")
