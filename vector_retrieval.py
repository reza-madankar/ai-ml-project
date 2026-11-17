"""Gemini Embedding 2 and a resumable local NumPy cosine-similarity store."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv

from retrieval import CORPUS, INDEX_DIR, ROOT, chunk_text

MODEL = "gemini-embedding-2"
DIMENSIONS = 768
DEFAULT_LIMIT = 5000
MIN_REQUEST_INTERVAL = 0.65
_key_lock = threading.Lock()
_key_index = 0
_last_request = 0.0


def google_keys():
    load_dotenv(ROOT / ".env")
    keys = [value.strip() for value in os.getenv("GEMINI_KEYS", "").split(",")]
    keys.append(os.getenv("GOOGLE_API_KEY", "").strip())
    keys = list(dict.fromkeys(key for key in keys if key))
    if not keys:
        raise RuntimeError("Set GEMINI_KEYS or GOOGLE_API_KEY in the project's .env file.")
    return keys


def embed(texts, input_type, model=MODEL):
    global _key_index, _last_request
    if not texts:
        raise ValueError("At least one text is required")
    if input_type not in ("search_document", "search_query"):
        raise ValueError("Invalid embedding input type")
    keys = google_keys()
    with _key_lock:
        start = _key_index % len(keys)
        _key_index += 1
    prefix = "title: Biomedical passage | text: " if input_type == "search_document" else "task: search result | query: "
    payload = {"requests": [{"model": f"models/{model}",
                            "content": {"parts": [{"text": prefix + text}]},
                            "outputDimensionality": DIMENSIONS} for text in texts]}
    attempts = max(len(keys) * 3, 6)
    for attempt in range(attempts):
        key = keys[(start + attempt) % len(keys)]
        with _key_lock:
            delay = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request)
            if delay > 0:
                time.sleep(delay)
            _last_request = time.monotonic()
        try:
            response = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents",
                                     headers={"x-goog-api-key": key}, json=payload, timeout=(15, 120))
        except requests.RequestException:
            response = None
        if response is not None and response.ok:
            try:
                vectors = np.asarray([row["values"] for row in response.json()["embeddings"]], dtype="float32")
                if len(vectors) != len(texts):
                    raise ValueError("Embedding count mismatch")
                return normalize(vectors)
            except (KeyError, TypeError, ValueError):
                raise RuntimeError("Google returned invalid embedding data.") from None
        status = response.status_code if response is not None else None
        if status == 400 and response is not None:
            try:
                if "API key not valid" in response.json().get("error", {}).get("message", ""):
                    continue
            except (ValueError, TypeError, AttributeError):
                pass
        if status not in (None, 401, 403, 429, 500, 502, 503, 504):
            raise RuntimeError(f"Google embeddings returned HTTP {status}. Completed batches are saved; rerun to resume.")
        if attempt + 1 < attempts and (attempt + 1) % len(keys) == 0:
            delay = min(60, 30 * (attempt // len(keys) + 1))
            if response is not None:
                try:
                    delay = min(60, max(delay, float(response.headers.get("Retry-After", 0))))
                except ValueError:
                    pass
            time.sleep(delay)
    raise RuntimeError("Google embedding requests failed or reached a rate limit. Completed batches are saved; retry later to resume.")


def normalize(vectors):
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    if vectors.ndim != 2 or not vectors.shape[1] or not np.isfinite(vectors).all():
        raise ValueError("Expected a finite two-dimensional embedding matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Zero-length embeddings cannot be used for cosine similarity")
    return vectors / norms


def prepare_store(corpus=CORPUS, chunk_size=500, overlap=75, index_dir=INDEX_DIR, model=MODEL, limit=DEFAULT_LIMIT):
    list(chunk_text("", chunk_size, overlap))
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    settings = {"version": 2, "model": model, "dimensions_requested": DIMENSIONS, "chunk_size": chunk_size,
                "overlap": overlap, "limit": limit, "skip_literal_nan": True,
                "document_input_type": "search_document", "query_input_type": "search_query",
                "sha256": hashlib.sha256(Path(corpus).read_bytes()).hexdigest()}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:20]
    store = Path(index_dir) / f"numpy-{key}"
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
                chunks.append(dict(chunk_id=len(chunks), passage_id=str(passage["id"]), chunk_number=number, text=chunk))
                if settings["limit"] and len(chunks) >= settings["limit"]:
                    return chunks
    return chunks


def build_vector_index(corpus=CORPUS, chunk_size=500, overlap=75, index_dir=INDEX_DIR,
                       model=MODEL, batch_size=32, limit=DEFAULT_LIMIT, progress=None):
    if batch_size < 1 or batch_size > 100:
        raise ValueError("batch_size must be between 1 and 100")
    store, settings = prepare_store(corpus, chunk_size, overlap, index_dir, model, limit)
    with (store / "build.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This index is already being built by another process.") from None
        if (store / "ready.json").exists():
            return store
        chunks = collect_chunks(corpus, settings)
        if not chunks:
            raise ValueError("No usable passages in corpus")
        batches = store / f"batches-{batch_size}"
        batches.mkdir(exist_ok=True)
        matrix = None
        for start in range(0, len(chunks), batch_size):
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
            if matrix is None:
                matrix = np.lib.format.open_memmap(store / "vectors.tmp.npy", mode="w+", dtype="float32", shape=(len(chunks), vectors.shape[1]))
            if vectors.shape[1] != matrix.shape[1]:
                raise ValueError("Embedding dimensions changed; cannot combine batches")
            matrix[start:end] = vectors
            if progress:
                progress(end, len(chunks))
        matrix.flush()
        dimensions = matrix.shape[1]
        del matrix
        (store / "vectors.tmp.npy").replace(store / "vectors.npy")
        (store / "chunks.json").write_text(json.dumps(chunks))
        (store / "ready.tmp.json").write_text(json.dumps({**settings, "chunks": len(chunks), "dimensions": dimensions}, indent=2))
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
    vectors = np.load(store / "vectors.npy", mmap_mode="r", allow_pickle=False)
    query_vector = embed([query], "search_query", metadata["model"])[0]
    if query_vector.shape[0] != vectors.shape[1] or len(chunks) != len(vectors):
        raise ValueError("Vector store dimensions or chunk count do not match")
    scores = vectors @ query_vector
    positions = np.argsort(-scores, kind="stable")[:top_k]
    return [{**chunks[int(pos)], "score": float(scores[pos])} for pos in positions]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Only index this many chunks for a smoke test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query", help="Search after building")
    args = parser.parse_args()
    try:
        store = build_vector_index(limit=args.limit, batch_size=args.batch_size,
                                   progress=lambda done, total: print(f"Embedded {done:,}/{total:,}", flush=True))
        print(f"NumPy store: {store}")
        if args.query:
            print(json.dumps(vector_search(store, args.query), indent=2))
    except (RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")
