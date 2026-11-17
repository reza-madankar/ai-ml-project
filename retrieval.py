"""Chunk BioASQ passages and search them with SQLite FTS5 BM25."""

from pathlib import Path
import hashlib
import json
import re
import sqlite3
import tempfile


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data/rag-mini-bioasq/data/passages.parquet/part.0.parquet"
INDEX_DIR = ROOT / ".cache/retrieval"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 75):
    """Yield (chunk number, text) using whitespace-separated word windows."""
    if chunk_size < 1 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    words = text.split()
    for number, start in enumerate(range(0, len(words), chunk_size - overlap)):
        yield number, " ".join(words[start : start + chunk_size])
        if start + chunk_size >= len(words):
            break


def build_index(corpus=CORPUS, chunk_size=500, overlap=75, index_dir=INDEX_DIR):
    """Build once per corpus/settings combination; reuse the on-disk index."""
    import pyarrow.parquet as pq

    # Validate settings even when the corpus is empty.
    list(chunk_text("", chunk_size, overlap))
    corpus = Path(corpus)
    fingerprint = hashlib.sha256(corpus.read_bytes()).hexdigest()
    settings = {"version": 2, "sha256": fingerprint,
                "chunk_size": chunk_size, "overlap": overlap}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:20]
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    destination = index_dir / f"bioasq-{key}.sqlite3"
    if destination.exists():
        return destination

    # Publish a complete database atomically, including when two sessions build it.
    with tempfile.NamedTemporaryFile(dir=index_dir, suffix=".sqlite3", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with sqlite3.connect(temporary) as connection:
            connection.execute(
                "CREATE VIRTUAL TABLE chunks USING fts5("
                "text, passage_id UNINDEXED, chunk_number UNINDEXED, "
                "tokenize='porter unicode61')"
            )
            for batch in pq.ParquetFile(corpus).iter_batches(columns=["id", "passage"]):
                rows = []
                for passage in batch.to_pylist():
                    text = passage["passage"]
                    if not isinstance(text, str) or not text.strip() or text.strip().lower() == "nan":
                        continue
                    for number, chunk in chunk_text(text, chunk_size, overlap):
                        rows.append((chunk, str(passage["id"]), number))
                connection.executemany(
                    "INSERT INTO chunks(text, passage_id, chunk_number) VALUES (?, ?, ?)", rows
                )
            connection.execute("CREATE TABLE metadata (settings TEXT)")
            connection.execute("INSERT INTO metadata VALUES (?)", (json.dumps(settings),))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def search(index, query: str, top_k: int = 5):
    """Return matching chunks ranked by BM25; higher displayed scores are better.

    Query terms are ORed so partial matches can rank. Quoting each token treats
    user input as text, including punctuation and FTS query operators.
    """
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    terms = list(dict.fromkeys(re.findall(r"[^\W_]+", query.lower(), re.UNICODE)))
    if not terms:
        return []
    expression = " OR ".join('"' + term + '"' for term in terms)
    with sqlite3.connect(index) as connection:
        rows = connection.execute(
            "SELECT rowid, passage_id, chunk_number, text, bm25(chunks) AS score "
            "FROM chunks WHERE chunks MATCH ? ORDER BY score, rowid LIMIT ?",
            (expression, top_k),
        ).fetchall()
    return [dict(chunk_id=row[0], passage_id=row[1], chunk_number=row[2],
                 text=row[3], score=-row[4]) for row in rows]


def index_stats(index):
    with sqlite3.connect(index) as connection:
        count = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        settings = json.loads(connection.execute("SELECT settings FROM metadata").fetchone()[0])
    return {"chunks": count, **settings}
