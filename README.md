# ai-ml-project
AI Learning

## Chat dashboard

Run `uv run streamlit run chat_dashboard.py` from the project directory.
The chat uses Google AI Studio; put `GOOGLE_API_KEY=your_key` in the project's
`.env` file (or export it in your shell). Choose the chat model under
`providers.gemini.model` in `config.yaml`. If the key is missing, the app
shows setup instructions. Streamlit uses polling for file changes to avoid
exhausting Linux inotify instances; automatic reload remains available.

## Compare IR and RAG

In `uv run streamlit run retrieval_dashboard.py`, choose **IR only**, **RAG only**, or **Compare IR and RAG**, then enter a question. One local BM25 search returns ranked passages for the IR side; the RAG side sends excerpts from those same results to Gemini via Google AI Studio to produce an answer with source labels. The sidebar controls chunk size, overlap, number of results, and the context budget. Set `GOOGLE_API_KEY` in `.env` for generation. BM25 retrieval itself requires no API key. The comparison is separate from the detailed pipeline below it.

## Complete RAG dashboard

Run `uv run streamlit run retrieval_dashboard.py` and open http://localhost:8501.
The dashboard now implements the full pipeline using **GPT-4o mini**
(`openai/gpt-4o-mini`) through the existing OpenRouter connection. The model's
[official documentation](https://developers.openai.com/api/docs/models/gpt-4o-mini)
describes GPT-4o mini; this project uses OpenRouter's Chat Completions endpoint.
The detailed pipeline uses `OPENAI_ROUTER_KEY` for generation and Google AI Studio keys for embeddings.

The numbered stages correspond to the supplied Advanced-RAG diagram, included
under **Your architecture diagram and stage mapping**:

1. **Indexing:** corpus, chunk settings, embedding model, NumPy vector count and dimensions.
2. **Query:** the entered question and selected retrieval strategy; query rewriting is not performed.
3. **Retrieve:** ranked source chunks, scores, hybrid branch ranks, and which excerpts enter the prompt.
4. **Augment:** exact assembled context with source labels and a configurable body-word budget.
5. **Prompt:** system instructions, user question plus context, and the complete API payload.
6. **Generate:** GPT-4o mini's answer, source reference key, finish reason, latency and token usage.

**Run full RAG** retrieves and generates in one step. **Retrieve only** allows
inspection before pressing **Generate answer**. Hybrid defaults to an untuned 50% BM25 / 50% embedding weight for the new Gemini index; the previous model’s tuning does not apply.
All three retrieval methods remain available.

Context is assembled in retrieval order. Repeated identical source chunks are
removed; excerpts are truncated or omitted when the context-word budget is
exhausted, with this shown alongside the full retrieved chunks. The exact
displayed API payload is sent to the model. The prompt requests answers only
from supplied excerpts, treats source text as untrusted evidence, and requests
citations such as `[S1]`. Citation-label validation checks that referenced labels
exist; it does not independently verify factual grounding.

The saved run persists across reruns, downloads, and sidebar changes. Changed
controls are flagged; the saved prompt keeps its original settings until
retrieval is rerun. Generation errors preserve context and prompt for retry.
Queries returning no chunks do not trigger generation. **Download complete RAG
trace** exports the query, retrieval settings/results, context, exact prompt,
answer and usage without API credentials.

## Retrieval implementation

From this project directory, run:

```bash
uv sync
uv run streamlit run retrieval_dashboard.py
```

The vector retrieval method embeds up to the first **5,000 chunks** using
`gemini-embedding-2` through Google AI Studio and searches a local NumPy
cosine-similarity store. BM25 searches the full corpus; hybrid combines BM25
results from the full corpus with vector results from this limited subset.
The subset follows corpus order, so vector and hybrid results do not cover the
whole dataset equally. Chunk size is 500 words with 75 words of overlap.

Set `GEMINI_KEYS` in `.env` to a comma-separated list of Google AI Studio API
keys, or set `GOOGLE_API_KEY` for a single key. Keys are rotated across requests;
Google rate limits apply per project, so keys from the same project share quota.
The indexer paces requests and saves each completed batch for resumption. Usage
may incur charges and consume quota on the projects behind the keys.

Build using **Build / resume vector index** in the sidebar, or run:

```bash
uv run python vector_retrieval.py --limit 5000
```

If interrupted or rate-limited, run the same command again to reuse completed
batches. Keep the same batch size. A small smoke test can use `--limit 32`;
it creates a separate store. The dashboard uses the 5,000-chunk store.
Query text and corpus chunks are sent to Google AI Studio for embedding.

In BM25 mode, the first search splits passages and builds a SQLite FTS5 index in
`.cache/retrieval/`. Later searches reuse it. A different corpus or chunking
configuration gets its own index. Search uses SQLite's BM25 with English
stemming and OR matching across query terms. Higher displayed scores rank
better; scores are not probabilities or comparable across queries. This is
keyword retrieval, so synonyms without matching terms may be missed.
The question/answer test set is kept out of the search index.

### Hybrid retrieval and ground-truth tuning

Select **Hybrid** as the third retrieval method. It combines the top 100 chunks
from each retriever using weighted reciprocal-rank fusion:

```text
score = alpha / (60 + BM25 rank) + (1 - alpha) / (60 + embedding rank)
```

A chunk absent from a retriever's top 100 contributes zero for that branch.
The two weights sum to one. Alpha = 1 means BM25 only, and alpha = 0 means
embedding only. Fusion uses ranks because raw BM25 and cosine scores have
different scales. The rank constant 60 is fixed; see the
[original RRF paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).

To tune alpha against the downloaded BioASQ relevance labels:

```bash
uv run python tune_hybrid.py
```

The existing saved report was computed with the previous full-corpus OpenRouter/FAISS index and does not describe the new limited Gemini/NumPy store. Retuning sends all 4,719 questions to Google for embedding and uses additional quota.

The tuning script uses all 4,719 labeled questions, splitting by normalized question text
into development (80%) and held-out evaluation (20%), with seed 42. Duplicate
question text stays in the same split. The dataset's original `test` split is
being repurposed for this local experiment; this is not an untouched official
benchmark evaluation. Answer text is never indexed or used for tuning.

The grid tests alpha from 0 to 1 in steps of 0.05 and maximizes development
**passage nDCG@10**. Source passage IDs are deduplicated before computing
nDCG@10, Recall@10, and MRR@10 because ground truth labels passages, not chunks.
Labels pointing to missing or placeholder passages stay in the metric
denominators; corpus coverage is reported separately. Held-out metrics compare
BM25, embeddings, and the selected hybrid. Paired bootstrap intervals describe
held-out nDCG differences, conditional on this split and the selected weight.

The report, development grid, exact split IDs, and per-query held-out metrics
are saved in `reports/hybrid-tuning/`. Query embeddings and rankings are cached
in `.cache/retrieval/`, so repeating the experiment reuses API results.
The dashboard automatically loads the tuned alpha when model, corpus,
chunk settings, candidate count, and fusion constant match. It shows a warning
for settings without a matching tuned weight; 0.5 is then an untuned default.
The weight slider permits manual experiments without changing the saved
evaluation. Held-out numbers describe only the saved tuned configuration.

BM25 needs no model API or API key. To use BM25 retrieval in Python:

```python
from retrieval import build_index, search

index = build_index(chunk_size=500, overlap=75)
for result in search(index, "Hirschsprung disease genes", top_k=5):
    print(result["passage_id"], result["score"], result["text"])
```
