# ai-ml-project
AI Learning

## Complete RAG dashboard

Run `uv run streamlit run retrieval_dashboard.py` and open http://localhost:8501.
The dashboard now implements the full pipeline using **GPT-4o mini**
(`openai/gpt-4o-mini`) through the existing OpenRouter connection. The model's
[official documentation](https://developers.openai.com/api/docs/models/gpt-4o-mini)
describes GPT-4o mini; this project uses OpenRouter's Chat Completions endpoint.
The local `.env` variable `OPENAI_ROUTER_KEY` is used for embedding and generation.

The numbered stages correspond to the supplied Advanced-RAG diagram, included
under **Your architecture diagram and stage mapping**:

1. **Indexing:** corpus, chunk settings, embedding model, FAISS vector count and dimensions.
2. **Query:** the entered question and selected retrieval strategy; query rewriting is not performed.
3. **Retrieve:** ranked source chunks, scores, hybrid branch ranks, and which excerpts enter the prompt.
4. **Augment:** exact assembled context with source labels and a configurable body-word budget.
5. **Prompt:** system instructions, user question plus context, and the complete API payload.
6. **Generate:** GPT-4o mini's answer, source reference key, finish reason, latency and token usage.

**Run full RAG** retrieves and generates in one step. **Retrieve only** allows
inspection before pressing **Generate answer**. Hybrid is the default and uses
the saved 75% BM25 / 25% embedding weight for the current 500/75-word settings.
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

The default retrieval method embeds chunks and queries using
`perplexity/pplx-embed-v1-0.6b` through OpenRouter and searches a local FAISS
index with cosine similarity. Set `OPENAI_ROUTER_KEY` in `.env` (ignored by Git).
Chunk size is 500 words with 75 words of overlap. Blank and literal `nan`
passages are excluded from both indexes. Corpus chunks and search queries
are sent to OpenRouter. Generation uses the selected context as described above.

Build once using **Build / resume vector index** in the sidebar, or:

```bash
uv run python vector_retrieval.py
```

Completed embedding batches are cached under `.cache/retrieval/` and reused
after interruption or rate limiting. Keep the same batch size when resuming.
Model, corpus, and chunk settings determine which store is used. To run a small
live test without building the entire corpus:

```bash
uv run python vector_retrieval.py --limit 32 --query "Hirschsprung disease genes"
```

The limited test uses its own store; the dashboard searches the full corpus.
Select **BM25** in the sidebar to use the existing local keyword retrieval.

To search the full vector store from Python after building it:

```python
from vector_retrieval import build_vector_index, vector_search

store = build_vector_index()  # Reuses the completed store.
results = vector_search(store, "Hirschsprung disease genes", top_k=5)
```

Enter an English biomedical query to get ranked chunks from the downloaded
BioASQ corpus. Adjust the number of results, chunk size (default 500 words),
and overlap (default 75 words) in the sidebar. Each result shows its source
passage ID, chunk number, text, and cosine similarity or BM25 score.
Results can be downloaded as JSON.

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

This uses all 4,719 labeled questions, splitting by normalized question text
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
