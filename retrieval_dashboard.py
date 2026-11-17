"""Run: uv run streamlit run retrieval_dashboard.py"""

from contextlib import nullcontext
from datetime import datetime, timezone
import json
import time
import uuid

import streamlit as st
from retrieval import CORPUS, ROOT, build_index, search
from vector_retrieval import MODEL, build_vector_index, prepare_store, vector_search
from hybrid_retrieval import TUNING_REPORT, hybrid_search, load_tuned_alpha
from rag import LLM_MODEL, assemble_context, citation_status, generate_answer, make_payload
from ir_rag_comparison import answer_from_context


st.set_page_config(page_title="BioASQ RAG Lab", page_icon="🔎", layout="wide")
st.title("BioASQ RAG Lab")
st.caption("Follow a question from retrieval to a grounded GPT-4o mini answer. Inspect every context excerpt and the exact prompt.")


@st.cache_resource(show_spinner=False)
def load_index(size, overlap_words, source_mtime, source_size, index_version=2):
    return build_index(chunk_size=size, overlap=overlap_words)


with st.sidebar:
    st.header("Retrieval settings")
    method = st.selectbox("Retrieval method", ["NumPy + Gemini", "BM25", "Hybrid"], index=2)
    chunk_size = st.slider("Chunk size (words)", 50, 500, 500, step=25)
    overlap = st.slider("Overlap (words)", 0, chunk_size - 1, min(75, chunk_size - 1))
    top_k = st.slider("Number of results", 1, 20, 5)
    st.caption("Changing chunk size or overlap uses a separate index.")
    store, vector_metadata = None, None
    if CORPUS.exists():
        store, _ = prepare_store(chunk_size=chunk_size, overlap=overlap)
        if (store / "ready.json").exists():
            vector_metadata = json.loads((store / "ready.json").read_text())
    alpha = None
    if method == "Hybrid":
        tuned_alpha = load_tuned_alpha(store) if vector_metadata else None
        alpha = st.slider("BM25 weight (α)", 0.0, 1.0, float(tuned_alpha) if tuned_alpha is not None else 0.5, step=0.05)
        st.caption(f"BM25: {alpha:.0%} · Embedding: {1-alpha:.0%}")
        if tuned_alpha is None:
            st.info("No tuned weight for these settings. Run: uv run python tune_hybrid.py")
        else:
            st.caption(f"Ground-truth tuned BM25 weight: {tuned_alpha:.2f}")
            with st.expander("Held-out passage evaluation"):
                report = json.loads(TUNING_REPORT.read_text())
                st.dataframe([{"strategy": name, **values} for name, values in report["held_out"].items()], hide_index=True)
                st.caption("Metrics describe the saved tuned weight. Manual overrides do not change these results.")
    if method != "BM25":
        st.caption(f"Embedding model: {MODEL}")
        st.caption("Vector search covers up to the first 5,000 chunks; BM25 searches the full corpus. Hybrid combines both scopes.")
        st.caption("The first 5,000 corpus chunks and search queries are embedded through Google AI Studio; vectors are stored locally in NumPy.")
        build_clicked = st.button("Build / resume vector index")
    else:
        build_clicked = False
        st.caption("Local BM25 keyword search with English stemming.")
    st.header("Generation settings")
    st.write("GPT-4o mini")
    st.caption(f"OpenRouter · {LLM_MODEL}")
    context_budget = st.slider("Context budget (body words)", 500, 10000, 2500, step=500)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.1)
    max_tokens = st.slider("Maximum answer tokens", 256, 2048, 1024, step=256)
    st.caption("The query, selected context, and displayed prompt are sent to OpenRouter for generation.")
    if st.button("Clear last run"):
        st.session_state.pop("rag_run", None)
        st.session_state.pop("comparison_run", None)
        st.rerun()

settings = {"method": method, "chunk_size": chunk_size, "overlap": overlap,
            "top_k": top_k, "alpha": alpha, "context_budget": context_budget,
            "temperature": temperature, "max_tokens": max_tokens}

if build_clicked:
    progress = st.progress(0, text="Preparing chunks…")
    try:
        build_vector_index(chunk_size=chunk_size, overlap=overlap,
                           progress=lambda done, total: progress.progress(done / total, text=f"Embedded {done:,} / {total:,} chunks"))
        st.rerun()
    except (RuntimeError, ValueError, OSError) as error:
        st.error(str(error))
    finally:
        progress.empty()

st.subheader("Compare IR and RAG using BM25")
st.markdown("**Information retrieval (IR)** ranks and returns matching passages. **RAG** takes those same passages, adds them to a prompt, and asks a language model to answer with citations.")
st.caption("One local BM25 search supplies both sides. Only RAG sends the question and selected excerpts to Google AI Studio. The biomedical corpus is research material, not medical advice.")
comparison_mode = st.radio("Choose a system", ["Compare IR and RAG", "IR only", "RAG only"], horizontal=True)
with st.form("comparison_query"):
    comparison_query = st.text_input("Question for IR and RAG", placeholder="Which genes are associated with Hirschsprung disease?")
    compare_clicked = st.form_submit_button("Run selected system", type="primary")

if compare_clicked:
    st.session_state.pop("comparison_run", None)
    if not comparison_query.strip():
        st.warning("Enter a question to compare the two systems.")
    elif not CORPUS.exists():
        st.error("The BioASQ passage file is missing.")
    else:
        try:
            with st.spinner("Searching the corpus with BM25…"):
                stat = CORPUS.stat()
                index = load_index(chunk_size, overlap, stat.st_mtime_ns, stat.st_size)
                results = search(index, comparison_query, top_k)
                context = assemble_context(results, context_budget)
            comparison = {"query": comparison_query, "results": results, "context": context, "mode": comparison_mode}
            st.session_state["comparison_run"] = comparison
            if context["sources"] and comparison_mode != "IR only":
                try:
                    with st.spinner("Generating an answer from the same BM25 passages…"):
                        comparison["answer"] = answer_from_context(comparison_query, context)
                except RuntimeError as error:
                    comparison["error"] = str(error)
        except (RuntimeError, ValueError, OSError) as error:
            st.error(str(error))

comparison = st.session_state.get("comparison_run")
if comparison:
    st.markdown("**Executed question:** " + comparison["query"])
    if comparison_mode != comparison["mode"]:
        st.info("The result below uses the previous selection. Run the selected system to update it.")
    mode = comparison["mode"]
    ir_col, rag_col = st.columns(2) if mode == "Compare IR and RAG" else (nullcontext(), nullcontext())
    if mode != "RAG only":
      with ir_col:
        st.markdown("### IR · Ranked passages")
        st.caption("BM25 returns source text ranked by keyword relevance; it does not compose an answer.")
        if not comparison["results"]:
            st.info("No matching passages were found.")
        for rank, result in enumerate(comparison["results"], 1):
            with st.expander(f"{rank}. Passage {result['passage_id']} · BM25 {result['score']:.4g}", expanded=rank == 1):
                st.write(result["text"])
    if mode != "IR only":
      with rag_col:
        st.markdown("### RAG · Generated answer")
        st.caption("Gemini answers from the BM25 excerpts within the context budget and cites their labels.")
        if not comparison["context"]["sources"]:
            st.info("No source context was found, so no answer was generated.")
        elif comparison.get("error"):
            st.error(comparison["error"])
        elif comparison.get("answer"):
            st.markdown(comparison["answer"])
            citations = citation_status(comparison["answer"], comparison["context"]["sources"])
            if citations["unknown"]:
                st.warning("Unknown citation labels: " + ", ".join(citations["unknown"]))
            if not citations["has_citations"]:
                st.warning("The answer did not cite its sources.")
            with st.expander("Excerpts sent to Gemini and citation key"):
                for source in comparison["context"]["sources"]:
                    st.markdown(f"**[{source['label']}]** Passage {source['passage_id']} · Chunk {source['chunk_number'] + 1}" + (" · truncated" if source['truncated'] else ""))
                    st.write(source["excerpt"])
            st.caption("Citation labels identify supplied excerpts; they do not verify that claims are supported.")
st.divider()

st.subheader("The RAG pipeline")
st.caption("These stages map directly to your Advanced-RAG diagram.")
for column, number, title, detail in zip(st.columns(6), range(1, 7),
        ["Indexing", "Query", "Retrieve", "Augment", "Prompt", "LLM → Response"],
        ["Documents → chunks → index", "Question → query vector", "Rank relevant source chunks",
         "Assemble the retrieved context", "Question + context + instructions", "GPT-4o mini + source citations"]):
    with column:
        with st.container(border=True):
            st.markdown(f"**{number:02d} · {title}**")
            st.caption(detail)
with st.expander("Your architecture diagram and stage mapping"):
    diagram = ROOT / "assets/advanced-rag.png"
    if diagram.exists():
        st.image(str(diagram), caption="Reference: Advanced-RAG.png", width="stretch")
    st.markdown("**01 Indexing** is the top row and vector store. **02 Query** is the user/query branch. "
                "**03 Retrieve** is search into the store and returned contexts. **04–05 Augment / Prompt** "
                "is the lower-right box combining query and relevant contexts. **06 LLM / Response** is the return path to the user.")

st.subheader("01 · Indexing — Documents → Chunks → Embedding model → Vector store")
metrics = st.columns(4)
metrics[0].metric("Original passages", "40,221" if CORPUS.exists() else "Missing dataset")
metrics[1].metric("Chunk size / overlap", f"{chunk_size} / {overlap} words")
metrics[2].metric("NumPy indexed chunks", f"{vector_metadata['chunks']:,}" if vector_metadata else "Not built")
metrics[3].metric("Vector dimensions", vector_metadata["dimensions"] if vector_metadata else "—")
st.caption("Corpus: BioASQ passages. Blank / literal 'nan' passages are excluded. Question-answer labels are reserved for evaluation.")
with st.expander("Inspect corpus and index details"):
    st.write(f"**Embedding model:** {MODEL}")
    st.write("**Stores:** local NumPy cosine search and SQLite FTS5 for BM25.")
    st.write("**Chunks:** overlapping windows of whitespace-separated words; the final chunk can be shorter.")
    if vector_metadata:
        st.json(vector_metadata, expanded=False)

st.subheader("02 · Query — User → Query → Query embedding")
with st.form("rag_query"):
    query = st.text_area("Your query", placeholder="Which genes are associated with Hirschsprung disease?", height=85, max_chars=4000)
    buttons = st.columns([1, 1, 3])
    run_full = buttons[0].form_submit_button("Run full RAG", type="primary")
    retrieve_only = buttons[1].form_submit_button("Retrieve only")


def complete_run(run):
    if not run["context"]["sources"]:
        run["generation_error"] = "No source context was retrieved. Try another query before generating an answer."
        return
    run.pop("generation_error", None)
    try:
        run["generation"] = generate_answer(run["payload"])
    except RuntimeError as error:
        run["generation_error"] = str(error)


if run_full or retrieve_only:
    if not query.strip():
        st.warning("Enter a query to start the pipeline.")
    elif not CORPUS.exists():
        st.error("The BioASQ passage file is missing.")
    else:
        st.session_state.pop("rag_run", None)
        try:
            started = time.perf_counter()
            with st.spinner("03 · Retrieving source chunks…"):
                if method != "BM25" and not vector_metadata:
                    raise RuntimeError("Build / resume the vector index for these settings first.")
                if method in ("BM25", "Hybrid"):
                    stat = CORPUS.stat()
                    bm25_index = load_index(chunk_size, overlap, stat.st_mtime_ns, stat.st_size)
                if method == "Hybrid":
                    results = hybrid_search(bm25_index, store, query, alpha, top_k)
                elif method == "BM25":
                    results = search(bm25_index, query, top_k)
                else:
                    results = vector_search(store, query, top_k)
            context = assemble_context(results, context_budget)
            run = {"id": uuid.uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(),
                   "query": query, "settings": settings.copy(), "retrieved_chunks": results,
                   "retrieval_seconds": time.perf_counter() - started, "context": context,
                   "payload": make_payload(query, context, temperature, max_tokens)}
            st.session_state["rag_run"] = run
            if run_full:
                with st.spinner("06 · GPT-4o mini is answering from the retrieved context…"):
                    complete_run(run)
        except (RuntimeError, ValueError, OSError) as error:
            st.error(str(error))

run = st.session_state.get("rag_run")
if not run:
    st.info("Enter a question and choose Run full RAG, or use Retrieve only to inspect context and prompt before generation.")
    st.stop()
if settings != run["settings"]:
    st.info("Controls have changed. The saved run below keeps its original settings and prompt. Run retrieval again to apply the new settings.")
st.markdown("**Executed query**")
st.text(run["query"])
st.caption(f"Executed strategy: {run['settings']['method']} · Query rewriting: none · "
           f"Query embedding: {'not used for BM25' if run['settings']['method'] == 'BM25' else MODEL}")

st.subheader("03 · Retrieve — Search → Retrieved context")
st.caption(f"{len(run['retrieved_chunks'])} chunks returned in {run['retrieval_seconds']:.2f}s. Complete retriever outputs before the context budget is applied.")
source_by_key = {(source["passage_id"], source["chunk_number"]): source for source in run["context"]["sources"]}
if not run["retrieved_chunks"]:
    st.warning("No matching chunks. Generation is paused until source context is available.")
for rank, result in enumerate(run["retrieved_chunks"], 1):
    source = source_by_key.get((result["passage_id"], result["chunk_number"]))
    with st.container(border=True):
        title, score = st.columns([4, 1])
        title.markdown(f"**{rank}. Passage {result['passage_id']} · Chunk {result['chunk_number'] + 1}**")
        score.caption(f"Score: {result['score']:.6g}")
        score_name = {"Hybrid": "Weighted RRF", "BM25": "BM25", "NumPy + Gemini": "Cosine similarity"}[run["settings"]["method"]]
        included = f"Included as [{source['label']}]" if source else "Excluded by context budget"
        st.caption(f"{score_name} · {len(result['text'].split())} words · {included}" + (" · excerpt truncated" if source and source["truncated"] else ""))
        if run["settings"]["method"] == "Hybrid":
            st.caption(f"BM25 rank: {result['bm25_rank'] or 'outside top 100'} · Embedding rank: {result['embedding_rank'] or 'outside top 100'}")
        st.text(result["text"])

st.subheader("04 · Augment — Assembled context")
st.caption(f"{len(run['context']['sources'])} labeled excerpts · {run['context']['words']:,} body words / {run['context']['max_words']:,} budget. This exact context is included in the prompt.")
st.text_area("Context sent to GPT-4o mini", value=run["context"]["text"], height=300, disabled=True, key=f"context-{run['id']}")
st.subheader("05 · Prompt — Instructions + Query + Relevant contexts")
prompt_tab, api_tab = st.tabs(["Readable prompt", "Exact API payload"])
with prompt_tab:
    st.text_area("System instructions", value=run["payload"]["messages"][0]["content"], height=180, disabled=True, key=f"system-{run['id']}")
    st.text_area("User prompt: query + retrieved context", value=run["payload"]["messages"][1]["content"], height=320, disabled=True, key=f"prompt-{run['id']}")
with api_tab:
    st.json(run["payload"], expanded=True)
st.caption(f"Requested model: {run['payload']['model']} · Temperature: {run['payload']['temperature']} · Maximum answer tokens: {run['payload']['max_tokens']}")
st.subheader("06 · Generate — LLM → Response → User")
if st.button("Regenerate answer" if run.get("generation") else "Generate answer", disabled=not run["context"]["sources"], type="primary"):
    run.pop("generation", None)
    with st.spinner("GPT-4o mini is answering from the saved context and prompt…"):
        complete_run(run)
if run.get("generation_error"):
    st.error(run["generation_error"])
if run.get("generation"):
    generated = run["generation"]
    st.markdown(generated["answer"])
    st.caption(f"Model: {generated['model']} · Generation time: {generated['seconds']:.2f}s · Finish reason: {generated['finish_reason']}")
    if generated["finish_reason"] == "length":
        st.warning("The answer reached its output-token limit. Increase the answer limit and run the pipeline again for a longer answer.")
    citations = citation_status(generated["answer"], run["context"]["sources"])
    if citations["unknown"]:
        st.warning("Unknown source labels in the answer: " + ", ".join(citations["unknown"]))
    if not citations["has_citations"]:
        st.caption("No source labels were cited in this answer.")
    with st.expander("Source reference key and token usage", expanded=True):
        for source in run["context"]["sources"]:
            st.markdown(f"**[{source['label']}]** Passage `{source['passage_id']}` · Chunk {source['chunk_number'] + 1}")
        st.caption("Labels map to the excerpts above. Label checks do not verify factual support.")
        st.json(generated["usage"], expanded=False)
else:
    st.info("The context and prompt are ready. Choose Generate answer to send this exact prompt to GPT-4o mini.")
st.download_button("Download complete RAG trace", data=json.dumps(run, indent=2), file_name="rag_trace.json", mime="application/json")
