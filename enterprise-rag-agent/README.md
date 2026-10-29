# Enterprise RAG & Agentic AI Knowledge Assistant

A working, locally-runnable knowledge assistant that combines **Retrieval-Augmented
Generation** with **agentic tool use**: instead of a fixed "retrieve then answer"
pipeline, Claude itself decides when to search the knowledge base, when to run a
calculation, and when it already has enough information to answer — and it always
cites its sources.

## Features

- 📄 **Multi-format ingestion**: PDF, DOCX, TXT, MD
- ✂️ **Smart chunking**: sentence-boundary-aware, configurable size/overlap
- 🔍 **Hybrid retrieval**: FAISS dense vector search + BM25 keyword search, blended
- 🧠 **Graceful embedding fallback**: uses `sentence-transformers` when available,
  automatically falls back to a TF-IDF vectorizer if offline — the app never breaks
- 🤖 **Agentic orchestration**: Claude's native tool-use loop decides which tools to
  call and how many times, not a hardcoded pipeline
- 🛠️ **Tools**: knowledge base search, safe calculator, document listing
- 💬 **Streamlit chat UI** with visible tool-call traces and source citations
- 💾 **Persistent index**: build once, reload on next launch
- ✅ **Test suite** covering every component that doesn't require a live API call

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Documents   │────▶│  Ingestion   │────▶│    Chunking    │
│ (PDF/DOCX/TXT)     │ (text extract)     │ (sentence-aware) │
└─────────────┘     └──────────────┘     └───────┬───────┘
                                                    ▼
                                          ┌───────────────────┐
                                          │   Embeddings        │
                                          │ (sentence-transf.    │
                                          │  or TF-IDF fallback)│
                                          └─────────┬─────────┘
                                                    ▼
                                    ┌───────────────────────────┐
                                    │   Hybrid Vector Store        │
                                    │  FAISS (dense) + BM25 (sparse)│
                                    └─────────────┬─────────────┘
                                                    ▼
     User ──▶ Streamlit Chat ──▶  KnowledgeAgent (tool-use loop)
                                    │        │           │
                                    ▼        ▼           ▼
                          kb_search   calculator   list_documents
                                    │
                                    ▼
                          Claude (claude-sonnet-5) generates final,
                          cited answer
```

## Project layout

```
enterprise-rag-agent/
├── app.py                    # Streamlit UI (entry point)
├── config.py                 # all tunables in one place
├── requirements.txt
├── .env.example
├── src/
│   ├── ingestion.py           # file loading + chunking
│   ├── embeddings.py          # embedding backend with offline fallback
│   ├── vectorstore.py         # FAISS + BM25 hybrid retrieval
│   ├── tools.py                # agent tool definitions
│   └── agent.py                # Claude tool-use orchestration loop
├── data/sample_docs/           # 3 sample enterprise docs (HR, IT, Product FAQ)
├── storage/                    # persisted index (created on first build)
└── tests/test_pipeline.py       # pytest suite, no API key needed
```

## Setup

1. **Install dependencies**

   ```bash
   cd enterprise-rag-agent
   pip install -r requirements.txt
   ```

2. **Add your Anthropic API key**

   ```bash
   cp .env.example .env
   # edit .env and paste your key from console.anthropic.com
   ```

   (You can also paste the key directly into the sidebar at runtime instead.)

3. **Run it**

   ```bash
   streamlit run app.py
   ```

4. In the sidebar: keep "Include bundled sample documents" checked (or upload your
   own), click **Build / Rebuild Index**, then start chatting.

   Try asking:
   - "How many days of annual leave do employees get, and does unused leave carry over?"
   - "What's the API rate limit on the Enterprise tier, and what would that be per second?" *(forces the agent to chain knowledge_base_search → calculator)*
   - "What documents do you have access to?"

## Running the tests

```bash
pytest tests/ -v
```

All 8 tests exercise ingestion, chunking, embeddings, hybrid retrieval, index
persistence, and the calculator's safety guard — none require an API key, so
you can verify the pipeline works before ever touching billing.

## Notes on the embedding fallback

`sentence-transformers` downloads model weights from Hugging Face on first use.
If that download isn't possible (offline, locked-down network), `EmbeddingProvider`
automatically falls back to a `scikit-learn` TF-IDF vectorizer — no code changes
needed, no crash. You can see which backend is active in the sidebar under
"Index Stats". On a normal internet connection, the real embedding model loads
automatically and gives noticeably better semantic recall.

## Extending it

- **Add a tool**: write a function + JSON schema in `src/tools.py`, register it in
  `build_tool_registry`, add the schema to `TOOL_SCHEMAS`. The agent will start
  using it automatically — no changes to `agent.py` needed.
- **Swap the LLM**: change `CLAUDE_MODEL` in `.env` (e.g. to `claude-opus-4-8` for
  harder reasoning, or `claude-haiku-4-5-20251001` for lower latency/cost).
- **Add a web-search tool**: wire in Tavily/Serper/Bing the same way as the other
  tools — same registration pattern.
- **Swap vector DB**: `VectorStore` is a thin wrapper; swapping FAISS for
  Pinecone/Weaviate/Chroma only touches `src/vectorstore.py`.

## Known limitations

- The calculator only supports `+ - * / % ** ()` on numeric literals (by design,
  for safety — no `eval` of arbitrary Python).
- BM25 and FAISS indices are rebuilt from scratch on "Build / Rebuild Index"
  rather than incrementally updated — fine for demo/portfolio scale, would need
  incremental indexing for a large production corpus.
- Single-node, in-process FAISS — for enterprise-scale (millions of chunks) you'd
  want a managed vector DB with sharding.
