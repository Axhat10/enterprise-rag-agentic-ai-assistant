"""
Enterprise RAG & Agentic AI Knowledge Assistant — Streamlit UI.

Run with:  streamlit run app.py
"""
import shutil
import tempfile
from pathlib import Path

import streamlit as st

import config
from src.ingestion import ingest_files
from src.embeddings import EmbeddingProvider
from src.vectorstore import VectorStore
from src.agent import KnowledgeAgent

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "store" not in st.session_state:
    embedder = EmbeddingProvider()
    store = VectorStore(embedder)
    loaded = store.load()
    st.session_state.store = store
    st.session_state.index_loaded = loaded

if "agent" not in st.session_state:
    st.session_state.agent = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: role, content, tool_calls


def get_agent(api_key: str) -> KnowledgeAgent:
    if st.session_state.agent is None:
        st.session_state.agent = KnowledgeAgent(st.session_state.store, api_key=api_key)
    return st.session_state.agent


# ---------------------------------------------------------------------------
# Sidebar: setup, upload, index management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Knowledge Assistant")
    st.caption("Enterprise RAG & Agentic AI")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=config.ANTHROPIC_API_KEY,
        help="Get one at console.anthropic.com. Stored only for this session.",
    )

    st.divider()
    st.subheader("📁 Knowledge Base")

    use_samples = st.checkbox("Include bundled sample documents", value=True)

    uploaded_files = st.file_uploader(
        "Upload documents (.pdf, .docx, .txt, .md)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("🔨 Build / Rebuild Index", use_container_width=True, type="primary"):
        with st.spinner("Ingesting documents and building index..."):
            tmp_dir = Path(tempfile.mkdtemp())
            paths = []

            if use_samples:
                for f in config.SAMPLE_DOCS_DIR.glob("*"):
                    paths.append(f)

            for uf in uploaded_files or []:
                dest = tmp_dir / uf.name
                dest.write_bytes(uf.getbuffer())
                paths.append(dest)

            if not paths:
                st.error("No documents selected. Upload files or enable sample docs.")
            else:
                try:
                    chunks = ingest_files(paths)
                    embedder = EmbeddingProvider()
                    store = VectorStore(embedder)
                    store.build(chunks)
                    store.save()
                    st.session_state.store = store
                    st.session_state.agent = None  # force re-bind to new store
                    st.session_state.index_loaded = True
                    st.success(f"Indexed {len(chunks)} chunks from {len(paths)} document(s).")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

            shutil.rmtree(tmp_dir, ignore_errors=True)

    st.divider()
    st.subheader("📊 Index Stats")
    stats = st.session_state.store.stats
    st.metric("Documents indexed", stats["num_documents"])
    st.metric("Chunks indexed", stats["num_chunks"])
    st.caption(f"Embedding backend: `{stats['embedding_backend'] or 'not built yet'}`")
    if stats["documents"]:
        with st.expander("View indexed documents"):
            for d in stats["documents"]:
                st.write(f"• {d}")

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        if st.session_state.agent:
            st.session_state.agent.reset()
        st.rerun()

# ---------------------------------------------------------------------------
# Main area: Chat
# ---------------------------------------------------------------------------
st.title("Enterprise RAG & Agentic AI Knowledge Assistant")
st.caption(
    "Ask questions about company policy, product docs, or anything you've uploaded. "
    "The agent decides on its own whether to search the knowledge base, run a "
    "calculation, or both — and always cites its sources."
)

if not stats["num_chunks"]:
    st.info("👈 Build the knowledge base index from the sidebar to get started.")

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("tool_calls"):
            with st.expander(f"🔧 {len(turn['tool_calls'])} tool call(s) used"):
                for tc in turn["tool_calls"]:
                    st.markdown(f"**{tc.name}**  \n`input: {tc.input}`")
                    st.text(tc.output[:1500] + ("..." if len(tc.output) > 1500 else ""))

user_input = st.chat_input("Ask a question about your knowledge base...")

if user_input:
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar first.")
    elif not stats["num_chunks"]:
        st.error("Please build the knowledge base index first (see sidebar).")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    agent = get_agent(api_key)
                    response = agent.ask(user_input)
                    st.markdown(response.answer)
                    if response.tool_calls:
                        with st.expander(f"🔧 {len(response.tool_calls)} tool call(s) used"):
                            for tc in response.tool_calls:
                                st.markdown(f"**{tc.name}**  \n`input: {tc.input}`")
                                st.text(
                                    tc.output[:1500]
                                    + ("..." if len(tc.output) > 1500 else "")
                                )
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": response.answer,
                            "tool_calls": response.tool_calls,
                        }
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
