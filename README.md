# enterprise-rag-agentic-ai-assistant
Enterprise RAG knowledge assistant with hybrid retrieval and agentic AI. 

Large Language Models are powerful but don't know an organization's private/internal documents, and they
can hallucinate facts. The goal was to build a system that grounds LLM answers in real retrieved documents
(RAG), while also giving the model the ability to reason about *when* to retrieve vs. use a tool vs. answer
directly — making it agentic rather than a simple fixed retrieve-then-answer pipeline.

Tech Stack used - FAISS, BM25, Claude API(tool-use), Streamlit, SQLite, Docker
