"""
Tools the agent can call. Each tool has:
  1. A JSON schema (Anthropic tool-use format) describing it to the model.
  2. A Python function that executes it and returns a string result.

Adding a new capability to the agent is just: write a function + schema,
then register it in TOOL_REGISTRY.
"""
from __future__ import annotations

import ast
import operator
from typing import Callable, Dict

from src.vectorstore import VectorStore

# --- Tool schemas (sent to Claude) -----------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "knowledge_base_search",
        "description": (
            "Search the internal enterprise knowledge base (uploaded company "
            "documents) for information relevant to a query. Use this whenever "
            "the user asks something that might be answered by company policy, "
            "product docs, or any uploaded material. Returns the most relevant "
            "text passages along with their source document names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, rephrased for retrieval if helpful.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many passages to retrieve (default 5).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate a basic arithmetic expression (+, -, *, /, %, **, parentheses). "
            "Use this for any numeric computation instead of doing math yourself, "
            "to guarantee accuracy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression, e.g. '(1250 * 0.18) / 12'.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "list_knowledge_base_documents",
        "description": (
            "List the documents currently indexed in the knowledge base. Use this "
            "if the user asks what documents are available, or to sanity-check "
            "coverage before answering."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


# --- Tool implementations ---------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported or unsafe expression.")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


def make_knowledge_base_search(store: VectorStore) -> Callable:
    def knowledge_base_search(query: str, top_k: int = 5) -> str:
        results = store.search(query, top_k=top_k)
        if not results:
            return "No relevant documents found in the knowledge base."

        lines = []
        for i, (chunk, score) in enumerate(results, start=1):
            lines.append(
                f"[Passage {i} | source: {chunk.source} | relevance: {score:.2f}]\n"
                f"{chunk.text}"
            )
        return "\n\n".join(lines)

    return knowledge_base_search


def make_list_documents(store: VectorStore) -> Callable:
    def list_knowledge_base_documents() -> str:
        stats = store.stats
        if not stats["documents"]:
            return "The knowledge base is currently empty. No documents indexed."
        doc_list = "\n".join(f"- {d}" for d in stats["documents"])
        return (
            f"{stats['num_documents']} document(s), {stats['num_chunks']} chunk(s) indexed:\n"
            f"{doc_list}"
        )

    return list_knowledge_base_documents


def build_tool_registry(store: VectorStore) -> Dict[str, Callable]:
    """Wire up tool name -> callable, bound to a specific vector store instance."""
    return {
        "knowledge_base_search": make_knowledge_base_search(store),
        "calculator": calculator,
        "list_knowledge_base_documents": make_list_documents(store),
    }
