"""
The agentic core: drives a multi-step Claude tool-use loop.

Flow per user turn:
  1. Send conversation + tool schemas to Claude.
  2. If Claude responds with tool_use blocks, execute each tool locally,
     append the results as tool_result blocks, and send back.
  3. Repeat until Claude returns a plain text answer (or we hit the
     iteration safety cap).

This is what makes it "agentic" rather than a fixed RAG pipeline: the
model itself decides *whether* to search the knowledge base, *how many*
times, whether to also use the calculator, and when it has enough
information to answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any

import anthropic

import config
from src.tools import TOOL_SCHEMAS, build_tool_registry
from src.vectorstore import VectorStore

SYSTEM_PROMPT = """You are an Enterprise Knowledge Assistant. You help employees \
find accurate answers from the company's internal knowledge base and perform \
simple calculations when needed.

Rules:
- Prefer information retrieved via knowledge_base_search over your own general \
knowledge whenever the question could relate to company-specific material.
- Always cite the source document name(s) for any claim drawn from the \
knowledge base, in the form (Source: filename).
- If the knowledge base does not contain relevant information, say so plainly \
rather than guessing.
- Use the calculator tool for arithmetic instead of computing it yourself.
- Be concise and business-appropriate in tone.
"""


@dataclass
class ToolCallRecord:
    name: str
    input: Dict[str, Any]
    output: str


@dataclass
class AgentResponse:
    answer: str
    tool_calls: List[ToolCallRecord] = field(default_factory=list)


class KnowledgeAgent:
    def __init__(self, store: VectorStore, api_key: str = None):
        self.store = store
        self.client = anthropic.Anthropic(api_key=api_key or config.ANTHROPIC_API_KEY)
        self.tool_registry = build_tool_registry(store)
        self.history: List[Dict[str, Any]] = []

    def reset(self):
        self.history = []

    def ask(self, user_message: str) -> AgentResponse:
        self.history.append({"role": "user", "content": user_message})
        tool_calls: List[ToolCallRecord] = []

        for _ in range(config.MAX_AGENT_TOOL_ITERATIONS):
            response = self.client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=config.MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=self.history,
            )

            # Record assistant turn (may contain text + tool_use blocks)
            self.history.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                final_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                return AgentResponse(answer=final_text, tool_calls=tool_calls)

            # Execute every tool_use block, collect tool_result blocks
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                fn = self.tool_registry.get(block.name)
                if fn is None:
                    output = f"Error: unknown tool '{block.name}'"
                else:
                    try:
                        output = fn(**block.input)
                    except Exception as e:
                        output = f"Error running tool '{block.name}': {e}"

                tool_calls.append(
                    ToolCallRecord(name=block.name, input=block.input, output=output)
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )

            self.history.append({"role": "user", "content": tool_results})

        return AgentResponse(
            answer="I wasn't able to reach a final answer within the tool-use step "
            "limit. Try rephrasing your question or breaking it into parts.",
            tool_calls=tool_calls,
        )
