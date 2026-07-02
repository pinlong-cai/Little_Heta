"""Outer agent loop with two tools: recall_memory and query_kb."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from heta.config.schema import HetaConfig
from heta.mem.client import build_chat_model
from heta.mem.kb_writer import remember_kb_insights
from heta.mem.recall import LayerEvidence, format_evidence, retrieve_evidence
from heta.providers.model_protocols import ChatCompletionRequest, ChatMessage, ChatModelOptions, ChatModelProtocol
from heta.query.models import QueryResult

logger = logging.getLogger(__name__)

MAX_OUTER_STEPS = 5
TEXT_TOOL_CALL_RE = re.compile(r"<tool_call\b(?P<body>.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
TEXT_TOOL_FUNCTION_RE = re.compile(r"<function\s*=\s*['\"]?(?P<name>[A-Za-z_][\w]*)['\"]?\s*>", re.IGNORECASE)
TEXT_TOOL_PARAMETER_RE = re.compile(
    r"<parameter\s*=\s*['\"]?(?P<name>[A-Za-z_][\w]*)['\"]?\s*>(?P<value>.*?)(?=(?:<parameter\s*=|</function>|</tool_call>|$))",
    re.IGNORECASE | re.DOTALL,
)

_NO_INFO_PHRASES = [
    "no relevant",
    "not found",
    "unable to find",
    "cannot find",
    "找不到",
    "没有相关",
    "无法找到",
]

OUTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Search personal memory layers (past conversation turns, episodic events, "
                "atomic facts, and previously cached KB insights). Fast, no LLM calls. "
                "Returns formatted evidence grouped by layer; '(no results)' means a layer is empty."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query to search memory with."}
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_kb",
            "description": (
                "Run a deep wiki knowledge-base search via a sub-agent that can read pages "
                "and perform semantic search. Slower but authoritative. Use only when memory is "
                "insufficient. Returns a synthesized answer string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to ask the KB sub-agent."}
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
]

OUTER_SYSTEM_PROMPT = """\
You are Little Heta, a knowledge management assistant with access to two
information sources via tools.

Tools:
- recall_memory(query): fast search over personal memory (past conversations,
  episodes, facts, and previously cached KB insights). Use first.
- query_kb(question): deep search over the project knowledge base via a
  sub-agent that reads pages. Slower but authoritative. Use only when memory
  is insufficient.

Decision strategy:
1. Always call recall_memory FIRST with the user's question (unless the question
   is a trivial greeting or meta-message that needs no retrieval).
2. Read the evidence carefully. A section showing "(no results)" means that
   layer is empty. A score below ~0.3 usually means weak relevance.
3. If memory contains the specific information the question asks for, answer
   directly from memory. Do NOT call query_kb.
4. If memory is empty or only thematically related (mentions the topic but
   not the specific answer), call query_kb.
5. Special case: if the question is about a personal experience ("what did I
   do yesterday", "我上次去哪了") and memory has no hits, answer that you
   don't have that information. Do NOT search the KB for personal events.
6. After getting tool results, produce the final answer as plain Markdown in
   the SAME language as the question. Do not mention the tools or your
   internal reasoning. Do not include a "Sources" section.
"""


@dataclass
class SmartQueryResult:
    answer: str
    source: Literal["memory", "kb", "both"]
    memory_evidence: list[LayerEvidence] = field(default_factory=list)
    kb_result: QueryResult | None = None
    written_back: int = 0
    agent_steps: list[str] = field(default_factory=list)
    usage: dict[str, Any] | None = None


@dataclass
class _State:
    memory_evidence: list[LayerEvidence] = field(default_factory=list)
    kb_result: QueryResult | None = None
    written_back: int = 0
    used_memory: bool = False
    used_kb: bool = False
    agent_steps: list[str] = field(default_factory=list)
    outer_tokens: int = 0
    started_at: float = field(default_factory=time.time)


def smart_query(
    question: str,
    config: HetaConfig,
    top_k: int = 5,
    base_dir: Path | None = None,
) -> SmartQueryResult:
    """Outer agent loop: lets an LLM decide when to recall memory vs. query KB."""
    state = _State()
    chat_model = build_chat_model(config)
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

    for _ in range(MAX_OUTER_STEPS):
        resp = _chat(chat_model, messages, tools=OUTER_TOOLS)
        _record_outer_usage(state, resp)
        msg = resp.message
        tool_calls = list(msg.tool_calls or [])

        if not tool_calls:
            tool_calls = _parse_text_tool_calls(msg.content or "")
            if not tool_calls:
                return _build_result(state, answer=msg.content or "")

        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if msg.content:
            assistant_msg["content"] = msg.content
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ]
        messages.append(assistant_msg)

        for tc in tool_calls:
            result = _exec_tool(tc, config, top_k, base_dir, state)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Step limit reached — force a final answer with no tools
    messages.append(
        {"role": "user", "content": "Step limit reached. Answer with the evidence already gathered, or say you don't know."}
    )
    final = _chat(chat_model, messages, tools=None)
    _record_outer_usage(state, final)
    return _build_result(state, answer=final.message.content or "")


def _build_result(state: _State, *, answer: str) -> SmartQueryResult:
    memory_has_hits = any(layer.items for layer in state.memory_evidence)
    if state.used_kb and state.used_memory and memory_has_hits:
        source: Literal["memory", "kb", "both"] = "both"
    elif state.used_kb:
        source = "kb"
    else:
        source = "memory"
    return SmartQueryResult(
        answer=answer,
        source=source,
        memory_evidence=state.memory_evidence,
        kb_result=state.kb_result,
        written_back=state.written_back,
        agent_steps=list(state.agent_steps),
        usage={
            "outer_tokens": state.outer_tokens,
            "kb_tokens": (state.kb_result.usage or {}).get("tokens", 0) if state.kb_result else 0,
            "tokens": state.outer_tokens + ((state.kb_result.usage or {}).get("tokens", 0) if state.kb_result else 0),
            "elapsed_s": round(time.time() - state.started_at, 3),
        },
    )


def _exec_tool(tool_call: Any, config: HetaConfig, top_k: int, base_dir: Path | None, state: _State) -> str:
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return f"error: invalid tool arguments: {exc}"

    if name == "recall_memory":
        return _exec_recall_memory(str(args.get("query", "")), config, top_k, state)
    if name == "query_kb":
        return _exec_query_kb(str(args.get("question", "")), config, top_k, base_dir, state)
    return f"error: unknown tool {name}"


def _parse_text_tool_calls(content: str) -> list[Any]:
    """Parse XML-ish tool calls emitted as assistant text by some compatible APIs."""
    calls: list[Any] = []
    for index, match in enumerate(TEXT_TOOL_CALL_RE.finditer(content), start=1):
        body = match.group("body")
        function_match = TEXT_TOOL_FUNCTION_RE.search(body)
        if function_match is None:
            continue
        name = function_match.group("name")
        if name not in {"recall_memory", "query_kb"}:
            continue
        args = {
            parameter.group("name"): _strip_text_tool_markup(parameter.group("value"))
            for parameter in TEXT_TOOL_PARAMETER_RE.finditer(body)
        }
        if not args:
            continue
        calls.append(
            SimpleNamespace(
                id=f"text_tool_call_{index}",
                function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
            )
        )
    return calls


def _strip_text_tool_markup(value: str) -> str:
    return re.sub(r"</?(?:function|parameter)[^>]*>", "", value).strip()


def _exec_recall_memory(query: str, config: HetaConfig, top_k: int, state: _State) -> str:
    if not query.strip():
        return "error: empty query"
    try:
        evidence = retrieve_evidence(query, config, top_k=top_k)
    except Exception as exc:
        logger.exception("recall_memory failed")
        return f"error: {exc}"
    state.memory_evidence = evidence
    state.used_memory = True
    state.agent_steps.append("recall_memory")
    return format_evidence(evidence)


def _exec_query_kb(question: str, config: HetaConfig, top_k: int, base_dir: Path | None, state: _State) -> str:
    if not question.strip():
        return "error: empty question"
    from heta.query.agent import run_query_agent

    try:
        kb_result = run_query_agent(
            question=question,
            config=config,
            base_dir=base_dir,
            top_k=top_k,
        )
    except Exception as exc:
        logger.exception("query_kb failed")
        return f"error: {exc}"

    state.kb_result = kb_result
    state.used_kb = True
    state.agent_steps.append("query_kb")

    if _kb_has_info(kb_result.answer) and kb_result.insights:
        try:
            state.written_back = remember_kb_insights(
                question=question,
                insights=kb_result.insights,
                sources=kb_result.sources,
                config=config,
                base_dir=base_dir,
            )
        except Exception:
            logger.exception("kb write-back failed")

    return kb_result.answer


def _chat(chat_model: ChatModelProtocol, messages: list[dict[str, Any]], *, tools):
    return chat_model.complete(
        ChatCompletionRequest(
            messages=[ChatMessage(role="system", content=OUTER_SYSTEM_PROMPT), *messages],
            tools=tools,
            tool_choice="auto" if tools else None,
            options=ChatModelOptions(temperature=0.2),
        )
    )


def _record_outer_usage(state: _State, response: Any) -> None:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    state.outer_tokens += prompt_tokens + completion_tokens


def _kb_has_info(answer: str) -> bool:
    lower = answer.lower()
    return not any(phrase in lower for phrase in _NO_INFO_PHRASES)
