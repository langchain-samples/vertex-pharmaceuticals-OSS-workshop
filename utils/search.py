"""Web research helper built on Anthropic's server-side `web_search` tool.

The shape: call `resilient_web_research(query)` from inside your own `@tool`
function. The helper asks Claude to answer the query with live web search
(Anthropic runs the searches; there's no separate search API or key), and
returns the answer followed by the sources it cited. It retries on failure
with linear backoff, and if every retry fails it returns a canned response
matched against topic keywords — chosen so the agent's downstream synthesis
still produces a useful answer during a workshop with a flaky network.

Usage (from a notebook or module):

    from langchain_core.tools import tool
    from utils.search import resilient_web_research

    @tool(parse_docstring=True)
    def web_research(query: str) -> str:
        \"\"\"Research a question on the web and return a cited answer.

        Args:
            query: The question or topic to research.
        \"\"\"
        return resilient_web_research(query, max_retries=2)

The `@tool` decorator stays at the call site so the tool's name, docstring,
and signature remain visible to the reader. Only the API and resilience
plumbing is hidden behind the util.

The research call always goes to the Anthropic API with `ANTHROPIC_API_KEY`,
whichever chat model `utils/models.py` points the agent at. Web search is not
available on Amazon Bedrock.
"""

from __future__ import annotations

import time
from typing import Optional

from anthropic import Anthropic

# Model that runs the web search. Independent of the agent's own model.
RESEARCH_MODEL = "claude-haiku-5-5"

RESEARCH_INSTRUCTIONS = (
    "Answer the query using web search. Be concise: at most three short "
    "paragraphs of facts, no preamble. Prefer primary and recent sources. "
    "Don't list sources at the end; citations are attached automatically."
)

# The basic variant, not `web_search_20260209`: the newer one filters results
# inside a code-execution step, so the answer carries no citations, and it used
# ~2.5x the input tokens on our test queries.
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}

# A long server-tool turn can stop with `pause_turn`; we re-send to continue.
MAX_CONTINUATIONS = 3


# --------------------------------------------------------------------------- #
# Canned fallbacks — based on patterns we observed in real traces.
# Each entry is a list of (title, url, content) tuples. Keys are lowercase
# phrases matched as substrings. Order matters: longer/more-specific keys are
# checked first.
# --------------------------------------------------------------------------- #

_FALLBACK_RESULTS: list[tuple[tuple[str, ...], list[tuple[str, str, str]]]] = [
    (("difference between langchain and langgraph", "langchain vs langgraph"), [
        (
            "LangChain vs LangGraph — when to use each",
            "https://docs.langchain.com/oss/python/concepts/products",
            "LangChain provides the high-level agent and chain abstractions (create_agent, "
            "tools, middleware) for assembling LLM applications. LangGraph is the lower-level "
            "runtime that LangChain agents compile to — a stateful graph with nodes, edges, "
            "checkpointers, and human-in-the-loop interrupts. Use LangChain when you want a "
            "ReAct-style agent in a few lines; use LangGraph when you need custom state, "
            "multi-agent topologies, or fine-grained control of the execution loop.",
        ),
    ]),
    (("create_agent", "create agent"), [
        (
            "create_agent — LangChain v1 prebuilt agent",
            "https://docs.langchain.com/oss/python/langchain/agents",
            "`create_agent(model, tools, system_prompt, middleware=...)` returns a ReAct-style "
            "agent as a compiled LangGraph. It handles the model→tool→model loop, accepts "
            "middleware for cross-cutting concerns (logging, HITL, structured output), and "
            "can be passed a checkpointer for persistence. Returned object exposes "
            ".invoke / .stream / .ainvoke and is itself a Runnable.",
        ),
    ]),
    (("langchain v1", "langchain 1.0"), [
        (
            "LangChain v1.0 release notes",
            "https://blog.langchain.com/langchain-v1/",
            "LangChain v1 introduces a redesigned agent API centered on `create_agent` and a "
            "first-class middleware system for customizing the agent loop. Other highlights: "
            "consolidated message types in langchain-core, an init_chat_model factory for "
            "provider-agnostic model loading, and improved structured output support via "
            "with_structured_output. Built on top of LangGraph for the runtime."
        ),
        (
            "Migrating to LangChain v1",
            "https://docs.langchain.com/oss/python/releases/langchain-v1",
            "v1 simplifies the agent surface area: AgentExecutor and the v0 agent constructors "
            "are replaced by `create_agent`, middleware replaces callbacks for most flows, and "
            "tool definitions remain compatible with the existing `@tool` decorator."
        ),
    ]),
    (("langgraph",), [
        (
            "LangGraph — stateful, multi-agent workflows",
            "https://docs.langchain.com/oss/python/langgraph/overview",
            "LangGraph is an open-source framework for building stateful, multi-agent "
            "applications with LLMs. You model your workflow as a StateGraph: define a "
            "TypedDict State, register nodes (Python functions) that read state and return "
            "updates, and connect them with normal or conditional edges. Built-in support for "
            "checkpointers (persistence), interrupts (human-in-the-loop), and stores "
            "(long-term memory)."
        ),
    ]),
    (("langsmith",), [
        (
            "LangSmith — observability and evaluation for LLM apps",
            "https://docs.smith.langchain.com/",
            "LangSmith is a framework-agnostic platform for tracing, evaluating, and monitoring "
            "LLM applications. Set LANGSMITH_TRACING=true and every LLM/tool/chain call lands "
            "in your project. Build datasets, run offline evaluations with `client.evaluate`, "
            "or configure online evaluators (run rules) that score new traces automatically. "
            "Annotation queues route flagged runs to human reviewers."
        ),
    ]),
]


_GENERIC_FALLBACK = [
    (
        "Search temporarily unavailable",
        "https://fallback.example/no-results",
        "The web search service didn't return results for this query. Use what you already "
        "know to answer; if information is missing, say so plainly rather than guessing.",
    ),
]


def _format_results(results: list[tuple[str, str, str]]) -> str:
    return "\n\n".join(f"**{t}**\n{u}\n{c}" for t, u, c in results)


def _pick_fallback(query: str) -> str:
    q = query.lower()
    for keys, results in _FALLBACK_RESULTS:
        if any(k in q for k in keys):
            return _format_results(results)
    return _format_results(_GENERIC_FALLBACK)


def _format_response(response) -> str:
    """Return the answer text followed by its de-duplicated web citations."""
    # Text written between searches ("Let me search for...") isn't part of the
    # answer; keep only the text after the last search result.
    blocks = list(response.content)
    last_result = max(
        (i for i, b in enumerate(blocks) if b.type == "web_search_tool_result"), default=-1
    )
    sources: dict[str, str] = {}
    parts: list[str] = []
    for block in blocks[last_result + 1:]:
        if block.type != "text":
            continue
        parts.append(block.text)
        for cite in getattr(block, "citations", None) or []:
            if cite.type == "web_search_result_location":
                sources.setdefault(cite.url, cite.title or cite.url)

    answer = "".join(parts).strip()
    if not sources:
        return answer
    cited = "\n".join(f"[{i}] {title} — {url}" for i, (url, title) in enumerate(sources.items(), 1))
    return f"{answer}\n\nSources:\n{cited}"


def _research(client: Anthropic, query: str, model: str):
    messages = [{"role": "user", "content": query}]
    for _ in range(MAX_CONTINUATIONS + 1):
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=RESEARCH_INSTRUCTIONS,
            tools=[WEB_SEARCH_TOOL],
            messages=messages,
            output_config={"effort": "low"},
        )
        if response.stop_reason != "pause_turn":
            return response
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response.content},
        ]
    return response


# --------------------------------------------------------------------------- #
# Public helper
# --------------------------------------------------------------------------- #

# Lazy singleton — instantiated once on first call, then reused.
_default_client: Optional[Anthropic] = None


def resilient_web_research(
    query: str,
    *,
    max_retries: int = 2,
    model: str = RESEARCH_MODEL,
    base_backoff_seconds: float = 1.0,
    client: Optional[Anthropic] = None,
) -> str:
    """Research a query with Claude web search; fall back to canned content on failure.

    Returns Claude's answer followed by a `Sources:` list of the URLs it
    cited, so the caller can return it from a `@tool` function as-is.

    Args:
        query: question or topic to research.
        max_retries: additional attempts after the first call. `max_retries=2`
            means up to 3 total attempts before falling back.
        model: Claude model that runs the search.
        base_backoff_seconds: first retry sleeps this long; subsequent retries
            sleep `base * attempt` (linear backoff).
        client: optional Anthropic client to use. If `None`, a lazily-initialized
            module-level client is used.
    """
    global _default_client
    if client is None:
        if _default_client is None:
            _default_client = Anthropic(timeout=120, max_retries=0)
        client = _default_client

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = _research(client, query, model)
            if response.stop_reason == "refusal":
                # Retrying a decline won't change it; tell the agent plainly.
                category = getattr(response.stop_details, "category", None)
                return f"[research declined by Claude's safety classifiers (category: {category})]"
            answer = _format_response(response)
            if answer:
                return answer
            last_error = RuntimeError(f"web search returned an empty answer ({response.stop_reason})")
        except Exception as exc:
            last_error = exc
        if attempt < max_retries:
            time.sleep(base_backoff_seconds * (attempt + 1))

    notice = (
        f"[fallback content; live search unavailable -- "
        f"{type(last_error).__name__}: {str(last_error)[:120]}]\n\n"
    )
    return notice + _pick_fallback(query)
