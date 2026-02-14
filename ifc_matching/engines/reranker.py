"""LLM-based reranking of embedding retrieval candidates.

Implements the "Retrieve & Rerank" strategy described in the improvement
roadmap.  An LLM receives the IFC element description together with N
candidate database entries and returns a physically-plausible ranking.

When an ÖKOBAUDAT search tool is provided, the LLM can call it to fetch
additional EPD candidates from the ÖKOBAUDAT API before ranking.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Literal

from ifc_matching.engines.base import BaseMatcher, MatchResult

# Max tool-call iterations to prevent runaway loops
MAX_TOOL_ITERATIONS = 3

_RERANK_SYSTEM = (
    "You are a building-materials expert. "
    "Given an IFC element description and a numbered list of candidate database entries, "
    "rank the candidates from most to least physically plausible match. "
    "Return ONLY a JSON array of objects with keys \"candidate\" (the integer number "
    "from the list) and \"score\" (a float 0-1 indicating confidence), "
    "ordered by descending score. "
    "Do not include any text outside the JSON array."
)

_RERANK_SYSTEM_WITH_TOOLS = (
    "You are a building-materials expert matching IFC materials to EPD databases. "
    "You have access to an ÖKOBAUDAT search tool to look up additional EPDs if the "
    "provided candidates seem insufficient or off-topic. Use the tool when you need "
    "more or better candidates (e.g. different search terms, category hints). "
    "When you have enough candidates, rank them from most to least physically plausible. "
    "Return ONLY a JSON array of objects with keys \"candidate\" (the integer number "
    "from the list) and \"score\" (0-1 confidence), ordered by descending score."
)

_RERANK_USER = (
    "IFC product name: {query}\n"
    "Element type: {element_type}\n\n"
    "Candidate database entries:\n{candidate_list}\n\n"
    "Which of these database entries is the most physically plausible match "
    "for this IFC product used in a {element_type}? "
    "Rank all candidates by number."
)

# Tool definition for ÖKOBAUDAT search (OpenAI format)
_OEKOBAUDAT_SEARCH_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "oekobaudat_search",
        "description": "Search the ÖKOBAUDAT EPD database for building materials. Use when initial candidates are insufficient or you need more specific matches.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term (e.g. Beton, Stahl, Gipskarton)"},
                "element_type": {"type": "string", "description": "IFC element type hint (e.g. IfcWall, IfcSlab)"},
                "class_hint": {"type": "string", "description": "ÖKOBAU.DAT class ID hint (e.g. 1.4.01 for Beton)"},
                "limit": {"type": "integer", "description": "Max results to return (default 30)"},
            },
            "required": ["query"],
        },
    },
}

# Tool definition for Anthropic
_OEKOBAUDAT_SEARCH_TOOL_ANTHROPIC = {
    "name": "oekobaudat_search",
    "description": "Search the ÖKOBAUDAT EPD database for building materials. Use when initial candidates are insufficient or you need more specific matches.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search term (e.g. Beton, Stahl, Gipskarton)"},
            "element_type": {"type": "string", "description": "IFC element type hint (e.g. IfcWall, IfcSlab)"},
            "class_hint": {"type": "string", "description": "ÖKOBAU.DAT class ID hint (e.g. 1.4.01 for Beton)"},
            "limit": {"type": "integer", "description": "Max results to return (default 30)"},
        },
        "required": ["query"],
    },
}


def _default_oekobaudat_search(
    query: str,
    element_type: str | None = None,
    class_hint: str | None = None,
    limit: int = 30,
) -> list[str]:
    """Default ÖKOBAUDAT search implementation using OekobaudatClient."""
    from ifc_matching.databases.oekobaudat_client import OekobaudatClient

    client = OekobaudatClient()
    results = client.search_processes(
        name=query,
        class_id=class_hint,
        limit=limit,
    )
    return [r.display_name for r in results]


class LLMReranker(BaseMatcher):
    """Stage-2 reranker that uses a chat-completion LLM for contextual ranking.

    Supports three backends:

    * ``"openai"`` – Chat Completions API (GPT-4o etc.)
    * ``"openai-responses"`` – Responses API (GPT-5.2 etc.)
    * ``"anthropic"`` – Anthropic Messages API (Claude)

    When ``oekobaudat_search_fn`` is provided, the reranker can use tool-calling
    to search ÖKOBAUDAT for additional candidates before ranking.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        backend: Literal["openai", "openai-responses", "anthropic"] = "openai",
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        oekobaudat_search_fn: Callable[..., list[str]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self._api_key = api_key
        self.temperature = temperature
        self._client = None
        self._oekobaudat_search_fn = oekobaudat_search_fn

    # ------------------------------------------------------------------
    # Client lazy-init
    # ------------------------------------------------------------------

    def _get_openai_client(self):
        if self._client is None:
            import openai

            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def _get_anthropic_client(self):
        if self._client is None:
            import anthropic

            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        query: str,
        candidates: list[str],
        element_type: str | None,
    ) -> tuple[str, str]:
        cand_lines = "\n".join(
            f"  {i + 1}. {c}" for i, c in enumerate(candidates)
        )
        user_msg = _RERANK_USER.format(
            query=query,
            element_type=element_type or "unknown",
            candidate_list=cand_lines,
        )
        return _RERANK_SYSTEM, user_msg

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_openai(self, system: str, user: str) -> str:
        client = self._get_openai_client()
        resp = client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def _call_openai_with_tools(
        self,
        system: str,
        user: str,
        candidates: list[str],
        search_fn: Callable[..., list[str]],
    ) -> tuple[str, list[str]]:
        """Call OpenAI with tool support; returns (final_text, merged_candidates)."""
        client = self._get_openai_client()
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        merged = list(candidates)
        seen = set(candidates)

        for _ in range(MAX_TOOL_ITERATIONS):
            resp = client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=messages,
                tools=[_OEKOBAUDAT_SEARCH_TOOL_OPENAI],
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            if msg.content:
                return (msg.content or "", merged)

            if not getattr(msg, "tool_calls", None):
                return ("", merged)

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                if tc.function.name != "oekobaudat_search":
                    continue
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                q = args.get("query", "")
                et = args.get("element_type")
                ch = args.get("class_hint")
                lim = args.get("limit", 30)
                new_names = search_fn(query=q, element_type=et, class_hint=ch, limit=lim)
                for n in new_names:
                    if n and n not in seen:
                        seen.add(n)
                        merged.append(n)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps({"found": len(new_names), "names": new_names}),
            })

        return ("", merged)

    def _call_anthropic_with_tools(
        self,
        system: str,
        user: str,
        candidates: list[str],
        search_fn: Callable[..., list[str]],
    ) -> tuple[str, list[str]]:
        """Call Anthropic with tool support; returns (final_text, merged_candidates)."""
        import time as _time

        client = self._get_anthropic_client()
        merged = list(candidates)
        seen = set(candidates)
        messages: list[dict] = [{"role": "user", "content": user}]

        for _ in range(MAX_TOOL_ITERATIONS):
            try:
                resp = client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    temperature=self.temperature,
                    system=system,
                    messages=messages,
                    tools=[_OEKOBAUDAT_SEARCH_TOOL_ANTHROPIC],
                )
            except Exception as e:
                if "overloaded" in str(e).lower() or "529" in str(e):
                    _time.sleep(5)
                    continue
                raise

            text_parts: list[str] = []
            tool_results: list[dict] = []

            for block in resp.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif getattr(block, "type", None) == "tool_use":
                    tool_id = getattr(block, "id", "")
                    name = getattr(block, "name", "")
                    inp = getattr(block, "input", {}) or {}
                    if name == "oekobaudat_search":
                        q = inp.get("query", "")
                        et = inp.get("element_type")
                        ch = inp.get("class_hint")
                        lim = inp.get("limit", 30)
                        new_names = search_fn(query=q, element_type=et, class_hint=ch, limit=lim)
                        for n in new_names:
                            if n and n not in seen:
                                seen.add(n)
                                merged.append(n)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": json.dumps({"found": len(new_names), "names": new_names}),
                        })

            if text_parts:
                return ("".join(text_parts), merged)

            if not tool_results:
                return ("", merged)

            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": tool_results})

        return ("", merged)

    def _call_anthropic(self, system: str, user: str) -> str:
        import time as _time
        client = self._get_anthropic_client()
        for attempt in range(5):
            try:
                resp = client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    temperature=self.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return resp.content[0].text
            except Exception as e:
                if "overloaded" in str(e).lower() or "529" in str(e):
                    wait = 2 ** attempt * 5
                    print(f"    [Anthropic overloaded, retry {attempt+1}/5 in {wait}s]")
                    _time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Anthropic API overloaded after 5 retries")

    def _call_openai_responses(self, system: str, user: str) -> str:
        """Call OpenAI Responses API (gpt-5.2 etc.)."""
        client = self._get_openai_client()
        resp = client.responses.create(
            model=self.model_name,
            instructions=system,
            input=user,
            temperature=self.temperature,
        )
        return resp.output_text or ""

    def _call_llm(self, system: str, user: str) -> str:
        if self.backend == "anthropic":
            return self._call_anthropic(system, user)
        if self.backend == "openai-responses":
            return self._call_openai_responses(system, user)
        return self._call_openai(system, user)

    # ------------------------------------------------------------------
    # Parse LLM response
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str, candidates: list[str]) -> list[MatchResult]:
        """Extract structured ranking from the LLM response."""
        # Find JSON array in the response
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            # Fallback: return candidates in original order with decaying scores
            return [
                MatchResult(candidate=c, score=1.0 - i * 0.05, method="rerank:fallback")
                for i, c in enumerate(candidates)
            ]

        try:
            items = json.loads(match.group())
        except json.JSONDecodeError:
            return [
                MatchResult(candidate=c, score=1.0 - i * 0.05, method="rerank:fallback")
                for i, c in enumerate(candidates)
            ]

        # Build a lookup to resolve numeric references (1-indexed) back to names
        idx_lookup = {i + 1: c for i, c in enumerate(candidates)}
        name_set = set(candidates)

        results: list[MatchResult] = []
        seen = set()
        for item in items:
            raw_cand = item.get("candidate", "")
            score = float(item.get("score", 0.0))
            # Resolve numeric index to candidate name
            if isinstance(raw_cand, int) and raw_cand in idx_lookup:
                cand = idx_lookup[raw_cand]
            elif isinstance(raw_cand, str) and raw_cand.isdigit() and int(raw_cand) in idx_lookup:
                cand = idx_lookup[int(raw_cand)]
            else:
                cand = str(raw_cand)
            if cand and cand not in seen:
                seen.add(cand)
                results.append(
                    MatchResult(candidate=cand, score=score, method="rerank:llm")
                )

        # Append any candidates the LLM missed
        for c in candidates:
            if c not in seen:
                results.append(
                    MatchResult(candidate=c, score=0.0, method="rerank:missed")
                )

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
        *,
        element_type: str | None = None,
    ) -> list[MatchResult]:
        if not candidates:
            return []

        search_fn = self._oekobaudat_search_fn
        use_tools = search_fn is not None and self.backend in ("openai", "anthropic")

        if use_tools:
            system = _RERANK_SYSTEM_WITH_TOOLS
        else:
            system = _RERANK_SYSTEM

        cand_lines = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(candidates))
        user = _RERANK_USER.format(
            query=query,
            element_type=element_type or "unknown",
            candidate_list=cand_lines,
        )

        if use_tools:
            if self.backend == "openai":
                raw, merged = self._call_openai_with_tools(
                    system, user, candidates, search_fn
                )
            else:
                raw, merged = self._call_anthropic_with_tools(
                    system, user, candidates, search_fn
                )
            candidates = merged
            if not raw or merged != candidates:
                cand_lines = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(merged))
                user = _RERANK_USER.format(
                    query=query,
                    element_type=element_type or "unknown",
                    candidate_list=cand_lines,
                )
                raw = self._call_llm(_RERANK_SYSTEM, user)
        else:
            raw = self._call_llm(system, user)

        results = self._parse_response(raw, candidates)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
