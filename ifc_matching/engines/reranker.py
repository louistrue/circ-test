"""LLM-based reranking of embedding retrieval candidates.

Implements the "Retrieve & Rerank" strategy described in the improvement
roadmap.  An LLM receives the IFC element description together with N
candidate database entries and returns a physically-plausible ranking.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from ifc_matching.engines.base import BaseMatcher, MatchResult


_RERANK_SYSTEM = (
    "You are a building-materials expert. "
    "Given an IFC element description and a list of candidate database entries, "
    "rank the candidates from most to least physically plausible match. "
    "Return ONLY a JSON array of objects with keys \"candidate\" and \"score\" "
    "(a float 0-1 indicating confidence), ordered by descending score. "
    "Do not include any text outside the JSON array."
)

_RERANK_USER = (
    "IFC product name: {query}\n"
    "Element type: {element_type}\n\n"
    "Candidate database entries (numbered):\n{candidate_list}\n\n"
    "Which of these database entries is the most physically plausible match? "
    "Rank all candidates."
)


class LLMReranker(BaseMatcher):
    """Stage-2 reranker that uses a chat-completion LLM for contextual ranking.

    Supports ``"openai"`` (GPT-4o etc.) and ``"anthropic"`` (Claude) backends.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        backend: Literal["openai", "anthropic"] = "openai",
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self._api_key = api_key
        self.temperature = temperature
        self._client = None

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

    def _call_anthropic(self, system: str, user: str) -> str:
        client = self._get_anthropic_client()
        resp = client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    def _call_llm(self, system: str, user: str) -> str:
        if self.backend == "anthropic":
            return self._call_anthropic(system, user)
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

        results: list[MatchResult] = []
        seen = set()
        for item in items:
            cand = item.get("candidate", "")
            score = float(item.get("score", 0.0))
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

        system, user = self._build_prompt(query, candidates, element_type)
        raw = self._call_llm(system, user)
        results = self._parse_response(raw, candidates)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
