"""Embedding-based retrieval using sentence-transformers or OpenAI."""

from __future__ import annotations

import re
from typing import Literal

import numpy as np

from ifc_matching.engines.base import BaseMatcher, MatchResult


# BGE-M3: state-of-the-art multilingual embedding model (BAAI).
# 100+ languages, 8K context, ~568M params.  Best open-source model for
# multilingual RAG / retrieve-rerank pipelines in the construction domain.
DEFAULT_MODEL = "BAAI/bge-m3"

# Lightweight multilingual alternative (~118M params, 50+ languages).
MULTILINGUAL_MINI_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Legacy English-only model kept for benchmark comparison.
LEGACY_MODEL = "all-MiniLM-L6-v2"


class EmbeddingMatcher(BaseMatcher):
    """Stage-1 retriever that ranks candidates via cosine similarity.

    Supports two back-ends:

    * ``"sentence-transformers"`` – local models (BERT, fine-tuned HF models).
    * ``"openai"`` – OpenAI embedding API (``text-embedding-3-large`` etc.).

    The default model is ``BAAI/bge-m3``, a state-of-the-art multilingual
    embedding model supporting 100+ languages with 8K context window.
    This eliminates the cross-lingual failures observed with the
    English-only ``all-MiniLM-L6-v2``.

    The original ifcProductMatching logic (whole-term *and* token-level
    cosine similarity, keeping the higher score) is preserved and can be
    toggled with ``use_token_matching``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        backend: Literal["sentence-transformers", "openai"] = "sentence-transformers",
        *,
        use_token_matching: bool = True,
        openai_api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self.use_token_matching = use_token_matching
        self._model = None
        self._openai_client = None
        self._openai_api_key = openai_api_key

    # ------------------------------------------------------------------
    # Lazy initialisation so the object is lightweight until first use
    # ------------------------------------------------------------------

    def _get_st_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _get_openai_client(self):
        if self._openai_client is None:
            import openai

            kwargs = {}
            if self._openai_api_key:
                kwargs["api_key"] = self._openai_api_key
            self._openai_client = openai.OpenAI(**kwargs)
        return self._openai_client

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_st(self, texts: list[str]) -> np.ndarray:
        model = self._get_st_model()
        return model.encode(texts, convert_to_numpy=True)

    def _encode_openai(self, texts: list[str]) -> np.ndarray:
        client = self._get_openai_client()
        resp = client.embeddings.create(model=self.model_name, input=texts)
        return np.array([d.embedding for d in resp.data])

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "openai":
            return self._encode_openai(texts)
        return self._encode_st(texts)

    # ------------------------------------------------------------------
    # Core matching (preserves original token-level logic)
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in re.split(r"[ :\-_]", text) if t]

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

        # Encode query + all candidates in one batch for efficiency
        all_texts = [query] + list(candidates)
        embeddings = self.encode(all_texts)
        query_emb = embeddings[0]
        cand_embs = embeddings[1:]

        results: list[MatchResult] = []
        for idx, cand in enumerate(candidates):
            term_score = self._cosine(query_emb, cand_embs[idx])
            best_score = term_score
            method = "term"

            if self.use_token_matching:
                query_tokens = self._tokenize(query)
                cand_tokens = self._tokenize(cand)
                if query_tokens and cand_tokens:
                    # Encode all tokens in one call
                    all_tokens = query_tokens + cand_tokens
                    tok_embs = self.encode(all_tokens)
                    q_tok_embs = tok_embs[: len(query_tokens)]
                    c_tok_embs = tok_embs[len(query_tokens) :]

                    # Improved scoring: average best-match per query token,
                    # weighted by coverage. Prevents single-token collapse
                    # (e.g., "Metall" matching everything with "Metall" at 1.0).
                    q_best = []
                    for qt in q_tok_embs:
                        best_for_q = max(self._cosine(qt, ct) for ct in c_tok_embs)
                        q_best.append(best_for_q)
                    # Average of per-query-token best matches
                    avg_tok = sum(q_best) / len(q_best)
                    # Blend: 60% average coverage + 40% best single match
                    max_tok = max(q_best)
                    tok_score = 0.6 * avg_tok + 0.4 * max_tok
                    if tok_score > best_score:
                        best_score = tok_score
                        method = "token"

            results.append(
                MatchResult(
                    candidate=cand,
                    score=best_score,
                    method=f"embedding:{method}",
                    metadata={"model": self.model_name, "backend": self.backend},
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
