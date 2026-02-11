"""Two-stage Retrieve & Rerank hybrid matching engine.

This is the primary improvement over the original single-shot cosine
similarity approach.  It combines the multilingual robustness of large
embedding models with the contextual reasoning of an LLM reranker.

Pipeline
--------
1. **Retrieve** – Use ``EmbeddingMatcher`` to fetch the top-N candidates
   by cosine similarity (fast, handles multilingual names well).
2. **Rerank** – Pass the shortlist to ``LLMReranker`` which considers
   element type and physical plausibility to produce a final ranking.
"""

from __future__ import annotations

from ifc_matching.engines.base import BaseMatcher, MatchResult
from ifc_matching.engines.embedding import EmbeddingMatcher
from ifc_matching.engines.reranker import LLMReranker


class HybridMatcher(BaseMatcher):
    """Retrieve & Rerank pipeline combining embeddings + LLM.

    Parameters
    ----------
    retriever:
        An ``EmbeddingMatcher`` instance (stage 1).
    reranker:
        An ``LLMReranker`` instance (stage 2).
    retrieval_k:
        Number of candidates to pass from stage 1 to stage 2.
    """

    def __init__(
        self,
        retriever: EmbeddingMatcher | None = None,
        reranker: LLMReranker | None = None,
        *,
        retrieval_k: int = 10,
    ) -> None:
        self.retriever = retriever or EmbeddingMatcher()
        self.reranker = reranker or LLMReranker()
        self.retrieval_k = retrieval_k

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

        # Stage 1: fast embedding retrieval
        shortlist = self.retriever.match(
            query, candidates, top_k=self.retrieval_k, element_type=element_type
        )

        shortlist_names = [r.candidate for r in shortlist]
        retrieval_scores = {r.candidate: r.score for r in shortlist}

        # Stage 2: LLM rerank
        reranked = self.reranker.match(
            query, shortlist_names, top_k=top_k, element_type=element_type
        )

        # Merge scores: combine embedding similarity and reranker confidence
        for result in reranked:
            emb_score = retrieval_scores.get(result.candidate, 0.0)
            result.metadata["embedding_score"] = emb_score
            result.metadata["rerank_score"] = result.score
            # Weighted combination (reranker is given more weight since
            # it has contextual understanding)
            result.score = 0.3 * emb_score + 0.7 * result.score
            result.method = "hybrid"

        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k]
