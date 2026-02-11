"""Tests for the hybrid matching engine."""

import pytest
from unittest.mock import MagicMock

from ifc_matching.engines.hybrid import HybridMatcher
from ifc_matching.engines.base import MatchResult


class TestHybridMatcher:
    def _make_hybrid(self):
        retriever = MagicMock()
        retriever.match.return_value = [
            MatchResult(candidate="cement", score=0.9, method="embedding:term"),
            MatchResult(candidate="brick", score=0.7, method="embedding:term"),
            MatchResult(candidate="wood", score=0.5, method="embedding:term"),
        ]

        reranker = MagicMock()
        reranker.match.return_value = [
            MatchResult(candidate="cement", score=0.95, method="rerank:llm"),
            MatchResult(candidate="wood", score=0.6, method="rerank:llm"),
            MatchResult(candidate="brick", score=0.3, method="rerank:llm"),
        ]

        return HybridMatcher(retriever=retriever, reranker=reranker, retrieval_k=10)

    def test_combines_scores(self):
        matcher = self._make_hybrid()
        results = matcher.match("concrete", ["cement", "brick", "wood"], top_k=3)

        # Should have combined scores
        for r in results:
            assert "embedding_score" in r.metadata
            assert "rerank_score" in r.metadata
            assert r.method == "hybrid"

    def test_weighted_scoring(self):
        matcher = self._make_hybrid()
        results = matcher.match("concrete", ["cement", "brick", "wood"], top_k=3)

        cement = next(r for r in results if r.candidate == "cement")
        # 0.3 * 0.9 + 0.7 * 0.95 = 0.27 + 0.665 = 0.935
        assert cement.score == pytest.approx(0.935, abs=0.01)

    def test_top_k_respected(self):
        matcher = self._make_hybrid()
        results = matcher.match("concrete", ["cement", "brick", "wood"], top_k=2)
        assert len(results) == 2

    def test_empty_candidates(self):
        matcher = self._make_hybrid()
        assert matcher.match("test", []) == []

    def test_results_sorted_descending(self):
        matcher = self._make_hybrid()
        results = matcher.match("concrete", ["cement", "brick", "wood"])
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retriever_called_with_retrieval_k(self):
        matcher = self._make_hybrid()
        matcher.match("q", ["a", "b"], top_k=5)
        _, kwargs = matcher.retriever.match.call_args
        assert kwargs["top_k"] == 10  # retrieval_k passed to retriever
