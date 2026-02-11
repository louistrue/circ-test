"""Tests for the embedding matching engine."""

import pytest

import numpy as np

from ifc_matching.engines.embedding import EmbeddingMatcher
from ifc_matching.engines.base import MatchResult


class TestEmbeddingMatcherTokenize:
    def test_basic_split(self):
        assert EmbeddingMatcher._tokenize("rigid insulation") == ["rigid", "insulation"]

    def test_hyphen_split(self):
        assert EmbeddingMatcher._tokenize("XPS-board") == ["XPS", "board"]

    def test_underscore_split(self):
        assert EmbeddingMatcher._tokenize("gypsum_board") == ["gypsum", "board"]

    def test_mixed_delimiters(self):
        tokens = EmbeddingMatcher._tokenize("fire-rated:gypsum_board type")
        assert tokens == ["fire", "rated", "gypsum", "board", "type"]

    def test_empty(self):
        assert EmbeddingMatcher._tokenize("") == []


class TestCosine:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert EmbeddingMatcher._cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert EmbeddingMatcher._cosine(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert EmbeddingMatcher._cosine(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 0.0])
        assert EmbeddingMatcher._cosine(a, b) == 0.0


class TestEmbeddingMatcherMatch:
    def _make_matcher_with_mock(self):
        """Create a matcher with a mocked encode method."""
        matcher = EmbeddingMatcher(use_token_matching=False)

        # Deterministic embeddings: each text gets a unique unit-ish vector
        _vectors = {}
        _counter = [0]

        def mock_encode(texts):
            results = []
            for t in texts:
                if t not in _vectors:
                    v = np.zeros(10)
                    v[_counter[0] % 10] = 1.0
                    _vectors[t] = v
                    _counter[0] += 1
                results.append(_vectors[t])
            return np.array(results)

        matcher.encode = mock_encode
        return matcher

    def test_returns_correct_count(self):
        matcher = self._make_matcher_with_mock()
        results = matcher.match("test", ["a", "b", "c"], top_k=2)
        assert len(results) == 2

    def test_results_are_sorted_descending(self):
        matcher = self._make_matcher_with_mock()
        results = matcher.match("test", ["a", "b", "c"])
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_candidates(self):
        matcher = self._make_matcher_with_mock()
        assert matcher.match("test", []) == []

    def test_result_type(self):
        matcher = self._make_matcher_with_mock()
        results = matcher.match("test", ["candidate"])
        assert all(isinstance(r, MatchResult) for r in results)
