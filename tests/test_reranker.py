"""Tests for the LLM reranker."""

import json
from unittest.mock import patch

import pytest

from ifc_matching.engines.reranker import LLMReranker


class TestParseResponse:
    def test_valid_json(self):
        raw = json.dumps([
            {"candidate": "cement", "score": 0.95},
            {"candidate": "brick", "score": 0.3},
        ])
        results = LLMReranker._parse_response(raw, ["cement", "brick"])
        assert len(results) == 2
        assert results[0].candidate == "cement"
        assert results[0].score == 0.95
        assert results[1].candidate == "brick"

    def test_json_with_surrounding_text(self):
        raw = 'Here is the ranking:\n[{"candidate": "A", "score": 0.9}]\nDone.'
        results = LLMReranker._parse_response(raw, ["A", "B"])
        assert results[0].candidate == "A"
        # "B" should be appended as missed
        assert any(r.candidate == "B" for r in results)

    def test_invalid_json_fallback(self):
        raw = "I think cement is the best match."
        results = LLMReranker._parse_response(raw, ["cement", "brick"])
        assert len(results) == 2
        assert results[0].method == "rerank:fallback"

    def test_deduplication(self):
        raw = json.dumps([
            {"candidate": "cement", "score": 0.9},
            {"candidate": "cement", "score": 0.8},  # duplicate
        ])
        results = LLMReranker._parse_response(raw, ["cement"])
        cement_results = [r for r in results if r.candidate == "cement"]
        assert len(cement_results) == 1

    def test_empty_candidates(self):
        reranker = LLMReranker()
        assert reranker.match("test", []) == []


class TestBuildPrompt:
    def test_includes_query(self):
        system, user = LLMReranker._build_prompt("concrete", ["a", "b"], "IfcWall")
        assert "concrete" in user

    def test_includes_element_type(self):
        system, user = LLMReranker._build_prompt("concrete", ["a"], "IfcSlab")
        assert "IfcSlab" in user

    def test_includes_candidates(self):
        system, user = LLMReranker._build_prompt("concrete", ["cement", "brick"], None)
        assert "cement" in user
        assert "brick" in user

    def test_unknown_element_type_default(self):
        _, user = LLMReranker._build_prompt("concrete", ["a"], None)
        assert "unknown" in user


class TestRerankerFallback:
    """Without oekobaudat_search_fn, reranker uses standard JSON ranking (no tools)."""

    def test_no_tools_when_search_fn_none(self):
        reranker = LLMReranker(oekobaudat_search_fn=None)
        with patch.object(reranker, "_call_llm", return_value='[{"candidate": 1, "score": 0.9}]'):
            results = reranker.match("Beton", ["Beton C20/25", "Stahl"], top_k=2)
        assert len(results) >= 1
        assert results[0].candidate == "Beton C20/25"

    def test_with_search_fn_uses_tool_path(self):
        def fake_search(query, element_type=None, class_hint=None, limit=30):
            return ["Extra EPD from search"]

        reranker = LLMReranker(
            oekobaudat_search_fn=fake_search,
            backend="openai",
        )
        with patch.object(reranker, "_call_openai_with_tools") as mock_tools:
            mock_tools.return_value = ('[{"candidate": 1, "score": 0.9}]', ["Beton C20/25", "Stahl"])
            results = reranker.match("Beton", ["Beton C20/25", "Stahl"], top_k=2)
        mock_tools.assert_called_once()
        assert len(results) >= 1
