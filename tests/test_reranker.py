"""Tests for the LLM reranker."""

import json
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
