"""Tests for llm.json_extract -- tolerating chatty / thinking-model output."""
from __future__ import annotations

from price_predictor.llm.json_extract import extract_json


class TestExtractJson:
    def test_clean_json_passes_through(self):
        raw = '{"sentiment": "neutral", "score": 0.1}'
        assert extract_json(raw) == raw

    def test_strips_think_block_before_json(self):
        raw = (
            "<think>We need to produce ImpactAssessment. News is mixed, "
            "lean neutral. Now output.</think>\n"
            '{"sentiment": "neutral"}'
        )
        assert extract_json(raw) == '{"sentiment": "neutral"}'

    def test_strips_leading_prose_without_tags(self):
        # qwen3 sometimes narrates WITHOUT <think> tags (litellm stripped them
        # or the model never emitted them). We still find the JSON object.
        raw = 'We need to produce Impact... neutral.\n\nNow output.\n{"sentiment": "bearish"}'
        assert extract_json(raw) == '{"sentiment": "bearish"}'

    def test_unwraps_markdown_json_fence(self):
        raw = '```json\n{"sentiment": "bullish"}\n```'
        assert extract_json(raw) == '{"sentiment": "bullish"}'

    def test_unwraps_bare_fence(self):
        raw = '```\n{"a": 1}\n```'
        assert extract_json(raw) == '{"a": 1}'

    def test_brace_inside_string_value_is_ignored(self):
        # A '}' inside a string must not end the object early.
        raw = 'prose {"reasoning": "target near }resistance{ zone", "n": 2} trailing'
        assert extract_json(raw) == '{"reasoning": "target near }resistance{ zone", "n": 2}'

    def test_nested_objects(self):
        raw = 'x {"a": {"b": {"c": 1}}, "d": 2} y'
        assert extract_json(raw) == '{"a": {"b": {"c": 1}}, "d": 2}'

    def test_escaped_quote_inside_string(self):
        raw = r'{"q": "he said \"hi\"", "n": 1}'
        assert extract_json(raw) == raw

    def test_no_json_returns_cleaned_text(self):
        # No object at all -> return cleaned text so the caller's parser raises
        # an informative error (this helper never swallows the failure).
        raw = "<think>only reasoning, no output</think>Now output."
        assert extract_json(raw) == "Now output."

    def test_none_and_empty(self):
        assert extract_json(None) == ""
        assert extract_json("") == ""

    def test_truncated_object_returns_cleaned_text(self):
        # Unbalanced braces (truncated stream) -> no object found -> cleaned
        # text passes through for the parser to reject.
        raw = '{"sentiment": "neutral", "score":'
        assert extract_json(raw) == raw
