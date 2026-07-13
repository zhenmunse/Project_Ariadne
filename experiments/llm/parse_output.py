"""Strict minimal parser for one LLM sequence JSON object."""

from __future__ import annotations

import json
import re

from experiments.llm.models import ParseResult


_OUTER_FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```\Z", re.IGNORECASE)


def _error(code: str, detail: str, *, parsed: bool = False) -> ParseResult:
    return ParseResult(parsed, False, None, code, detail)


def parse_output(response_text: str) -> ParseResult:
    if not isinstance(response_text, str):
        raise TypeError("response_text must be a string")
    text = response_text.strip()
    if not text:
        return _error("empty_response", "Response is empty")
    fence = _OUTER_FENCE.fullmatch(text)
    if fence:
        text = fence.group(1).strip()
    elif text.startswith("```") or text.endswith("```"):
        return _error("outer_text_present", "Code fence is not the sole outer wrapper")
    if not text.startswith("{"):
        return _error("outer_text_present", "Text exists outside the JSON object")
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(text)
    except json.JSONDecodeError as error:
        return _error("invalid_json", str(error))
    remainder = text[end:].strip()
    if remainder:
        code = "multiple_json_objects" if remainder.startswith("{") else "outer_text_present"
        return _error(code, "Content remains after the JSON object")
    if not isinstance(value, dict):
        return _error("root_not_object", "JSON root must be an object", parsed=True)
    if "sequence" not in value:
        return _error("missing_sequence", "Required sequence field is absent", parsed=True)
    if set(value) != {"sequence"}:
        return _error("unexpected_fields", "Only the sequence field is allowed", parsed=True)
    sequence = value["sequence"]
    if not isinstance(sequence, list):
        return _error("sequence_not_array", "sequence must be an array", parsed=True)
    if any(not isinstance(item, str) for item in sequence):
        return _error("sequence_item_not_string", "Every sequence item must be a string", parsed=True)
    return ParseResult(True, True, tuple(sequence), None, None)
