"""Deterministic no-network provider with validity and transport fixtures."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any

from experiments.llm.models import ProviderRequest, ProviderResponse
from experiments.llm.providers.base import CapabilityReport, TransportError


class MockProvider:
    provider_name = "mock"

    def __init__(
        self,
        *,
        fixture: str = "valid",
        fixtures_by_run: dict[str, str] | None = None,
        transport_failures_before_success: int = 0,
    ) -> None:
        self.fixture = fixture
        self.fixtures_by_run = fixtures_by_run or {}
        self.transport_failures_before_success = transport_failures_before_success
        self.calls: list[str] = []
        self.attempts: defaultdict[str, int] = defaultdict(int)

    def capability_report(self) -> CapabilityReport:
        return CapabilityReport(True, False, False, True, True, True)

    @staticmethod
    def _prompt_graph(prompt: str) -> tuple[list[str], list[tuple[str, str]], str]:
        target_match = re.search(r"Target concept:\n(C\d+):", prompt)
        concept_match = re.search(r"Concepts:\n(.*?)\n\nPrerequisite edges:", prompt, re.DOTALL)
        edge_match = re.search(
            r"Prerequisite edges:\n(.*?)(?:\n\n(?:Aggregate historical statistics:|No historical))",
            prompt,
            re.DOTALL,
        )
        if not target_match or not concept_match or not edge_match:
            raise ValueError("Mock provider could not parse frozen prompt structure")
        concepts = [line.split(":", 1)[0] for line in concept_match.group(1).splitlines()]
        edges = []
        if edge_match.group(1).strip() != "(none)":
            edges = [tuple(line.split(" -> ", 1)) for line in edge_match.group(1).splitlines()]
        return concepts, edges, target_match.group(1)

    @classmethod
    def _valid_sequence(cls, prompt: str) -> list[str]:
        concepts, edges, target = cls._prompt_graph(prompt)
        remaining = set(concepts)
        predecessors = {node: set() for node in concepts}
        for source, destination in edges:
            predecessors[destination].add(source)
        result = []
        while remaining - {target}:
            available = sorted(
                node for node in remaining - {target}
                if predecessors[node].issubset(result)
            )
            if not available:
                raise ValueError("Mock prompt graph is cyclic or target is a required prerequisite")
            selected = available[0]
            result.append(selected)
            remaining.remove(selected)
        if not predecessors[target].issubset(result):
            raise ValueError("Mock target prerequisites are not satisfiable")
        return result + [target]

    def _response_text(self, request: ProviderRequest, fixture: str) -> str:
        valid = self._valid_sequence(request.user_prompt)
        payload = json.dumps({"sequence": valid}, separators=(",", ":"))
        if fixture == "valid":
            return payload
        if fixture == "code_fenced":
            return f"```json\n{payload}\n```"
        if fixture == "outer_text":
            return f"Here is the answer: {payload}"
        if fixture == "invalid_json":
            return "{'sequence': []}"
        if fixture == "duplicate_node":
            invalid = valid.copy(); invalid[1] = invalid[0]
            return json.dumps({"sequence": invalid})
        if fixture == "missing_node":
            return json.dumps({"sequence": valid[:-1]})
        if fixture == "unknown_id":
            invalid = valid.copy(); invalid[0] = "ZZZ"
            return json.dumps({"sequence": invalid})
        if fixture == "prerequisite_violation":
            return json.dumps({"sequence": list(reversed(valid[:-1])) + [valid[-1]]})
        if fixture == "target_not_final":
            invalid = valid.copy(); invalid[-1], invalid[-2] = invalid[-2], invalid[-1]
            return json.dumps({"sequence": invalid})
        if fixture == "empty_response":
            return ""
        raise ValueError(f"Unknown mock fixture: {fixture}")

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.calls.append(request.logical_run_id)
        attempt = self.attempts[request.logical_run_id]
        self.attempts[request.logical_run_id] += 1
        if attempt < self.transport_failures_before_success:
            raise TransportError("simulated 429", retryable=True, status_code=429)
        fixture = self.fixtures_by_run.get(request.logical_run_id, self.fixture)
        if fixture == "timeout":
            raise TransportError("simulated timeout", retryable=True)
        if fixture == "fatal_transport":
            raise TransportError("simulated non-retryable transport error", retryable=False, status_code=400)
        digest = hashlib.sha256(request.logical_run_id.encode("utf-8")).hexdigest()[:16]
        text = self._response_text(request, fixture)
        raw: dict[str, Any] = {
            "fixture": fixture,
            "logical_run_id": request.logical_run_id,
            "response_text": text,
        }
        return ProviderResponse(
            response_text=text,
            requested_model_id=request.requested_model_id,
            response_model_id="mock-model-v1",
            provider_request_id=f"mock-{digest}",
            created_at_utc="2000-01-01T00:00:00+00:00",
            finish_reason="stop",
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            latency_seconds=0.0,
            raw_provider_payload=raw,
        )
