"""Construct capability-gated providers without importing data-preparation code."""

from __future__ import annotations

from experiments.llm.providers.closed_frontier import ClosedFrontierProvider
from experiments.llm.providers.mock import MockProvider
from experiments.llm.providers.open_weight import OpenWeightProvider


def build_provider(name: str, config: dict, fixture: str = "valid"):
    if name == "mock":
        return MockProvider(fixture=fixture)
    model = config["models"][name]
    cls = ClosedFrontierProvider if name == "closed_frontier" else OpenWeightProvider
    provider = cls(
        endpoint=model["endpoint"],
        requested_model_id=model["requested_model_id"],
        reasoning=model["reasoning"],
        api_key_env=model["api_key_env"],
        **({"thinking_enabled": model["thinking_enabled"]} if name == "open_weight" else {}),
    )
    provider.require_ready()
    return provider
