"""Provider interface, transport errors, and capability contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from experiments.llm.models import ProviderRequest, ProviderResponse


class ProviderConfigurationError(RuntimeError):
    pass


class TransportError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


@dataclass(frozen=True)
class CapabilityReport:
    supports_single_turn: bool
    tools_sent: bool
    previous_response_id_sent: bool
    native_reasoning_requested: bool
    requested_model_id_frozen: bool
    endpoint_frozen: bool

    @property
    def ready(self) -> bool:
        return (
            self.supports_single_turn
            and not self.tools_sent
            and not self.previous_response_id_sent
            and self.native_reasoning_requested
            and self.requested_model_id_frozen
            and self.endpoint_frozen
        )


class LLMProvider(Protocol):
    provider_name: str

    def capability_report(self) -> CapabilityReport:
        ...

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        ...
