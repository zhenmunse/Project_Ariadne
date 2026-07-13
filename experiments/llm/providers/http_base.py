"""Minimal injectable JSON-over-HTTP transport for formal adapters."""

from __future__ import annotations

import json
import http.client
import os
import ssl
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from experiments.llm.providers.base import TransportError


JSONTransport = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]


def urllib_json_transport(
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, allow_nan=False).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        retryable = error.code == 429 or 500 <= error.code < 600
        raise TransportError(
            f"Provider HTTP error {error.code}",
            retryable=retryable,
            status_code=error.code,
        ) from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise TransportError(str(error), retryable=True) from error
    except (
        ConnectionResetError,
        ConnectionAbortedError,
        BrokenPipeError,
        http.client.IncompleteRead,
        ssl.SSLError,
    ) as error:
        # The request may have reached the provider and a response may have
        # started before the connection failed. Retrying could create a second
        # model response for the same logical run, so fail closed.
        raise TransportError(
            f"Ambiguous provider response transport failure: {error}",
            retryable=False,
        ) from error


def authorization_headers(api_key_env: str) -> dict[str, str]:
    secret = os.environ.get(api_key_env)
    if not secret:
        raise TransportError(
            f"Required API key environment variable is not set: {api_key_env}",
            retryable=False,
        )
    return {"Authorization": f"Bearer {secret}"}
