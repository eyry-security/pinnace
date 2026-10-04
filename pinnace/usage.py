"""Exact, provider-reported inference usage metering.

The meter never estimates tokens. Missing or ambiguous provider metadata is
recorded explicitly and left unpriced rather than turning an estimate into a
billing fact.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4


class UsageMeterError(RuntimeError):
    """A usage record could not be made durable."""


def _token(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_token(*values: Any) -> int | None:
    for value in values:
        parsed = _token(value)
        if parsed is not None:
            return parsed
    return None


def _decimal(value: Decimal | str | int) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("cost rates must be finite and non-negative")
    return result


def _decimal_string(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


@dataclass(frozen=True)
class CostBasis:
    """Rates in USD per million tokens, captured with every priced call."""

    input_per_million: Decimal | str | int
    output_per_million: Decimal | str | int
    cache_read_per_million: Decimal | str | int
    cache_write_5m_per_million: Decimal | str | int
    cache_write_1h_per_million: Decimal | str | int
    source: str
    effective_date: str

    def __post_init__(self) -> None:
        for name in (
            "input_per_million",
            "output_per_million",
            "cache_read_per_million",
            "cache_write_5m_per_million",
            "cache_write_1h_per_million",
        ):
            object.__setattr__(self, name, _decimal(getattr(self, name)))

    def to_dict(self) -> dict[str, str]:
        return {
            "currency": "USD",
            "unit": "per_million_tokens",
            "input": _decimal_string(self.input_per_million),
            "output": _decimal_string(self.output_per_million),
            "cache_read": _decimal_string(self.cache_read_per_million),
            "cache_write_5m": _decimal_string(self.cache_write_5m_per_million),
            "cache_write_1h": _decimal_string(self.cache_write_1h_per_million),
            "source": self.source,
            "effective_date": self.effective_date,
        }


# Snapshot of the published direct-provider rates. The complete basis is copied
# into every record so historical charges do not depend on a mutable table.
_OPUS_46_DIRECT = CostBasis(
    input_per_million="5",
    output_per_million="25",
    cache_read_per_million="0.50",
    cache_write_5m_per_million="6.25",
    cache_write_1h_per_million="10",
    source="provider-published",
    effective_date="2026-10-04",
)


def default_cost_basis(model_ref: str | None) -> CostBasis | None:
    """Return a built-in basis only for an exact, known provider/model pair."""
    if not model_ref:
        return None
    provider, separator, model = model_ref.lower().partition(":")
    if separator and provider == "anthropic" and (
        model == "claude-opus-4-6" or model.startswith("claude-opus-4-6-")
    ):
        return _OPUS_46_DIRECT
    return None


@dataclass(frozen=True)
class TokenUsage:
    """Normalized exact token counts from a model response."""

    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cache_read_tokens: int
    cache_write_5m_tokens: int
    cache_write_1h_tokens: int
    cache_write_unclassified_tokens: int
    uncached_input_tokens: int | None

    @property
    def available(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None

    @property
    def exactly_priceable(self) -> bool:
        return self.available and self.cache_write_unclassified_tokens == 0

    def to_dict(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "uncached_input_tokens": self.uncached_input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_5m_tokens": self.cache_write_5m_tokens,
            "cache_write_1h_tokens": self.cache_write_1h_tokens,
            "cache_write_unclassified_tokens": self.cache_write_unclassified_tokens,
        }

    def cost(self, basis: CostBasis | None) -> Decimal | None:
        if basis is None or not self.exactly_priceable:
            return None
        assert self.uncached_input_tokens is not None
        assert self.output_tokens is not None
        numerator = (
            Decimal(self.uncached_input_tokens) * basis.input_per_million
            + Decimal(self.output_tokens) * basis.output_per_million
            + Decimal(self.cache_read_tokens) * basis.cache_read_per_million
            + Decimal(self.cache_write_5m_tokens) * basis.cache_write_5m_per_million
            + Decimal(self.cache_write_1h_tokens) * basis.cache_write_1h_per_million
        )
        return numerator / Decimal(1_000_000)

    @classmethod
    def from_response(cls, response: Any) -> "TokenUsage":
        """Read standardized metadata plus provider TTL details, without estimates."""
        standard = _mapping(getattr(response, "usage_metadata", None))
        response_metadata = _mapping(getattr(response, "response_metadata", None))
        native = _mapping(
            response_metadata.get("usage") or response_metadata.get("token_usage")
        )
        details = _mapping(standard.get("input_token_details"))
        native_creation = _mapping(native.get("cache_creation"))

        cache_read = _first_token(
            details.get("cache_read"),
            native.get("cache_read_input_tokens"),
        ) or 0
        cache_write_reported = _first_token(
            details.get("cache_creation"),
            native.get("cache_creation_input_tokens"),
        )
        cache_write_5m = _first_token(
            native_creation.get("ephemeral_5m_input_tokens"),
            native.get("cache_creation_5m_input_tokens"),
        ) or 0
        cache_write_1h = _first_token(
            native_creation.get("ephemeral_1h_input_tokens"),
            native.get("cache_creation_1h_input_tokens"),
        ) or 0
        classified_writes = cache_write_5m + cache_write_1h
        # Some provider adapters expose only the TTL split. Treat that exact
        # split as the aggregate floor so writes cannot also be billed as base
        # input. Any reported surplus remains explicitly unclassified.
        cache_write_total = max(cache_write_reported or 0, classified_writes)
        cache_write_unclassified = cache_write_total - classified_writes

        standard_input = _first_token(standard.get("input_tokens"))
        native_input = _first_token(native.get("input_tokens"), native.get("prompt_tokens"))
        if standard_input is not None:
            input_tokens = standard_input
        elif native_input is not None:
            # Native Claude metadata separates base, read, and creation tokens.
            if "input_tokens" in native and (
                "cache_read_input_tokens" in native
                or "cache_creation_input_tokens" in native
            ):
                input_tokens = native_input + cache_read + cache_write_total
            else:
                input_tokens = native_input
        else:
            input_tokens = None

        output_tokens = _first_token(
            standard.get("output_tokens"),
            native.get("output_tokens"),
            native.get("completion_tokens"),
        )
        total_tokens = _first_token(standard.get("total_tokens"), native.get("total_tokens"))
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        uncached_input = None
        if input_tokens is not None:
            uncached_input = input_tokens - cache_read - cache_write_total
            if uncached_input < 0:
                # Contradictory metadata is retained but never priced.
                cache_write_unclassified += -uncached_input
                uncached_input = None

        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_read_tokens=cache_read,
            cache_write_5m_tokens=cache_write_5m,
            cache_write_1h_tokens=cache_write_1h,
            cache_write_unclassified_tokens=cache_write_unclassified,
            uncached_input_tokens=uncached_input,
        )


def response_model(response: Any, fallback: str | None = None) -> str:
    metadata = _mapping(getattr(response, "response_metadata", None))
    for key in ("model_name", "model", "model_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback or "unknown"


class UsageMeter:
    """Append one JSON record per inference and optionally forward it to a sink."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if path is None:
            configured = os.environ.get("PINNACE_USAGE_LOG")
            if configured:
                path = configured
            else:
                home = Path(os.environ.get("PINNACE_HOME", "~/.pinnace")).expanduser()
                path = home / "usage.jsonl"
        self.path = Path(path).expanduser()
        self.sink = sink

    def record(
        self,
        response: Any,
        *,
        model: str,
        model_ref: str | None,
        call_kind: str,
        agent_id: str,
        customer_id: str | None,
        session_id: str | None,
        cost_basis: CostBasis | None = None,
    ) -> dict[str, Any]:
        usage = TokenUsage.from_response(response)
        basis = cost_basis if cost_basis is not None else default_cost_basis(model_ref)
        amount = usage.cost(basis)
        if not usage.available:
            status = "usage_unavailable"
        elif basis is None:
            status = "cost_basis_unavailable"
        elif not usage.exactly_priceable:
            status = "cache_write_ttl_unavailable"
        else:
            status = "priced"

        record: dict[str, Any] = {
            "schema_version": 1,
            "id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "customer_id": customer_id,
            "session_id": session_id,
            "call_kind": call_kind,
            "model": model,
            "model_ref": model_ref,
            "usage": usage.to_dict(),
            "cost_basis": basis.to_dict() if basis is not None else None,
            "cost": {
                "currency": "USD",
                "amount": _decimal_string(amount) if amount is not None else None,
                "status": status,
            },
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as output:
                output.write(line)
                output.flush()
                os.fsync(output.fileno())
            if self.sink is not None:
                self.sink(record)
        except (OSError, ValueError, TypeError) as error:
            raise UsageMeterError(f"could not persist usage record: {error}") from error
        return record
