"""Known model catalog and provider wiring.

Only exact catalog entries are priced. Unknown refs stay unpriced rather than
falling back to another provider's rate table.
"""

from __future__ import annotations

from dataclasses import dataclass

from .usage import CostBasis

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to price and construct a cataloged model."""

    provider: str
    model: str
    chat_provider: str
    cost_basis: CostBasis
    base_url: str | None = None
    api_key_env: str | None = None


_ANTHROPIC_OPUS_46 = ModelSpec(
    provider="anthropic",
    model="claude-opus-4-6",
    chat_provider="anthropic",
    cost_basis=CostBasis(
        input_per_million="5",
        output_per_million="25",
        cache_read_per_million="0.50",
        cache_write_5m_per_million="6.25",
        cache_write_1h_per_million="10",
        source="provider-published",
        effective_date="2026-10-04",
    ),
)

_DEEPSEEK_FLASH = ModelSpec(
    provider="deepseek",
    model="deepseek-flash",
    chat_provider="openai",
    base_url=DEEPSEEK_BASE_URL,
    api_key_env=DEEPSEEK_API_KEY_ENV,
    cost_basis=CostBasis(
        input_per_million="0.15",
        output_per_million="0.60",
        cache_read_per_million="0.003",
        cache_write_5m_per_million="0",
        cache_write_1h_per_million="0",
        source="provider-published",
        effective_date="2026-10-24",
    ),
)

_DEEPSEEK_V4_PRO = ModelSpec(
    provider="deepseek",
    model="deepseek-v4-pro",
    chat_provider="openai",
    base_url=DEEPSEEK_BASE_URL,
    api_key_env=DEEPSEEK_API_KEY_ENV,
    cost_basis=CostBasis(
        input_per_million="0.66",
        output_per_million="1.98",
        # No separate cache-hit rate was published for this model; price cache
        # reads at the full input rate rather than inventing a discount.
        cache_read_per_million="0.66",
        cache_write_5m_per_million="0",
        cache_write_1h_per_million="0",
        source="provider-published",
        effective_date="2026-10-24",
    ),
)

_MODEL_CATALOG: dict[str, ModelSpec] = {
    "anthropic:claude-opus-4-6": _ANTHROPIC_OPUS_46,
    "deepseek:deepseek-flash": _DEEPSEEK_FLASH,
    "deepseek:deepseek-v4-pro": _DEEPSEEK_V4_PRO,
}


def get_model_spec(model_ref: str | None) -> ModelSpec | None:
    """Return the exact catalog entry for a provider/model ref, if any."""
    if not isinstance(model_ref, str):
        return None
    normalized = model_ref.strip().lower()
    spec = _MODEL_CATALOG.get(normalized)
    if spec is not None:
        return spec
    provider, separator, model = normalized.partition(":")
    if separator and provider == "anthropic" and model.startswith("claude-opus-4-6-"):
        return _ANTHROPIC_OPUS_46
    return None
