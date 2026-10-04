"""Visible, versioned prompt packs and overrides for Pinnace runs."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

from .config import DEFAULT_SYSTEM

_PROMPT_TOKEN = re.compile(r"\{prompt\}")


class PromptError(ValueError):
    """A prompt pack name, template, or override file is invalid."""


@dataclass(frozen=True)
class PromptPack:
    """One immutable, versioned pair of system and per-run prompts."""

    name: str
    version: int
    system_prompt: str
    run_prompt_template: str
    overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise PromptError("prompt pack name cannot be empty")
        if self.version < 1:
            raise PromptError("prompt pack version must be positive")
        if not self.system_prompt.strip():
            raise PromptError("system prompt cannot be empty")
        if not self.run_prompt_template.strip():
            raise PromptError("run prompt template cannot be empty")
        if "{prompt}" not in self.run_prompt_template:
            raise PromptError("run prompt template must contain {prompt}")

    @property
    def identifier(self) -> str:
        """Stable built-in ID, with an explicit suffix for local overrides."""
        identifier = f"{self.name}-v{self.version}"
        if self.overrides:
            identifier += "+" + "+".join(self.overrides)
        return identifier

    def render(self, prompt: str) -> str:
        """Render only the documented token; unrelated braces remain untouched."""
        return _PROMPT_TOKEN.sub(lambda _match: prompt, self.run_prompt_template)


DEFAULT_PROMPT_PACK_ID = "general-v1"
_GENERAL_V1 = PromptPack(
    name="general",
    version=1,
    system_prompt=DEFAULT_SYSTEM,
    run_prompt_template="{prompt}",
)
_PROMPT_PACKS = {DEFAULT_PROMPT_PACK_ID: _GENERAL_V1}


def available_prompt_packs() -> tuple[PromptPack, ...]:
    """Return built-in packs in stable display order."""
    return tuple(_PROMPT_PACKS.values())


def get_prompt_pack(identifier: str | None = None) -> PromptPack:
    """Resolve a built-in prompt pack; ``default`` tracks the current default."""
    requested = identifier or DEFAULT_PROMPT_PACK_ID
    if requested == "default":
        requested = DEFAULT_PROMPT_PACK_ID
    try:
        return _PROMPT_PACKS[requested]
    except KeyError as exc:
        choices = ", ".join(_PROMPT_PACKS)
        raise PromptError(
            f"unknown prompt pack {identifier!r}; available: {choices}"
        ) from exc


def _read_override(path: str, label: str) -> str:
    try:
        content = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PromptError(f"cannot read {label} {path!r}: {exc}") from exc
    if not content.strip():
        raise PromptError(f"{label} {path!r} is empty")
    return content


def load_prompt_pack(
    identifier: str | None = None,
    *,
    system_prompt_file: str | None = None,
    run_prompt_file: str | None = None,
) -> PromptPack:
    """Resolve a built-in pack and apply optional UTF-8 text-file overrides."""
    pack = get_prompt_pack(identifier)
    overrides: list[str] = []
    changes: dict[str, object] = {}
    if system_prompt_file:
        changes["system_prompt"] = _read_override(
            system_prompt_file, "system prompt file"
        )
        overrides.append("system")
    if run_prompt_file:
        changes["run_prompt_template"] = _read_override(
            run_prompt_file, "run prompt file"
        )
        overrides.append("run")
    if not changes:
        return pack
    changes["overrides"] = tuple(overrides)
    return replace(pack, **changes)
