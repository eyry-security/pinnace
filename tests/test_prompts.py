"""Prompt-pack controls work without model credentials, Docker, or network."""

from __future__ import annotations

import json

import pytest

from pinnace import (
    AgentResult,
    DEFAULT_PROMPT_PACK_ID,
    PromptError,
    PromptPack,
    available_prompt_packs,
    get_prompt_pack,
    load_prompt_pack,
)
from pinnace import cli
from pinnace.config import DEFAULT_SYSTEM


def test_default_pack_is_named_versioned_and_discoverable():
    pack = get_prompt_pack()
    assert pack.identifier == DEFAULT_PROMPT_PACK_ID == "general-v1"
    assert pack.system_prompt == DEFAULT_SYSTEM
    assert pack.render("do the thing") == "do the thing"
    assert get_prompt_pack("default") is pack
    assert available_prompt_packs() == (pack,)


def test_unknown_pack_lists_available_ids():
    with pytest.raises(PromptError, match="general-v1"):
        get_prompt_pack("missing-v9")


def test_custom_template_renders_documented_token_only():
    pack = PromptPack(
        name="review",
        version=2,
        system_prompt="Use the review policy.",
        run_prompt_template='Task:\n{prompt}\nKeep JSON: {"mode": "safe"}',
    )
    rendered = pack.render("literal {prompt} input")
    assert rendered == 'Task:\nliteral {prompt} input\nKeep JSON: {"mode": "safe"}'


def test_prompt_pack_validation():
    with pytest.raises(PromptError, match="name cannot be empty"):
        PromptPack(" ", 1, "system", "{prompt}")
    with pytest.raises(PromptError, match="version must be positive"):
        PromptPack("bad", 0, "system", "{prompt}")
    with pytest.raises(PromptError, match="system prompt cannot be empty"):
        PromptPack("bad", 1, " ", "{prompt}")
    with pytest.raises(PromptError, match=r"must contain \{prompt\}"):
        PromptPack("bad", 1, "system", "no task token")


def test_file_overrides_are_loaded_as_utf8_and_identified(tmp_path):
    system_file = tmp_path / "system.txt"
    run_file = tmp_path / "run.txt"
    system_file.write_text("Réponds clairement.\n", encoding="utf-8")
    run_file.write_text("Before\n{prompt}\nAfter", encoding="utf-8")

    pack = load_prompt_pack(
        "general-v1",
        system_prompt_file=str(system_file),
        run_prompt_file=str(run_file),
    )
    assert pack.identifier == "general-v1+system+run"
    assert pack.system_prompt == "Réponds clairement.\n"
    assert pack.render("inspect") == "Before\ninspect\nAfter"


def test_missing_empty_invalid_and_non_utf8_override_files_fail_cleanly(tmp_path):
    with pytest.raises(PromptError, match="cannot read system prompt file"):
        load_prompt_pack(system_prompt_file=str(tmp_path / "missing.txt"))

    empty = tmp_path / "empty.txt"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(PromptError, match="is empty"):
        load_prompt_pack(system_prompt_file=str(empty))

    invalid = tmp_path / "invalid.txt"
    invalid.write_text("No task token", encoding="utf-8")
    with pytest.raises(PromptError, match=r"must contain \{prompt\}"):
        load_prompt_pack(run_prompt_file=str(invalid))

    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"\xff")
    with pytest.raises(PromptError, match="cannot read system prompt file"):
        load_prompt_pack(system_prompt_file=str(binary))


def test_prompts_command_lists_and_displays_full_pack(capsys):
    assert cli.main(["prompts"]) == 0
    assert capsys.readouterr().out.strip() == "general-v1 (default)"

    assert cli.main(["prompts", "--pack", "general-v1", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "general-v1"
    assert payload["system_prompt"] == DEFAULT_SYSTEM
    assert payload["run_prompt_template"] == "{prompt}"


def test_run_parser_exposes_prompt_controls():
    args = cli.build_parser().parse_args([
        "run",
        "--prompt",
        "inspect",
        "--prompt-pack",
        "general-v1",
        "--system-prompt-file",
        "system.txt",
        "--run-prompt-file",
        "run.txt",
    ])
    assert args.prompt_pack == "general-v1"
    assert args.system_prompt_file == "system.txt"
    assert args.run_prompt_file == "run.txt"


class FakeSandbox:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeAgent:
    def __init__(self, seen):
        self.seen = seen

    def run(self, prompt):
        self.seen.append(prompt)
        return AgentResult(prompt, None, 1, 0, None)


def test_run_uses_custom_system_and_run_prompts(monkeypatch, tmp_path, capsys):
    system_file = tmp_path / "system.txt"
    run_file = tmp_path / "run.txt"
    system_file.write_text("CUSTOM SYSTEM", encoding="utf-8")
    run_file.write_text("CUSTOM RUN: {prompt}", encoding="utf-8")
    sandbox = FakeSandbox()
    configs = []
    prompts = []

    monkeypatch.setattr(cli, "_make_sandbox", lambda args: sandbox)

    def from_config(config):
        configs.append(config)
        return FakeAgent(prompts)

    monkeypatch.setattr(cli.PinnaceAgent, "from_config", from_config)
    code = cli.main([
        "run",
        "--prompt",
        "inspect",
        "--system-prompt-file",
        str(system_file),
        "--run-prompt-file",
        str(run_file),
        "--quiet",
    ])
    assert code == 0
    assert configs[0].system_prompt == "CUSTOM SYSTEM"
    assert prompts == ["CUSTOM RUN: inspect"]
    assert sandbox.closed
    capsys.readouterr()


def test_unknown_pack_is_rejected_before_sandbox_creation(monkeypatch, capsys):
    started = False

    def make_sandbox(args):
        nonlocal started
        started = True
        return FakeSandbox()

    monkeypatch.setattr(cli, "_make_sandbox", make_sandbox)
    assert cli.main([
        "run", "--prompt", "inspect", "--prompt-pack", "unknown-v1"
    ]) == 2
    assert not started
    assert "unknown prompt pack" in capsys.readouterr().err
