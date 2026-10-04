"""Command-line interface.

Subcommands:
  run       hand the agent a prompt and let it work
  prompts   inspect built-in versioned prompt packs
  sessions  list saved sessions
  tools     list the built-in tools
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from langchain_core.messages import SystemMessage

from . import __version__
from .agent import PinnaceAgent, PinnaceError, _make_model
from .compaction import SUMMARIZE_PROMPT
from .compaction_controls import CompactionConfig, compact_with_config
from .config import AgentConfig, DEFAULT_MODEL
from .prompts import (
    DEFAULT_PROMPT_PACK_ID,
    PromptError,
    available_prompt_packs,
    get_prompt_pack,
    load_prompt_pack,
)
from .sandbox import DockerSandbox, LocalSandbox, SandboxError
from .session import SessionStore
from .tools import builtin_tools
from .terminal import TerminalRenderer
from .usage import UsageMeter, response_model


def _log(msg: str) -> None:
    TerminalRenderer(stream=sys.stderr)(msg)


def _make_sandbox(args):
    if args.sandbox == "local":
        if not args.unsafe_ok:
            _log("local sandbox runs model-generated commands on YOUR machine.")
            _log("re-run with --unsafe-ok if that's really what you want (dev/tests only).")
            raise SystemExit(2)
        return LocalSandbox(args.workdir, unsafe_ok=True)
    try:
        return DockerSandbox(image=args.image, network=args.no_net and "none" or None)
    except SandboxError as e:
        _log(f"error: {e}")
        raise SystemExit(1)


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("must be an integer") from e
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pinnace",
        description="General multi-turn agent runtime. Hand it a prompt; it runs in a sandbox.",
    )
    p.add_argument("--version", action="version", version=f"pinnace {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("run", help="run the agent on a prompt")
    src = sp.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="the task, as a string")
    src.add_argument("--prompt-file", help="read the task from a file")
    sp.add_argument(
        "--model", default=None,
        help=f"provider:model ref (default: $PINNACE_MODEL or {DEFAULT_MODEL})",
    )
    sp.add_argument(
        "--prompt-pack",
        default=DEFAULT_PROMPT_PACK_ID,
        help=f"versioned prompt pack (default: {DEFAULT_PROMPT_PACK_ID})",
    )
    system = sp.add_mutually_exclusive_group()
    system.add_argument("--system", default=None, help="override the system prompt inline")
    system.add_argument(
        "--system-prompt-file",
        help="UTF-8 file replacing the selected pack's system prompt",
    )
    sp.add_argument(
        "--run-prompt-file",
        help="UTF-8 template replacing the selected pack's run template; must include {prompt}",
    )
    sp.add_argument("--max-turns", type=int, default=30)
    sp.add_argument("--compaction-tokens", type=int, default=100_000,
                    help="rough token tripwire for context compaction")
    sp.add_argument("--sandbox", choices=["docker", "local"], default="docker")
    sp.add_argument("--image", default="python:3.12-slim", help="docker image for the sandbox")
    sp.add_argument("--no-net", action="store_true", help="cut sandbox network egress")
    sp.add_argument("--workdir", default="./pinnace-work", help="workdir for the local sandbox")
    sp.add_argument("--unsafe-ok", action="store_true", help="allow the local sandbox (dev/tests only)")
    sp.add_argument("--session", default=None, help="persist/resume a named session")
    sp.add_argument("--agent-id", default=None,
                    help="usage attribution (default: $PINNACE_AGENT_ID, session, or pinnace)")
    sp.add_argument("--customer-id", default=None,
                    help="optional usage attribution (default: $PINNACE_CUSTOMER_ID)")
    sp.add_argument("--usage-log", default=None,
                    help="append-only usage JSONL (default: $PINNACE_USAGE_LOG or ~/.pinnace/usage.jsonl)")
    sp.add_argument("--json", action="store_true", help="emit the result as JSON on stdout")
    sp.add_argument("--quiet", action="store_true", help="no turn-by-turn narration on stderr")

    sp = sub.add_parser("prompts", help="list or inspect versioned prompt packs")
    sp.add_argument("--pack", help="show one prompt pack in full")
    sp.add_argument("--json", action="store_true", help="emit the selected pack as JSON")

    sp = sub.add_parser("compact", help="manually compact a saved session")
    sp.add_argument("session", help="saved session name")
    sp.add_argument("--root", default=None,
                    help="session store root (default: $PINNACE_HOME or ~/.pinnace)")
    sp.add_argument("--model", default=None,
                    help=f"summarizer provider:model (default: $PINNACE_MODEL or {DEFAULT_MODEL})")
    sp.add_argument("--keep-last", type=_non_negative_int, default=8,
                    help="recent messages to retain verbatim (default: 8)")
    sp.add_argument("--summarize-prompt-file",
                    help="UTF-8 file replacing the compaction prompt")
    sp.add_argument("--agent-id", default=None,
                    help="usage attribution (default: $PINNACE_AGENT_ID or session name)")
    sp.add_argument("--customer-id", default=None,
                    help="optional usage attribution (default: $PINNACE_CUSTOMER_ID)")
    sp.add_argument("--usage-log", default=None,
                    help="append-only usage JSONL (default: $PINNACE_USAGE_LOG or ~/.pinnace/usage.jsonl)")
    sp.add_argument("--json", action="store_true", help="emit the compaction report as JSON")

    sp = sub.add_parser("sessions", help="list saved sessions")
    sp.add_argument("--root", default=None, help="session store root (default: $PINNACE_HOME or ~/.pinnace)")

    sub.add_parser("tools", help="list the built-in tools")

    return p


def cmd_run(args) -> int:
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read()
    pack = load_prompt_pack(
        args.prompt_pack,
        system_prompt_file=args.system_prompt_file,
        run_prompt_file=args.run_prompt_file,
    )
    rendered_prompt = pack.render(prompt)
    system_prompt = args.system if args.system is not None else pack.system_prompt

    sandbox = _make_sandbox(args)
    try:
        config = AgentConfig.resolve(
            model=args.model,
            sandbox=sandbox,
            system_prompt=system_prompt,
            max_turns=args.max_turns,
            compaction_tokens=args.compaction_tokens,
            session_id=args.session,
            log=(lambda *a: None) if args.quiet else _log,
            usage_meter=UsageMeter(args.usage_log) if args.usage_log else None,
            agent_id=args.agent_id,
            customer_id=args.customer_id,
        )
        agent = PinnaceAgent.from_config(config)
    except PinnaceError as e:
        sandbox.close()
        _log(f"error: {e}")
        return 1
    try:
        result = agent.run(rendered_prompt)
    finally:
        sandbox.close()
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        if result.structured is not None:
            print(json.dumps(result.structured, indent=2))
        elif result.final:
            print(result.final)
        _log(f"[pinnace] done in {result.turns} turns"
             + (f", compacted {result.compacted}x" if result.compacted else ""))
    return 0


def cmd_prompts(args) -> int:
    if not args.pack:
        for pack in available_prompt_packs():
            suffix = " (default)" if pack.identifier == DEFAULT_PROMPT_PACK_ID else ""
            print(f"{pack.identifier}{suffix}")
        return 0

    pack = get_prompt_pack(args.pack)
    payload = {
        "id": pack.identifier,
        "name": pack.name,
        "version": pack.version,
        "system_prompt": pack.system_prompt,
        "run_prompt_template": pack.run_prompt_template,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"id: {pack.identifier}")
        print("\n[system_prompt]")
        print(pack.system_prompt)
        print("\n[run_prompt_template]")
        print(pack.run_prompt_template)
    return 0


def cmd_compact(args) -> int:
    store = SessionStore(args.root)
    messages = store.load(args.session)
    if messages is None:
        _log(f"error: session {args.session!r} not found")
        return 1

    body_start = 1 if messages and isinstance(messages[0], SystemMessage) else 0
    messages_summarizable = max(0, len(messages) - body_start - args.keep_last)
    if messages_summarizable == 0:
        payload = {
            "session_id": args.session,
            "compacted": False,
            "message_count": len(messages),
            "keep_last": args.keep_last,
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(
                f"session {args.session!r}: nothing to compact "
                f"({len(messages)} messages, keeping {args.keep_last})"
            )
        return 0

    summarize_prompt = SUMMARIZE_PROMPT
    if args.summarize_prompt_file:
        try:
            with open(args.summarize_prompt_file, encoding="utf-8") as f:
                summarize_prompt = f.read()
        except (OSError, UnicodeError) as e:
            _log(f"error: cannot read compaction prompt: {e}")
            return 2

    model_ref = AgentConfig.resolve(model=args.model).model
    try:
        model = _make_model(model_ref)
    except PinnaceError as e:
        _log(f"error: {e}")
        return 1

    meter = UsageMeter(args.usage_log)
    usage_records: list[dict] = []

    def record_usage(response) -> None:
        usage_records.append(meter.record(
            response,
            model=response_model(response, model_ref),
            model_ref=model_ref,
            call_kind="compaction",
            agent_id=(args.agent_id or os.environ.get("PINNACE_AGENT_ID") or args.session),
            customer_id=(args.customer_id or os.environ.get("PINNACE_CUSTOMER_ID")),
            session_id=args.session,
        ))

    config = CompactionConfig(
        max_tokens=0,
        keep_last=args.keep_last,
        summarize_prompt=summarize_prompt,
    )
    try:
        compacted, report = compact_with_config(
            model,
            messages,
            config,
            force=True,
            on_response=record_usage,
        )
        if report is None:  # Defensive: candidate count above should prevent this.
            _log(f"error: session {args.session!r} could not be compacted")
            return 1
        path = store.save(args.session, compacted)
    except Exception as e:  # noqa: BLE001 - summary/meter failures leave the session unchanged
        _log(f"error: compaction failed: {e}")
        return 1

    payload = {
        "session_id": args.session,
        "compacted": True,
        **report.to_dict(),
        "message_count_after": len(compacted),
        "path": str(path),
        "usage": usage_records,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"session {args.session!r}: {report.one_liner()}; saved to {path}")
    return 0


def cmd_sessions(args) -> int:
    store = SessionStore(args.root)
    for s in store.list():
        print(s)
    return 0


def cmd_tools(args) -> int:
    print("built-in tools (bound to the run's sandbox):")
    for name, doc in [
        ("shell", "Run a shell command inside the sandbox. Returns stdout, stderr, exit code."),
        ("read_file", "Read a file from the sandbox workdir (relative path)."),
        ("write_file", "Write a file into the sandbox workdir (relative path). Creates parent dirs."),
        ("fetch_url", "GET a URL and return the body as text. Plain GET, no JS."),
        ("finish", "End the run with a structured result: a JSON object as a string."),
    ]:
        print(f"  {name:12} {doc}")
    return 0


_DISPATCH = {
    "run": cmd_run,
    "prompts": cmd_prompts,
    "compact": cmd_compact,
    "sessions": cmd_sessions,
    "tools": cmd_tools,
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _DISPATCH[args.cmd](args)
    except PromptError as e:
        _log(f"error: {e}")
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
