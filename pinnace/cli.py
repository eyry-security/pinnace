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
import sys

from . import __version__
from .agent import PinnaceAgent, PinnaceError
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
from .usage import UsageMeter


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
