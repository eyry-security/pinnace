"""Command-line interface.

Subcommands:
  run       hand the agent a prompt and let it work
  sessions  list saved sessions
  tools     list the built-in tools
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .agent import PinnaceAgent, PinnaceError
from .sandbox import DockerSandbox, LocalSandbox, SandboxError
from .session import SessionStore
from .tools import builtin_tools


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


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
    sp.add_argument("--model", default=None, help="provider:model ref (default: $PINNACE_MODEL or anthropic:claude-sonnet-4-5)")
    sp.add_argument("--system", default=None, help="override the system prompt")
    sp.add_argument("--max-turns", type=int, default=30)
    sp.add_argument("--compaction-tokens", type=int, default=100_000,
                    help="rough token tripwire for context compaction")
    sp.add_argument("--sandbox", choices=["docker", "local"], default="docker")
    sp.add_argument("--image", default="python:3.12-slim", help="docker image for the sandbox")
    sp.add_argument("--no-net", action="store_true", help="cut sandbox network egress")
    sp.add_argument("--workdir", default="./pinnace-work", help="workdir for the local sandbox")
    sp.add_argument("--unsafe-ok", action="store_true", help="allow the local sandbox (dev/tests only)")
    sp.add_argument("--session", default=None, help="persist/resume a named session")
    sp.add_argument("--json", action="store_true", help="emit the result as JSON on stdout")
    sp.add_argument("--quiet", action="store_true", help="no turn-by-turn narration on stderr")

    sp = sub.add_parser("sessions", help="list saved sessions")
    sp.add_argument("--root", default=None, help="session store root (default: $PINNACE_HOME or ~/.pinnace)")

    sp = sub.add_parser("tools", help="list the built-in tools")

    return p


def cmd_run(args) -> int:
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read()
    sandbox = _make_sandbox(args)
    try:
        agent = PinnaceAgent(
            model=args.model,
            sandbox=sandbox,
            system_prompt=args.system,
            max_turns=args.max_turns,
            compaction_tokens=args.compaction_tokens,
            session_id=args.session,
            log=(lambda *a: None) if args.quiet else _log,
        )
    except PinnaceError as e:
        _log(f"error: {e}")
        return 1
    try:
        result = agent.run(prompt)
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


_DISPATCH = {"run": cmd_run, "sessions": cmd_sessions, "tools": cmd_tools}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _DISPATCH[args.cmd](args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
