"""Command-line interface.

Subcommands:
  run     run an agent with a prompt (interactive or one-shot)
  tools   list the registered tools
  config  show resolved configuration
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .config import AgentConfig


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--model", default=None,
                    help="LLM model name (env PINNACE_MODEL, default gpt-4o)")
    sp.add_argument("--api-key", default=None,
                    help="OpenAI API key (env OPENAI_API_KEY)")
    sp.add_argument("--base-url", default=None,
                    help="OpenAI-compatible base URL (env OPENAI_BASE_URL)")
    sp.add_argument("--sandbox-image", default=None,
                    help="Docker image for the sandbox (env PINNACE_SANDBOX_IMAGE)")
    sp.add_argument("--max-turns", type=int, default=None,
                    help="maximum agent turns (default 30)")


def _config(args) -> AgentConfig:
    return AgentConfig.resolve(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        sandbox_image=getattr(args, "sandbox_image", None),
        max_turns=getattr(args, "max_turns", None),
        system_prompt=getattr(args, "system_prompt", None),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pinnace",
        description="Multi-turn agent runtime with tool use and a Docker sandbox.",
    )
    p.add_argument("--version", action="version", version=f"pinnace {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("run", help="run an agent with a prompt")
    _add_common(sp)
    sp.add_argument("prompt", nargs="?", default=None,
                    help="the task / prompt (reads stdin if omitted)")
    sp.add_argument("--system-prompt", default=None,
                    help="override the default system prompt")
    sp.add_argument("--no-sandbox", action="store_true",
                    help="disable Docker sandbox (tools run on the host — be careful)")
    sp.add_argument("--network", action="store_true",
                    help="allow network access inside the sandbox")
    sp.add_argument("--quiet", action="store_true",
                    help="only print the final answer, not intermediate tool output")

    sp = sub.add_parser("tools", help="list registered tools")

    sp = sub.add_parser("config", help="show resolved configuration")
    _add_common(sp)
    sp.add_argument("--json", action="store_true")

    return p


def _log(msg: str) -> None:
    print(f"[pinnace] {msg}", file=sys.stderr, flush=True)


def _on_message(quiet: bool):
    def callback(role: str, content: str) -> None:
        if quiet and role != "assistant":
            return
        if role == "assistant":
            print(content, flush=True)
        else:
            print(f"  {content}", file=sys.stderr, flush=True)
    return callback


def cmd_run(args) -> int:
    from .agent import Agent

    prompt = args.prompt
    if prompt is None:
        if sys.stdin.isatty():
            _log("reading prompt from stdin (ctrl-d to end):")
        prompt = sys.stdin.read().strip()
    if not prompt:
        _log("no prompt given")
        return 2

    cfg = _config(args)
    if args.network:
        cfg.sandbox_network = True

    agent = Agent(config=cfg)

    if args.no_sandbox:
        _log(f"running without sandbox — model={cfg.model} turns={cfg.max_turns}")
        result = agent.run(prompt, on_message=_on_message(args.quiet))
    else:
        _log(f"running with sandbox={cfg.sandbox_image} — model={cfg.model} turns={cfg.max_turns}")
        result = agent.run_sandboxed(prompt, on_message=_on_message(args.quiet))

    return 0


def cmd_tools(args) -> int:
    from .tools import default_tools
    for tool in default_tools():
        print(f"  {tool.name:16} {tool.description}")
    return 0


def cmd_config(args) -> int:
    from dataclasses import asdict
    cfg = _config(args)
    d = asdict(cfg)
    # Mask the API key for display.
    if d.get("api_key"):
        d["api_key"] = d["api_key"][:4] + "..." + d["api_key"][-4:] if len(d["api_key"]) > 8 else "***"
    if getattr(args, "json", False):
        print(json.dumps(d, indent=2))
    else:
        for k, v in d.items():
            print(f"  {k:28} {v}")
    return 0


_DISPATCH = {
    "run": cmd_run,
    "tools": cmd_tools,
    "config": cmd_config,
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _DISPATCH[args.cmd](args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        _log(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
