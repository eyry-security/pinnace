# pinnace

> General multi-turn agent runtime with compaction, tools, and a Docker sandbox (Python).

Part of **[Eyry](https://eyry.io)**: open-source, agentic recon and offensive security tooling for
bug bounty hunters, red teamers, and pentesters. Pinnace is the agent engine the rest of the suite
is built on — Aplomado (the AI scanner) and Quarterdeck (the control plane) both run on it. It is
deliberately general-purpose: hand it a prompt or a seed and it runs.

## What it does

- **Multi-turn agent loop** with tool calling and a hard turn cap
- **Context compaction** — long runs get summarized mid-flight instead of blowing the window
- **Built-in tools** (`shell`, `read_file`, `write_file`, `fetch_url`) executed inside a sandbox
- **Docker sandbox** per run (throwaway container, optional no-egress); local fallback for dev
- **`finish()` convention** — the agent ends a run with a structured JSON result, not vibes
- **Persistent sessions** — transcripts saved as JSONL, resumable by name

## Install

```bash
pip install pinnace
# plus a model provider, e.g.:
pip install "pinnace[anthropic]"   # or pinnace[openai]
```

Needs a model: set `PINNACE_MODEL` (e.g. `anthropic:claude-sonnet-4-5`) or pass `--model`.
Docker is required for the real sandbox; without it, use `--sandbox local --unsafe-ok`
(dev only — it runs model-generated commands on your machine).

## Quickstart

```bash
# one-shot task, docker sandbox
pinnace run --prompt "clone https://github.com/example/tool, build it, and summarize what it does"

# keep going later in the same session
pinnace run --session recon-1 --prompt "now check its open issues for security-relevant ones"

# machine-readable result
pinnace run --prompt "audit this repo for hardcoded secrets" --json > result.json

# no docker on this box (dev only)
pinnace run --prompt "list files" --sandbox local --unsafe-ok
```

Python API:

```python
from pinnace import PinnaceAgent, LocalSandbox

agent = PinnaceAgent(
    model="anthropic:claude-sonnet-4-5",
    sandbox=LocalSandbox("./work", unsafe_ok=True),  # dev only
    session_id="my-run",
    max_turns=30,
)
result = agent.run("write a fuzzer for the login endpoint and run it")
print(result.structured or result.final)
```

## How it works

Each turn: estimate context size → compact if over the tripwire → call the model with tools
bound → execute tool calls in the sandbox → repeat. The loop ends when the model stops calling
tools, when it calls `finish(result_json)`, or at `max_turns`.

**Compaction** summarizes everything except the trailing messages into one system message,
keeping the original system prompt verbatim. The tripwire is a rough chars/4 estimate —
it just needs to fire before the real window does.

**Sessions** live under `~/.pinnace/sessions/<name>/transcript.jsonl` (or `$PINNACE_HOME`).
Resume with `--session <name>`; list with `pinnace sessions`.

## CLI

```
pinnace run --prompt TEXT | --prompt-file FILE [--model provider:model]
    [--max-turns N] [--sandbox docker|local] [--image IMG] [--no-net]
    [--session NAME] [--json] [--quiet]
pinnace sessions [--root DIR]
pinnace tools
```

## The Eyry suite

- **Vedette**: fast, multi-threaded HTTP prober (Rust)
- **Pinnace**: general multi-turn agent runtime with compaction, tools, and a Docker sandbox
- **Aplomado**: AI security scanner and reviewer built on Pinnace
- **Purser**: Redis-backed priority queue and work distributor (hot/warm/cold/DLQ)
- **Foretop**: configurable producer of new hosts from pluggable feeds (certstream first)
- **Quarterdeck**: agent control plane, scheduler, events, IRC-style chat, and pipeline orchestration

## License

MIT, Eyry.
