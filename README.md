# pinnace

> The ship's boat, dispatched to do independent work: a general multi-turn agent runtime with compaction, tools, and a Docker sandbox.

Part of **[Eyry](https://eyry.io)** — open-source, agentic recon and offensive security tooling for bug bounty hunters, red teamers, and pentesters.

*Status: converging. This README documents the converged v0 branch (`claude/pinnace-converge`, v0.1.0, 29 tests). The public API contract lives in `API.md`.*

## What it does

- **Multi-turn agent loop** with tool calling and a hard turn cap (`max_turns=30` default)
- **Context compaction** — when estimated context passes the tripwire (`compaction_tokens`, default 100,000), everything except the trailing messages is summarized mid-run instead of blowing the window
- **Five built-in tools** bound to the run's sandbox: `shell`, `read_file`, `write_file`, `fetch_url`, `finish`. Tool failures are returned to the model as text, never raised
- **Docker sandbox** — one throwaway container per run (`python:3.12-slim` default, 1g memory cap); `network="none"` cuts egress. `LocalSandbox` is the dev fallback and refuses to run without `unsafe_ok=True`
- **`finish()` convention** — the agent ends a run with a structured JSON result instead of vibes
- **Persistent sessions** — transcripts saved as JSONL under `~/.pinnace/sessions/<name>/transcript.jsonl` (`$PINNACE_HOME` overrides), resumable by name
- **Provider-agnostic models** — any LangChain chat model via `provider:model` refs. Default is `anthropic:claude-opus-4-6` (the suite-wide model decision); `$PINNACE_MODEL` or an explicit `model=` overrides it

Deliberately *not* a security tool. Pinnace is the keystone everything agentic in the suite builds on, and its generality is the point: point a sandboxed agent at anything.

## Install

Requires Python 3.10+. Docker for the real sandbox. A model provider key in your environment (e.g. `ANTHROPIC_API_KEY`).

```bash
pip install pinnace "pinnace[anthropic]"   # or pinnace[openai]
```

From source:

```bash
git clone https://github.com/eyry-security/pinnace
cd pinnace && pip install -e . "pinnace[anthropic]"
```

## Quickstart

```bash
# one-shot task, docker sandbox
pinnace run --prompt "clone https://github.com/example/tool, build it, and summarize what it does"

# keep going later in the same session
pinnace run --session recon-1 --prompt "now check its open issues for security-relevant ones"

# machine-readable result
pinnace run --prompt "audit this repo for hardcoded secrets" --json > result.json

# prompt from a file, custom system prompt, no sandbox network
pinnace run --prompt-file task.md --system "you are a terse auditor" --no-net

# inspect the exact versioned prompts, then override either template from UTF-8 files
pinnace prompts
pinnace prompts --pack general-v1 --json
pinnace run --prompt "audit this repo" \
  --system-prompt-file system.txt --run-prompt-file task-template.txt

# no docker on this box (dev only — runs model-generated commands on YOUR machine)
pinnace run --prompt "list files" --sandbox local --unsafe-ok

# list saved sessions and built-in tools
pinnace sessions
pinnace tools
```

Python API:

```python
from pinnace import PinnaceAgent, AgentConfig, LocalSandbox

agent = PinnaceAgent(
    model="anthropic:claude-opus-4-6",   # or a langchain chat model, or omit -> $PINNACE_MODEL
    sandbox=LocalSandbox("./work", unsafe_ok=True),  # dev only
    session_id="my-run",
    max_turns=30,
)
result = agent.run("write a fuzzer for the login endpoint and run it")
print(result.structured or result.final)
```

For reusable settings, build `AgentConfig.resolve(...)` and pass it to `PinnaceAgent.from_config(config)`. The public `Message` type is LangChain's `BaseMessage` — the in-memory representation used by the loop and the session store.

## How it works

Each turn: estimate context size → compact if over the tripwire → call the model with tools bound → execute tool calls in the sandbox → repeat. The loop ends when the model stops calling tools, when it calls `finish(result_json)`, or at `max_turns`.

**Compaction** summarizes everything except the last `compaction_keep_last` (default 8) messages into one system message, keeping the original system prompt verbatim. The tripwire is a rough chars/4 estimate — it just needs to fire before the real window does.

**Sessions** are JSONL transcripts on disk. Pass `--session <name>` (or `session_id=` in the API) and the transcript persists; run again with the same name and the agent picks up where it left off.

## CLI

```
pinnace run --prompt TEXT | --prompt-file FILE [--model provider:model]
    [--system TEXT] [--max-turns N] [--compaction-tokens N]
    [--sandbox docker|local] [--image IMG] [--no-net] [--workdir DIR]
    [--unsafe-ok] [--session NAME] [--json] [--quiet]
pinnace sessions [--root DIR]
pinnace tools
```

`--json` prints `result.to_dict()` on stdout; narration goes to stderr. `AgentResult` carries `final` (last message text), `structured` (the `finish()` payload), `turns`, `compacted`, `session_id`, `finished`, and `transcript` (serialized message dicts).

## Where it fits

Pinnace is the agent engine Aplomado and Quarterdeck run on: Aplomado wraps it in a security-review policy and a findings schema; Quarterdeck wakes it on a schedule and gives it an identity, memory, and a chat room.

## The Eyry suite

- **eyry**: one CLI that wires the data plane together — discover → queue → probe → store
- **vedette**: fast, multi-threaded HTTP prober (Rust) — confirms what's live and fingerprints it
- **foretop**: pluggable live feed of new hosts, starting with Certificate Transparency logs
- **purser**: Redis-backed priority work queue — hot/warm/cold lanes, retries, dead-letter queue
- **rutt**: Postgres store for the host lifecycle (discovered → probed → reviewed) with an append-only scan log
- **pinnace**: general multi-turn agent runtime — compaction, tools, Docker sandbox, resumable sessions
- **aplomado**: AI security reviewer built on Pinnace — target in, structured findings out
- **quarterdeck**: agent control plane — scheduler, wake/sleep, identity and memory, IRC-style chat, ChatOps, pipeline orchestration

## Roadmap

- Merge the converged v0 to `main` and cut the first public release (API contract in `API.md` is treated as stable from here)
- Verify DigitalOcean serverless inference as a cost-down provider option (`anthropic-claude-opus-4.6` via OpenAI-compatible endpoint) — needs a cheap test call first
- Quarterdeck wake/sleep and agent identity build on Pinnace sessions — keep that path load-bearing
- Keep the core general: security behavior lives in Aplomado, not here

## License

MIT © Eyry.
