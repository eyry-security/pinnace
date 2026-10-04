# Pinnace v0 API contract

Downstream repos (aplomado, quarterdeck) build on this. Treat it as stable;
breaking changes need a version bump and a note here.

## Install

```bash
pip install pinnace "pinnace[anthropic]"   # or pinnace[openai]
```

## Core usage

```python
from pinnace import (
    PinnaceAgent, AgentConfig, AgentResult, Message,
    Sandbox, DockerSandbox, LocalSandbox, SandboxError,
    SessionStore, builtin_tools,
    UsageMeter, CostBasis, TokenUsage, UsageMeterError,
)

agent = PinnaceAgent(
    model="anthropic:claude-opus-4-6",  # or a langchain chat model instance,
                                          # or omit -> $PINNACE_MODEL
    sandbox=DockerSandbox(),              # default if omitted; raises SandboxError
                                          # with a clear message when Docker is absent
    tools=[...],                          # extra langchain tools, appended AFTER builtins
    system_prompt="...",                  # default: DEFAULT_SYSTEM in pinnace.config
    max_turns=30,
    compaction_tokens=100_000,            # rough chars/4 tripwire
    compaction_keep_last=8,
    session_store=SessionStore(),         # default ~/.pinnace/sessions (or $PINNACE_HOME)
    session_id="run-name",                # persist + resume transcript; None = ephemeral
    agent_id="reviewer-1",               # copied into every inference usage record
    customer_id="team-1",                # optional tenant attribution
    usage_meter=UsageMeter("usage.jsonl"), # default: $PINNACE_USAGE_LOG or ~/.pinnace/usage.jsonl
    cost_basis=CostBasis(...),             # optional explicit rates for custom models/providers
    log=print,                            # turn-by-turn narration callable
)
result: AgentResult = agent.run("do the thing")
```

`AgentConfig.resolve(...)` provides the same constructor values in a reusable
object, including `PINNACE_MODEL` resolution. Construct with
`PinnaceAgent.from_config(config)`. `Message` aliases LangChain's `BaseMessage`,
the in-memory representation used by the agent loop and `SessionStore`.
`AgentResult.transcript` remains a list of serialized message dictionaries.

### Convergence note

The pre-merge v0 branch used `anthropic:claude-sonnet-4-5` as its fallback.
The converged 0.1.0 contract defaults to `anthropic:claude-opus-4-6`, per the
suite-wide model decision. `PINNACE_MODEL` and explicit `model=` values still
override the fallback.

## AgentResult

- `result.final: str | None` — last message text (None if it ended via finish() silently)
- `result.structured: dict | str | None` — payload from the `finish()` tool
- `result.turns: int`, `result.compacted: int`, `result.session_id: str | None`
- `result.finished: bool` — True when finish() was called
- `result.to_dict()` — JSON-serializable (transcript = list of serialized messages)
- `result.transcript` — list of langchain `dumpd` message dicts
- `result.usage` — one structured record per model call, including compaction

## Exact usage metering

Every inference appends one schema-versioned JSON object to
`$PINNACE_USAGE_LOG` or `~/.pinnace/usage.jsonl`. Records contain the actual
response model, call kind, agent/customer/session attribution, provider-reported
input/output/cache-read/cache-write tokens, the complete USD-per-million cost
basis, and an exact decimal charge. `AgentResult.usage` contains the same records
for that run. Prompts and completions are never copied into the usage log.

The meter does not estimate billing data. If a provider omits usage metadata,
a custom model has no configured `CostBasis`, or cache-write tokens lack the
5-minute/1-hour split needed for exact pricing, the call is still logged with a
specific unpriced status and `amount: null`. `CostBasis` lets callers provide
explicit rates for custom provider/model pairs. The built-in basis is selected
only for the exact default direct-provider model ref. Meter write failures raise
`UsageMeterError` so silent unmetered operation is impossible.

## Built-in tools (bound to the run's sandbox)

`shell(command)`, `read_file(path)`, `write_file(path, content)`,
`fetch_url(url)`, `finish(result_json)`. Tool failures are returned to the model
as text (`error: ...`), never raised. Unknown tool names are reported the same way.
`pinnace.tools.parse_finish(output)` extracts a finish payload: dict when the
output is a finish marker (wrapping non-JSON in `{"result": ...}`), else None.

## Sandbox

- `DockerSandbox(image="python:3.12-slim", workdir="/work", network=None, mem_limit="1g")`
  — one throwaway container per instance; `network="none"` cuts egress.
- `LocalSandbox(workdir=None, unsafe_ok=False)` — host execution; requires `unsafe_ok=True`.
  Workdir defaults to a fresh tempdir.
- Both: `exec(cmd, timeout=120.0) -> ExecResult(stdout, stderr, exit_code, truncated)`,
  `read_file(path)`, `write_file(path, content)`, `close()`. Output capped at 200k chars.
  Paths are workdir-relative; `..` and absolute paths raise `SandboxError`.

## Sessions

`SessionStore(root=None)` — `save(id, messages)`, `load(id) -> list | None`,
`list() -> [ids]`, `delete(id)`. Transcripts are JSONL of serialized messages.

## CLI

```
pinnace run --prompt TEXT | --prompt-file FILE [--model provider:model]
    [--max-turns N] [--sandbox docker|local] [--image IMG] [--no-net]
    [--session NAME] [--agent-id ID] [--customer-id ID] [--usage-log PATH]
    [--json] [--quiet]
pinnace compact SESSION [--root DIR] [--model provider:model] [--keep-last N]
    [--summarize-prompt-file FILE] [--agent-id ID] [--customer-id ID]
    [--usage-log PATH] [--json]
pinnace sessions [--root DIR]
pinnace tools
```

`pinnace compact` is the manual compaction trigger for a persisted session. It
bypasses the automatic token tripwire, retains the requested tail, replaces the
transcript only after a successful summary, and prints the same before/after estimate and message counts exposed by `CompactionReport`. The
summarizer call is recorded by the exact usage meter. Sessions with no messages
before the retained tail are reported as no-ops without constructing a model.

Interactive CLI narration is written to stderr with TTY-aware color and
progress markers for turns, tool calls, compaction, sessions, completion, and
errors. Set `NO_COLOR` to disable ANSI styling; redirected output is always
escape-free.

## Conventions for downstream repos

- Same `pyproject.toml` shape as purser/pinnace (setuptools, `[project.scripts]`
  entry point, `__init__.py`/`__main__.py`/`cli.py` layout).
- `from __future__ import annotations` everywhere; stderr narration prefixed `[name]`.
- Tests must run with no API keys and no Docker (scripted fake models, LocalSandbox).
- Builder-to-builder voice in docs: direct, technical, no buzzwords.
- Work on `vector/dev-<topic>` branches. Commit locally. NEVER push.

## Prompt packs

`PromptPack` is an immutable, versioned pair of `system_prompt` and
`run_prompt_template`. The run template must contain `{prompt}`; rendering
replaces only that token, so braces in operator input or JSON examples remain
literal. `get_prompt_pack()` resolves the current built-in default
(`general-v1`), `available_prompt_packs()` lists built-ins, and
`load_prompt_pack()` applies optional UTF-8 system/run text-file overrides.
Override identifiers include `+system` and/or `+run` so result provenance stays
visible.

Use `pinnace prompts` to list packs and
`pinnace prompts --pack general-v1 [--json]` to inspect the complete prompts.
`pinnace run` accepts `--prompt-pack`, `--system-prompt-file`, and
`--run-prompt-file`; an explicit `--system` remains supported and takes the
inline system text. Invalid packs or override files fail before a sandbox is
started.
