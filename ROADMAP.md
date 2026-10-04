# Pinnace Roadmap

Planned capabilities for the agent harness. Owner: Claude (Kiro) — Ben' sets priority.

## Terminal UX

- [ ] **Prettier terminal output** — richer live display of the agent loop:
  streaming responses, tool calls, color, progress indicators.
- [ ] **Thinking output** — stream and display the model's thinking blocks
  (collapsible/verbose modes), so you can follow the agent's reasoning.

## Prompting & context

- [ ] **Better prompting control** — prompt templates that are visible,
  overridable, and versioned per agent and task.
- [ ] **Compaction controls** — make compaction configurable and transparent:
  when it triggers, what got summarized, plus a manual trigger.

## Cost & performance

- [ ] **Prompt caching** — use Anthropic prompt caching for the large static
  prefixes (system prompts, tool definitions) to cut cost and latency on
  multi-turn scans. Wire cache breakpoints through Pinnace's model init.

## Shipped

- Agent loop, sessions, sandbox, tool system (v0)
- Convergence onto v0 base (merged to main 2026-10-04)
