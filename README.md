# pinnace

> General multi-turn agent runtime with compaction, tools, and a Docker sandbox (Python).

Part of **[Eyry](https://eyry.io)**: open-source, agentic recon and offensive security tooling for
bug bounty hunters, red teamers, and pentesters. Pinnace is the agent engine the rest of the suite is built on. Hand it a prompt or seed and it runs.

## Status

🚧 **Early development.** Structure and APIs will change. Star the repo to follow along, and
see [eyry.io](https://eyry.io).

## What it does

- Persistent multi-turn agent loop with context compaction
- Built-in tool use, executed inside a Docker sandbox
- Runs standalone or driven by Quarterdeck and Aplomado

## Install

Coming soon.

## The Eyry suite

- **Vedette**: fast, multi-threaded HTTP prober (Rust)
- **Foretop**: configurable producer of new hosts from pluggable feeds (certstream first)
- **Purser**: Redis-backed priority queue and work distributor (hot/warm/cold/DLQ)
- **Pinnace**: general multi-turn agent runtime with compaction, tools, and a Docker sandbox
- **Aplomado**: AI security scanner and reviewer built on Pinnace
- **Quarterdeck**: agent control plane, scheduler, events, IRC-style chat, and pipeline orchestration

## License

MIT, Eyry.
