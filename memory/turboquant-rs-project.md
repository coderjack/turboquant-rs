---
name: TurboQuant-RS Project
description: Rust implementation of Google's TurboQuant paper — workspace structure, algorithm pipeline, and current state
type: project
---

Rust workspace implementing Google's TurboQuant paper (arXiv:2504.19874).

**Why:** Building a high-performance vector compression library for embedding search (agent memory, code search).

**How to apply:** The TurboQuant pipeline is 3 steps — rotation, PolarQuant, QJL residual correction. All code changes must preserve this structure.

## Workspace (as of 2026-04-04)

- `crates/turboquant/` — core compression library (rewritten 2026-04-04)
- `crates/codesearch-mcp/` — MCP server for semantic code search (independent)
- `agent-memory` and `memory-mcp` were removed — to be rebuilt in a separate codebase

## TurboQuant Pipeline

1. **Rotation** (`compression/rotation.rs`) — random orthogonal matrix, spreads energy
2. **PolarQuant** (`compression/polarquant.rs`) — polar coordinate quantization (angle + radius per pair), NO baked-in rotation
3. **QJL** (`compression/qjl.rs`) — applied to the RESIDUAL after PolarQuant, 1-bit sign correction

Composed in `turboquant.rs` with `PreparedQuery` for O(d²)-once, O(d)-per-candidate search.

## Key Numbers (384-dim)

- 344 bytes/vector (4.5x compression vs f32)
- PolarQuant: 288 bytes, QJL signs: 48 bytes, norms: 8 bytes
- Brute-force search (no pre-filter)

## GitHub

Private repo: github.com/coderjack/turboquant-rs
