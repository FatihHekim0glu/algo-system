# ADR-0005 — Simulated execution

- **Status:** Accepted
- **Context:** the "live" path of the pipeline must be honest about what it is — and
  is not — a connection to.

## Context

The project name and the parity oracle invite the assumption that there is a real
broker on the other end. There is not, and overstating that would be dishonest and a
security/operational risk (a real broker key in a public, deployed request path).

## Decision

- Execution is **SIMULATED**. The "live" path is an internal paper broker
  (`execution/paper_broker.py`) that replays bars with next-bar-open fills + fixed
  per-side basis-point costs + slippage. It tracks position, cash, and an equity
  curve.
- There is **no** live Alpaca / broker connection and **no broker key** anywhere in
  the package, the deployed route, or CI. Nothing places a real order.
- The deployed request path runs the full pipeline on the **synthetic default**
  (cheap, offline, never trains). Real point-in-time data is reachable only via the
  offline Polygon-PIT CLI path, using the existing Polygon key — never a broker key.

## Consequences

- The README headline, the Limitations section, and the frontend caption all state
  "SIMULATED execution, NOT a live broker" explicitly.
- Import purity holds: no network / broker at import time; the Polygon client is
  lazy.
- The fills are idealized (no partial fills, market impact, queue position, or
  latency) — documented as a limitation, not hidden.
