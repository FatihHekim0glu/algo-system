# ADR-0003 — Bar-finality guard

- **Status:** Accepted
- **Context:** in live/simulated execution, the most recent bar may still be forming;
  acting on it is a look-ahead that a backtest on closed bars cannot reproduce.

## Context

A vectorized backtest only ever sees closed bars. A real or simulated execution loop
receives the current bar **as it forms** — its high/low/close are not final until the
bar closes. If the strategy reacts to a partial bar, it is trading on information
that did not exist at any tradeable instant, and the result can never be reproduced
by the backtest (breaking ADR-0002 parity).

## Decision

- `backtest/bar_finality.py` provides the only-act-on-**closed**-bars guard: a
  partial / unclosed bar can **never** trigger an order. Order generation consumes
  only finalized bars.

## Consequences

- A unit test asserts that feeding a partial / forming bar emits no order.
- `serve.run_system` reports `bar_finality_ok` in the summary (and the frontend
  shows a "Bar-finality ✓" badge).
- Together with ADR-0001 and ADR-0002, this closes the loop: signals are causal,
  fills are next-bar, partial bars are inert, and the parity oracle proves the
  backtest and live paths cannot have diverged.
