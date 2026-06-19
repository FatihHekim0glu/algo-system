# ADR-0002 — Backtest↔live parity oracle

- **Status:** Accepted
- **Context:** a vectorized backtest and a bar-by-bar execution engine are two
  implementations of the same strategy; they must agree, or one of them is wrong.

## Context

The pipeline has two equity paths:

- the **vectorized backtest** (`backtest/engine.py`) — fast, array-based, used for
  metrics and walk-forward; and
- the **simulated paper broker** (`execution/paper_broker.py`) — a bar-by-bar replay
  that fills at the next bar's open with costs + slippage (the "live" path).

If these diverge, either the backtest peeks at the future or there is a fill-timing
/ cost bug. A single number — the max absolute difference between the two equity
curves — captures the entire class of integration look-ahead bugs.

## Decision

- `execution/parity.py` asserts the vectorized backtest equity curve **equals** the
  paper-broker equity curve to an absolute `max-diff ≤ PARITY_TOL = 1e-10`, for any
  signal/param sequence.
- This is the **load-bearing** correctness artifact of the project. It is exercised
  by a Hypothesis property test over random position sequences (`tests/parity`).
- Costs and slippage are applied **identically** in both paths so the only thing the
  oracle can catch is a timing / leakage discrepancy.

## Consequences

- A deliberately-**leaky backtester negative control** is committed; the oracle
  catches it (parity FAILS with `ParityError`) in both a unit and an integration
  test — proving the oracle can actually fail, not just pass vacuously.
- On the deployed default, `backtest_live_parity_max_diff = 0.0` — the curves match
  to the cent.
