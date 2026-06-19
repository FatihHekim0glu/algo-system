# ADR-0001 — Causal signal & next-bar fill

- **Status:** Accepted
- **Context:** signal → backtest → execution pipeline; the most common backtest bug
  is look-ahead — acting on information not yet available at the decision time.

## Context

A backtest is only trustworthy if every order is decided from information that was
actually available when the order would have been placed. Two classic leaks:

1. **Reading the future / forming bar.** Computing a signal at `t` from a bar that
   has not closed (or from `t+1`) inflates results and cannot be traded.
2. **Same-bar fill.** Filling an order at the same bar's close that the signal was
   computed on assumes you could act on a price you only knew at the close — a
   look-ahead.

## Decision

- The signal at bar `t` reads **only closed bars `≤ t`** and maps that history to a
  **target position for bar `t+1`**. Signals (`ma_crossover`, `momentum`, `flat`)
  are pure functions of the closed history.
- Positions are lagged (`signal.shift(1)`) so the signal at `t` earns the
  **`t→t+1`** return, and returns use `pct_change(fill_method=None)` to avoid
  forward-fill leakage.
- Orders fill at the **next bar's open**, never the same bar's close.

## Consequences

- A Hypothesis property test perturbs the forming / future bar and asserts the order
  emitted at `t` is unchanged (`tests/property`).
- This invariant is what the parity oracle (ADR-0002) ultimately verifies
  end-to-end: if the backtest secretly peeked at the future, its equity curve would
  diverge from the next-bar-open paper broker.
