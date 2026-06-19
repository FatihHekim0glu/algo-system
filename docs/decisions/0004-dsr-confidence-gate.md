# ADR-0004 — DSR confidence gate

- **Status:** Accepted
- **Context:** the headline `system_has_edge` verdict must not be foolable by a
  positive point-estimate Sharpe found among many trials.

## Context

The Deflated Sharpe Ratio (Bailey & López de Prado, 2014) is a **probability** in
`[0, 1]` — the probability that the true Sharpe exceeds a multiplicity-adjusted
benchmark, given how many configurations were tried. A naïve gate of
`deflated_sharpe > 0` is trivially satisfied by any positive Sharpe and so never
binds; the multiplicity deflation would have no teeth.

Because the DSR is a probability, the correct significance call is a **confidence
level**, not a sign test.

## Decision

- The verdict gates the DSR at the `1 − α` **confidence** level:
  `deflated_sharpe > 1 − α` (with the default `α = 0.05`, i.e. `> 0.95`) — the
  standard Bailey–López de Prado significance call.
- The DSR uses the **honest** multiplicity: `n_trials` = #signals × #param configs
  (`n_effective_trials`), not 1.
- `system_has_edge` is a **pure function** (`evaluation/verdict.py`) that is `True`
  only if **all three** gates pass: the Diebold–Mariano test vs. buy-and-hold is
  significant and signed in the system's favour (`dm_pvalue < α` AND
  `dm_statistic > 0`), **AND** `deflated_sharpe > 1 − α`, **AND** `pbo < 0.5`. Any
  failure ⇒ `NO_ROBUST_EDGE`.

## Consequences

- The truth table is unit-tested; the verdict cannot be narrated to `True`.
- On the deployed default, `deflated_sharpe ≈ 0.003 ≤ 0.95`, so the DSR gate fails
  (as do the DM and PBO gates) and `system_has_edge = False` — the honest null.
