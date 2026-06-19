# Design

`algo-system` is a torch-free, leakage-free, single-asset algo trading **pipeline**
whose deliverable is rigor, not alpha: a strictly-causal signal feeds a purged
walk-forward backtest **and** a simulated bar-by-bar paper-broker, and the two equity
curves are proven equal to `1e-10` by a parity oracle. The strategy itself shows no
robust out-of-sample edge after costs, a Deflated-Sharpe correction, and a PBO
overfitting check — and the verdict that says so is a pure function.

## Goals

- **Correctness over returns.** The load-bearing artifact is the backtest↔live
  parity oracle, not a Sharpe number. Every claim is gated by a test (see the
  [Validation table](../README.md#validation)).
- **No leakage.** The signal at bar `t` reads only closed bars `≤ t` and is applied
  to the `t→t+1` return; orders fill at the **next** bar's open, never the same
  bar's close.
- **Honest null, honestly judged.** `system_has_edge` is a pure function that is
  `False` unless three independent gates all pass (DM-significance, DSR confidence,
  PBO). The synthetic default is constructed so the null holds; a tradeable-trend
  sanity fixture proves the machinery can detect a real edge, so the null is not
  vacuous.
- **Import purity.** `import algosystem` has zero side effects: no network, no
  broker, no Polygon at import time. Clients are lazy; demos live behind
  `__main__`.

## Non-goals

- Not a live trading system. Execution is **simulated** — there is no broker
  connection and no broker key.
- Not an alpha claim. No profit is asserted or implied.
- Not a torch / ML serving stack. Pure numpy / scipy / statsmodels throughout; no
  torch / onnx / onnxruntime / sklearn anywhere.
- Not multi-asset. One instrument at a time.

## Pipeline

```
synthetic OHLC bars  (data/synthetic.py, seeded _rng)
        │
        ▼
causal signal        (signals/library.py: ma_crossover | momentum | flat)
   position target for t+1 from closed bars ≤ t
        │
        ├─────────────────────────────┐
        ▼                             ▼
vectorized backtest          simulated paper broker
(backtest/engine.py:         (execution/paper_broker.py:
 purge/embargo, positions     next-bar-open fills,
 shifted t→t+1, costs)        costs + slippage, equity)
        │                             │
        └───────────► parity ◄────────┘
            (execution/parity.py: assert equity curves
             equal to 1e-10 — the LOAD-BEARING oracle)
        │
        ▼
evaluation  (metrics.py OOS Sharpe/drawdown/turnover; diebold_mariano.py vs buy-hold
             with Newey-West HAC; dsr.py Deflated Sharpe; pbo.py CSCV)
        │
        ▼
verdict     (verdict.py: PURE system_has_edge — False unless DM-significant
             AND DSR > 1−α AND PBO < 0.5, net of costs)
        │
        ▼
serve.run_system  →  JSON-safe summary + backtest-vs-live equity overlay + drawdown
```

## Module map

| Module | Responsibility |
| --- | --- |
| `data/synthetic.py` | Synthetic OHLC bars (honest-null default), `learnable_trend` / `regime_trend` sanity fixtures, `pure_noise`. |
| `data/loaders.py` | Synthetic default; lazy Polygon PIT bars. |
| `signals/library.py` | Pure, strictly-causal signals: `ma_crossover`, `momentum`, `flat`. |
| `backtest/engine.py` | Vectorized no-lookahead purged walk-forward; positions shifted so the signal at `t` earns the `t→t+1` return. |
| `backtest/bar_finality.py` | The only-act-on-closed-bars guard. |
| `execution/paper_broker.py` | Simulated bar-by-bar execution: next-bar-open fills + costs + slippage; tracks position, cash, equity (the "live" path). |
| `execution/parity.py` | The parity oracle: backtest equity == paper-broker equity to `1e-10`. |
| `evaluation/metrics.py` | OOS net Sharpe, max drawdown, turnover, net PnL. |
| `evaluation/diebold_mariano.py` + `hac.py` | DM of the system-vs-buy-hold per-bar net-return differential (Newey-West HAC denominator). |
| `evaluation/dsr.py` | Deflated Sharpe; `n_trials` = #signals × #param configs. |
| `evaluation/pbo.py` | Probability of Backtest Overfitting via CSCV. |
| `evaluation/verdict.py` | The PURE `system_has_edge` verdict. |
| `serve.py` | `run_system` entrypoint: signal → backtest + paper replay → parity → metrics/DSR/PBO → verdict + figures. |
| `plots.py` | Lazy plotly: backtest-vs-live equity overlay + buy-hold + drawdown. |
| `cli.py` | Typer `backtest` / `paper` / `compare`. |

## Key invariants

1. **Causal signal + next-bar fill.** Perturbing the forming / future bar never
   changes the order emitted at `t`.
2. **Bar-finality.** A partial / unclosed bar can never trigger an order.
3. **Parity.** Vectorized backtest equity == paper-broker equity to `1e-10` for any
   signal/param sequence; a deliberately-leaky backtester negative control is caught.
4. **Cost symmetry.** Costs + slippage are applied identically in the backtest and
   the paper broker.
5. **Pure verdict.** `system_has_edge` is a deterministic function of the inference
   outputs; it cannot be narrated to `True`.

## Decisions

Architecture decisions are recorded as ADRs under
[`docs/decisions/`](decisions/):

- [ADR-0001 — Causal signal & next-bar fill](decisions/0001-causal-signal-next-bar-fill.md)
- [ADR-0002 — Backtest↔live parity oracle](decisions/0002-backtest-live-parity-oracle.md)
- [ADR-0003 — Bar-finality guard](decisions/0003-bar-finality-guard.md)
- [ADR-0004 — DSR confidence gate](decisions/0004-dsr-confidence-gate.md)
- [ADR-0005 — Simulated execution](decisions/0005-simulated-execution.md)

## References

See the [References section of the README](../README.md#references) — Bailey &
López de Prado (2014, DSR); Bailey et al. (2017, PBO/CSCV); López de Prado (2018,
purged CV); Diebold & Mariano (1995); Newey & West (1987).
