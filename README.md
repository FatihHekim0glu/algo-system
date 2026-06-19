# algo-system

**A complete, leakage-free single-asset algo trading system — and the backtest you
can trust because it matches the live execution to the cent.**

> **Honest headline.** A causal signal (MA-crossover / momentum) drives a purged
> walk-forward, no-lookahead vectorized backtest **and** a SIMULATED bar-by-bar
> paper-broker execution engine (next-bar-open fills + costs + slippage). The
> backtest and the live-simulated equity curves **agree to within costs** — the
> backtest↔live **parity oracle** passes to `1e-10` — and a **bar-finality guard**
> prevents acting on an unclosed bar. Judged honestly out-of-sample net of costs
> with Diebold-Mariano vs. buy-and-hold, a **Deflated-Sharpe** correction, and a
> **Probability-of-Backtest-Overfitting (PBO/CSCV)** check, the strategy itself
> shows **NO robust out-of-sample edge after costs** (`system_has_edge = False`).
> The deliverable is the rigorous, leakage-free **integration and the backtest↔live
> parity**, not a profit claim.

The deployed default runs the full pipeline on a synthetic OHLC bar process where,
by construction, the simple signals have no exploitable edge net of costs (so the
honest null holds); real data is available via the Polygon-PIT CLI path.
**Execution is SIMULATED** — an internal cost+slippage paper broker, **not a live
broker** (there is no Alpaca / broker connection and no broker key).

## What this is (and is not)

- **Is:** a torch-free, leakage-free **signal → backtest → simulated-execution**
  pipeline whose load-bearing artifact is the **backtest↔live parity oracle** — the
  vectorized backtest equity curve must equal the paper-broker replay equity curve
  to `1e-10`, which catches any look-ahead or fill-timing bug.
- **Is not:** an alpha claim, a live trading system, or a torch/ML serving stack.
  Pure numpy / scipy / statsmodels throughout.

## Causality & correctness guards

| Guard | What it enforces |
| --- | --- |
| **Causal signal + next-bar fill** | The signal at bar `t` reads ONLY closed bars `≤ t` and applies to the `t→t+1` return; orders fill at the NEXT bar's OPEN, never the same bar's close. Perturbing the forming/future bar does not change the order at `t` (property test). |
| **Bar-finality guard** | A partial / unclosed bar can NEVER trigger an order (tested). |
| **Backtest↔live parity oracle** | The vectorized backtest equity curve equals the paper-broker equity curve to `1e-10` for any signal/param (Hypothesis test). A deliberately-leaky backtester negative control is CAUGHT by the oracle. |
| **Purged walk-forward** | Rolling in-sample/out-of-sample folds with a purge + embargo; costs + slippage applied IDENTICALLY in backtest and paper execution. |
| **Honest multiplicity** | DSR `n_trials` = #signals × #param configs; PBO via CSCV. |

## Validation

| Check | Status | Tolerance / criterion |
| --- | --- | --- |
| Backtest↔live equity parity (the oracle) | _to be filled_ | `1e-10` |
| Leaky-backtester negative control caught | _to be filled_ | parity FAILS |
| DSR vs. `dsr.py` reference | _to be filled_ | `1e-10` |
| Diebold-Mariano correctness | _to be filled_ | hand reference |
| PBO / CSCV vs. reference | _to be filled_ | reference |
| Sharpe / drawdown / turnover | _to be filled_ | hand references |
| `learnable_trend` SANITY (MA beats buy-hold) | _to be filled_ | the machinery works |
| Honest-null (`system_has_edge = False` after costs + DSR + PBO) | _to be filled_ | deterministic across `PYTHONHASHSEED` |
| Coverage | _to be filled_ | `≥ 85%` |

_(This is a scaffold; the surrounding modules are typed stubs. The PURE honesty
kernels — DSR, Diebold-Mariano, and the `system_has_edge` verdict — are fully
implemented and locked by tests.)_

## Quickstart

```bash
uv venv
uv pip install -e '.[data,viz,dev]'
uv run pytest -q -m "not slow"
```

## Limitations

- **SIMULATED execution, NOT a live broker.** The paper broker is an internal
  cost+slippage simulator; there is no live Alpaca / broker connection and no
  broker key.
- **Synthetic default.** The deployed default runs on a synthetic OHLC bar process
  designed so the honest null holds; it is not historical market data.
- **Single-asset.** One instrument at a time; no cross-sectional or portfolio
  effects.
- **Idealized fills.** Next-bar-open fills with fixed per-side basis-point costs and
  slippage; no partial fills, no market impact, no queue position.
- **PIT / survivorship.** The Polygon-PIT path is point-in-time but does not model
  delisting / survivorship beyond the provider's coverage.

## References

- Bailey, D. H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting, and Non-Normality.* Journal of Portfolio
  Management.
- Bailey, D. H., Borwein, J. M., López de Prado, M. & Zhu, Q. J. (2017). *The
  Probability of Backtest Overfitting* (PBO via CSCV). Journal of Computational
  Finance.
- López de Prado, M. (2018). *Advances in Financial Machine Learning* (purged /
  embargoed cross-validation). Wiley.
- Diebold, F. X. & Mariano, R. S. (1995). *Comparing Predictive Accuracy.* Journal
  of Business & Economic Statistics.
- Newey, W. K. & West, K. D. (1987). *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.*
  Econometrica.

## License

MIT — see [LICENSE](LICENSE).
