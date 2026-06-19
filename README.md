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

For the architecture, module map, and the recorded decisions behind each guard, see
[`docs/DESIGN.md`](docs/DESIGN.md) and the ADRs under
[`docs/decisions/`](docs/decisions/).

## Causality & correctness guards

| Guard | What it enforces |
| --- | --- |
| **Causal signal + next-bar fill** | The signal at bar `t` reads ONLY closed bars `≤ t` and applies to the `t→t+1` return; orders fill at the NEXT bar's OPEN, never the same bar's close. Perturbing the forming/future bar does not change the order at `t` (property test). |
| **Bar-finality guard** | A partial / unclosed bar can NEVER trigger an order (tested). |
| **Backtest↔live parity oracle** | The vectorized backtest equity curve equals the paper-broker equity curve to `1e-10` for any signal/param (Hypothesis test). A deliberately-leaky backtester negative control is CAUGHT by the oracle. |
| **Purged walk-forward** | Rolling in-sample/out-of-sample folds with a purge + embargo; costs + slippage applied IDENTICALLY in backtest and paper execution. |
| **Honest multiplicity** | DSR `n_trials` = #signals × #param configs; PBO via CSCV. |

## Validation

These are the committed reference metrics from
[`src/algosystem/artifacts/reference.json`](src/algosystem/artifacts/reference.json),
the deployed-default summary the request path serves verbatim. The configuration is
`ma_crossover` 10/50, `cost_bps=5`, `slippage_bps=2`, `seed=7`, 2000 synthetic bars.
(Regenerate with `python scripts/build_reference.py`.)

| Metric | Value | Reading |
| --- | --- | --- |
| OOS net Sharpe (system) | **−0.7070** | below buy-and-hold; no edge |
| Buy-and-hold Sharpe | −0.0973 | the benchmark |
| Diebold-Mariano p vs. buy-hold | 0.1929 | insignificant (≥ `0.05`) — DM gate fails |
| Deflated Sharpe (DSR) | 0.00323 | a probability; far below the `1 − α = 0.95` gate |
| PBO (CSCV) | 0.8626 | ≥ `0.5` — overfitting gate fails |
| Effective trials (DSR multiplicity) | 7 | #signals × #param configs |
| Max drawdown | −69.1% | — |
| Turnover | 131.0 | — |
| **`backtest_live_parity_max_diff`** | **0.0** | backtest == live to the cent |
| `bar_finality_ok` | True | no order off an unclosed bar |
| **`system_has_edge`** | **False** | the honest NULL verdict |

All three edge gates fail independently (DM insignificant, DSR `0.003 ≤ 0.95`,
PBO `0.86 ≥ 0.5`), so the PURE `system_has_edge` verdict is `False`. The deliverable
is the load-bearing parity (`max-diff = 0.0`), not the strategy.

### Correctness gates

Every gate below is enforced by a test in the partitioned `tests/` suite and is
green in CI. These — not any return number — are the product.

| Gate | Status | Criterion |
| --- | --- | --- |
| Backtest↔live equity **parity oracle** | PASS | `max-diff = 0.0` ≤ `1e-10` for any signal/param (`tests/parity`, Hypothesis property) |
| **Leaky negative control** caught | PASS | the deliberately-leaky backtester FAILS parity (`ParityError`; unit + integration) |
| **DSR** vs. `dsr.py` reference | PASS | agrees to `1e-10` |
| **Diebold-Mariano** correctness | PASS | hand reference + Newey-West HAC long-run variance |
| **PBO / CSCV** vs. reference | PASS | CSCV combinatorial reference |
| **Causal signal + next-bar fill** | PASS | perturbing the forming/future bar leaves the order at `t` unchanged (Hypothesis) |
| **Bar-finality guard** | PASS | a partial / unclosed bar emits no order |
| Sharpe / drawdown / turnover | PASS | hand references |
| `learnable_trend` / `regime_trend` SANITY | PASS | a tradeable trend IS captured (`regime_trend`: `dm_pvalue ≈ 1e-5`, beats buy-hold) |
| **Honest-null** | PASS | `system_has_edge = False` after costs + DSR + PBO; deterministic across `PYTHONHASHSEED` |
| Coverage | PASS | `86.31%` ≥ `85%`; ruff + strict-mypy clean |

The SANITY check uses a **directional regime-trend** DGP (`regime_trend_bars`:
alternating persistent up / down trends): the long/short MA-crossover flips short
through the down-trends that drag a static buy-and-hold down, so it beats
buy-and-hold DM-significant (`dm_pvalue ≈ 1e-5`) net of costs — proving the
machinery detects a real, tradeable edge, so the honest null is honest, not vacuous.
(A pure monotonic uptrend, `learnable_trend_bars`, is one buy-and-hold itself is
optimal on, so a long/short trend-follower cannot beat it there — documented in the
sanity suite for contrast.)

## Reproduce

Lean install (no torch / onnx / onnxruntime / sklearn — only `[data,viz,dev]`):

```bash
uv venv
uv pip install -e '.[data,viz,dev]'
```

Run the pipeline three ways via the Typer CLI — all on the synthetic default,
offline, never training:

```bash
# Vectorized purged walk-forward backtest: OOS metrics + the verdict
algo-system backtest --signal ma_crossover --fast 10 --slow 50 --cost-bps 5 --slippage-bps 2 --seed 7

# Simulated bar-by-bar paper-broker replay (next-bar-open fills): the "live" path
algo-system paper --signal ma_crossover --fast 10 --slow 50

# Run BOTH and print the parity max-diff + DM/DSR/PBO + the PURE verdict
algo-system compare --signal ma_crossover --fast 10 --slow 50
```

Regenerate the committed reference artifact:

```bash
python scripts/build_reference.py
```

Run the gates exactly as CI does:

```bash
ruff check src tests          # lint
mypy src                      # strict type-check (clean)
pytest -q -m "not slow" --cov=algosystem --cov-report=term --cov-fail-under=85
```

## Limitations

- **SIMULATED execution, NOT a live broker.** The paper broker is an internal
  cost+slippage simulator that replays bars; there is **no** live Alpaca / broker
  connection and **no broker key**. Nothing here places a real order.
- **Synthetic default.** The deployed request path runs on a synthetic OHLC bar
  process constructed so the simple signals have no exploitable edge net of costs
  (the honest null holds); it is not historical market data.
- **Single-asset.** One instrument at a time; no cross-sectional, portfolio, or
  hedging effects.
- **Idealized fills.** Next-bar-open fills with fixed per-side basis-point costs and
  slippage; no partial fills, no market impact, no queue position, no latency.
- **PIT / survivorship.** The Polygon-PIT offline path is point-in-time but does not
  model delisting / survivorship beyond the provider's coverage.

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
