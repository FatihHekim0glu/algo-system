# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- The CLI `run_pipeline` computed the headline OOS Sharpe + Diebold-Mariano (and the
  buy-hold baseline, turnover, drawdown, and DSR observed Sharpe) on FULL-SAMPLE net
  returns, while `serve.run_system` correctly used the purged walk-forward OOS folds
  — so the two entry points could report different "honest" numbers for the same
  config (leaked-vs-honest divergence). The CLI now computes the headline metrics +
  DM + DSR observed Sharpe on the purged walk-forward OUT-OF-SAMPLE folds, exactly
  like serve; the backtest↔live parity oracle stays on the full sample (a
  fill-accounting check) and the PBO/CSCV stays on the full-sample grid (CSCV does
  its own in-sample/out-of-sample splitting). Added
  `tests/regression/test_cli_serve_parity.py` asserting the CLI and serve produce the
  IDENTICAL verdict + headline metrics for the same config. Corrected the README
  **Validation** table to the committed purged-OOS reference numbers (OOS Sharpe
  −0.7040, buy-hold −0.0467, DM p 0.1798, deflated Sharpe 0.00561, max drawdown
  −64.1%, turnover 123.0).

### Added
- Documentation pass: filled the README **Validation** section with the actual
  committed metrics from `src/algosystem/artifacts/reference.json` (OOS Sharpe
  −0.7040 vs. buy-hold −0.0467, DM p 0.1798, deflated Sharpe 0.00561, PBO 0.8626,
  `n_effective_trials` 7, `backtest_live_parity_max_diff` 0.0, `system_has_edge`
  False) and a correctness-gates table (parity oracle `1e-10`, leaky negative
  control caught, DSR `1e-10`, DM, PBO/CSCV, causal-signal/next-bar-fill,
  bar-finality, the `learnable_trend` / `regime_trend` sanity, the honest-null).
  Added a **Reproduce** block (lean install + `backtest`/`paper`/`compare` CLI +
  the `ruff`/`mypy`/`pytest` gates), tightened **Limitations** (SIMULATED execution
  / no broker key, synthetic default, single-asset, idealized fills, PIT
  survivorship).
- Added `docs/DESIGN.md` (goals / non-goals, the pipeline diagram, the module map,
  the key invariants) and `docs/decisions/` ADRs: causal-signal-next-bar-fill,
  backtest-live-parity-oracle, bar-finality-guard, dsr-confidence-gate, and
  simulated-execution.
- Wired the full serve-time pipeline in `serve.run_system`: synthetic bars -> causal
  signal -> vectorized backtest + simulated paper-broker replay -> the backtest<->live
  PARITY ORACLE (asserted to `1e-10`) -> OOS metrics + Diebold-Mariano + Deflated
  Sharpe + PBO/CSCV -> the PURE `system_has_edge` verdict -> the JSON-safe summary +
  the backtest-vs-live equity overlay + drawdown Plotly figures. The request path is
  offline-safe and never trains.
- Added a directional regime-trend synthetic DGP (`data.synthetic.regime_trend_bars`)
  — the tradeable SANITY fixture the FULL long/short pipeline beats buy-and-hold on,
  DM-significant net of costs (proving the machinery detects a real edge so the honest
  null is not vacuous); routed through `synthetic_default_bars` and exported.
- Precomputed + committed the synthetic reference summary
  (`src/algosystem/artifacts/reference.json`) holding the deployed-default honest-NULL
  summary plus the learnable_trend / regime_trend sanity numbers and the pure_noise
  honest-null numbers, regenerable via `scripts/build_reference.py`.
- Integration + regression suites: the end-to-end pipeline (no network), the
  honest-null regression (`system_has_edge = False`, deterministic across
  `PYTHONHASHSEED`), the regime-trend SANITY (the system DOES beat buy-hold), the
  leaky-backtester integration negative control (the parity oracle CATCHES it), and
  the reference-artifact lock test.
- Scaffolded the `algosystem` src-layout package (import-pure, typed, `py.typed`).
- Reused infrastructure verbatim from the HRP repo (renamed `hrp` -> `algosystem`):
  `_constants`, `_rng`, `_validation`, `py.typed`, `evaluation/dsr.py`,
  `backtest/costs.py`, and `data_providers/polygon.py`; reframed `_typing`,
  `_exceptions` (`AlgoSystemError` base), and `_manifest` for the
  signal -> backtest -> simulated-execution domain.
- Vendored the Newey-West HAC standard error from `pairs_trading` into
  `evaluation/hac.py` (the Diebold-Mariano denominator).
- Fully implemented the PURE honesty kernels so they cannot regress: the
  Probabilistic / Deflated Sharpe ratios (reused), the Diebold-Mariano test of the
  system-vs-buy-hold per-bar net-return differential, and the PURE
  `system_has_edge` verdict (gated at DM-significance AND DSR > 1 - alpha AND
  PBO < 0.5, net of costs).
- Typed stubs (signatures + docstrings + `NotImplementedError`; frozen `slots`
  dataclasses + `to_dict`) for every module-map module: `data/{synthetic,loaders}`,
  `signals/library`, `backtest/{engine,bar_finality}`,
  `execution/{paper_broker,parity}`, `evaluation/{metrics,pbo}`, `serve`, `plots`,
  `cli`, plus a deliberately-leaky negative-control backtester the parity oracle
  must catch.
- Partitioned `tests/` (unit / parity / property / regression / integration) with
  seeded conftest fixtures (`synthetic_bars`, `learnable_trend`, `pure_noise`), an
  import-purity smoke test, and the honesty-kernel lock tests.
- `pyproject.toml` with the lean extras (`[data]`, `[viz]`, `[dev]`; NO `[all]`; NO
  torch / onnx / onnxruntime / sklearn), CI (py3.11-3.13, lean extras, mypy
  continue-on-error, coverage gate >= 85), and the `no-ai-attribution` CI guard.

[Unreleased]: https://github.com/FatihHekim0glu/algo-system/commits/main
