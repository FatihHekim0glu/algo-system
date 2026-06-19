# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
