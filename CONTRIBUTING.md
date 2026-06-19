# Contributing

Thanks for your interest in `algo-system`. This is a research-grade, honest-NULL
benchmark; contributions that strengthen the leakage guards, the backtest<->live
parity oracle, or the honest statistics are especially welcome.

## Development setup

```bash
uv venv
uv pip install -e '.[data,viz,dev]'
```

## Quality bar (all must pass)

```bash
uv run ruff check src tests        # lint (zero findings)
uv run mypy src                    # strict mypy (zero errors)
uv run pytest -q -m "not slow" \
  --cov=algosystem --cov-report=term --cov-fail-under=85
```

- **Import purity.** `src/algosystem/` has ZERO import-time side effects. No
  network / broker / Polygon access at import; heavy dependencies (httpx,
  statsmodels, plotly, typer) are imported LAZILY inside the functions that need
  them. The import-purity smoke test enforces this in a fresh interpreter.
- **Torch-free.** This is a TORCH-FREE integration capstone. Do NOT add
  `torch` / `onnx` / `onnxruntime` / `sklearn` / `stable-baselines3` / `gymnasium`
  anywhere. Signals, the backtester, the paper broker, and the DSR / PBO / DM / HAC
  statistics are pure numpy / scipy / statsmodels.
- **No look-ahead.** The signal at bar `t` uses ONLY closed bars `<= t` and applies
  to the `t -> t+1` return; orders fill at the NEXT bar's OPEN (never the same
  bar's close). The bar-finality guard forbids acting on an unclosed bar.
- **The parity oracle is load-bearing.** The vectorized backtest equity curve MUST
  equal the simulated paper-broker equity curve to `1e-10` for any signal/param.
  Any change that breaks parity is a look-ahead or fill-timing bug — fix the bug,
  do not relax the tolerance.
- **The verdict is PURE.** `system_has_edge` is a deterministic function of the
  evidence (DM-vs-buy-hold AND DSR > 1 - alpha AND PBO < 0.5, net of costs). Do not
  narrate an edge the statistics do not support.

## Commit hygiene

Write clear, conventional commit messages. **Do NOT add AI-attribution trailers**
(no `Co-Authored-By: Claude`, no "Generated with Claude", no robot emoji
attribution). The `no-ai-attribution` CI guard rejects PRs that carry them.
