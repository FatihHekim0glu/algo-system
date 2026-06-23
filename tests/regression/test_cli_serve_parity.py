"""Regression: the CLI and ``serve.run_system`` agree (no leaked-vs-honest divergence).

The OFFLINE Typer CLI (:func:`algosystem.cli.run_pipeline`) and the DEPLOYED request
path (:func:`algosystem.serve.run_system`) assemble the SAME leakage-free pipeline
from the SAME primitives. The headline summary numbers — the OOS net Sharpe, the
buy-and-hold Sharpe, the Diebold-Mariano p-value, the Deflated Sharpe, the PBO, the
turnover, the max drawdown, the effective-trial count, the backtest<->live parity
max-diff — and the PURE ``system_has_edge`` verdict MUST be IDENTICAL for the same
config.

This pins the fix for the leaked-full-sample-metrics defect: previously the CLI
computed the headline Sharpe + DM on FULL-SAMPLE net returns while serve correctly
used the purged walk-forward OOS folds, so the two paths could report DIFFERENT
honest numbers for the same config. Both now compute the headline metrics + DM + DSR
observed-Sharpe on the purged walk-forward OUT-OF-SAMPLE folds (the parity oracle and
the PBO/CSCV matrix stay on the full sample — a fill-accounting check and a self-
splitting overfit estimate respectively). Any reappearance of the divergence — e.g.
one path silently reverting to leaked full-sample metrics — fails these assertions.
"""

from __future__ import annotations

import pytest

from algosystem.cli import run_pipeline
from algosystem.execution.parity import PARITY_TOL
from algosystem.serve import run_system

#: Configs covering both signals, an in-grid and a grid-extending selection.
_CONFIGS: tuple[dict[str, object], ...] = (
    {"signal": "ma_crossover", "fast": 10, "slow": 50},  # the deployed default.
    {"signal": "momentum", "lookback": 20},  # the momentum path.
    {"signal": "ma_crossover", "fast": 7, "slow": 33},  # a grid-EXTENDING config.
)


@pytest.mark.regression
@pytest.mark.parametrize("config", _CONFIGS)
def test_cli_and_serve_agree_on_the_verdict(config: dict[str, object]) -> None:
    """The CLI and serve derive the IDENTICAL ``system_has_edge`` verdict + verdict enum."""
    cli = run_pipeline(seed=7, **config)  # type: ignore[arg-type]
    srv = run_system(seed=7, **config)  # type: ignore[arg-type]
    assert cli.verdict.system_has_edge == srv.summary.system_has_edge
    # On the deployed honest-null DGP both must read False (no leaked edge sneaks in).
    assert cli.verdict.system_has_edge is False
    assert srv.summary.system_has_edge is False


@pytest.mark.regression
@pytest.mark.parametrize("config", _CONFIGS)
def test_cli_and_serve_agree_on_the_headline_metrics(config: dict[str, object]) -> None:
    """Every headline summary number matches EXACTLY (the no-divergence contract).

    All headline metrics are computed on the SAME purged walk-forward OOS folds with
    the SAME geometry on both paths, so they must agree bit-for-bit (no tolerance).
    """
    cli = run_pipeline(seed=7, **config)  # type: ignore[arg-type]
    srv = run_system(seed=7, **config).summary  # type: ignore[arg-type]
    # PURGED-OOS headline metrics (the previously-leaked-on-the-CLI quantities).
    assert cli.oos_sharpe == srv.oos_sharpe
    assert cli.buyhold_sharpe == srv.buyhold_sharpe
    assert cli.dm_pvalue == srv.dm_pvalue_vs_buyhold
    assert cli.deflated_sharpe == srv.deflated_sharpe
    assert cli.turnover == srv.turnover
    assert cli.max_drawdown == srv.max_drawdown
    # Full-sample, self-splitting / fill-accounting quantities (already aligned).
    assert cli.pbo == srv.pbo
    assert cli.n_trials == srv.n_effective_trials


@pytest.mark.regression
@pytest.mark.parametrize("config", _CONFIGS)
def test_cli_and_serve_agree_on_the_parity_oracle(config: dict[str, object]) -> None:
    """Both paths run the FULL-SAMPLE backtest<->live parity oracle and it passes.

    Parity is a fill-accounting property of the full path, independent of the
    train/test folding, so both report the SAME (``~0``) max-diff.
    """
    cli = run_pipeline(seed=7, **config)  # type: ignore[arg-type]
    srv = run_system(seed=7, **config).summary  # type: ignore[arg-type]
    assert cli.parity_max_diff == srv.backtest_live_parity_max_diff
    assert cli.parity_ok is True
    assert cli.parity_max_diff <= PARITY_TOL


@pytest.mark.regression
def test_cli_headline_sharpe_is_purged_oos_not_full_sample() -> None:
    """The CLI headline OOS Sharpe is the PURGED-OOS value, not the leaked full-sample one.

    Guards directly against a regression to leaked full-sample metrics: the OOS
    folds drop the train + purge/embargo span, so the purged-OOS Sharpe differs from
    the full-sample vectorized Sharpe on the deployed default. We assert the CLI
    reports the purged-OOS value (matching serve) and that it is genuinely distinct
    from the full-sample number (so a silent revert to full-sample would be caught).
    """
    import numpy as np

    from algosystem.backtest.engine import vectorized_backtest, walk_forward_signal_backtest
    from algosystem.data.loaders import synthetic_default_bars
    from algosystem.evaluation.metrics import strategy_metrics
    from algosystem.serve import _align_positions
    from algosystem.signals.library import SignalSpec, build_signal

    bars, returns, _ = synthetic_default_bars(n_obs=2000, seed=7, kind="gbm_regime")
    ret = np.asarray(returns.to_numpy(dtype="float64"), dtype="float64")
    pos = _align_positions(
        build_signal(SignalSpec("ma_crossover", {"fast": 10, "slow": 50}), bars["close"]),
        ret.size,
    )
    full = vectorized_backtest(ret, pos, cost_bps=5.0, slippage_bps=2.0)
    full_sharpe = strategy_metrics(full.net_returns, full.positions).oos_sharpe
    wf = walk_forward_signal_backtest(ret, pos, cost_bps=5.0, slippage_bps=2.0)
    oos_sharpe = strategy_metrics(wf.net_returns, wf.positions).oos_sharpe

    cli = run_pipeline(signal="ma_crossover", fast=10, slow=50, seed=7)
    # The CLI must report the PURGED-OOS Sharpe (the honest number) ...
    assert cli.oos_sharpe == oos_sharpe
    # ... and that must be DISTINCT from the leaked full-sample Sharpe (regression guard).
    assert cli.oos_sharpe != full_sharpe
