"""Regression: the SANITY check — the FULL pipeline DOES capture a real tradeable edge.

The honest NULL is only meaningful if the machinery CAN detect an edge when one
exists; otherwise the ``system_has_edge=False`` verdict is vacuous. These tests pin
that the SAME leakage-free pipeline the serve path wires together — a strictly-causal
long/short ``ma_crossover`` signal -> the no-lookahead vectorized backtest -> OOS
metrics + Diebold-Mariano vs. buy-and-hold — DOES beat buy-and-hold, DM-significant
net of costs, on a directional regime-trend DGP (:func:`regime_trend_bars`).

A directional regime-trend (alternating persistent up / down trends) is the
tradeable structure a long/short trend-follower SHOULD win on: it flips short
through the down-trends that drag a static buy-and-hold position down — whereas the
pure monotonic :func:`learnable_trend_bars` is one buy-and-hold itself is optimal on
(documented below for contrast). The parity oracle holds throughout, so the edge is
earned on the SAME backtest<->live-consistent curve, not a leaky one.
"""

from __future__ import annotations

import numpy as np
import pytest

from algosystem.backtest.engine import vectorized_backtest
from algosystem.data import compute_returns
from algosystem.data.synthetic import learnable_trend_bars, regime_trend_bars
from algosystem.evaluation.diebold_mariano import diebold_mariano, dm_favours_system
from algosystem.evaluation.metrics import strategy_metrics
from algosystem.execution.parity import PARITY_TOL, check_parity
from algosystem.serve import _align_positions
from algosystem.signals.library import SignalSpec, build_signal

_COST_BPS = 5.0
_SLIPPAGE_BPS = 2.0


def _signal_vs_buyhold(
    close: object, *, fast: int = 10, slow: int = 50
) -> tuple[float, float, float, float]:
    """Score the long/short ma_crossover vs. buy-and-hold (the serve primitives)."""
    ret = np.asarray(
        compute_returns(close).to_numpy(dtype="float64"), dtype="float64"  # type: ignore[arg-type]
    )
    positions = _align_positions(
        build_signal(SignalSpec("ma_crossover", {"fast": fast, "slow": slow}), close), ret.size
    )
    strat = vectorized_backtest(ret, positions, cost_bps=_COST_BPS, slippage_bps=_SLIPPAGE_BPS)
    buyhold = vectorized_backtest(
        ret, np.ones(ret.size, dtype="float64"), cost_bps=_COST_BPS, slippage_bps=_SLIPPAGE_BPS
    )
    strat_m = strategy_metrics(strat.net_returns, strat.positions)
    buyhold_m = strategy_metrics(buyhold.net_returns, buyhold.positions)
    dm_stat, dm_p = diebold_mariano(strat.net_returns, buyhold.net_returns)
    return strat_m.oos_sharpe, buyhold_m.oos_sharpe, dm_stat, dm_p


@pytest.mark.regression
def test_regime_trend_pipeline_beats_buyhold_dm_significant() -> None:
    """On the directional regime-trend the long/short pipeline beats buy-and-hold (DM-significant)."""
    close = regime_trend_bars(n_obs=2000, seed=7).bars["close"]
    strat_sharpe, buyhold_sharpe, dm_stat, dm_p = _signal_vs_buyhold(close)
    # The strategy's OOS net Sharpe clearly beats buy-and-hold...
    assert strat_sharpe > buyhold_sharpe
    assert strat_sharpe > 0.5
    # ...and the Diebold-Mariano test confirms it is SIGNED in the system's favour
    # AND significant net of costs (the machinery detects a real, tradeable edge).
    assert dm_favours_system(dm_stat, dm_p) is True
    assert dm_stat > 0.0
    assert dm_p < 0.05


@pytest.mark.regression
@pytest.mark.parametrize("seed", [1, 7, 42, 123, 2024])
def test_regime_trend_edge_is_robust_across_seeds(seed: int) -> None:
    """The tradeable edge on the regime-trend is robust across several seeds."""
    close = regime_trend_bars(n_obs=2000, seed=seed).bars["close"]
    strat_sharpe, buyhold_sharpe, dm_stat, dm_p = _signal_vs_buyhold(close)
    assert strat_sharpe > buyhold_sharpe
    assert dm_favours_system(dm_stat, dm_p) is True


@pytest.mark.regression
def test_regime_trend_parity_holds_so_the_edge_is_not_leaky() -> None:
    """The SANITY edge is earned on a parity-consistent curve (not a look-ahead one)."""
    close = regime_trend_bars(n_obs=2000, seed=7).bars["close"]
    ret = np.asarray(compute_returns(close).to_numpy(dtype="float64"), dtype="float64")
    positions = _align_positions(
        build_signal(SignalSpec("ma_crossover", {"fast": 10, "slow": 50}), close), ret.size
    )
    report = check_parity(ret, positions, cost_bps=_COST_BPS, slippage_bps=_SLIPPAGE_BPS)
    assert report.passed is True
    assert report.max_abs_diff <= PARITY_TOL


@pytest.mark.regression
def test_monotonic_trend_is_not_beaten_by_longshort_crossover() -> None:
    """For contrast: on a PURE uptrend buy-and-hold is optimal — the crossover cannot win.

    This documents WHY the regime-trend (not the monotonic trend) is the right SANITY
    DGP for the long/short pipeline: on a single persistent uptrend, buy-and-hold is
    always long (the optimal exposure) and a long/short crossover that flips short on
    pullbacks and pays costs can never beat it on Sharpe.
    """
    close = learnable_trend_bars(n_obs=2000, seed=7).bars["close"]
    strat_sharpe, buyhold_sharpe, dm_stat, dm_p = _signal_vs_buyhold(close)
    # Buy-and-hold IS the better strategy on a pure uptrend; the crossover does not
    # beat it DM-significant (so the verdict would correctly be no-edge here too).
    assert buyhold_sharpe > strat_sharpe
    assert dm_favours_system(dm_stat, dm_p) is False
