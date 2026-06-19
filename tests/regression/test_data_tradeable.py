"""Regression: the synthetic DGPs have the INTENDED tradeability structure.

These tests pin the load-bearing property of the data layer that makes the honest
NULL honest rather than vacuous:

- on :func:`learnable_trend_bars` a long/flat MA-crossover earns a CLEARLY positive
  net-of-cost Sharpe and is meaningfully long — the machinery CAN detect a real
  edge (the SANITY fixture);
- on :func:`pure_noise_bars` and :func:`gbm_regime_bars` (the deployed honest-null
  default) the SAME crossover earns a NON-positive net-of-cost Sharpe — no
  exploitable edge net of costs.

The MA-crossover is reimplemented here as an INDEPENDENT reference (the production
``algosystem.signals`` library is authored by another group), so this test pins the
data structure, not a particular signal implementation. Costs/slippage are applied
to the lagged position so the comparison is honest and no-lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algosystem.data.synthetic import (
    gbm_regime_bars,
    learnable_trend_bars,
    pure_noise_bars,
)

_N_OBS = 2000
_SEED = 7
_COST_BPS = 5.0
_WARMUP = 60  # drop the slow-MA warm-up before scoring (no-edge during warmup).
_ANNUALIZATION = float(np.sqrt(252.0))


def _ma_crossover_positions(close: pd.Series, *, fast: int = 10, slow: int = 50) -> pd.Series:
    """Independent reference: long when fast SMA > slow SMA, else flat (causal)."""
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()
    pos = pd.Series(0.0, index=close.index)
    pos[sma_fast > sma_slow] = 1.0
    pos[sma_slow.isna()] = 0.0  # no position before the slow window is full.
    return pos


def _net_return_sharpe(close: pd.Series, pos: pd.Series, *, cost_bps: float = _COST_BPS) -> float:
    """Net-of-cost annualized Sharpe of the lagged (no-lookahead) MA-crossover."""
    ret = close.pct_change(fill_method=None).fillna(0.0)
    pos_lag = pos.shift(1).fillna(0.0)  # signal at t applies to the t -> t+1 return.
    gross = pos_lag * ret
    turnover = pos_lag.diff().abs().fillna(0.0)
    net = gross - turnover * (cost_bps / 10_000.0)
    scored = net.iloc[_WARMUP:]
    sd = float(scored.std())
    if sd == 0.0:
        return 0.0
    return _ANNUALIZATION * float(scored.mean()) / sd


@pytest.mark.regression
def test_learnable_trend_is_tradeable() -> None:
    """The MA-crossover earns a clearly positive net Sharpe on the learnable trend."""
    close = learnable_trend_bars(n_obs=_N_OBS, seed=_SEED).bars["close"]
    pos = _ma_crossover_positions(close)
    sharpe = _net_return_sharpe(close, pos)
    # Clearly positive net-of-cost Sharpe: the trend IS captured (machinery works).
    assert sharpe > 0.5, f"learnable_trend net Sharpe should be clearly positive, got {sharpe:.3f}"
    # The crossover spends most of its scored life LONG (it is following the trend).
    avg_position = float(pos.iloc[_WARMUP:].mean())
    assert avg_position > 0.45, f"crossover should be meaningfully long, got {avg_position:.3f}"


@pytest.mark.regression
def test_pure_noise_is_not_tradeable() -> None:
    """The MA-crossover earns a NON-positive net Sharpe on driftless noise."""
    close = pure_noise_bars(n_obs=_N_OBS, seed=_SEED).bars["close"]
    pos = _ma_crossover_positions(close)
    sharpe = _net_return_sharpe(close, pos)
    assert sharpe <= 0.0, f"pure_noise net Sharpe should be non-positive, got {sharpe:.3f}"


@pytest.mark.regression
def test_gbm_regime_null_is_not_tradeable() -> None:
    """The deployed honest-null default yields a NON-positive net-of-cost Sharpe."""
    close = gbm_regime_bars(n_obs=_N_OBS, seed=_SEED).bars["close"]
    pos = _ma_crossover_positions(close)
    sharpe = _net_return_sharpe(close, pos)
    assert sharpe <= 0.0, f"gbm_regime net Sharpe should be non-positive, got {sharpe:.3f}"


@pytest.mark.regression
def test_trend_dominates_noise_by_a_clear_margin() -> None:
    """At the same seed, the trend's net Sharpe clearly exceeds the noise's."""
    trend_close = learnable_trend_bars(n_obs=_N_OBS, seed=_SEED).bars["close"]
    noise_close = pure_noise_bars(n_obs=_N_OBS, seed=_SEED).bars["close"]
    trend_sharpe = _net_return_sharpe(trend_close, _ma_crossover_positions(trend_close))
    noise_sharpe = _net_return_sharpe(noise_close, _ma_crossover_positions(noise_close))
    assert trend_sharpe - noise_sharpe > 0.5, (
        f"trend should dominate noise: trend={trend_sharpe:.3f} noise={noise_sharpe:.3f}"
    )


@pytest.mark.regression
def test_cross_seed_separation_is_robust() -> None:
    """Across several seeds the trend's median net Sharpe stays clearly positive."""
    seeds = (1, 7, 42, 123, 2024)
    # A modestly stronger explicit drift gives a robust, seed-independent edge —
    # exactly the regime the SANITY fixture is meant to certify.
    trend_sharpes = [
        _net_return_sharpe(
            learnable_trend_bars(n_obs=_N_OBS, seed=s, drift=0.0015).bars["close"],
            _ma_crossover_positions(
                learnable_trend_bars(n_obs=_N_OBS, seed=s, drift=0.0015).bars["close"]
            ),
        )
        for s in seeds
    ]
    assert all(x > 0.5 for x in trend_sharpes), f"trend Sharpe positive every seed: {trend_sharpes}"
