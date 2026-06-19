"""Validate the OOS metric kernels against independent hand-computed references.

Each metric (annualized OOS net Sharpe, max drawdown, one-way turnover, compounded
net PnL) is checked against a value computed by hand / a textbook closed form on a
small fixed series, plus the boundary contracts (NaN on a flat / single-bar Sharpe,
``0.0`` drawdown on a monotone-up curve, empty / non-finite rejection). These are
the scalars the PURE verdict and the API summary consume, so pinning them to 1e-12
against a reference guards the whole honest-statistics layer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from algosystem._exceptions import ValidationError
from algosystem.evaluation.metrics import (
    StrategyMetrics,
    max_drawdown,
    net_pnl,
    oos_sharpe,
    strategy_metrics,
    turnover,
)

_TOL = 1e-12


@pytest.mark.parity
def test_oos_sharpe_matches_hand_reference() -> None:
    """Annualized Sharpe equals ``mean / std(ddof=1) * sqrt(ppy)`` to 1e-12."""
    r = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1))
    # Non-annualized (ppy=1) closed form.
    assert oos_sharpe(r, periods_per_year=1) == pytest.approx(mean / std, abs=_TOL)
    # Annualized at the daily factor.
    expected_annual = mean / std * math.sqrt(252)
    assert oos_sharpe(r, periods_per_year=252) == pytest.approx(expected_annual, abs=_TOL)


@pytest.mark.parity
def test_oos_sharpe_risk_free_subtracts_from_mean() -> None:
    """A per-bar risk-free rate is subtracted from the mean before scaling."""
    r = np.array([0.01, 0.02, 0.0, 0.03, -0.01])
    rf = 0.002
    std = float(np.std(r, ddof=1))
    expected = (float(np.mean(r)) - rf) / std * math.sqrt(252)
    assert oos_sharpe(r, risk_free=rf, periods_per_year=252) == pytest.approx(expected, abs=_TOL)


@pytest.mark.parity
def test_oos_sharpe_is_nan_on_flat_and_single_bar() -> None:
    """A (numerically) flat or single-observation series has an undefined Sharpe."""
    assert math.isnan(oos_sharpe(np.array([0.01, 0.01, 0.01, 0.01])))
    assert math.isnan(oos_sharpe(np.array([0.05])))
    # A genuine zero series is flat too.
    assert math.isnan(oos_sharpe(np.zeros(10)))


@pytest.mark.parity
def test_max_drawdown_matches_hand_reference() -> None:
    """Max drawdown is the most negative ``W_t / running_peak - 1``."""
    # wealth: 1.10, 0.88, 0.924 -> peak 1.10 -> dd at idx1 = 0.88/1.10 - 1 = -0.20.
    r = np.array([0.10, -0.20, 0.05])
    assert max_drawdown(r) == pytest.approx(-0.20, abs=_TOL)


@pytest.mark.parity
def test_max_drawdown_is_zero_for_monotone_up_curve() -> None:
    """A never-declining equity curve has drawdown exactly ``0.0``."""
    r = np.array([0.01, 0.02, 0.0, 0.03])
    assert max_drawdown(r) == 0.0


@pytest.mark.parity
def test_max_drawdown_two_drawdowns_takes_the_worst() -> None:
    """With two separate troughs the deeper one is returned."""
    # +20% -> -10% (dd1) -> +5% -> -40% (dd2, the worst).
    r = np.array([0.20, -0.10, 0.05, -0.40])
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    expected = float(np.min(wealth / peak - 1.0))
    assert max_drawdown(r) == pytest.approx(expected, abs=_TOL)
    assert max_drawdown(r) < -0.39  # the deep second trough dominates.


@pytest.mark.parity
def test_turnover_matches_hand_reference() -> None:
    """One-way turnover sums absolute position changes, first vs the opening book."""
    pos = np.array([1.0, 1.0, -1.0, 0.0])
    # From flat: |1-0| + |1-1| + |-1-1| + |0-(-1)| = 1 + 0 + 2 + 1 = 4.
    assert turnover(pos) == pytest.approx(4.0, abs=_TOL)
    # From an already-long book: |1-1| + 0 + 2 + 1 = 3.
    assert turnover(pos, initial_position=1.0) == pytest.approx(3.0, abs=_TOL)


@pytest.mark.parity
def test_turnover_flat_book_is_zero() -> None:
    """A constant flat position incurs no turnover."""
    assert turnover(np.zeros(20)) == 0.0


@pytest.mark.parity
def test_net_pnl_matches_compounded_reference() -> None:
    """Net PnL is ``prod(1 + r) - 1``."""
    r = np.array([0.10, -0.10, 0.05])
    expected = 1.10 * 0.90 * 1.05 - 1.0
    assert net_pnl(r) == pytest.approx(expected, abs=_TOL)


@pytest.mark.parity
def test_strategy_metrics_bundle_is_consistent_with_scalars() -> None:
    """The frozen bundle reuses the same scalar kernels (no divergent re-derivation)."""
    r = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    pos = np.array([1.0, 1.0, 0.0, -1.0, -1.0])
    bundle = strategy_metrics(r, pos, periods_per_year=252)
    assert isinstance(bundle, StrategyMetrics)
    assert bundle.oos_sharpe == pytest.approx(oos_sharpe(r, periods_per_year=252), abs=_TOL)
    assert bundle.max_drawdown == pytest.approx(max_drawdown(r), abs=_TOL)
    assert bundle.turnover == pytest.approx(turnover(pos), abs=_TOL)
    assert bundle.net_pnl == pytest.approx(net_pnl(r), abs=_TOL)
    assert bundle.n_bars == 5
    # The JSON view round-trips the same scalars.
    d = bundle.to_dict()
    assert set(d) == {"oos_sharpe", "max_drawdown", "turnover", "net_pnl", "n_bars"}
    assert d["n_bars"] == 5


@pytest.mark.parity
def test_strategy_metrics_rejects_length_mismatch() -> None:
    """The return and position series must index the same scored OOS window."""
    with pytest.raises(ValidationError, match="same length"):
        strategy_metrics(np.array([0.01, 0.02, 0.0]), np.array([1.0, 1.0]))


@pytest.mark.parity
@pytest.mark.parametrize("kernel", [oos_sharpe, max_drawdown, net_pnl, turnover])
def test_metrics_reject_empty_series(kernel: object) -> None:
    """Every scalar metric rejects an empty input through the shared boundary."""
    with pytest.raises(ValidationError, match="non-empty"):
        kernel(np.array([]))  # type: ignore[operator]


@pytest.mark.parity
@pytest.mark.parametrize("kernel", [oos_sharpe, max_drawdown, net_pnl, turnover])
def test_metrics_reject_non_finite_series(kernel: object) -> None:
    """Every scalar metric rejects NaN / inf rather than propagating it."""
    with pytest.raises(ValidationError, match="non-finite"):
        kernel(np.array([0.01, np.nan, 0.02]))  # type: ignore[operator]
    with pytest.raises(ValidationError, match="non-finite"):
        kernel(np.array([0.01, np.inf, 0.02]))  # type: ignore[operator]


@pytest.mark.parity
def test_turnover_rejects_non_finite_initial_position() -> None:
    """The opening book position must itself be finite."""
    with pytest.raises(ValidationError, match="initial_position must be finite"):
        turnover(np.array([1.0, 0.0]), initial_position=float("nan"))


@pytest.mark.parity
def test_oos_sharpe_sign_tracks_mean_return() -> None:
    """A net-positive (net-negative) mean yields a positive (negative) Sharpe."""
    assert oos_sharpe(np.array([0.02, 0.01, 0.03, 0.0, 0.015])) > 0.0
    assert oos_sharpe(np.array([-0.02, -0.01, -0.03, 0.0, -0.015])) < 0.0
