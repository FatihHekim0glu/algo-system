"""Integration negative control: a leaky backtester wired into the pipeline is CAUGHT.

The serve path asserts backtest<->live parity with :func:`assert_parity`, which
RAISES :class:`algosystem._exceptions.ParityError` on any divergence beyond
``1e-10``. To prove that guard has teeth at the INTEGRATION level (not just the unit
level), this test substitutes the honest vectorized backtester with the
deliberately-leaky :func:`leaky_vectorized_backtest` (it earns the SAME-bar return
``pi_t * r_t`` instead of the causal ``pi_t * r_{t+1}``) inside a serve-shaped run,
and confirms the parity oracle reports a divergence far above tolerance — i.e. the
pipeline would refuse to ship a leaky equity curve.

A parity check that can never fail proves nothing; this is the proof it CAN fail
when a real look-ahead bug is introduced into the integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from algosystem.data.loaders import synthetic_default_bars
from algosystem.execution.paper_broker import PaperBrokerConfig, replay
from algosystem.execution.parity import (
    PARITY_TOL,
    ParityError,
    assert_parity,
    leaky_vectorized_backtest,
)
from algosystem.serve import _align_positions, run_system
from algosystem.signals.library import SignalSpec, build_signal


def _serve_shaped_returns_positions(
    *, seed: int = 7, fast: int = 10, slow: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Build the (returns, positions) the serve path would compute for the default config."""
    bars, returns, _src = synthetic_default_bars(n_obs=2000, seed=seed, kind="gbm_regime")
    close = bars["close"]
    ret = np.asarray(returns.to_numpy(dtype="float64"), dtype="float64")
    positions = _align_positions(
        build_signal(SignalSpec("ma_crossover", {"fast": fast, "slow": slow}), close), ret.size
    )
    return ret, positions


@pytest.mark.integration
def test_honest_pipeline_passes_parity_but_leaky_one_is_caught() -> None:
    """The honest serve config passes parity; the leaky variant diverges far above tol."""
    ret, positions = _serve_shaped_returns_positions()

    # HONEST: the serve path's parity oracle passes (curves coincide to 1e-10).
    honest_curve = assert_parity(ret, positions, cost_bps=5.0, slippage_bps=2.0)
    assert honest_curve.size > 0

    # LEAKY: swap in the same-bar-return backtester and compare to the honest live
    # paper broker — the divergence is orders of magnitude above the parity tol, so
    # the oracle (which the serve path runs) would CATCH this look-ahead bug.
    leaky = leaky_vectorized_backtest(ret, positions, cost_bps=5.0, slippage_bps=2.0)
    live = replay(ret, positions, PaperBrokerConfig(cost_bps=5.0, slippage_bps=2.0))
    max_diff = float(np.max(np.abs(leaky.equity_curve - live.equity_curve)))
    assert max_diff > PARITY_TOL
    assert max_diff > 1e-6


@pytest.mark.integration
def test_serve_parity_oracle_raises_when_the_backtester_leaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a look-ahead backtester were wired into the serve parity check, it RAISES.

    We monkeypatch the parity module's vectorized backtester to the leaky variant so
    the parity comparison sees a same-bar-fill ("future-peeking") curve against the
    honest paper broker, and confirm ``run_system`` surfaces a :class:`ParityError`
    rather than silently shipping a leaky result.
    """
    import algosystem.execution.parity as parity_mod

    # Wire the deliberately-leaky backtester into BOTH parity seams the serve path
    # uses (``check_parity`` and ``assert_parity`` both read this name).
    monkeypatch.setattr(parity_mod, "vectorized_backtest", leaky_vectorized_backtest)

    with pytest.raises(ParityError, match="parity FAILED"):
        run_system(signal="ma_crossover", fast=10, slow=50, seed=7)
