"""Unit + property tests for the vectorized no-lookahead backtest engine.

These lock the load-bearing accounting contracts the parity oracle relies on:

- positions are LAGGED — the position decided at the close of bar ``t`` earns the
  ``t -> t+1`` return, NEVER the same bar's close (no look-ahead / same-bar fill);
- cost accounting — friction is charged on ``|pi_t - pi_{t-1}|`` (the position
  CHANGE) at ``(cost_bps + slippage_bps)/1e4`` per unit of one-way turnover, and is
  monotone (more turnover => weakly lower net return);
- the purged walk-forward fold geometry — the first OOS bar starts a full
  ``train_window + purge + embargo`` in, and only OOS bars are scored.

The backtest<->live PARITY ORACLE itself lives in the parity test group; here we
additionally pin the vectorized curve against an explicit bar-by-bar reference
replay so a same-bar-fill regression in this engine is caught locally too.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from algosystem._exceptions import InsufficientDataError, ValidationError
from algosystem.backtest.engine import (
    BacktestResult,
    _safe_float,
    equity_curve,
    vectorized_backtest,
    walk_forward_signal_backtest,
)


def _sequential_reference(
    returns: np.ndarray,
    positions: np.ndarray,
    *,
    cost_bps: float,
    slippage_bps: float,
    initial_position: float = 0.0,
) -> np.ndarray:
    """A transparent bar-by-bar replay of the fill-timing contract (the reference).

    Mirrors what the simulated paper broker does step by step: at the close of bar
    ``t`` the target ``positions[t]`` is read, the change ``|target - prev|`` is
    charged, and the position earns ``returns[t+1]`` over the next bar. Returns the
    cumulative-wealth curve over the ``N - 1`` scored bars.
    """
    friction = (cost_bps + slippage_bps) / 1e4
    prev = initial_position
    equity = 1.0
    curve = []
    for t in range(len(positions) - 1):
        target = float(positions[t])
        cost = friction * abs(target - prev)
        net = target * float(returns[t + 1]) - cost
        equity *= 1.0 + net
        curve.append(equity)
        prev = target
    return np.asarray(curve, dtype="float64")


# --------------------------------------------------------------------------- #
# Positions are lagged: the signal at t earns the t->t+1 return (no same-bar)   #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_positions_are_lagged_to_next_bar_return() -> None:
    """``positions[t]`` earns ``returns[t+1]``; it NEVER touches ``returns[t]``."""
    # A return spike sits at index 2; only a position held at index 1 (the bar
    # BEFORE the spike) can earn it. A same-bar-fill bug would credit index 2.
    returns = np.array([0.0, 0.0, 0.50, 0.0, 0.0])
    held_before = np.array([0.0, 1.0, 0.0, 0.0, 0.0])  # long at t=1 -> earns r[2]
    held_on = np.array([0.0, 0.0, 1.0, 0.0, 0.0])  # long at t=2 (same bar)

    earns = vectorized_backtest(returns, held_before, cost_bps=0.0, slippage_bps=0.0)
    misses = vectorized_backtest(returns, held_on, cost_bps=0.0, slippage_bps=0.0)

    # The position one bar EARLY captures the spike in its gross return...
    assert earns.gross_returns[1] == pytest.approx(0.50)
    # ...while the same-bar position earns 0 from the spike (it is already over).
    assert misses.gross_returns[2] == pytest.approx(0.0)


@pytest.mark.unit
def test_last_position_is_dropped_no_phantom_return() -> None:
    """With ``N`` bars there are exactly ``N - 1`` scored steps (last pi has no r)."""
    returns = np.array([0.0, 0.01, 0.02, 0.03])
    positions = np.array([1.0, 1.0, 1.0, 1.0])
    res = vectorized_backtest(returns, positions, cost_bps=0.0, slippage_bps=0.0)
    assert res.n_bars == returns.size - 1
    assert res.net_returns.size == returns.size - 1
    assert res.positions.size == returns.size - 1


@pytest.mark.unit
def test_vectorized_matches_sequential_reference() -> None:
    """The vectorized curve equals an explicit bar-by-bar replay to 1e-12."""
    rng = np.random.default_rng(7)
    returns = rng.standard_normal(200) * 0.02
    positions = rng.choice([-1.0, 0.0, 1.0], size=200)
    res = vectorized_backtest(returns, positions, cost_bps=8.0, slippage_bps=3.0)
    ref = _sequential_reference(returns, positions, cost_bps=8.0, slippage_bps=3.0)
    assert np.max(np.abs(res.equity_curve - ref)) <= 1e-12


@pytest.mark.property
@given(
    returns=st.lists(
        st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=120,
    ),
    cost_bps=st.floats(min_value=0.0, max_value=25.0),
    slippage_bps=st.floats(min_value=0.0, max_value=15.0),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=120, deadline=None)
def test_vectorized_equals_sequential_for_random_paths(
    returns: list[float], cost_bps: float, slippage_bps: float, seed: int
) -> None:
    """For any path/friction the vectorized curve matches the bar-by-bar replay."""
    r = np.asarray(returns, dtype="float64")
    pi = np.random.default_rng(seed).choice([-1.0, 0.0, 1.0], size=r.size)
    res = vectorized_backtest(r, pi, cost_bps=cost_bps, slippage_bps=slippage_bps)
    ref = _sequential_reference(r, pi, cost_bps=cost_bps, slippage_bps=slippage_bps)
    assert np.max(np.abs(res.equity_curve - ref)) <= 1e-10


# --------------------------------------------------------------------------- #
# Cost accounting                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_cost_is_charged_on_position_change() -> None:
    """Friction is charged on ``|pi_t - pi_{t-1}|`` against ``initial_position``."""
    returns = np.array([0.0, 0.0, 0.0, 0.0])
    positions = np.array([1.0, 1.0, -1.0, -1.0])
    # friction = (10 + 0)/1e4 = 1e-3 per unit of |Δposition|.
    res = vectorized_backtest(
        returns, positions, cost_bps=10.0, slippage_bps=0.0, initial_position=0.0
    )
    # t=0: |1 - 0| = 1 -> 1e-3 ; t=1: |1 - 1| = 0 ; t=2: |-1 - 1| = 2 -> 2e-3.
    assert res.costs == pytest.approx(np.array([1e-3, 0.0, 2e-3]))
    assert res.turnover == pytest.approx(3.0)  # 1 + 0 + 2


@pytest.mark.unit
def test_cost_and_slippage_add_linearly() -> None:
    """``cost_bps`` and ``slippage_bps`` sum into the single per-trade friction."""
    returns = np.array([0.0, 0.0, 0.0])
    positions = np.array([1.0, 0.0, 0.0])  # one unit traded at t=0 only.
    split = vectorized_backtest(returns, positions, cost_bps=6.0, slippage_bps=4.0)
    pooled = vectorized_backtest(returns, positions, cost_bps=10.0, slippage_bps=0.0)
    assert np.allclose(split.costs, pooled.costs)
    assert split.costs[0] == pytest.approx(10.0 / 1e4)


@pytest.mark.unit
def test_zero_friction_means_net_equals_gross() -> None:
    """With zero cost and slippage, net returns equal gross returns exactly."""
    rng = np.random.default_rng(3)
    returns = rng.standard_normal(50) * 0.01
    positions = rng.choice([-1.0, 0.0, 1.0], size=50)
    res = vectorized_backtest(returns, positions, cost_bps=0.0, slippage_bps=0.0)
    np.testing.assert_array_equal(res.net_returns, res.gross_returns)
    assert np.all(res.costs == 0.0)


@pytest.mark.property
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    low_bps=st.floats(min_value=0.0, max_value=5.0),
    extra_bps=st.floats(min_value=0.0, max_value=20.0),
)
@settings(max_examples=80, deadline=None)
def test_cost_monotonicity_more_cost_lower_net(
    seed: int, low_bps: float, extra_bps: float
) -> None:
    """Raising the cost (holding the path/positions fixed) cannot raise total net."""
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal(120) * 0.01
    positions = rng.choice([-1.0, 0.0, 1.0], size=120)
    cheap = vectorized_backtest(returns, positions, cost_bps=low_bps, slippage_bps=0.0)
    dear = vectorized_backtest(
        returns, positions, cost_bps=low_bps + extra_bps, slippage_bps=0.0
    )
    # Total net return is non-increasing in the cost (turnover is identical).
    assert float(np.sum(dear.net_returns)) <= float(np.sum(cheap.net_returns)) + 1e-12


# --------------------------------------------------------------------------- #
# equity_curve helper                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_safe_float_maps_finite_and_scrubs_the_rest() -> None:
    """``_safe_float`` coerces finite numbers and maps NaN/Inf/None/garbage to None."""
    assert _safe_float(1.5) == 1.5
    assert _safe_float("2.0") == 2.0
    assert _safe_float(np.float64(3.0)) == 3.0
    assert _safe_float(float("nan")) is None
    assert _safe_float(float("inf")) is None
    assert _safe_float(None) is None
    assert _safe_float("not-a-number") is None


@pytest.mark.unit
def test_equity_curve_is_cumprod_of_one_plus_net() -> None:
    """``equity_curve`` returns ``cumprod(1 + net)``."""
    net = np.array([0.1, -0.05, 0.02])
    np.testing.assert_allclose(equity_curve(net), np.cumprod(1.0 + net))


@pytest.mark.unit
def test_equity_curve_rejects_empty_and_nonfinite() -> None:
    """Empty / non-finite net return series are rejected before any cumulation."""
    with pytest.raises(ValidationError):
        equity_curve(np.array([]))
    with pytest.raises(ValidationError):
        equity_curve(np.array([0.1, np.nan]))


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_length_mismatch_and_short_path_are_rejected() -> None:
    """Mismatched lengths raise; a single-bar path cannot score a causal step."""
    with pytest.raises(ValidationError):
        vectorized_backtest(np.array([0.0, 0.1, 0.2]), np.array([1.0, 0.0]))
    with pytest.raises(InsufficientDataError):
        vectorized_backtest(np.array([0.0]), np.array([1.0]))


@pytest.mark.unit
def test_negative_and_nonfinite_friction_are_rejected() -> None:
    """Negative or non-finite cost / slippage / initial position are rejected."""
    r = np.array([0.0, 0.1])
    p = np.array([1.0, 1.0])
    with pytest.raises(ValidationError):
        vectorized_backtest(r, p, cost_bps=-1.0)
    with pytest.raises(ValidationError):
        vectorized_backtest(r, p, slippage_bps=np.inf)
    with pytest.raises(ValidationError):
        vectorized_backtest(r, p, initial_position=np.nan)


@pytest.mark.unit
def test_nonfinite_inputs_are_rejected() -> None:
    """A NaN/Inf in returns or positions would silently break parity; reject it."""
    with pytest.raises(ValidationError):
        vectorized_backtest(np.array([0.0, np.nan, 0.1]), np.array([1.0, 1.0, 1.0]))
    with pytest.raises(ValidationError):
        vectorized_backtest(np.array([0.0, 0.1, 0.2]), np.array([1.0, np.inf, 1.0]))


@pytest.mark.unit
def test_result_to_dict_is_json_plain() -> None:
    """``BacktestResult.to_dict`` yields a plain, list/float/int structure."""
    res = vectorized_backtest(np.array([0.0, 0.1, -0.2]), np.array([1.0, 0.0, 1.0]))
    assert isinstance(res, BacktestResult)
    d = res.to_dict()
    assert isinstance(d["net_returns"], list)
    assert isinstance(d["turnover"], float)
    assert isinstance(d["n_bars"], int)
    assert all(isinstance(x, float) for x in d["equity_curve"])


# --------------------------------------------------------------------------- #
# Purged walk-forward geometry                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_walk_forward_scores_only_out_of_sample_bars() -> None:
    """Only bars from ``train_window + purge + embargo`` onward are scored."""
    n = 300
    rng = np.random.default_rng(11)
    returns = rng.standard_normal(n) * 0.01
    positions = rng.choice([-1.0, 0.0, 1.0], size=n)

    train_window, test_window, purge, embargo = 100, 25, 1, 1
    wf = walk_forward_signal_backtest(
        returns,
        positions,
        train_window=train_window,
        test_window=test_window,
        purge=purge,
        embargo=embargo,
    )

    n_scored = n - 1
    first_test = train_window + purge + embargo
    # The number of OOS scored bars equals every scored index from first_test on.
    assert wf.n_bars == n_scored - first_test
    assert wf.meta["train_window"] == train_window
    assert wf.meta["purge"] == purge
    assert wf.meta["embargo"] == embargo
    # The equity curve is internally consistent over the concatenated OOS folds.
    np.testing.assert_allclose(wf.equity_curve, np.cumprod(1.0 + wf.net_returns))


@pytest.mark.unit
def test_walk_forward_excludes_the_purged_embargoed_warmup() -> None:
    """A spike inside the purge/embargo warm-up never enters the OOS score."""
    n = 220
    returns = np.zeros(n)
    # A large return at an index that falls strictly inside the warm-up boundary.
    train_window, purge, embargo = 100, 2, 3
    warmup_spike_idx = train_window + 1  # < train_window + purge + embargo (=105)
    returns[warmup_spike_idx] = 5.0
    positions = np.ones(n)  # always long: would capture the spike if it were scored

    wf = walk_forward_signal_backtest(
        returns,
        positions,
        train_window=train_window,
        test_window=20,
        purge=purge,
        embargo=embargo,
    )
    # The spike's return is earned by the position at warmup_spike_idx - 1, which is
    # inside the excluded warm-up region, so no scored OOS gross return equals it.
    assert not np.any(np.isclose(wf.gross_returns, 5.0))


@pytest.mark.unit
def test_walk_forward_purge_embargo_shrink_the_oos_set() -> None:
    """Larger purge/embargo strictly delay the first OOS bar (fewer scored bars)."""
    n = 260
    rng = np.random.default_rng(5)
    returns = rng.standard_normal(n) * 0.01
    positions = rng.choice([-1.0, 0.0, 1.0], size=n)
    tight = walk_forward_signal_backtest(
        returns, positions, train_window=100, test_window=20, purge=0, embargo=0
    )
    loose = walk_forward_signal_backtest(
        returns, positions, train_window=100, test_window=20, purge=5, embargo=5
    )
    assert loose.n_bars < tight.n_bars
    assert loose.n_bars == tight.n_bars - 10  # 5 purge + 5 embargo bars excluded


@pytest.mark.unit
def test_walk_forward_too_short_path_raises() -> None:
    """A path too short for even one train/test split is rejected."""
    returns = np.zeros(50)
    positions = np.ones(50)
    with pytest.raises(InsufficientDataError):
        walk_forward_signal_backtest(
            returns, positions, train_window=100, test_window=20, purge=1, embargo=1
        )


@pytest.mark.unit
def test_walk_forward_invalid_params_raise() -> None:
    """Invalid cost / window / purge parameters are rejected up front."""
    returns = np.zeros(300)
    positions = np.ones(300)
    with pytest.raises(ValidationError):
        walk_forward_signal_backtest(returns, positions, cost_bps=-1.0)
    with pytest.raises(ValidationError):
        walk_forward_signal_backtest(returns, positions, slippage_bps=-1.0)
    with pytest.raises(ValidationError):
        walk_forward_signal_backtest(returns, positions, train_window=0)
    with pytest.raises(ValidationError):
        walk_forward_signal_backtest(returns, positions, purge=-1)
    with pytest.raises(ValidationError):
        walk_forward_signal_backtest(
            np.zeros(10), np.ones(9), train_window=2, test_window=2
        )
