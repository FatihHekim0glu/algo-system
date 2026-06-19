"""The LOAD-BEARING backtest<->live PARITY ORACLE tests.

These pin the integration look-ahead catch: the vectorized backtest equity curve
(:func:`algosystem.backtest.engine.vectorized_backtest`) MUST equal the simulated
bar-by-bar paper-broker equity curve (:func:`algosystem.execution.paper_broker.replay`)
to ``1e-10`` for ANY signal / param sequence, because the two charge friction and
accrue wealth identically with the SAME next-bar-open fill timing.

The suite also exercises the proof-of-teeth negative control: a deliberately-leaky
backtester that earns the SAME-bar return ``pi_t * r_t`` (look-ahead) instead of the
causal ``pi_t * r_{t+1}`` — the oracle MUST catch it (``passed=False`` /
:class:`algosystem._exceptions.ParityError`), so a parity test that can never fail
is impossible here.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from algosystem._exceptions import ParityError, ValidationError
from algosystem.backtest.engine import vectorized_backtest
from algosystem.execution.paper_broker import PaperBrokerConfig, replay
from algosystem.execution.parity import (
    PARITY_TOL,
    ParityReport,
    assert_parity,
    check_parity,
    leaky_vectorized_backtest,
)

# --------------------------------------------------------------------------- #
# Hypothesis strategies: random return paths + random target-position sequences #
# --------------------------------------------------------------------------- #
_RETURN = st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False)
_POSITION = st.sampled_from([-1.0, 0.0, 1.0])
_CONT_POSITION = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_BPS = st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False)


@st.composite
def _returns_and_positions(
    draw: st.DrawFn,
    *,
    min_size: int = 2,
    max_size: int = 200,
    position: st.SearchStrategy[float] = _POSITION,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw an aligned ``(returns, positions)`` pair of a Hypothesis-chosen length."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    returns = draw(st.lists(_RETURN, min_size=n, max_size=n))
    positions = draw(st.lists(position, min_size=n, max_size=n))
    return (
        np.asarray(returns, dtype="float64"),
        np.asarray(positions, dtype="float64"),
    )


# --------------------------------------------------------------------------- #
# The oracle: vectorized backtest == paper-broker equity to 1e-10 (random)      #
# --------------------------------------------------------------------------- #
@pytest.mark.parity
@given(data=_returns_and_positions(), cost_bps=_BPS, slippage_bps=_BPS)
@settings(max_examples=200, deadline=None)
def test_backtest_equals_live_for_random_signal_params(
    data: tuple[np.ndarray, np.ndarray], cost_bps: float, slippage_bps: float
) -> None:
    """For ANY signal/param sequence the two equity curves agree to ``1e-10``."""
    returns, positions = data
    report = check_parity(
        returns, positions, cost_bps=cost_bps, slippage_bps=slippage_bps
    )
    assert report.passed is True
    assert report.max_abs_diff <= PARITY_TOL
    assert report.tol == PARITY_TOL


@pytest.mark.parity
@given(
    data=_returns_and_positions(position=_CONT_POSITION),
    cost_bps=_BPS,
    slippage_bps=_BPS,
)
@settings(max_examples=150, deadline=None)
def test_parity_holds_for_continuous_target_weights(
    data: tuple[np.ndarray, np.ndarray], cost_bps: float, slippage_bps: float
) -> None:
    """Parity also holds for continuous target weights in ``[-1, 1]`` (not just {-1,0,1})."""
    returns, positions = data
    report = check_parity(
        returns, positions, cost_bps=cost_bps, slippage_bps=slippage_bps
    )
    assert report.passed is True
    assert report.max_abs_diff <= PARITY_TOL


@pytest.mark.parity
def test_parity_exact_on_a_hand_built_path() -> None:
    """A small explicit path: the two equity curves are bit-for-bit identical."""
    returns = np.array([0.0, 0.02, -0.01, 0.03, 0.01])
    positions = np.array([0.0, 1.0, 1.0, -1.0, 0.0])
    bt = vectorized_backtest(returns, positions, cost_bps=5.0, slippage_bps=2.0)
    live = replay(returns, positions, PaperBrokerConfig(cost_bps=5.0, slippage_bps=2.0))
    # The equity curves coincide EXACTLY (max diff is a true zero, not merely < tol).
    np.testing.assert_array_equal(bt.equity_curve, live.equity_curve)
    np.testing.assert_array_equal(bt.net_returns, live.net_returns)
    np.testing.assert_array_equal(bt.positions, live.positions)
    assert bt.turnover == pytest.approx(live.turnover, abs=0.0)
    assert bt.n_bars == live.n_bars


# --------------------------------------------------------------------------- #
# assert_parity: returns the agreed curve on success, raises on a fabricated fail #
# --------------------------------------------------------------------------- #
@pytest.mark.parity
def test_assert_parity_returns_agreed_curve() -> None:
    """On an honest pair, ``assert_parity`` returns the (shared) equity curve."""
    returns = np.array([0.0, 0.01, 0.02, -0.01, 0.015, 0.0])
    positions = np.array([0.0, 1.0, 0.0, -1.0, 1.0, 0.0])
    curve = assert_parity(returns, positions, cost_bps=5.0, slippage_bps=2.0)
    bt = vectorized_backtest(returns, positions, cost_bps=5.0, slippage_bps=2.0)
    np.testing.assert_array_equal(curve, bt.equity_curve)


@pytest.mark.parity
def test_assert_parity_raises_on_a_simulated_fill_timing_bug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a fill-timing bug ever desynced the live path, ``assert_parity`` raises.

    We can't make the honest paths disagree, so we inject a divergent "live" path
    (a stand-in for a fill-timing regression) and confirm the oracle's failure
    branch raises :class:`ParityError` rather than silently passing.
    """
    import algosystem.execution.parity as parity_mod
    from algosystem.execution.paper_broker import PaperBrokerResult

    returns = np.array([0.0, 0.02, -0.01, 0.03, 0.0])
    positions = np.array([0.0, 1.0, 1.0, -1.0, 0.0])

    real_replay = parity_mod.replay

    def _buggy_replay(r: object, p: object, cfg: object = None) -> PaperBrokerResult:
        honest = real_replay(r, p, cfg)  # type: ignore[arg-type]
        # Corrupt the live equity curve to mimic a look-ahead / fill-timing bug.
        bad_equity = honest.equity_curve + 1.0
        return PaperBrokerResult(
            net_returns=honest.net_returns,
            equity_curve=bad_equity,
            positions=honest.positions,
            fills=honest.fills,
            turnover=honest.turnover,
            n_bars=honest.n_bars,
            meta=honest.meta,
        )

    monkeypatch.setattr(parity_mod, "replay", _buggy_replay)
    with pytest.raises(ParityError, match="parity FAILED"):
        assert_parity(returns, positions, cost_bps=5.0, slippage_bps=2.0)


@pytest.mark.parity
def test_parity_report_to_dict_is_json_shaped() -> None:
    """``ParityReport.to_dict`` exposes the four scalar fields for the API boundary."""
    returns = np.array([0.0, 0.01, 0.02, 0.0])
    positions = np.array([0.0, 1.0, 0.0, 0.0])
    report = check_parity(returns, positions, cost_bps=5.0, slippage_bps=2.0)
    d = report.to_dict()
    assert set(d) == {"max_abs_diff", "tol", "passed", "n_bars"}
    assert d["passed"] is True
    assert d["tol"] == PARITY_TOL


@pytest.mark.parity
@pytest.mark.parametrize("field", ["cost_bps", "slippage_bps"])
def test_leaky_backtester_rejects_negative_friction(field: str) -> None:
    """The leaky negative control still validates its friction inputs."""
    returns = np.array([0.0, 0.01, 0.02])
    positions = np.array([1.0, 0.0, -1.0])
    with pytest.raises(ValidationError):
        leaky_vectorized_backtest(returns, positions, **{field: -1.0})


@pytest.mark.parity
def test_leaky_backtester_rejects_length_mismatch() -> None:
    """The leaky negative control rejects misaligned returns / positions."""
    with pytest.raises(ValidationError, match="same length"):
        leaky_vectorized_backtest(np.array([0.0, 0.01, 0.02]), np.array([1.0, 0.0]))


@pytest.mark.parity
def test_assert_parity_raises_when_tolerance_is_impossible() -> None:
    """A negative-impossible path cannot exist, so force a fail via a tiny equity gap.

    We cannot make the honest paths disagree (that is the whole point), so we drive
    the failure seam directly: a ``ParityReport`` with ``passed=False`` formatted by
    the shared raiser must produce a :class:`ParityError`.
    """
    from algosystem.execution.parity import _raise_parity_error

    bad = ParityReport(max_abs_diff=1.0, tol=PARITY_TOL, passed=False, n_bars=10)
    with pytest.raises(ParityError, match="parity FAILED"):
        _raise_parity_error(bad)


# --------------------------------------------------------------------------- #
# NEGATIVE CONTROL: the leaky backtester trips the oracle (proof of teeth)      #
# --------------------------------------------------------------------------- #
@pytest.mark.parity
def test_leaky_backtester_is_caught_by_the_oracle() -> None:
    """The deliberately-leaky (same-bar-fill) backtester diverges far beyond ``1e-10``."""
    rng = np.random.default_rng(11)
    # A return path with a clear non-degenerate structure so the same-bar vs
    # next-bar distinction actually moves the curve.
    returns = rng.standard_normal(300) * 0.02
    positions = rng.integers(-1, 2, size=300).astype("float64")

    leaky = leaky_vectorized_backtest(returns, positions, cost_bps=5.0, slippage_bps=2.0)
    live = replay(returns, positions, PaperBrokerConfig(cost_bps=5.0, slippage_bps=2.0))

    max_diff = float(np.max(np.abs(leaky.equity_curve - live.equity_curve)))
    # The look-ahead bug is caught: the divergence is orders of magnitude above tol.
    assert max_diff > 1e-6
    assert max_diff > PARITY_TOL
    assert leaky.meta["leaky"] is True


@pytest.mark.parity
def test_leaky_backtester_earns_same_bar_return() -> None:
    """The leaky variant earns ``pi_t * r_t`` while the honest one earns ``pi_t * r_{t+1}``."""
    returns = np.array([0.0, 0.10, 0.0, -0.05, 0.0])
    positions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    leaky = leaky_vectorized_backtest(returns, positions, cost_bps=0.0, slippage_bps=0.0)
    honest = vectorized_backtest(returns, positions, cost_bps=0.0, slippage_bps=0.0)
    # Leaky gross at scored index t == pi_t * r_t (the SAME bar, look-ahead).
    np.testing.assert_array_equal(leaky.gross_returns, positions[:-1] * returns[:-1])
    # Honest gross at scored index t == pi_t * r_{t+1} (the next, causal bar).
    np.testing.assert_array_equal(honest.gross_returns, positions[:-1] * returns[1:])


@pytest.mark.parity
@given(data=_returns_and_positions(min_size=4), cost_bps=_BPS, slippage_bps=_BPS)
@settings(max_examples=120, deadline=None)
def test_leaky_costs_match_honest_costs(
    data: tuple[np.ndarray, np.ndarray], cost_bps: float, slippage_bps: float
) -> None:
    """The leak is ONLY in the fill timing — friction/turnover are charged identically."""
    returns, positions = data
    leaky = leaky_vectorized_backtest(
        returns, positions, cost_bps=cost_bps, slippage_bps=slippage_bps
    )
    honest = vectorized_backtest(
        returns, positions, cost_bps=cost_bps, slippage_bps=slippage_bps
    )
    # Same costs, same positions, same turnover — only gross/net differ (the leak).
    np.testing.assert_array_equal(leaky.costs, honest.costs)
    np.testing.assert_array_equal(leaky.positions, honest.positions)
    assert leaky.turnover == pytest.approx(honest.turnover, abs=0.0)


# --------------------------------------------------------------------------- #
# Next-bar-OPEN fill timing (the causal contract the parity oracle protects)    #
# --------------------------------------------------------------------------- #
@pytest.mark.parity
def test_next_bar_open_fill_timing_in_paper_broker() -> None:
    """A position held one bar BEFORE a spike captures it; the same-bar position misses."""
    returns = np.array([0.0, 0.0, 0.50, 0.0, 0.0])
    held_before = np.array([0.0, 1.0, 0.0, 0.0, 0.0])  # long at t=1 -> fills t=2 open
    held_on = np.array([0.0, 0.0, 1.0, 0.0, 0.0])  # long at t=2 (already over)

    earns = replay(returns, held_before, PaperBrokerConfig(cost_bps=0.0, slippage_bps=0.0))
    misses = replay(returns, held_on, PaperBrokerConfig(cost_bps=0.0, slippage_bps=0.0))

    # The fill for the decision at bar t lands at the NEXT bar's open (index t+1).
    assert earns.fills[1].bar_index == 2
    # The bar BEFORE the spike captures it in its next-bar net return...
    assert earns.net_returns[1] == pytest.approx(0.50)
    # ...the same-bar position earns nothing from the (already over) spike.
    assert misses.net_returns[2] == pytest.approx(0.0)


@pytest.mark.parity
def test_paper_broker_drops_the_last_position_no_phantom_fill() -> None:
    """With ``N`` bars there are exactly ``N - 1`` fills (the last pi has no next bar)."""
    returns = np.array([0.0, 0.01, 0.02, 0.03])
    positions = np.array([1.0, 1.0, 1.0, 1.0])
    live = replay(returns, positions, PaperBrokerConfig(cost_bps=0.0, slippage_bps=0.0))
    assert live.n_bars == 3
    assert len(live.fills) == 3
    assert live.equity_curve.size == 3


# --------------------------------------------------------------------------- #
# Cost + slippage accounting (charged on |Δposition|, identically to backtest)  #
# --------------------------------------------------------------------------- #
@pytest.mark.parity
def test_cost_and_slippage_charged_on_position_change() -> None:
    """Friction is ``(cost_bps + slippage_bps)/1e4 * |Δposition|``, additive in the two bps."""
    returns = np.array([0.0, 0.0, 0.0])  # zero returns isolate the friction.
    positions = np.array([1.0, 0.0, 0.0])  # one round-trip: 0->1 then 1->0.

    live = replay(returns, positions, PaperBrokerConfig(cost_bps=10.0, slippage_bps=5.0))
    friction = (10.0 + 5.0) / 1e4
    # Bar 0: trade from initial 0 -> 1 (|Δ|=1) charged; bar 1: 1 -> 0 (|Δ|=1) charged.
    assert live.net_returns[0] == pytest.approx(-friction * 1.0)
    assert live.net_returns[1] == pytest.approx(-friction * 1.0)
    # The per-fill cost field reflects the same charge.
    assert live.fills[0].cost == pytest.approx(friction * 1.0)
    assert live.fills[0].traded == pytest.approx(1.0)
    assert live.fills[1].traded == pytest.approx(-1.0)
    # Total one-way turnover counts both legs of the round-trip.
    assert live.turnover == pytest.approx(2.0)


@pytest.mark.parity
def test_cost_monotonicity_more_friction_lowers_net_return() -> None:
    """Holding turnover fixed, raising the bps weakly lowers the cumulative net return."""
    rng = np.random.default_rng(3)
    returns = rng.standard_normal(120) * 0.01
    positions = rng.integers(-1, 2, size=120).astype("float64")

    cheap = replay(returns, positions, PaperBrokerConfig(cost_bps=1.0, slippage_bps=0.0))
    dear = replay(returns, positions, PaperBrokerConfig(cost_bps=20.0, slippage_bps=10.0))
    assert dear.equity_curve[-1] <= cheap.equity_curve[-1]
    # Turnover is identical (only the bps changed), so the gap is pure friction.
    assert dear.turnover == pytest.approx(cheap.turnover, abs=0.0)


# --------------------------------------------------------------------------- #
# Input validation parity (both paths reject the same malformed inputs)         #
# --------------------------------------------------------------------------- #
@pytest.mark.parity
def test_length_mismatch_is_rejected_by_both_paths() -> None:
    """Both the vectorized path and the paper broker reject misaligned inputs."""
    returns = np.array([0.0, 0.01, 0.02])
    positions = np.array([1.0, 0.0])
    with pytest.raises(ValidationError):
        replay(returns, positions)
    with pytest.raises(ValidationError):
        check_parity(returns, positions)


@pytest.mark.parity
def test_check_parity_rejects_negative_tolerance() -> None:
    """A negative tolerance is a programming error, rejected up front."""
    returns = np.array([0.0, 0.01])
    positions = np.array([1.0, 0.0])
    with pytest.raises(ValidationError, match="tol must be"):
        check_parity(returns, positions, tol=-1.0)
