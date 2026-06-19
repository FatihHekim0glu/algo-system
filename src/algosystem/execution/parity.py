"""Parity oracle: vectorized backtest == simulated paper-broker equity to 1e-10.

[TYPED STUB — signatures, docstrings, the ``PARITY_TOL`` constant, and the frozen
``ParityReport`` are final; the oracle bodies (and the deliberately-leaky negative
control) raise :class:`NotImplementedError` for a sequential author to fill.]

THE LOAD-BEARING BACKTEST<->LIVE LOOK-AHEAD GUARD. The vectorized backtester
(:func:`algosystem.backtest.engine.vectorized_backtest`) and the simulated
bar-by-bar paper-broker replay (:func:`algosystem.execution.paper_broker.replay`)
must produce the SAME per-bar net-return / equity curve for ANY signal/param
sequence, to ``1e-10``. A Hypothesis property test drives random position
sequences through both paths; any mismatch beyond the tolerance indicates the
vectorized path peeked at a future bar (a look-ahead bug) or a fill-timing bug and
FAILS the build. This module provides the assertion seam the property suite and the
serve-time parity probe call.

NEGATIVE CONTROL: :func:`leaky_vectorized_backtest` is a DELIBERATELY look-ahead
backtester (it earns the SAME-bar return ``pi_t * r_t`` instead of the next bar's
``pi_t * r_{t+1}``). The parity oracle MUST catch it (the parity check FAILS), so
the oracle is proven capable of failing — a parity test that can never fail proves
nothing.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from algosystem._exceptions import ParityError, ValidationError
from algosystem._typing import FloatArray, PositionSequence, ReturnSeries
from algosystem.backtest.engine import BacktestResult

#: The parity tolerance: the two paths must agree to this absolute max-diff.
PARITY_TOL: float = 1e-10


@dataclass(frozen=True, slots=True)
class ParityReport:
    """Immutable report of a backtest<->live parity check.

    Attributes
    ----------
    max_abs_diff:
        The maximum absolute per-bar difference between the vectorized backtest
        equity curve and the simulated paper-broker equity curve.
    tol:
        The tolerance the check was run against (``1e-10``).
    passed:
        ``True`` iff ``max_abs_diff <= tol`` (no look-ahead / fill-timing bug).
    n_bars:
        The number of bars compared.
    """

    max_abs_diff: float
    tol: float
    passed: bool
    n_bars: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this report."""
        return asdict(self)


def check_parity(
    returns: ReturnSeries,
    positions: PositionSequence,
    *,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    tol: float = PARITY_TOL,
) -> ParityReport:
    """Compare the vectorized backtest to the simulated paper-broker replay.

    Runs the SAME ``(returns, positions, cost_bps, slippage_bps)`` through both the
    vectorized backtester and the simulated bar-by-bar paper broker, and reports
    the maximum absolute per-bar difference between their equity curves against
    ``tol``. The check PASSES iff the two agree to ``tol`` (the look-ahead /
    fill-timing guard); a failure means the vectorized path peeked at the future or
    a fill-timing bug crept in.

    Parameters
    ----------
    returns:
        The single-asset per-bar close-return path.
    positions:
        The per-bar target-position sequence replayed through both paths.
    cost_bps:
        Per-side transaction cost in basis points (applied IDENTICALLY to both).
    slippage_bps:
        Per-trade slippage in basis points (applied IDENTICALLY to both).
    tol:
        The absolute max-diff tolerance (default ``1e-10``).

    Returns
    -------
    ParityReport
        The max abs diff, the tolerance, the pass flag, and the bar count.

    Raises
    ------
    ValidationError
        If the inputs are malformed, length-mismatched, or ``tol`` is invalid.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    if tol < 0.0:
        raise ValidationError(f"tol must be >= 0, got {tol!r}.")
    raise NotImplementedError("check_parity: typed stub — body to be authored.")


def assert_parity(
    returns: ReturnSeries,
    positions: PositionSequence,
    *,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    tol: float = PARITY_TOL,
) -> FloatArray:
    """Assert backtest<->live parity to ``tol`` and return the agreed equity curve.

    Convenience wrapper over :func:`check_parity` that RAISES
    :class:`algosystem._exceptions.ParityError` when the two paths disagree beyond
    ``tol`` (so the serve-time probe and the property suite fail loudly on any
    look-ahead / fill-timing bug). On success returns the agreed per-bar equity
    curve (both paths produce it identically).

    Parameters
    ----------
    returns:
        The single-asset per-bar close-return path.
    positions:
        The per-bar target-position sequence.
    cost_bps, slippage_bps:
        Frictions applied IDENTICALLY to both paths.
    tol:
        The absolute max-diff tolerance (default ``1e-10``).

    Returns
    -------
    FloatArray
        The agreed per-bar equity curve (both paths produce this).

    Raises
    ------
    ParityError
        If the parity check fails (a look-ahead / fill-timing bug).
    ValidationError
        If the inputs are malformed.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    raise NotImplementedError("assert_parity: typed stub — body to be authored.")


def leaky_vectorized_backtest(
    returns: ReturnSeries,
    positions: PositionSequence,
    *,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    initial_position: float = 0.0,
) -> BacktestResult:
    r"""DELIBERATELY-LEAKY backtester (negative control the parity oracle MUST catch).

    Identical to :func:`algosystem.backtest.engine.vectorized_backtest` EXCEPT it
    earns the SAME-bar return ``pi_t * r_t`` instead of the strictly-causal next
    bar's ``pi_t * r_{t+1}`` — i.e. it acts on bar ``t`` using bar ``t``'s own
    (not-yet-known-at-decision-time) return. This is a look-ahead bug ON PURPOSE:
    the backtest<->live parity oracle MUST report a max-diff far above ``1e-10``
    against the honest paper broker, proving the oracle can actually fail. Used ONLY
    by the parity negative-control regression test — NEVER in the serve path.

    Parameters
    ----------
    returns:
        The single-asset per-bar close-return path.
    positions:
        The per-bar target-position sequence.
    cost_bps:
        Per-side transaction cost in basis points.
    slippage_bps:
        Per-trade slippage in basis points.
    initial_position:
        The position held before the first bar.

    Returns
    -------
    BacktestResult
        The (leaky) net/gross returns, equity curve, positions, turnover, costs.

    Raises
    ------
    ValidationError
        If ``returns`` and ``positions`` lengths are inconsistent.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    if not isinstance(cost_bps, float | int) or cost_bps < 0.0:
        raise ValidationError(f"cost_bps must be >= 0, got {cost_bps!r}.")
    raise NotImplementedError("leaky_vectorized_backtest: typed stub — body to be authored.")


def _raise_parity_error(report: ParityReport) -> None:
    """Raise a :class:`ParityError` describing a failed parity report.

    The single, shared formatter for the parity-failure message so the property
    suite and the serve probe report look-ahead the same way.

    Raises
    ------
    ParityError
        Always, describing the divergence in ``report``.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    raise NotImplementedError("_raise_parity_error: typed stub — body to be authored.")


# Re-export the parity-failure exception type for callers that catch it.
__all__ = [
    "PARITY_TOL",
    "ParityError",
    "ParityReport",
    "assert_parity",
    "check_parity",
    "leaky_vectorized_backtest",
]
