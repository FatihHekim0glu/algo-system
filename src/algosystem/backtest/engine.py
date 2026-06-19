"""Vectorized no-lookahead single-asset backtester (pure numpy; the parity reference).

[TYPED STUB — signatures, docstrings, and the frozen ``BacktestResult`` are final;
the engine bodies raise :class:`NotImplementedError` for a sequential author to
fill.]

A fast, fully-vectorized evaluator of a signal's target-position sequence over a
single-asset return path. For a per-bar position sequence ``pi_t`` and close
return path ``r_t`` the per-bar net return is

    net_t = pi_t * r_{t+1} - (cost_bps + slippage_bps)/1e4 * |pi_t - pi_{t-1}|

(the position decided at the CLOSE of bar ``t`` earns the NEXT bar's return —
STRICTLY CAUSAL, the order fills at bar ``t+1``'s OPEN, never the same bar's close)
and the equity curve is the cumulative product of ``1 + net_t``. Positions are
applied via ``signal.shift(1)`` and returns via ``pct_change(fill_method=None)``.
Costs + slippage are charged IDENTICALLY to the simulated paper broker, so this
vectorized equity curve MUST reproduce the paper-broker replay to 1e-10 (the
backtest<->live PARITY ORACLE). Any mismatch indicates the vectorized path peeked
at the future or a fill-timing bug.

The walk-forward variant drives the same accounting across rolling
in-sample/out-of-sample folds with a purge + embargo so a return earned in the OOS
fold was never seen by the signal that decided the position.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from algosystem._exceptions import ValidationError
from algosystem._typing import FloatArray, PositionSequence, ReturnSeries


def _safe_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, mapping NaN/Inf/None to ``None``."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Immutable result of a vectorized single-asset backtest.

    Attributes
    ----------
    net_returns:
        The per-bar net (after-cost, after-slippage) return series.
    gross_returns:
        The per-bar gross (before-cost) return series ``pi_t * r_{t+1}``.
    equity_curve:
        The cumulative-wealth curve ``cumprod(1 + net_returns)``.
    positions:
        The applied per-bar position sequence ``pi_t`` (shifted for the t->t+1 fill).
    turnover:
        Total one-way turnover ``sum |pi_t - pi_{t-1}|`` over the path.
    costs:
        The per-bar transaction-cost + slippage charge series.
    n_bars:
        The number of scored bars.
    """

    net_returns: FloatArray
    gross_returns: FloatArray
    equity_curve: FloatArray
    positions: FloatArray
    turnover: float
    costs: FloatArray
    n_bars: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "net_returns": [float(x) for x in np.asarray(self.net_returns).ravel()],
            "gross_returns": [float(x) for x in np.asarray(self.gross_returns).ravel()],
            "equity_curve": [float(x) for x in np.asarray(self.equity_curve).ravel()],
            "positions": [float(x) for x in np.asarray(self.positions).ravel()],
            "turnover": float(self.turnover),
            "costs": [float(x) for x in np.asarray(self.costs).ravel()],
            "n_bars": int(self.n_bars),
            "meta": dict(self.meta),
        }


def vectorized_backtest(
    returns: ReturnSeries,
    positions: PositionSequence,
    *,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    initial_position: float = 0.0,
) -> BacktestResult:
    r"""Evaluate a position sequence over a return path (vectorized, strictly causal).

    For per-bar positions ``pi_t`` and close returns ``r_t`` the per-bar net return
    is ``pi_t * r_{t+1} - (cost_bps + slippage_bps)/1e4 * |pi_t - pi_{t-1}|`` — the
    position decided at the CLOSE of bar ``t`` earns the NEXT bar's return (no
    look-ahead; the order fills at bar ``t+1``'s OPEN) and the trade friction is
    charged on the position CHANGE. The first position change is taken against
    ``initial_position``. Costs/slippage are charged IDENTICALLY to the simulated
    paper broker so this curve matches the paper-broker replay to 1e-10 (the parity
    oracle).

    Parameters
    ----------
    returns:
        The single-asset per-bar close-return path.
    positions:
        The per-bar target-position sequence (for the ``t -> t+1`` holding period).
    cost_bps:
        Per-side transaction cost in basis points on ``|Δposition|``.
    slippage_bps:
        Per-trade slippage in basis points on ``|Δposition|``.
    initial_position:
        The position held before the first bar (for the first turnover charge).

    Returns
    -------
    BacktestResult
        The net/gross returns, equity curve, positions, turnover, and costs.

    Raises
    ------
    ValidationError
        If ``returns`` and ``positions`` lengths are inconsistent, or a cost is
        negative.
    InsufficientDataError
        If there are fewer than two bars (no causal step can be scored).
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValidationError(f"cost_bps must be finite and >= 0, got {cost_bps!r}.")
    if not np.isfinite(slippage_bps) or slippage_bps < 0.0:
        raise ValidationError(f"slippage_bps must be finite and >= 0, got {slippage_bps!r}.")
    if not np.isfinite(initial_position):
        raise ValidationError(f"initial_position must be finite, got {initial_position!r}.")
    raise NotImplementedError("vectorized_backtest: typed stub — body to be authored.")


def walk_forward_signal_backtest(
    returns: ReturnSeries,
    positions: PositionSequence,
    *,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    train_window: int = 252,
    test_window: int = 63,
    purge: int = 1,
    embargo: int = 1,
) -> BacktestResult:
    r"""Run a purged walk-forward backtest of a precomputed position sequence.

    Drives the vectorized per-bar accounting across rolling
    in-sample/out-of-sample folds. With a daily horizon and ``shift(1)`` position
    application the embargo equals the return horizon (``embargo=1``) and the purge
    removes the single shared boundary observation (``purge=1``) so no OOS return
    is earned by a position that "saw" it. Only the concatenated OOS folds are
    scored and returned.

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
    train_window:
        The in-sample fold length (informational for this position-replay variant).
    test_window:
        The out-of-sample fold length stepped through the path.
    purge:
        Boundary observations purged between the train and test folds.
    embargo:
        Observations embargoed after each in-sample fold (= the return horizon).

    Returns
    -------
    BacktestResult
        The concatenated OOS net/gross returns, equity, positions, turnover, costs.

    Raises
    ------
    ValidationError
        If a cost / window parameter is invalid.
    InsufficientDataError
        If the path is too short for even one train/test split.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValidationError(f"cost_bps must be finite and >= 0, got {cost_bps!r}.")
    if not np.isfinite(slippage_bps) or slippage_bps < 0.0:
        raise ValidationError(f"slippage_bps must be finite and >= 0, got {slippage_bps!r}.")
    if train_window < 1 or test_window < 1:
        raise ValidationError(
            f"train_window and test_window must be >= 1, got "
            f"train_window={train_window}, test_window={test_window}."
        )
    if purge < 0 or embargo < 0:
        raise ValidationError(
            f"purge and embargo must be >= 0, got purge={purge}, embargo={embargo}."
        )
    raise NotImplementedError("walk_forward_signal_backtest: typed stub — body to be authored.")


def equity_curve(net_returns: FloatArray) -> FloatArray:
    """Return the cumulative-wealth curve ``cumprod(1 + net_returns)``.

    Parameters
    ----------
    net_returns:
        A per-bar net return series.

    Returns
    -------
    FloatArray
        The cumulative-wealth curve, same length as ``net_returns``.

    Raises
    ------
    ValidationError
        If ``net_returns`` is empty or non-finite.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    arr = np.asarray(net_returns, dtype="float64").ravel()
    if arr.size == 0:
        raise ValidationError("equity_curve: net_returns must be non-empty.")
    if not np.isfinite(arr).all():
        raise ValidationError("equity_curve: net_returns contains non-finite values.")
    raise NotImplementedError("equity_curve: typed stub — body to be authored.")
