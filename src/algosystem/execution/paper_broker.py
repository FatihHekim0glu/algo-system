"""SIMULATED bar-by-bar paper broker — next-bar-open fills + costs + slippage (the "live" path).

[TYPED STUB — signatures, docstrings, the frozen config + fill + result dataclasses
are final; the broker bodies raise :class:`NotImplementedError` for a sequential
author to fill.]

A SIMULATED bar-by-bar execution engine that replays a signal's target-position
sequence the way a live trader would, but against historical bars with simulated
friction — NEVER a live broker (there is no Alpaca / broker connection and no
broker key). THE FILL-TIMING CONTRACT:

- at the CLOSE of bar ``t`` the signal emits a target position for ``t+1``;
- the resulting order fills at the NEXT bar's OPEN (``open_{t+1}``) — NEVER the same
  bar's close (that would be look-ahead);
- the engine charges ``cost_bps`` + ``slippage_bps`` on the traded position change,
  IDENTICALLY to the vectorized backtester, and tracks position + cash + an equity
  curve (the "live" path);
- a forming / unclosed bar can NEVER trigger a fill (the bar-finality guard).

Because the friction and the next-bar-open fill timing are charged IDENTICALLY to
:func:`algosystem.backtest.engine.vectorized_backtest`, the paper-broker equity
curve MUST equal the vectorized backtest equity curve to ``1e-10`` — the
backtest<->live PARITY ORACLE (asserted in :mod:`algosystem.execution.parity`).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from algosystem._exceptions import ValidationError
from algosystem._typing import FloatArray, PositionSequence, ReturnSeries


@dataclass(frozen=True, slots=True)
class PaperBrokerConfig:
    """Immutable friction configuration for the simulated paper broker.

    Attributes
    ----------
    cost_bps:
        Per-side transaction cost in basis points on ``|Δposition|`` (charged
        IDENTICALLY to the vectorized backtester).
    slippage_bps:
        Per-trade slippage in basis points on ``|Δposition|``.
    initial_position:
        The position the book opens from (flat by default).
    initial_equity:
        The starting equity level of the "live" curve (``1.0`` => a wealth index).
    """

    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    initial_position: float = 0.0
    initial_equity: float = 1.0

    def __post_init__(self) -> None:
        """Validate the friction scalars are finite and non-negative where required."""
        for name in ("cost_bps", "slippage_bps"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValidationError(f"PaperBrokerConfig: {name} must be finite and >= 0.")
        if not np.isfinite(self.initial_position):
            raise ValidationError("PaperBrokerConfig: initial_position must be finite.")
        if not np.isfinite(self.initial_equity) or self.initial_equity <= 0.0:
            raise ValidationError("PaperBrokerConfig: initial_equity must be finite and > 0.")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Fill:
    """Immutable record of a single simulated fill at a bar's open.

    Attributes
    ----------
    bar_index:
        The index of the bar at whose OPEN the order filled (``t+1`` for a signal
        decided at the close of bar ``t``).
    target_position:
        The post-fill target position.
    traded:
        The signed position change executed (``target - prev``).
    cost:
        The cost + slippage charged on this fill (in return units).
    """

    bar_index: int
    target_position: float
    traded: float
    cost: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this fill."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PaperBrokerResult:
    """Immutable result of a simulated paper-broker replay (the "live" path).

    Attributes
    ----------
    net_returns:
        The per-bar net (after-cost, after-slippage) return series.
    equity_curve:
        The cumulative-wealth "live" curve tracked bar by bar.
    positions:
        The realized per-bar position sequence after each next-bar-open fill.
    fills:
        The per-trade fill records.
    turnover:
        Total one-way turnover executed.
    n_bars:
        The number of scored bars.
    """

    net_returns: FloatArray
    equity_curve: FloatArray
    positions: FloatArray
    fills: tuple[Fill, ...]
    turnover: float
    n_bars: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "net_returns": [float(x) for x in np.asarray(self.net_returns).ravel()],
            "equity_curve": [float(x) for x in np.asarray(self.equity_curve).ravel()],
            "positions": [float(x) for x in np.asarray(self.positions).ravel()],
            "fills": [f.to_dict() for f in self.fills],
            "turnover": float(self.turnover),
            "n_bars": int(self.n_bars),
            "meta": dict(self.meta),
        }


def replay(
    returns: ReturnSeries,
    positions: PositionSequence,
    config: PaperBrokerConfig | None = None,
) -> PaperBrokerResult:
    r"""Replay a target-position sequence bar by bar through the simulated paper broker.

    Steps through the bars the way a live trader would: at the CLOSE of bar ``t``
    the target position for ``t+1`` is read; the order fills at the NEXT bar's OPEN
    with ``cost_bps`` + ``slippage_bps`` charged on the traded change; the position,
    cash, and an equity curve are tracked. A forming bar can NEVER trigger a fill
    (the bar-finality guard). The friction + next-bar-open timing are IDENTICAL to
    :func:`algosystem.backtest.engine.vectorized_backtest`, so the equity curve
    matches the vectorized backtest to ``1e-10`` (the parity oracle).

    Parameters
    ----------
    returns:
        The single-asset per-bar close-return path.
    positions:
        The per-bar target-position sequence (for the ``t -> t+1`` holding period).
    config:
        The friction configuration; a default :class:`PaperBrokerConfig` when
        ``None``.

    Returns
    -------
    PaperBrokerResult
        The "live" net returns, equity curve, realized positions, fills, turnover.

    Raises
    ------
    ValidationError
        If ``returns`` and ``positions`` lengths are inconsistent.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    cfg = config if config is not None else PaperBrokerConfig()
    if not isinstance(cfg, PaperBrokerConfig):  # pragma: no cover - defensive type guard
        raise ValidationError("replay: config must be a PaperBrokerConfig.")
    raise NotImplementedError("replay: typed stub — body to be authored.")
