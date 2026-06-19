"""Plotly figure builders (LAZY plotly): backtest-vs-live equity overlay + drawdown.

[TYPED STUB — signatures, docstrings, and the ``FigureDict`` shape are final; the
figure-construction bodies raise :class:`NotImplementedError` for a sequential
author to fill.]

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}`` —
the same JSON shape the FastAPI layer serializes and the Next.js ``PlotlyChart``
component renders — so the figures cross the API boundary with no Plotly object
leaking through. Plotly is an OPTIONAL dependency (the ``viz`` extra) and is
imported lazily inside each builder; importing this module has no side effects and
does not require Plotly.

The serialization always routes through
``json.loads(plotly.io.to_json(fig, validate=False))`` so the emitted mapping is a
plain, JSON-safe ``dict`` (no numpy scalars, no Plotly classes) regardless of the
input container the caller passed. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from algosystem._exceptions import ValidationError
from algosystem._typing import FloatArray

if TYPE_CHECKING:
    import plotly.graph_objects as go

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]


def _finite_1d(values: object, *, name: str) -> FloatArray:
    """Coerce ``values`` to a non-empty, finite 1-D float64 array (or raise).

    The single input boundary every figure builder funnels its curves through:
    flatten to 1-D, require non-emptiness, and reject any NaN/Inf so a malformed
    series never silently produces a broken chart.

    Parameters
    ----------
    values:
        A sequence / ndarray of floats (an equity curve, a drawdown series).
    name:
        Human-readable label used in the error message.

    Returns
    -------
    FloatArray
        The coerced 1-D float64 array.

    Raises
    ------
    ValidationError
        If ``values`` is empty or contains any non-finite value.
    """
    arr = np.asarray(values, dtype="float64").ravel()
    if arr.size == 0:
        raise ValidationError(f"{name} must be non-empty.")
    if not np.isfinite(arr).all():
        raise ValidationError(f"{name} contains non-finite values.")
    return arr


def _serialize(fig: go.Figure) -> FigureDict:
    """Serialize a Plotly figure to a plain ``{data, layout}`` mapping.

    Routes through ``plotly.io.to_json(fig, validate=False)`` (then
    :func:`json.loads`) so the result is a JSON-safe ``dict`` with no numpy scalars
    or Plotly objects — exactly what the FastAPI layer returns and the frontend
    ``PlotlyChart`` renders. ``validate=False`` skips Plotly's schema validation
    (the figures are constructed in-house from trusted traces).

    Parameters
    ----------
    fig:
        The constructed Plotly figure.

    Returns
    -------
    FigureDict
        A plain ``{"data": [...], "layout": {...}}`` mapping.

    Raises
    ------
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    raise NotImplementedError("_serialize: typed stub — body to be authored.")


def equity_overlay_figure(
    backtest_equity: FloatArray,
    live_equity: FloatArray,
    buyhold_equity: FloatArray,
    *,
    title: str = "Backtest vs. live (paper-broker) equity",
) -> FigureDict:
    """Build the equity-overlay figure: backtest + paper-broker live + buy-hold.

    Overlays the vectorized backtest equity curve, the simulated paper-broker
    "live" equity curve, and the buy-and-hold curve. The backtest and live curves
    should COINCIDE to the eye (the parity oracle), visually proving the
    backtest<->live agreement; buy-and-hold is the bar the strategy must clear.

    Parameters
    ----------
    backtest_equity:
        The vectorized backtest cumulative-wealth curve.
    live_equity:
        The simulated paper-broker cumulative-wealth curve (should coincide with
        ``backtest_equity``).
    buyhold_equity:
        The buy-and-hold cumulative-wealth curve.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` line-chart mapping.

    Raises
    ------
    ValidationError
        If the curves are empty or length-mismatched.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    bt = _finite_1d(backtest_equity, name="backtest_equity")
    live = _finite_1d(live_equity, name="live_equity")
    bh = _finite_1d(buyhold_equity, name="buyhold_equity")
    if not (bt.size == live.size == bh.size):
        raise ValidationError(
            f"equity_overlay_figure: backtest ({bt.size}), live ({live.size}), and "
            f"buy-hold ({bh.size}) equity curves must have the same length."
        )
    raise NotImplementedError("equity_overlay_figure: typed stub — body to be authored.")


def drawdown_figure(
    net_returns: FloatArray,
    *,
    title: str = "Strategy drawdown",
) -> FigureDict:
    """Build the drawdown figure from a per-bar net-return series.

    Renders the running peak-to-trough drawdown ``W_t / max_{s<=t} W_s - 1`` of the
    cumulative-wealth curve as a filled area below zero — the depth-of-pain view of
    the strategy.

    Parameters
    ----------
    net_returns:
        The per-bar net-return series.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` area-chart mapping.

    Raises
    ------
    ValidationError
        If ``net_returns`` is empty or non-finite.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    _finite_1d(net_returns, name="net_returns")
    raise NotImplementedError("drawdown_figure: typed stub — body to be authored.")
