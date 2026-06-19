"""OOS performance metrics for a single-asset strategy net-return series.

[TYPED STUB — signatures, docstrings, and the frozen ``StrategyMetrics`` bundle are
final; the metric bodies raise :class:`NotImplementedError` for a sequential author
to fill.]

The scalar summaries the verdict + API consume, all judged net of simulated
transaction costs + slippage:

- :func:`oos_sharpe` — annualized OOS Sharpe of a per-bar net-return series;
- :func:`max_drawdown` — the worst peak-to-trough drawdown of the equity curve;
- :func:`turnover` — total one-way turnover of a position sequence;
- :func:`net_pnl` — total compounded net PnL of the equity curve.

Every builder here is pure numpy (no torch / sklearn / scipy at import or call), so
the serve path computes the system + buy-hold metrics live. Importing this module
has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from algosystem._constants import PERIODS_PER_YEAR
from algosystem._exceptions import ValidationError
from algosystem._typing import FloatArray


@dataclass(frozen=True, slots=True)
class StrategyMetrics:
    """Immutable bundle of OOS net-of-cost single-asset strategy metrics.

    Attributes
    ----------
    oos_sharpe:
        Annualized OOS Sharpe of the per-bar net-return series.
    max_drawdown:
        The worst peak-to-trough drawdown (``<= 0``) of the equity curve.
    turnover:
        Total one-way turnover of the position sequence.
    net_pnl:
        Total compounded net PnL (``prod(1 + r) - 1``).
    n_bars:
        The number of scored bars.
    """

    oos_sharpe: float
    max_drawdown: float
    turnover: float
    net_pnl: float
    n_bars: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of these metrics."""
        return asdict(self)


def _coerce_series(series: FloatArray, *, name: str = "series") -> FloatArray:
    """Coerce a per-bar series to a non-empty finite 1-D float64 vector.

    The single boundary every scalar metric (Sharpe, drawdown, net PnL, turnover)
    funnels its input through, so they all share one definition of "valid series":
    flattened to 1-D, non-empty, and finite. NaN/inf are rejected here rather than
    silently propagated into a Sharpe or an equity curve.

    Parameters
    ----------
    series:
        A per-bar net-return / position series.
    name:
        Human-readable label for error messages.

    Returns
    -------
    FloatArray
        The coerced 1-D float64 array.

    Raises
    ------
    ValidationError
        If ``series`` is empty or contains any non-finite value.
    """
    arr = np.asarray(series, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValidationError(f"{name} must be non-empty.")
    if not np.isfinite(arr).all():
        raise ValidationError(f"{name} contains non-finite values.")
    return arr


def oos_sharpe(
    net_returns: FloatArray,
    *,
    risk_free: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    r"""Annualized OOS Sharpe ratio of a per-bar net-return series.

    Computes :math:`\text{SR} = \dfrac{\bar{r} - r_f}{\sigma_r}\sqrt{\text{ppy}}`
    with the sample standard deviation (``ddof=1``). A (numerically) flat series
    has undefined Sharpe and returns NaN. The series MUST already be net of costs +
    slippage — there is no gross-Sharpe escape hatch.

    Parameters
    ----------
    net_returns:
        A per-bar NET (after-cost, after-slippage) return series.
    risk_free:
        Per-bar risk-free rate subtracted from the mean.
    periods_per_year:
        Annualization factor (``252`` for daily bars).

    Returns
    -------
    float
        The annualized OOS Sharpe (NaN if the return volatility is zero).

    Raises
    ------
    ValidationError
        If ``net_returns`` is empty or non-finite.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    _coerce_series(net_returns, name="net_returns")
    del risk_free, periods_per_year
    raise NotImplementedError("oos_sharpe: typed stub — body to be authored.")


def max_drawdown(net_returns: FloatArray) -> float:
    r"""Maximum drawdown of a per-bar net-return series (``<= 0``).

    Builds the cumulative wealth curve :math:`W_t = \prod_{s \le t}(1 + r_s)`,
    tracks its running peak, and returns the most negative value of
    :math:`W_t / \max_{s \le t} W_s - 1` (``0.0`` if the series never declines).

    Parameters
    ----------
    net_returns:
        A per-bar net-return series.

    Returns
    -------
    float
        The maximum drawdown (``<= 0``).

    Raises
    ------
    ValidationError
        If ``net_returns`` is empty or non-finite.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    _coerce_series(net_returns, name="net_returns")
    raise NotImplementedError("max_drawdown: typed stub — body to be authored.")


def turnover(positions: FloatArray, *, initial_position: float = 0.0) -> float:
    r"""Total one-way turnover of a per-bar position sequence.

    Returns :math:`\sum_t |\pi_t - \pi_{t-1}|` with the first change taken against
    ``initial_position`` (the book opens from flat by default). The cost model
    charges per-side basis points on this turnover, so net Sharpe must be
    non-increasing in turnover (the cost-monotonicity property).

    Parameters
    ----------
    positions:
        The per-bar position (target-weight) sequence.
    initial_position:
        The position held before the first bar.

    Returns
    -------
    float
        Total one-way turnover (``>= 0``).

    Raises
    ------
    ValidationError
        If ``positions`` is empty or non-finite.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    if not math.isfinite(initial_position):
        raise ValidationError(f"initial_position must be finite, got {initial_position!r}.")
    _coerce_series(positions, name="positions")
    raise NotImplementedError("turnover: typed stub — body to be authored.")


def net_pnl(net_returns: FloatArray) -> float:
    r"""Total compounded net PnL of a per-bar net-return series.

    Returns :math:`\prod_t (1 + r_t) - 1`, the total compounded return over the OOS
    window net of costs + slippage.

    Parameters
    ----------
    net_returns:
        A per-bar net-return series.

    Returns
    -------
    float
        The total compounded net PnL.

    Raises
    ------
    ValidationError
        If ``net_returns`` is empty or non-finite.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    _coerce_series(net_returns, name="net_returns")
    raise NotImplementedError("net_pnl: typed stub — body to be authored.")


def strategy_metrics(
    net_returns: FloatArray,
    positions: FloatArray,
    *,
    initial_position: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> StrategyMetrics:
    """Assemble the full OOS metric bundle (Sharpe, drawdown, turnover, net PnL).

    Parameters
    ----------
    net_returns:
        The per-bar NET return series.
    positions:
        The per-bar position sequence (for turnover).
    initial_position:
        The position held before the first bar.
    periods_per_year:
        Annualization factor for the Sharpe.

    Returns
    -------
    StrategyMetrics
        The frozen metric bundle.

    Raises
    ------
    ValidationError
        If the inputs are empty / non-finite / length-mismatched.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    net = _coerce_series(net_returns, name="net_returns")
    pos = _coerce_series(positions, name="positions")
    if net.size != pos.size:
        raise ValidationError(
            f"net_returns (len {net.size}) and positions (len {pos.size}) must have the "
            "same length; both index the scored OOS window."
        )
    del initial_position, periods_per_year
    raise NotImplementedError("strategy_metrics: typed stub — body to be authored.")
