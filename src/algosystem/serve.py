"""Serve entrypoint the backend calls (pure numpy/scipy/statsmodels, NO torch/onnx).

[TYPED STUB — signatures, docstrings, and the frozen summary + run dataclasses are
final; the orchestration body raises :class:`NotImplementedError` for a sequential
author to fill.]

The FastAPI router calls :func:`run_system` to run the FULL pipeline on the
synthetic default (or a real single-asset basket loaded via the CLI path): a causal
signal -> a purged walk-forward backtest -> a simulated bar-by-bar paper-broker
replay -> the backtest<->live PARITY ORACLE -> OOS metrics + DM + DSR + PBO -> the
PURE ``system_has_edge`` verdict -> the equity + drawdown figures. Everything is
pure numpy/scipy/statsmodels — torch / onnx / onnxruntime / sklearn are NEVER
imported. The honest verdict is the PURE function of the inference outputs
(DM-vs-buy-hold AND DSR > 1-alpha AND PBO < 0.5, net of costs).

The deployed default runs the pipeline live on the cheap synthetic bars (vectorized
backtest + a fast paper-broker replay) OR returns a committed precomputed-metrics
artifact; the request path NEVER trains a heavy model. Importing this module has no
side effects (plotly is imported lazily inside the figure path).

Honest headline: the backtest and the simulated live execution match to the cent
(the parity oracle passes) and the bar-finality guard holds; the strategy shows NO
robust edge after costs, a Deflated-Sharpe correction, and a PBO overfitting check
(``system_has_edge=False``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AlgoSystemSummary:
    """Immutable, JSON-safe summary of the signal-vs-buy-hold single-asset comparison.

    Attributes
    ----------
    oos_sharpe:
        The strategy's OOS net Sharpe (net of costs + slippage).
    buyhold_sharpe:
        The buy-and-hold OOS net Sharpe.
    dm_pvalue_vs_buyhold:
        The Diebold-Mariano p-value of the strategy net return vs. buy-and-hold.
    deflated_sharpe:
        The Deflated Sharpe (honest #signals x #param-config ``n_trials``) — a
        probability in ``[0, 1]``.
    pbo:
        The Probability of Backtest Overfitting (CSCV), in ``[0, 1]``.
    backtest_live_parity_max_diff:
        The max abs per-bar diff between the backtest and paper-broker equity curves
        (the parity oracle; ``~0`` when they coincide).
    bar_finality_ok:
        ``True`` iff no order was attributed to a forming/partial bar.
    turnover:
        The strategy's total one-way turnover.
    max_drawdown:
        The strategy's worst peak-to-trough drawdown (``<= 0``).
    system_has_edge:
        The PURE verdict: ``True`` iff the strategy beats buy-hold DM-significant AND
        DSR > 1-alpha AND PBO < 0.5, net of costs.
    n_effective_trials:
        The honest multiplicity count used for the DSR (#signals x #param configs).
    data_source:
        Provenance of the input bars (``"synthetic"`` / ``"polygon"``).
    """

    oos_sharpe: float
    buyhold_sharpe: float
    dm_pvalue_vs_buyhold: float
    deflated_sharpe: float
    pbo: float
    backtest_live_parity_max_diff: float
    bar_finality_ok: bool
    turnover: float
    max_drawdown: float
    system_has_edge: bool
    n_effective_trials: int
    data_source: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this summary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AlgoSystemRun:
    """Immutable bundle returned to the backend: summary + two Plotly figures.

    Attributes
    ----------
    summary:
        The :class:`AlgoSystemSummary`.
    equity_figure:
        A Plotly ``{data, layout}`` dict: the backtest equity overlaid on the
        paper-broker "live" equity + buy-hold (they should visually COINCIDE).
    drawdown_figure:
        A Plotly ``{data, layout}`` dict: the strategy drawdown curve.
    """

    summary: AlgoSystemSummary
    equity_figure: dict[str, Any] = field(default_factory=dict)
    drawdown_figure: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this run."""
        return {
            "summary": self.summary.to_dict(),
            "equity_figure": self.equity_figure,
            "drawdown_figure": self.drawdown_figure,
        }


def run_system(
    *,
    signal: str = "ma_crossover",
    fast: int = 10,
    slow: int = 50,
    lookback: int = 20,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> AlgoSystemRun:
    """Run the end-to-end pipeline; return a JSON-safe summary + figures.

    Builds (or loads) the single-asset OHLC bars, computes the causal signal's
    target positions, runs them through the purged walk-forward vectorized backtest
    AND the simulated paper-broker replay, ASSERTS backtest<->live parity (the
    oracle), computes the buy-and-hold baseline, runs the Diebold-Mariano test, the
    Deflated Sharpe (honest #signals x #param-config ``n_trials``), and the PBO/CSCV,
    derives the PURE ``system_has_edge`` verdict, and assembles the
    backtest-vs-live equity + drawdown Plotly figures. NEVER trains on the request
    path; everything is pure numpy/scipy/statsmodels.

    Parameters
    ----------
    signal:
        ``"ma_crossover"`` (default) or ``"momentum"``.
    fast:
        The fast MA window (for ``ma_crossover``; must be ``< slow``).
    slow:
        The slow MA window (for ``ma_crossover``).
    lookback:
        The momentum lookback (for ``momentum``).
    cost_bps:
        Per-side transaction cost in basis points.
    slippage_bps:
        Per-trade slippage in basis points.
    data_source_pref:
        ``"synthetic"`` (default) or ``"auto"`` for the real PIT path.
    seed:
        Master RNG seed for the synthetic path.

    Returns
    -------
    AlgoSystemRun
        The summary and figures for the backend response.

    Raises
    ------
    ValidationError
        If the request is invalid (e.g. ``fast >= slow``, negative cost).
    ParityError
        If the backtest<->live parity oracle fails (a look-ahead / fill-timing bug).
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    raise NotImplementedError("run_system: typed stub — body to be authored.")
