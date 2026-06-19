"""End-to-end integration tests for the serve-time pipeline (:func:`run_system`).

These exercise the FULL deployed request path with NO network: a synthetic OHLC bar
process -> a strictly-causal signal -> the vectorized backtest + the simulated
paper-broker replay -> the backtest<->live PARITY ORACLE (asserted to ``1e-10``) ->
OOS metrics + Diebold-Mariano + Deflated-Sharpe + PBO/CSCV -> the PURE
``system_has_edge`` verdict -> the equity + drawdown figures.

The headline assertions are the integration contract:

- on the deployed-default honest-null DGP the verdict is ``False`` (no robust edge
  after costs, DSR, and PBO) and the backtest<->live parity diff is ``~0`` (the
  curves coincide to the cent);
- the bar-finality guard holds (no order attributed to a forming bar);
- the equity figure overlays three coinciding curves and the whole run is JSON-safe.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from algosystem.execution.parity import PARITY_TOL
from algosystem.serve import AlgoSystemRun, AlgoSystemSummary, run_system

_SUMMARY_FIELDS = {
    "oos_sharpe",
    "buyhold_sharpe",
    "dm_pvalue_vs_buyhold",
    "deflated_sharpe",
    "pbo",
    "backtest_live_parity_max_diff",
    "bar_finality_ok",
    "turnover",
    "max_drawdown",
    "system_has_edge",
    "n_effective_trials",
    "data_source",
}


@pytest.mark.integration
def test_run_system_end_to_end_no_edge_on_the_synthetic_default() -> None:
    """The deployed default runs the full pipeline and reports the honest-NULL verdict."""
    run = run_system(signal="ma_crossover", fast=10, slow=50, seed=7)
    assert isinstance(run, AlgoSystemRun)
    assert isinstance(run.summary, AlgoSystemSummary)
    # The documented honest-NULL outcome: no robust OOS edge after costs.
    assert run.summary.system_has_edge is False
    assert run.summary.data_source == "synthetic"


@pytest.mark.integration
def test_run_system_parity_oracle_holds_to_tolerance() -> None:
    """The backtest<->live parity oracle passes: the curves coincide to ``1e-10``."""
    run = run_system(signal="ma_crossover", fast=10, slow=50, seed=7)
    # The whole point: the vectorized backtest and the simulated paper-broker live
    # curve agree to the cent (the load-bearing look-ahead catch).
    assert run.summary.backtest_live_parity_max_diff <= PARITY_TOL
    assert run.summary.backtest_live_parity_max_diff >= 0.0


@pytest.mark.integration
def test_run_system_bar_finality_guard_holds() -> None:
    """No order is ever attributed to a forming bar (the bar-finality guard)."""
    run = run_system(signal="ma_crossover", fast=10, slow=50, seed=7)
    assert run.summary.bar_finality_ok is True


@pytest.mark.integration
def test_run_system_summary_has_the_full_contract() -> None:
    """The summary exposes every field the backend response contract requires."""
    run = run_system(signal="ma_crossover", fast=10, slow=50, seed=7)
    summary = run.summary.to_dict()
    assert set(summary) == _SUMMARY_FIELDS
    # The DSR is a PROBABILITY and the PBO is a fraction, both in [0, 1].
    assert 0.0 <= summary["deflated_sharpe"] <= 1.0
    assert 0.0 <= summary["pbo"] <= 1.0
    assert summary["n_effective_trials"] >= 1
    # Drawdown <= 0, turnover >= 0 (sane signs).
    assert summary["max_drawdown"] <= 0.0
    assert summary["turnover"] >= 0.0


@pytest.mark.integration
def test_run_system_is_json_safe_end_to_end() -> None:
    """The entire run (summary + both figures) round-trips through ``json.dumps``."""
    run = run_system(signal="ma_crossover", fast=10, slow=50, seed=7)
    payload = run.to_dict()
    # No numpy scalars, no Plotly objects: the whole payload is plain JSON.
    serialized = json.dumps(payload)
    restored = json.loads(serialized)
    assert set(restored) == {"summary", "equity_figure", "drawdown_figure"}
    assert set(restored["summary"]) == _SUMMARY_FIELDS


@pytest.mark.integration
def test_run_system_equity_figure_overlays_three_curves() -> None:
    """The equity figure overlays backtest + live + buy-hold (they should coincide)."""
    run = run_system(signal="ma_crossover", fast=10, slow=50, seed=7)
    fig = run.equity_figure
    assert set(fig) == {"data", "layout"}
    # Three traces: backtest, live (paper broker), buy-and-hold.
    assert len(fig["data"]) == 3
    names = {trace.get("name") for trace in fig["data"]}
    assert "Backtest" in names
    assert "Live (paper broker)" in names
    assert "Buy & hold" in names
    # The backtest and live y-series coincide to the cent (the parity oracle).
    backtest = next(t for t in fig["data"] if t.get("name") == "Backtest")
    live = next(t for t in fig["data"] if t.get("name") == "Live (paper broker)")
    bt_y = np.asarray(backtest["y"], dtype="float64")
    live_y = np.asarray(live["y"], dtype="float64")
    assert bt_y.size == live_y.size
    assert float(np.max(np.abs(bt_y - live_y))) <= PARITY_TOL


@pytest.mark.integration
def test_run_system_drawdown_figure_is_non_positive() -> None:
    """The drawdown figure is an area curve at or below zero (depth of pain)."""
    run = run_system(signal="ma_crossover", fast=10, slow=50, seed=7)
    fig = run.drawdown_figure
    assert set(fig) == {"data", "layout"}
    assert len(fig["data"]) == 1
    drawdown = np.asarray(fig["data"][0]["y"], dtype="float64")
    assert drawdown.size > 0
    # Drawdown is W_t / peak - 1, always <= 0 (with a tiny float tolerance at peaks).
    assert float(np.max(drawdown)) <= 1e-12


@pytest.mark.integration
def test_run_system_supports_the_momentum_signal() -> None:
    """The momentum signal path runs end-to-end and still yields the honest null."""
    run = run_system(signal="momentum", lookback=20, seed=7)
    assert run.summary.system_has_edge is False
    assert run.summary.backtest_live_parity_max_diff <= PARITY_TOL


@pytest.mark.integration
def test_run_system_custom_config_extends_the_multiplicity_grid() -> None:
    """A selected config not in the shipped grid is appended (honest multiplicity)."""
    in_grid = run_system(signal="ma_crossover", fast=10, slow=50, seed=7)
    novel = run_system(signal="ma_crossover", fast=7, slow=33, seed=7)
    # The shipped grid has 7 configs; a config already in it is not double counted.
    assert in_grid.summary.n_effective_trials == 7
    # A novel selected config extends the grid (the multiplicity is honest).
    assert novel.summary.n_effective_trials == 8


@pytest.mark.integration
def test_run_system_auto_source_falls_back_to_synthetic_offline() -> None:
    """``data_source_pref='auto'`` is offline-safe: it resolves to synthetic in CI."""
    run = run_system(signal="ma_crossover", fast=10, slow=50, data_source_pref="auto", seed=7)
    # No Polygon key / no network in CI -> the loader falls back to synthetic bars.
    assert run.summary.data_source == "synthetic"
    assert run.summary.backtest_live_parity_max_diff <= PARITY_TOL


@pytest.mark.integration
@pytest.mark.parametrize("field", ["cost_bps", "slippage_bps"])
def test_run_system_rejects_negative_friction(field: str) -> None:
    """Negative transaction cost / slippage is rejected up front."""
    from algosystem._exceptions import ValidationError

    with pytest.raises(ValidationError):
        run_system(signal="ma_crossover", **{field: -1.0})  # type: ignore[arg-type]


@pytest.mark.integration
def test_run_system_rejects_fast_ge_slow() -> None:
    """``fast >= slow`` is an invalid MA-crossover config and is rejected."""
    from algosystem._exceptions import ValidationError

    with pytest.raises(ValidationError):
        run_system(signal="ma_crossover", fast=50, slow=10, seed=7)


@pytest.mark.integration
def test_run_system_rejects_unknown_signal() -> None:
    """An unknown signal name is rejected before any heavy work."""
    from algosystem._exceptions import ValidationError

    with pytest.raises(ValidationError, match="ma_crossover"):
        run_system(signal="bogus", seed=7)
