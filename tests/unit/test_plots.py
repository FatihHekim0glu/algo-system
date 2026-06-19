"""Tests for the LAZY-plotly figure builders (equity overlay + drawdown).

These pin the figure builders' JSON contract — every builder returns a plain
``{"data": [...], "layout": {...}}`` mapping (no numpy scalars, no Plotly classes)
the FastAPI layer serializes and the frontend ``PlotlyChart`` renders — plus the
overlay semantics (backtest + paper-broker live + buy-hold, the live curve dashed
ON TOP so the two COINCIDE, proving parity), the drawdown accounting (it mirrors
``evaluation.metrics.max_drawdown``), and the ``ValidationError`` boundary on
empty / length-mismatched / non-finite input.

Importing :mod:`algosystem.plots` must NOT pull in Plotly (it is imported lazily
inside each builder); a subprocess import-purity check enforces this.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from algosystem._exceptions import ValidationError
from algosystem.evaluation.metrics import max_drawdown
from algosystem.plots import drawdown_figure, equity_overlay_figure


def _is_json_safe(obj: object) -> bool:
    """Return ``True`` iff ``obj`` round-trips through ``json.dumps`` unchanged-shape."""
    try:
        json.dumps(obj)
    except (TypeError, ValueError):
        return False
    return True


# --------------------------------------------------------------------------- #
# equity_overlay_figure: shape, traces, and the parity-overlay semantics        #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_equity_overlay_returns_plain_data_layout_dict() -> None:
    """The builder returns a plain ``{data, layout}`` JSON-safe mapping."""
    bt = np.array([1.0, 1.01, 1.02, 1.0])
    live = bt.copy()
    bh = np.array([1.0, 1.005, 1.01, 1.012])
    fig = equity_overlay_figure(bt, live, bh)

    assert set(fig) == {"data", "layout"}
    assert isinstance(fig["data"], list)
    assert isinstance(fig["layout"], dict)
    assert _is_json_safe(fig)


@pytest.mark.unit
def test_equity_overlay_has_three_named_traces() -> None:
    """Backtest, live (paper broker), and buy-hold are the three overlaid traces."""
    bt = np.array([1.0, 1.02, 1.05])
    fig = equity_overlay_figure(bt, bt.copy(), np.array([1.0, 1.01, 1.02]))
    names = [trace["name"] for trace in fig["data"]]
    assert names == ["Backtest", "Live (paper broker)", "Buy & hold"]


@pytest.mark.unit
def test_equity_overlay_live_curve_is_dashed_on_top() -> None:
    """The live (paper-broker) curve is dashed so it visibly tracks over the backtest."""
    bt = np.array([1.0, 1.02, 1.05])
    fig = equity_overlay_figure(bt, bt.copy(), np.array([1.0, 1.01, 1.02]))
    live_trace = fig["data"][1]
    assert live_trace["name"] == "Live (paper broker)"
    assert live_trace["line"]["dash"] == "dash"


@pytest.mark.unit
def test_equity_overlay_y_values_round_trip_the_inputs() -> None:
    """The serialized y-values equal the input curves (no resampling / mangling)."""
    bt = np.array([1.0, 1.03, 0.98, 1.10])
    live = bt.copy()
    bh = np.array([1.0, 1.01, 1.02, 1.03])
    fig = equity_overlay_figure(bt, live, bh)
    np.testing.assert_allclose(fig["data"][0]["y"], bt)
    np.testing.assert_allclose(fig["data"][1]["y"], live)
    np.testing.assert_allclose(fig["data"][2]["y"], bh)


@pytest.mark.unit
def test_equity_overlay_coinciding_curves_overlap_exactly() -> None:
    """When backtest == live (the parity oracle), their serialized y-arrays coincide."""
    bt = np.array([1.0, 1.04, 1.02, 1.07, 1.05])
    fig = equity_overlay_figure(bt, bt.copy(), np.linspace(1.0, 1.1, bt.size))
    np.testing.assert_array_equal(
        np.asarray(fig["data"][0]["y"]), np.asarray(fig["data"][1]["y"])
    )


@pytest.mark.unit
def test_equity_overlay_custom_title_is_applied() -> None:
    """A custom title flows into the serialized layout."""
    bt = np.array([1.0, 1.01])
    fig = equity_overlay_figure(bt, bt.copy(), bt.copy(), title="Custom equity title")
    # Plotly serializes the title as either a string or a ``{"text": ...}`` mapping.
    title = fig["layout"]["title"]
    text = title["text"] if isinstance(title, dict) else title
    assert text == "Custom equity title"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("bt", "live", "bh"),
    [
        (np.array([]), np.array([]), np.array([])),
        (np.array([1.0, np.nan]), np.array([1.0, 1.0]), np.array([1.0, 1.0])),
        (np.array([1.0, np.inf]), np.array([1.0, 1.0]), np.array([1.0, 1.0])),
    ],
)
def test_equity_overlay_rejects_empty_or_nonfinite(
    bt: np.ndarray, live: np.ndarray, bh: np.ndarray
) -> None:
    """Empty / NaN / Inf curves are rejected up front with a ``ValidationError``."""
    with pytest.raises(ValidationError):
        equity_overlay_figure(bt, live, bh)


@pytest.mark.unit
def test_equity_overlay_rejects_length_mismatch() -> None:
    """Curves of unequal length cannot be overlaid; the builder rejects them."""
    with pytest.raises(ValidationError, match="same length"):
        equity_overlay_figure(
            np.array([1.0, 1.1, 1.2]), np.array([1.0, 1.1]), np.array([1.0, 1.1, 1.2])
        )


# --------------------------------------------------------------------------- #
# drawdown_figure: shape, accounting parity with metrics.max_drawdown           #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_drawdown_returns_plain_data_layout_dict() -> None:
    """The drawdown builder returns a plain, JSON-safe ``{data, layout}`` mapping."""
    fig = drawdown_figure(np.array([0.0, -0.05, 0.02, -0.01]))
    assert set(fig) == {"data", "layout"}
    assert isinstance(fig["data"], list) and len(fig["data"]) == 1
    assert _is_json_safe(fig)


@pytest.mark.unit
def test_drawdown_is_a_filled_area_below_zero() -> None:
    """The drawdown trace fills to zero (the depth-of-pain area view)."""
    fig = drawdown_figure(np.array([0.0, -0.05, -0.10, 0.02]))
    trace = fig["data"][0]
    assert trace["fill"] == "tozeroy"
    assert (np.asarray(trace["y"]) <= 1e-12).all()  # drawdown is non-positive.


@pytest.mark.unit
def test_drawdown_trough_matches_metrics_max_drawdown() -> None:
    """The serialized drawdown trough equals ``evaluation.metrics.max_drawdown``."""
    rng = np.random.default_rng(5)
    net = rng.standard_normal(200) * 0.01
    fig = drawdown_figure(net)
    trough = float(np.min(np.asarray(fig["data"][0]["y"])))
    assert trough == pytest.approx(max_drawdown(net), abs=1e-12)


@pytest.mark.unit
def test_drawdown_zero_returns_yield_flat_zero_curve() -> None:
    """A never-declining (all-zero) series has a flat-zero drawdown curve."""
    fig = drawdown_figure(np.zeros(10))
    np.testing.assert_allclose(np.asarray(fig["data"][0]["y"]), np.zeros(10), atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    "net",
    [np.array([]), np.array([0.0, np.nan]), np.array([0.0, np.inf])],
)
def test_drawdown_rejects_empty_or_nonfinite(net: np.ndarray) -> None:
    """Empty / NaN / Inf return series are rejected with a ``ValidationError``."""
    with pytest.raises(ValidationError):
        drawdown_figure(net)


@pytest.mark.unit
def test_drawdown_custom_title_is_applied() -> None:
    """A custom title flows into the serialized layout."""
    fig = drawdown_figure(np.array([0.0, -0.02]), title="My drawdown")
    title = fig["layout"]["title"]
    text = title["text"] if isinstance(title, dict) else title
    assert text == "My drawdown"


# --------------------------------------------------------------------------- #
# Import purity: importing algosystem.plots pulls in NO plotly (lazy import)     #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_importing_plots_does_not_import_plotly() -> None:
    """A fresh interpreter importing ``algosystem.plots`` does not load Plotly."""
    code = (
        "import sys\n"
        "import algosystem.plots\n"
        "assert 'plotly' not in sys.modules, 'plotly imported at module load'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout
