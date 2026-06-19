"""Unit tests for the serve-time helper functions (the JSON-safe edge cases).

These pin the small, defensive helpers :func:`run_system` relies on so the deployed
response is always valid JSON regardless of a degenerate (flat / tiny) net-return
series: the per-obs Sharpe and the sample moments fall back gracefully, and the
``_safe_float`` clamp maps NaN / Inf / non-numeric to ``0.0`` so no non-finite scalar
ever crosses the API boundary. The verdict itself is computed from the RAW
statistics, so these clamps never soften the honest gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from algosystem._exceptions import ValidationError
from algosystem.serve import (
    _align_positions,
    _per_obs_sharpe,
    _safe_float,
    _sample_moments,
    _selected_spec,
)
from algosystem.signals.library import SignalSpec


@pytest.mark.unit
def test_per_obs_sharpe_handles_tiny_and_flat_series() -> None:
    """A < 2-element or numerically-flat series has an undefined Sharpe -> 0.0."""
    assert _per_obs_sharpe(np.array([0.01])) == 0.0  # single observation.
    assert _per_obs_sharpe(np.array([0.0, 0.0, 0.0])) == 0.0  # zero dispersion.
    # A non-degenerate series returns a finite, sane per-obs Sharpe.
    out = _per_obs_sharpe(np.array([0.01, -0.005, 0.02, 0.0, 0.015]))
    assert np.isfinite(out)


@pytest.mark.unit
def test_sample_moments_fall_back_to_gaussian_for_tiny_series() -> None:
    """A < 3-element or flat series falls back to the Gaussian (0.0, 3.0) moments."""
    assert _sample_moments(np.array([0.01, 0.02])) == (0.0, 3.0)  # too short.
    assert _sample_moments(np.array([0.0, 0.0, 0.0, 0.0])) == (0.0, 3.0)  # flat.
    # A real series returns finite skew + FULL (non-excess) kurtosis.
    skew, kurt = _sample_moments(np.array([0.01, -0.02, 0.03, -0.01, 0.04, 0.0]))
    assert np.isfinite(skew)
    assert np.isfinite(kurt)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.5, 1.5),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (float("-inf"), 0.0),
        (None, 0.0),
        ("not-a-number", 0.0),
        (np.float64(2.0), 2.0),
    ],
)
def test_safe_float_clamps_non_finite_to_zero(value: object, expected: float) -> None:
    """``_safe_float`` maps NaN / Inf / non-numeric to 0.0 and passes finite floats."""
    assert _safe_float(value) == expected


@pytest.mark.unit
def test_selected_spec_builds_the_right_signal() -> None:
    """``_selected_spec`` maps the request signal to its parametrized SignalSpec."""
    assert _selected_spec("ma_crossover", 10, 50, 20) == SignalSpec(
        "ma_crossover", {"fast": 10, "slow": 50}
    )
    assert _selected_spec("momentum", 10, 50, 20) == SignalSpec("momentum", {"lookback": 20})


@pytest.mark.unit
def test_selected_spec_rejects_unknown_signal() -> None:
    """An unknown signal name is rejected up front."""
    with pytest.raises(ValidationError, match="ma_crossover"):
        _selected_spec("bogus", 10, 50, 20)


@pytest.mark.unit
def test_align_positions_drops_the_leading_position() -> None:
    """Aligning N positions to N-1 returns drops the first (warm-up) position."""
    positions = np.array([0.0, 1.0, 1.0, -1.0, 0.0])
    aligned = _align_positions(positions, 4)
    np.testing.assert_array_equal(aligned, np.array([1.0, 1.0, -1.0, 0.0]))
