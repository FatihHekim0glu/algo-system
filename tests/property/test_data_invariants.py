"""Property tests (Hypothesis) for the synthetic OHLC generators.

For arbitrary ``(seed, n_obs)`` drawn by Hypothesis, every generated bar panel
must satisfy the intrabar OHLC invariants (``low <= {open, close} <= high`` with
strictly-positive, finite prices) and must reproduce byte-for-byte under the same
seed (no dependence on the global RNG state). These invariants are what the
backtest / execution / parity layers downstream rely on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from algosystem.data.synthetic import (
    assert_ohlc_invariants,
    gbm_regime_bars,
    learnable_trend_bars,
    pure_noise_bars,
)

_GENERATORS = (gbm_regime_bars, learnable_trend_bars, pure_noise_bars)


@pytest.mark.property
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_obs=st.integers(min_value=2, max_value=400),
    which=st.integers(min_value=0, max_value=2),
)
@settings(max_examples=60, deadline=None)
def test_ohlc_invariants_hold_for_any_seed_and_size(seed: int, n_obs: int, which: int) -> None:
    """Any (seed, n_obs) yields a panel that satisfies the OHLC invariants."""
    path = _GENERATORS[which](n_obs=n_obs, seed=seed)
    assert_ohlc_invariants(path.bars)  # raises on any violation.
    assert path.bars.shape == (n_obs, 4)
    assert len(path.regime_labels) == n_obs


@pytest.mark.property
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_obs=st.integers(min_value=2, max_value=300),
    which=st.integers(min_value=0, max_value=2),
)
@settings(max_examples=40, deadline=None)
def test_generation_is_reproducible(seed: int, n_obs: int, which: int) -> None:
    """The same (seed, n_obs) reproduces the bars byte-for-byte (seeded RNG)."""
    generate = _GENERATORS[which]
    first = generate(n_obs=n_obs, seed=seed)
    second = generate(n_obs=n_obs, seed=seed)
    pd.testing.assert_frame_equal(first.bars, second.bars)
    assert first.regime_labels == second.regime_labels


@pytest.mark.property
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_obs=st.integers(min_value=2, max_value=300),
    which=st.integers(min_value=0, max_value=2),
)
@settings(max_examples=40, deadline=None)
def test_prices_are_finite_and_positive(seed: int, n_obs: int, which: int) -> None:
    """Every price in every generated panel is finite and strictly positive."""
    path = _GENERATORS[which](n_obs=n_obs, seed=seed)
    values = path.bars.to_numpy(dtype="float64")
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)
