"""Property tests (Hypothesis) for the CAUSALITY contract of the signal library.

The load-bearing invariant: the position the signal emits at bar ``t`` reads ONLY
closed bars ``<= t``. Therefore perturbing the forming / future bars ``> t`` (or
appending entirely new future bars) MUST NOT change any position at or before
``t``. This is the leakage catch at the signal layer — if a signal peeked at a
future bar, this property would fail. Also checks the output contract (length,
dtype, value range, warm-up flatness) holds for arbitrary causal price paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from algosystem.signals.library import flat, ma_crossover, momentum

# Strictly-positive close-price paths: exp of bounded i.i.d. log-returns anchored
# at a positive level (mirrors the synthetic generators' close process), so the
# series is finite and positive for any draw.
_LOG_RETURN = st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False)


@st.composite
def _close_paths(draw: st.DrawFn, min_size: int = 5, max_size: int = 256) -> pd.Series:
    """Draw a strictly-positive close-price Series of a Hypothesis-chosen length."""
    log_returns = draw(st.lists(_LOG_RETURN, min_size=min_size, max_size=max_size))
    levels = 100.0 * np.exp(np.cumsum(np.asarray(log_returns, dtype="float64")))
    return pd.Series(levels, dtype="float64")


# --------------------------------------------------------------------------- #
# Causality: perturbing future / forming bars leaves the prefix unchanged      #
# --------------------------------------------------------------------------- #
@pytest.mark.property
@given(
    close=_close_paths(min_size=30),
    cut=st.floats(min_value=0.1, max_value=0.9),
    shock=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
    fast=st.integers(min_value=1, max_value=8),
    extra=st.integers(min_value=2, max_value=18),
)
@settings(max_examples=80, deadline=None)
def test_ma_crossover_is_strictly_causal(
    close: pd.Series, cut: float, shock: float, fast: int, extra: int
) -> None:
    """Perturbing bars ``> t`` never changes ma_crossover's position at/before ``t``."""
    slow = fast + extra
    t = max(int(cut * close.size), 0)
    base = ma_crossover(close, fast=fast, slow=slow)

    perturbed = close.copy()
    perturbed.iloc[t + 1 :] = perturbed.iloc[t + 1 :] * shock  # forming/future bars only.
    after = ma_crossover(perturbed, fast=fast, slow=slow)

    # The position at every bar <= t depends only on closed bars <= t.
    np.testing.assert_array_equal(base[: t + 1], after[: t + 1])


@pytest.mark.property
@given(
    close=_close_paths(min_size=30),
    cut=st.floats(min_value=0.1, max_value=0.9),
    shock=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
    lookback=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=80, deadline=None)
def test_momentum_is_strictly_causal(
    close: pd.Series, cut: float, shock: float, lookback: int
) -> None:
    """Perturbing bars ``> t`` never changes momentum's position at/before ``t``."""
    t = max(int(cut * close.size), 0)
    base = momentum(close, lookback=lookback)

    perturbed = close.copy()
    perturbed.iloc[t + 1 :] = perturbed.iloc[t + 1 :] * shock
    after = momentum(perturbed, lookback=lookback)

    np.testing.assert_array_equal(base[: t + 1], after[: t + 1])


@pytest.mark.property
@given(
    close=_close_paths(min_size=30),
    n_future=st.integers(min_value=1, max_value=40),
    fast=st.integers(min_value=1, max_value=8),
    extra=st.integers(min_value=2, max_value=18),
    lookback=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=60, deadline=None)
def test_appending_future_bars_preserves_the_prefix(
    close: pd.Series, n_future: int, fast: int, extra: int, lookback: int
) -> None:
    """Appending entirely new future bars leaves every existing position unchanged."""
    slow = fast + extra
    extended = pd.concat(
        [close, close.iloc[-1] * pd.Series(np.linspace(1.01, 1.5, n_future))],
        ignore_index=True,
    ).astype("float64")

    n = close.size
    np.testing.assert_array_equal(
        ma_crossover(close, fast=fast, slow=slow),
        ma_crossover(extended, fast=fast, slow=slow)[:n],
    )
    np.testing.assert_array_equal(
        momentum(close, lookback=lookback),
        momentum(extended, lookback=lookback)[:n],
    )


# --------------------------------------------------------------------------- #
# Output contract for arbitrary causal paths                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.property
@given(
    close=_close_paths(),
    fast=st.integers(min_value=1, max_value=8),
    extra=st.integers(min_value=2, max_value=18),
    lookback=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=80, deadline=None)
def test_output_contract_holds_for_any_path(
    close: pd.Series, fast: int, extra: int, lookback: int
) -> None:
    """Every signal returns a same-length float64 vector valued in {-1, 0, +1}."""
    slow = fast + extra
    for pos in (
        ma_crossover(close, fast=fast, slow=slow),
        momentum(close, lookback=lookback),
        flat(close),
    ):
        assert pos.dtype == np.float64
        assert pos.shape == (close.size,)
        assert set(np.unique(pos)).issubset({-1.0, 0.0, 1.0})


@pytest.mark.property
@given(
    close=_close_paths(min_size=25),
    fast=st.integers(min_value=1, max_value=6),
    extra=st.integers(min_value=2, max_value=15),
    lookback=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=60, deadline=None)
def test_warmup_region_is_flat(close: pd.Series, fast: int, extra: int, lookback: int) -> None:
    """Bars before the window/lookback fills emit a zero (flat) position."""
    slow = fast + extra
    ma = ma_crossover(close, fast=fast, slow=slow)
    # ma_crossover is flat until the slow window is full (first ``slow - 1`` bars).
    assert np.all(ma[: slow - 1] == 0.0)

    mom = momentum(close, lookback=lookback)
    # momentum is flat until a full lookback window exists (first ``lookback`` bars).
    assert np.all(mom[:lookback] == 0.0)
