"""Unit tests for the causal signal library (:mod:`algosystem.signals.library`).

Covers the sign logic of each signal (``ma_crossover`` is long when the fast SMA
is above the slow SMA, short otherwise; ``momentum`` is long when the trailing
return is positive, short otherwise; ``flat`` is the zero baseline), the warm-up
flat region (no position before the window fills), the validation guards
(``fast >= 1``, ``slow > fast``, ``lookback >= 1``), output shape/dtype/range,
determinism, the ``SignalSpec`` config, and the ``build_signal`` dispatcher —
all without any network or heavy dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algosystem._exceptions import ValidationError
from algosystem.signals.library import (
    SignalSpec,
    build_signal,
    flat,
    ma_crossover,
    momentum,
)


def _ramp(start: float, stop: float, n: int) -> pd.Series:
    """A strictly-positive linear price ramp of length ``n``."""
    return pd.Series(np.linspace(start, stop, n), dtype="float64")


def _up_then_down(n_each: int = 60) -> pd.Series:
    """A clean up-trend followed by a clean down-trend (for sign logic tests)."""
    up = np.linspace(100.0, 150.0, n_each)
    down = np.linspace(150.0, 100.0, n_each)
    return pd.Series(np.concatenate([up, down]), dtype="float64")


# --------------------------------------------------------------------------- #
# Output contract (shape / dtype / range)                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_signals_return_float64_same_length_and_in_range() -> None:
    """Each signal returns a float64 vector the same length as ``close``, in {-1,0,1}."""
    close = _up_then_down()
    for pos in (
        ma_crossover(close, fast=5, slow=20),
        momentum(close, lookback=10),
        flat(close),
    ):
        assert isinstance(pos, np.ndarray)
        assert pos.dtype == np.float64
        assert pos.shape == (close.size,)
        assert set(np.unique(pos)).issubset({-1.0, 0.0, 1.0})


# --------------------------------------------------------------------------- #
# ma_crossover sign logic                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_ma_crossover_warmup_is_flat_until_slow_window_fills() -> None:
    """Bars before the slow window is full emit a zero (flat) position."""
    close = _ramp(100.0, 200.0, 100)
    pos = ma_crossover(close, fast=5, slow=20)
    # The slow SMA needs 20 observations: indices 0..18 are warm-up (flat).
    assert np.all(pos[:19] == 0.0)
    # From the first fully-formed slow window onward, a position is taken.
    assert np.all(pos[19:] != 0.0)


@pytest.mark.unit
def test_ma_crossover_long_in_uptrend_short_in_downtrend() -> None:
    """Long (+1) when the fast SMA leads the slow SMA (up-trend), short (-1) otherwise."""
    close = _up_then_down()
    pos = ma_crossover(close, fast=5, slow=20)
    # Deep into the up-trend the fast SMA is above the slow SMA -> long.
    assert pos[55] == 1.0
    # Deep into the down-trend the fast SMA is below the slow SMA -> short.
    assert pos[110] == -1.0


@pytest.mark.unit
def test_ma_crossover_equal_averages_emit_short_not_long() -> None:
    """A flat price (fast SMA == slow SMA) takes the non-strict branch (short)."""
    close = pd.Series(np.full(60, 100.0), dtype="float64")
    pos = ma_crossover(close, fast=5, slow=20)
    # Averages coincide everywhere; ``fast > slow`` is false, so it is short.
    assert np.all(pos[19:] == -1.0)


@pytest.mark.unit
def test_ma_crossover_matches_explicit_rolling_reference() -> None:
    """The +1/-1 output equals an independent rolling-mean crossover reference."""
    close = _up_then_down()
    fast, slow = 5, 20
    sma_fast = close.rolling(fast, min_periods=fast).mean().to_numpy()
    sma_slow = close.rolling(slow, min_periods=slow).mean().to_numpy()
    ready = np.isfinite(sma_fast) & np.isfinite(sma_slow)
    expected = np.zeros(close.size, dtype="float64")
    expected[ready & (sma_fast > sma_slow)] = 1.0
    expected[ready & (sma_fast <= sma_slow)] = -1.0
    np.testing.assert_array_equal(ma_crossover(close, fast=fast, slow=slow), expected)


@pytest.mark.unit
def test_ma_crossover_fast_one_slow_two_minimal_windows() -> None:
    """The smallest valid windows (fast=1, slow=2) score from the second bar on."""
    close = pd.Series([100.0, 101.0, 102.0, 101.0, 100.0], dtype="float64")
    pos = ma_crossover(close, fast=1, slow=2)
    assert pos[0] == 0.0  # slow window (2) not yet full at the first bar.
    assert pos.shape == (5,)


@pytest.mark.unit
def test_ma_crossover_rejects_bad_windows() -> None:
    """fast < 1 and slow <= fast are rejected with ValidationError."""
    close = _ramp(100.0, 110.0, 30)
    with pytest.raises(ValidationError, match="fast must be >= 1"):
        ma_crossover(close, fast=0, slow=10)
    with pytest.raises(ValidationError, match=r"slow .* must be > fast"):
        ma_crossover(close, fast=10, slow=10)
    with pytest.raises(ValidationError, match=r"slow .* must be > fast"):
        ma_crossover(close, fast=10, slow=5)


# --------------------------------------------------------------------------- #
# momentum sign logic                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_momentum_warmup_is_flat_until_lookback_full() -> None:
    """Bars before a full lookback window emit a zero (flat) position."""
    close = _ramp(100.0, 200.0, 100)
    pos = momentum(close, lookback=10)
    assert np.all(pos[:10] == 0.0)  # indices 0..9 have no full lookback yet.
    assert np.all(pos[10:] != 0.0)


@pytest.mark.unit
def test_momentum_long_when_trailing_return_positive() -> None:
    """Long (+1) when the trailing lookback return is positive, short (-1) otherwise."""
    close = _up_then_down()
    pos = momentum(close, lookback=10)
    assert pos[40] == 1.0  # up-trend: positive trailing return.
    assert pos[110] == -1.0  # down-trend: negative trailing return.


@pytest.mark.unit
def test_momentum_zero_trailing_return_emits_short() -> None:
    """A flat path (zero trailing return) takes the non-strict branch (short)."""
    close = pd.Series(np.full(40, 100.0), dtype="float64")
    pos = momentum(close, lookback=10)
    assert np.all(pos[10:] == -1.0)


@pytest.mark.unit
def test_momentum_matches_explicit_trailing_return_reference() -> None:
    """The +1/-1 output equals an independent trailing-return-sign reference."""
    close = _up_then_down()
    lookback = 10
    arr = close.to_numpy(dtype="float64")
    expected = np.zeros(arr.size, dtype="float64")
    trailing = arr[lookback:] / arr[:-lookback] - 1.0
    expected[lookback:] = np.where(trailing > 0.0, 1.0, -1.0)
    np.testing.assert_array_equal(momentum(close, lookback=lookback), expected)


@pytest.mark.unit
def test_momentum_all_flat_when_path_shorter_than_lookback() -> None:
    """If no full lookback window exists, every bar stays flat (no IndexError)."""
    close = _ramp(100.0, 110.0, 5)
    pos = momentum(close, lookback=10)
    assert pos.shape == (5,)
    assert np.all(pos == 0.0)
    # The boundary n == lookback also yields no scored bar.
    close_eq = _ramp(100.0, 110.0, 10)
    assert np.all(momentum(close_eq, lookback=10) == 0.0)


@pytest.mark.unit
def test_momentum_rejects_bad_lookback() -> None:
    """lookback < 1 is rejected with ValidationError."""
    close = _ramp(100.0, 110.0, 30)
    with pytest.raises(ValidationError, match="lookback must be >= 1"):
        momentum(close, lookback=0)


# --------------------------------------------------------------------------- #
# flat baseline                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_flat_is_all_zero() -> None:
    """The flat baseline emits a zero position at every bar (the zero-edge floor)."""
    close = _up_then_down()
    pos = flat(close)
    assert pos.shape == (close.size,)
    assert np.all(pos == 0.0)


# --------------------------------------------------------------------------- #
# Determinism                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_signals_are_deterministic() -> None:
    """Re-evaluating the same signal on the same input is byte-identical."""
    close = _up_then_down()
    np.testing.assert_array_equal(
        ma_crossover(close, fast=5, slow=20), ma_crossover(close, fast=5, slow=20)
    )
    np.testing.assert_array_equal(momentum(close, lookback=10), momentum(close, lookback=10))
    np.testing.assert_array_equal(flat(close), flat(close))


@pytest.mark.unit
def test_signals_do_not_mutate_input() -> None:
    """Signals are pure: the caller's ``close`` series is never mutated."""
    close = _up_then_down()
    snapshot = close.copy()
    ma_crossover(close, fast=5, slow=20)
    momentum(close, lookback=10)
    flat(close)
    pd.testing.assert_series_equal(close, snapshot)


@pytest.mark.unit
def test_signals_accept_ndarray_input() -> None:
    """A 1-D ndarray close (coerced at the boundary) yields the same result as a Series.

    The signal signatures declare ``close: pd.Series`` for documentation, but every
    body funnels through ``ensure_series`` which also accepts a 1-D ndarray; the
    ``type: ignore`` flags the intentional ndarray-at-the-boundary call.
    """
    arr = _up_then_down().to_numpy(dtype="float64")
    series = pd.Series(arr)
    np.testing.assert_array_equal(
        ma_crossover(arr, fast=5, slow=20),  # type: ignore[arg-type]
        ma_crossover(series, fast=5, slow=20),
    )
    np.testing.assert_array_equal(
        momentum(arr, lookback=10),  # type: ignore[arg-type]
        momentum(series, lookback=10),
    )


# --------------------------------------------------------------------------- #
# SignalSpec config                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_signal_spec_is_frozen_and_validates_name() -> None:
    """SignalSpec is immutable and rejects an unknown signal name."""
    spec = SignalSpec("ma_crossover", {"fast": 10, "slow": 50})
    with pytest.raises(AttributeError):
        spec.name = "momentum"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="unknown signal"):
        SignalSpec("not_a_signal")


@pytest.mark.unit
def test_signal_spec_to_dict_round_trips() -> None:
    """SignalSpec.to_dict emits a plain, JSON-serializable mapping."""
    spec = SignalSpec("momentum", {"lookback": 20})
    payload = spec.to_dict()
    assert payload == {"name": "momentum", "params": {"lookback": 20}}


@pytest.mark.unit
def test_signal_spec_default_params_is_empty_mapping() -> None:
    """A spec without params defaults to an empty parameter mapping (e.g. flat)."""
    spec = SignalSpec("flat")
    assert spec.params == {}


# --------------------------------------------------------------------------- #
# build_signal dispatcher                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_build_signal_dispatches_to_each_signal() -> None:
    """build_signal routes each spec to its function with the configured params."""
    close = _up_then_down()
    np.testing.assert_array_equal(
        build_signal(SignalSpec("ma_crossover", {"fast": 5, "slow": 20}), close),
        ma_crossover(close, fast=5, slow=20),
    )
    np.testing.assert_array_equal(
        build_signal(SignalSpec("momentum", {"lookback": 10}), close),
        momentum(close, lookback=10),
    )
    np.testing.assert_array_equal(build_signal(SignalSpec("flat"), close), flat(close))


@pytest.mark.unit
def test_build_signal_uses_function_defaults_when_params_empty() -> None:
    """An ma_crossover spec with no params uses the function's default windows."""
    close = _up_then_down()
    np.testing.assert_array_equal(
        build_signal(SignalSpec("ma_crossover"), close),
        ma_crossover(close),
    )


@pytest.mark.unit
def test_build_signal_rejects_params_on_flat() -> None:
    """A 'flat' spec carrying parameters is rejected (flat takes none)."""
    close = _ramp(100.0, 110.0, 30)
    spec = SignalSpec("flat")
    object.__setattr__(spec, "params", {"lookback": 5})  # 'flat' takes no params.
    with pytest.raises(ValidationError, match="'flat' takes no parameters"):
        build_signal(spec, close)


@pytest.mark.unit
def test_build_signal_rejects_externally_corrupted_spec() -> None:
    """A frozen-spec bypass with an unknown name hits the defensive dispatch guard."""
    spec = SignalSpec("flat")
    object.__setattr__(spec, "name", "mystery")  # bypass the frozen guard.
    close = _ramp(100.0, 110.0, 30)
    with pytest.raises(ValidationError, match="unknown signal"):
        build_signal(spec, close)
