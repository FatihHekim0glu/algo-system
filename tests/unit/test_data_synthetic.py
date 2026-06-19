"""Unit tests for the synthetic OHLC bar generators (:mod:`algosystem.data.synthetic`).

Covers determinism (same ``(seed, n_obs)`` -> byte-identical bars; different seed
-> different bars), the per-bar OHLC intrabar invariants (``low <= {open, close}
<= high`` with strictly-positive prices), the validation guards, the regime-label
contract, and the ``BarPath.to_dict`` metadata projection — all without any
network, heavy dependency, or import-time side effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algosystem._exceptions import ValidationError
from algosystem.data.synthetic import (
    DEFAULT_N_REGIMES,
    START_PRICE,
    BarPath,
    assert_ohlc_invariants,
    gbm_regime_bars,
    learnable_trend_bars,
    pure_noise_bars,
    regime_trend_bars,
)

_GENERATORS = (gbm_regime_bars, learnable_trend_bars, regime_trend_bars, pure_noise_bars)
_KINDS = {
    gbm_regime_bars: "gbm_regime",
    learnable_trend_bars: "learnable_trend",
    regime_trend_bars: "regime_trend",
    pure_noise_bars: "pure_noise",
}


# --------------------------------------------------------------------------- #
# Shape / structure                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("generate", _GENERATORS)
def test_shape_columns_and_kind(generate: object) -> None:
    """Every generator yields an (n_obs, 4) OHLC panel tagged with its kind."""
    path = generate(n_obs=256, seed=7)  # type: ignore[operator]
    assert isinstance(path, BarPath)
    assert path.kind == _KINDS[generate]  # type: ignore[index]
    assert list(path.bars.columns) == ["open", "high", "low", "close"]
    assert path.bars.shape == (256, 4)
    assert len(path.regime_labels) == 256
    assert isinstance(path.bars.index, pd.DatetimeIndex)


@pytest.mark.unit
@pytest.mark.parametrize("generate", _GENERATORS)
def test_anchored_at_start_price(generate: object) -> None:
    """The first bar's close anchors at START_PRICE (the strictly-positive anchor)."""
    path = generate(n_obs=64, seed=3)  # type: ignore[operator]
    assert path.bars["close"].iloc[0] == pytest.approx(START_PRICE)
    # First open equals the first close (gapless anchor convention).
    assert path.bars["open"].iloc[0] == pytest.approx(path.bars["close"].iloc[0])


# --------------------------------------------------------------------------- #
# Determinism                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("generate", _GENERATORS)
def test_same_seed_is_byte_identical(generate: object) -> None:
    """A given (seed, n_obs) reproduces the bars byte-for-byte (no global RNG)."""
    a = generate(n_obs=200, seed=11)  # type: ignore[operator]
    b = generate(n_obs=200, seed=11)  # type: ignore[operator]
    pd.testing.assert_frame_equal(a.bars, b.bars)
    assert a.regime_labels == b.regime_labels


@pytest.mark.unit
@pytest.mark.parametrize("generate", _GENERATORS)
def test_different_seed_changes_bars(generate: object) -> None:
    """A different seed produces a different path (the RNG actually drives it)."""
    a = generate(n_obs=200, seed=11)  # type: ignore[operator]
    c = generate(n_obs=200, seed=12)  # type: ignore[operator]
    assert not a.bars["close"].equals(c.bars["close"])


@pytest.mark.unit
def test_prefix_consistency_across_n_obs() -> None:
    """A longer request shares no surprising structure: still deterministic per n_obs."""
    short = gbm_regime_bars(n_obs=100, seed=5)
    short_again = gbm_regime_bars(n_obs=100, seed=5)
    pd.testing.assert_frame_equal(short.bars, short_again.bars)


# --------------------------------------------------------------------------- #
# OHLC invariants                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("generate", _GENERATORS)
def test_ohlc_invariants_hold(generate: object) -> None:
    """low <= {open, close} <= high, high >= low, all strictly positive."""
    path = generate(n_obs=1000, seed=7)  # type: ignore[operator]
    assert_ohlc_invariants(path.bars)  # raises on violation.

    o = path.bars["open"].to_numpy()
    h = path.bars["high"].to_numpy()
    low = path.bars["low"].to_numpy()
    c = path.bars["close"].to_numpy()
    assert np.all(h >= o) and np.all(h >= c)
    assert np.all(low <= o) and np.all(low <= c)
    assert np.all(h >= low)
    assert np.all((o > 0) & (h > 0) & (low > 0) & (c > 0))


@pytest.mark.unit
def test_assert_ohlc_invariants_rejects_missing_columns() -> None:
    """A frame missing an OHLC column is rejected with ValidationError."""
    bad = pd.DataFrame({"open": [1.0], "high": [2.0], "close": [1.5]})
    with pytest.raises(ValidationError, match="open/high/low/close"):
        assert_ohlc_invariants(bad)


@pytest.mark.unit
def test_assert_ohlc_invariants_rejects_high_below_close() -> None:
    """A high below the close (an impossible bar) is rejected."""
    bad = pd.DataFrame(
        {"open": [1.0], "high": [1.2], "low": [0.9], "close": [1.5]}  # close > high.
    )
    with pytest.raises(ValidationError, match="high must be"):
        assert_ohlc_invariants(bad)


@pytest.mark.unit
def test_assert_ohlc_invariants_rejects_nonpositive_price() -> None:
    """A non-positive price is rejected (prices are strictly positive by contract)."""
    bad = pd.DataFrame(
        {"open": [1.0], "high": [1.2], "low": [-0.1], "close": [1.1]}  # negative low.
    )
    with pytest.raises(ValidationError, match="positive"):
        assert_ohlc_invariants(bad)


@pytest.mark.unit
def test_assert_ohlc_invariants_rejects_nonfinite_price() -> None:
    """A non-finite price (NaN/inf) is rejected before any ordering check."""
    bad = pd.DataFrame(
        {"open": [1.0], "high": [np.inf], "low": [0.9], "close": [1.1]}  # inf high.
    )
    with pytest.raises(ValidationError, match="non-finite"):
        assert_ohlc_invariants(bad)


@pytest.mark.unit
def test_assert_ohlc_invariants_rejects_high_below_low() -> None:
    """A high below the low (a crossed bar) is rejected."""
    bad = pd.DataFrame(
        {"open": [1.0], "high": [0.8], "low": [1.1], "close": [1.0]}  # high < low.
    )
    with pytest.raises(ValidationError, match="high must be >= low"):
        assert_ohlc_invariants(bad)


@pytest.mark.unit
def test_assert_ohlc_invariants_rejects_low_above_open() -> None:
    """A low above the open (an impossible bar) is rejected."""
    bad = pd.DataFrame(
        {"open": [0.5], "high": [1.5], "low": [0.9], "close": [1.1]}  # low > open.
    )
    with pytest.raises(ValidationError, match="low must be"):
        assert_ohlc_invariants(bad)


# --------------------------------------------------------------------------- #
# Regime labels                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_gbm_regime_labels_in_range_and_switch() -> None:
    """The sticky chain visits >1 regime over a long path and stays in [0, n_regimes)."""
    path = gbm_regime_bars(n_obs=2000, seed=7, n_regimes=DEFAULT_N_REGIMES)
    labels = np.asarray(path.regime_labels)
    assert labels.min() >= 0 and labels.max() < DEFAULT_N_REGIMES
    assert len(set(path.regime_labels)) > 1  # the chain actually switches.


@pytest.mark.unit
def test_single_regime_path_has_one_label() -> None:
    """A 1-regime GBM has a single nominal regime label everywhere."""
    path = gbm_regime_bars(n_obs=500, seed=7, n_regimes=1)
    assert set(path.regime_labels) == {0}


@pytest.mark.unit
@pytest.mark.parametrize("generate", (learnable_trend_bars, pure_noise_bars))
def test_nonregime_paths_have_single_nominal_regime(generate: object) -> None:
    """The non-regime processes carry a single nominal regime label."""
    path = generate(n_obs=300, seed=7)  # type: ignore[operator]
    assert set(path.regime_labels) == {0}


@pytest.mark.unit
def test_regime_trend_labels_alternate_by_block() -> None:
    """regime_trend_bars flips its directional regime label every ``block`` bars."""
    block = 50
    path = regime_trend_bars(n_obs=200, seed=7, block=block)
    labels = path.regime_labels
    assert set(labels) == {0, 1}  # alternating up / down blocks.
    # The first block is the up regime (label 0), the second the down regime (1).
    assert labels[0] == 0
    assert labels[block] == 1
    assert labels[2 * block] == 0


@pytest.mark.unit
def test_regime_trend_rejects_bad_block_and_negative_vol() -> None:
    """regime_trend_bars rejects block < 1, negative vol, and negative intrabar range."""
    with pytest.raises(ValidationError, match="block"):
        regime_trend_bars(n_obs=100, block=0)
    with pytest.raises(ValidationError, match="vol"):
        regime_trend_bars(n_obs=100, vol=-0.01)
    with pytest.raises(ValidationError, match="intrabar_range_bps"):
        regime_trend_bars(n_obs=100, intrabar_range_bps=-1.0)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("generate", _GENERATORS)
def test_rejects_too_few_obs(generate: object) -> None:
    """n_obs < 2 is rejected (not a single causal step could be formed)."""
    with pytest.raises(ValidationError, match="n_obs"):
        generate(n_obs=1, seed=7)  # type: ignore[operator]


@pytest.mark.unit
def test_gbm_rejects_bad_n_regimes_and_negative_vol() -> None:
    """gbm_regime_bars rejects n_regimes < 1 and negative volatilities/ranges."""
    with pytest.raises(ValidationError, match="n_regimes"):
        gbm_regime_bars(n_obs=10, n_regimes=0)
    with pytest.raises(ValidationError, match="base_vol"):
        gbm_regime_bars(n_obs=10, base_vol=-0.1)
    with pytest.raises(ValidationError, match="intrabar_range_bps"):
        gbm_regime_bars(n_obs=10, intrabar_range_bps=-1.0)
    with pytest.raises(ValidationError, match="microstructure_bps"):
        gbm_regime_bars(n_obs=10, microstructure_bps=-1.0)


@pytest.mark.unit
@pytest.mark.parametrize("generate", (learnable_trend_bars, pure_noise_bars))
def test_nonregime_reject_negative_vol(generate: object) -> None:
    """The non-regime generators reject negative volatility."""
    with pytest.raises(ValidationError, match="vol"):
        generate(n_obs=10, vol=-0.01)  # type: ignore[operator]


@pytest.mark.unit
@pytest.mark.parametrize("generate", (learnable_trend_bars, pure_noise_bars))
def test_nonregime_reject_negative_intrabar_range(generate: object) -> None:
    """The non-regime generators reject a negative intrabar range."""
    with pytest.raises(ValidationError, match="intrabar_range_bps"):
        generate(n_obs=10, intrabar_range_bps=-1.0)  # type: ignore[operator]


# --------------------------------------------------------------------------- #
# Metadata projection                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_to_dict_is_json_serializable_metadata() -> None:
    """BarPath.to_dict emits plain shape metadata + regime labels (no bar panel)."""
    path = pure_noise_bars(n_obs=50, seed=7)
    payload = path.to_dict()
    assert payload["n_obs"] == 50
    assert payload["kind"] == "pure_noise"
    assert payload["columns"] == ["open", "high", "low", "close"]
    assert all(isinstance(x, int) for x in payload["regime_labels"])
    assert "bars" not in payload
