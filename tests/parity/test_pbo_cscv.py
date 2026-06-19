"""Validate the CSCV Probability-of-Backtest-Overfitting against references.

The Combinatorially-Symmetric Cross-Validation estimate (Bailey et al., 2017) is
pinned against:

- the EXACT combinatorial partition count :math:`\\binom{S}{S/2}` (the full set of
  symmetric IS/OOS splits, not a subsample);
- the closed-form relationship ``pbo == mean(logits <= 0)`` (the definition);
- the DISCRIMINATION the verdict relies on: a configuration that is genuinely best
  in every bar drives PBO -> 0, a pure-noise grid sits near 0.5, and an
  in-sample-overfit artifact (high IS Sharpe via outliers, poor OOS) drives
  PBO well above 0.5;
- the input-validation guards (2-D, ``N >= 2``, even ``n_splits >= 2``, enough bars,
  finite values).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from algosystem._exceptions import ValidationError
from algosystem.evaluation.pbo import PBOResult, probability_of_backtest_overfitting


@pytest.mark.parity
def test_pbo_partition_count_is_full_combinatorial_set() -> None:
    """The number of symmetric partitions equals C(S, S/2) exactly."""
    rng = np.random.default_rng(0)
    perf = rng.standard_normal((512, 6)) * 0.01
    for n_splits in (4, 6, 8, 10):
        res = probability_of_backtest_overfitting(perf, n_splits=n_splits)
        assert res.n_partitions == math.comb(n_splits, n_splits // 2)
        assert res.logits.size == res.n_partitions
        assert res.n_splits == n_splits
        assert res.n_configs == 6


@pytest.mark.parity
def test_pbo_equals_fraction_of_nonpositive_logits() -> None:
    """``pbo`` is exactly the fraction of partitions with ``lambda <= 0``."""
    rng = np.random.default_rng(1)
    perf = rng.standard_normal((1024, 10)) * 0.01
    res = probability_of_backtest_overfitting(perf, n_splits=10)
    assert isinstance(res, PBOResult)
    expected = float(np.mean(res.logits <= 0.0))
    assert res.pbo == pytest.approx(expected, abs=1e-12)
    assert 0.0 <= res.pbo <= 1.0
    assert np.isfinite(res.logits).all()  # omega in (0, 1) => finite logits.


@pytest.mark.parity
def test_pbo_is_low_for_a_genuinely_best_configuration() -> None:
    """A config dominant in EVERY bar (IS and OOS) is almost never overfit."""
    rng = np.random.default_rng(7)
    perf = rng.standard_normal((1024, 12)) * 0.01
    perf[:, 0] += 0.01  # config 0 dominates every bar in both halves.
    res = probability_of_backtest_overfitting(perf, n_splits=16)
    assert res.pbo < 0.05


@pytest.mark.parity
def test_pbo_is_near_one_half_for_pure_noise_on_average() -> None:
    """An i.i.d.-noise grid has no genuine best, so PBO averages near 0.5.

    A single CSCV draw on noise has large sampling variance (the IS-best config is
    chosen by chance and its OOS rank is near-uniform), so the honest, non-flaky
    statement is about the EXPECTATION: averaged over many independent noise grids
    the PBO concentrates around one half — the selection procedure is no better than
    a coin flip when there is nothing to select.
    """
    pbos = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        perf = rng.standard_normal((1024, 12)) * 0.01
        pbos.append(probability_of_backtest_overfitting(perf, n_splits=14).pbo)
    mean_pbo = float(np.mean(pbos))
    assert 0.4 < mean_pbo < 0.6


@pytest.mark.parity
def test_pbo_is_high_for_an_overfit_artifact() -> None:
    """A config that is IS-top via outliers but OOS-poor drives PBO well above 0.5."""
    rng = np.random.default_rng(123)
    t, n = 2048, 8
    perf = rng.standard_normal((t, n)) * 0.01
    # config 0: a negative central tendency (poor OOS Sharpe) with a handful of huge
    # positive outliers that pump whichever in-sample block contains them -> the
    # textbook overfit signature CSCV is designed to catch.
    perf[:, 0] -= 0.004
    spikes = rng.choice(t, size=40, replace=False)
    perf[spikes, 0] += 0.25
    res = probability_of_backtest_overfitting(perf, n_splits=12)
    assert res.pbo > 0.5


@pytest.mark.parity
def test_pbo_to_dict_is_json_serializable() -> None:
    """The logits serialize to a plain list of floats for the API boundary."""
    rng = np.random.default_rng(3)
    perf = rng.standard_normal((256, 4)) * 0.01
    res = probability_of_backtest_overfitting(perf, n_splits=4)
    d = res.to_dict()
    assert isinstance(d["logits"], list)
    assert all(isinstance(x, float) for x in d["logits"])
    assert len(d["logits"]) == res.n_partitions
    assert set(d) == {"pbo", "logits", "n_partitions", "n_configs", "n_splits"}


@pytest.mark.parity
def test_pbo_is_deterministic() -> None:
    """CSCV is a deterministic function of the input matrix (no internal RNG)."""
    rng = np.random.default_rng(11)
    perf = rng.standard_normal((600, 7)) * 0.01
    a = probability_of_backtest_overfitting(perf, n_splits=8)
    b = probability_of_backtest_overfitting(perf, n_splits=8)
    assert a.pbo == b.pbo
    np.testing.assert_array_equal(a.logits, b.logits)


@pytest.mark.parity
def test_pbo_rejects_non_2d_matrix() -> None:
    """A 1-D performance vector is rejected (CSCV needs N configurations)."""
    with pytest.raises(ValidationError, match="2-D"):
        probability_of_backtest_overfitting(np.zeros(100), n_splits=4)


@pytest.mark.parity
def test_pbo_rejects_single_configuration() -> None:
    """At least two configurations are required to rank an IS-best."""
    with pytest.raises(ValidationError, match=">= 2 configurations"):
        probability_of_backtest_overfitting(np.zeros((100, 1)), n_splits=4)


@pytest.mark.parity
@pytest.mark.parametrize("bad_splits", [3, 5, 1, 0])
def test_pbo_rejects_odd_or_subtwo_splits(bad_splits: int) -> None:
    """``n_splits`` must be even and at least two for a symmetric half-split."""
    with pytest.raises(ValidationError, match="even and >= 2"):
        probability_of_backtest_overfitting(np.zeros((100, 3)), n_splits=bad_splits)


@pytest.mark.parity
def test_pbo_rejects_too_few_bars_for_splits() -> None:
    """T must be large enough to form ``n_splits`` non-empty blocks."""
    with pytest.raises(ValidationError, match="must be >= n_splits"):
        probability_of_backtest_overfitting(np.zeros((6, 3)), n_splits=8)


@pytest.mark.parity
def test_pbo_rejects_half_with_under_two_rows() -> None:
    """An IS/OOS half with < 2 rows leaves the ddof=1 Sharpe undefined."""
    # 6 bars, 6 splits -> 1 row/block -> each 3-block half has 3 rows... use 4 bars,
    # 4 splits -> 1 row/block -> each half has 2 rows (ok); 2 bars, 2 splits -> each
    # half is a single 1-row block -> undefined.
    with pytest.raises(ValidationError, match="< 2 rows"):
        probability_of_backtest_overfitting(np.zeros((2, 3)), n_splits=2)


@pytest.mark.parity
def test_pbo_rejects_non_finite_values() -> None:
    """A NaN / inf anywhere in the performance matrix is rejected."""
    perf = np.zeros((100, 3))
    perf[10, 1] = np.nan
    with pytest.raises(ValidationError, match="non-finite"):
        probability_of_backtest_overfitting(perf, n_splits=4)
