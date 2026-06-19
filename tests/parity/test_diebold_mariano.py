"""Validate the Diebold-Mariano test of the system-vs-buy-hold net-return differential.

The DM kernel compares two per-bar net-return series. The differential
``d_t = system_t - baseline_t`` is a PERFORMANCE series (higher is better), so the
statistic ``d_bar / HAC_SE(d)`` is POSITIVE when the system out-returns the
baseline. These tests pin:

- the sign convention (a uniformly-higher mean net return => positive statistic);
- the identity null (pointwise-identical series => ``(0.0, 1.0)``);
- agreement of the denominator with the reused Newey-West HAC standard error;
- the scale-aware degeneracy guard (a CONSTANT non-zero differential has zero
  dispersion and an undefined statistic => rejected, NOT mis-reported as
  significant);
- the ``dm_favours_system`` gate (significant AND positively signed).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from algosystem._exceptions import ValidationError
from algosystem.evaluation.diebold_mariano import diebold_mariano, dm_favours_system
from algosystem.evaluation.hac import newey_west_se


def _norm_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


@pytest.mark.parity
def test_dm_statistic_matches_hac_closed_form() -> None:
    """The DM statistic equals ``mean(d) / newey_west_se(d)`` and the p-value the normal SF."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(400) * 0.01
    system = base + 0.001 + rng.standard_normal(400) * 0.0008
    diff = system - base

    stat, pvalue = diebold_mariano(system, base)
    expected_stat = float(np.mean(diff)) / newey_west_se(diff)
    assert stat == pytest.approx(expected_stat, rel=1e-12, abs=1e-12)
    expected_p = min(1.0, 2.0 * _norm_sf(abs(expected_stat)))
    assert pvalue == pytest.approx(expected_p, rel=1e-12, abs=1e-12)


@pytest.mark.parity
def test_dm_explicit_lag_overrides_andrews_rule() -> None:
    """Passing an explicit Bartlett lag reproduces the HAC SE at that lag."""
    rng = np.random.default_rng(5)
    base = rng.standard_normal(300) * 0.01
    system = base + 0.0015 + rng.standard_normal(300) * 0.0006
    diff = system - base
    stat, _ = diebold_mariano(system, base, lag=10)
    assert stat == pytest.approx(float(np.mean(diff)) / newey_west_se(diff, lag=10), abs=1e-12)


@pytest.mark.parity
def test_dm_positive_sign_when_system_out_returns_baseline() -> None:
    """A higher-mean-net-return system yields a positive, significant statistic."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal(500) * 0.01
    system = base + 0.002 + rng.standard_normal(500) * 0.0005
    stat, pvalue = diebold_mariano(system, base)
    assert stat > 0.0
    assert pvalue < 0.05
    assert dm_favours_system(stat, pvalue) is True


@pytest.mark.parity
def test_dm_negative_sign_when_system_underperforms() -> None:
    """A lower-mean system yields a negative statistic that never 'favours' it."""
    rng = np.random.default_rng(2)
    base = rng.standard_normal(500) * 0.01
    system = base - 0.002 + rng.standard_normal(500) * 0.0005
    stat, pvalue = diebold_mariano(system, base)
    assert stat < 0.0
    # Even with a tiny p-value (the difference IS significant) the gate is False
    # because the sign is against the system.
    assert dm_favours_system(stat, pvalue) is False


@pytest.mark.parity
def test_dm_identity_series_is_the_honest_null() -> None:
    """Pointwise-identical series have no differential: the exact null ``(0.0, 1.0)``."""
    rng = np.random.default_rng(3)
    series = rng.standard_normal(250) * 0.01
    stat, pvalue = diebold_mariano(series, series)
    assert (stat, pvalue) == (0.0, 1.0)
    assert dm_favours_system(stat, pvalue) is False


@pytest.mark.parity
def test_dm_insignificant_when_means_coincide() -> None:
    """Two series with the same mean but independent noise give an insignificant DM."""
    rng = np.random.default_rng(4)
    base = rng.standard_normal(600) * 0.01
    system = rng.standard_normal(600) * 0.01  # same DGP, independent draw.
    stat, pvalue = diebold_mariano(system, base)
    assert pvalue >= 0.05
    assert dm_favours_system(stat, pvalue) is False


@pytest.mark.parity
def test_dm_constant_nonzero_differential_is_rejected() -> None:
    """A zero-dispersion (constant offset) differential has an undefined statistic."""
    base = np.linspace(-0.01, 0.01, 200)
    system = base + 0.001  # uniform constant offset => zero-variance differential.
    with pytest.raises(ValidationError, match="zero dispersion"):
        diebold_mariano(system, base)


@pytest.mark.parity
def test_dm_rejects_empty_mismatched_and_short_inputs() -> None:
    """Empty, length-mismatched, and single-bar inputs are rejected before the statistic."""
    good = np.array([0.01, 0.02, 0.0])
    with pytest.raises(ValidationError, match="non-empty"):
        diebold_mariano(np.array([]), good)
    with pytest.raises(ValidationError, match="non-empty"):
        diebold_mariano(good, np.array([]))
    with pytest.raises(ValidationError, match="same length"):
        diebold_mariano(np.array([0.01, 0.02, 0.0]), np.array([0.01, 0.0]))
    with pytest.raises(ValidationError, match="at least two bars"):
        diebold_mariano(np.array([0.01]), np.array([0.0]))


@pytest.mark.parity
def test_dm_rejects_non_finite_inputs() -> None:
    """NaN / inf in either series is rejected at the boundary."""
    good = np.array([0.01, 0.0, 0.02, -0.01])
    with pytest.raises(ValidationError, match="non-finite"):
        diebold_mariano(np.array([0.01, np.nan, 0.02, 0.0]), good)
    with pytest.raises(ValidationError, match="non-finite"):
        diebold_mariano(good, np.array([0.01, np.inf, 0.02, 0.0]))


@pytest.mark.parity
@pytest.mark.parametrize(
    ("stat", "pvalue", "alpha", "expected"),
    [
        (3.0, 0.001, 0.05, True),  # significant + positive => favours.
        (3.0, 0.06, 0.05, False),  # positive but p >= alpha => not significant.
        (-3.0, 0.001, 0.05, False),  # significant but negative sign.
        (0.0, 1.0, 0.05, False),  # the null.
        (3.0, 0.04, 0.01, False),  # stricter alpha un-clears a borderline p.
    ],
)
def test_dm_favours_system_truth_table(
    stat: float, pvalue: float, alpha: float, expected: bool
) -> None:
    """``dm_favours_system`` requires BOTH significance and a positive sign."""
    assert dm_favours_system(stat, pvalue, alpha=alpha) is expected
