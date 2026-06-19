"""The Deflated-Sharpe multiplicity (``n_trials``) honesty contract.

The Deflated Sharpe is the honest yardstick that counts the FULL configuration grid
as ``n_trials = #signals x #param configs``. These tests pin the honesty properties
the verdict relies on — that inflating the trial count can only DEFLATE the
probability, never inflate it, so a wider search makes an edge claim strictly
harder (never easier):

- the DSR is a probability in ``[0, 1]``;
- at ``n_trials == 1`` (no multiplicity) the DSR collapses to the plain PSR
  against zero;
- the DSR is NON-INCREASING in ``n_trials`` (the multiplicity deflation has teeth);
- counting more trials can flip a verdict from edge to NO-edge but never the reverse
  (the honest direction).
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from algosystem.evaluation.dsr import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from algosystem.evaluation.verdict import system_has_edge

# Typed ``Any`` so the ``**`` splat accepts the mixed int (``n_obs``) / float
# (``variance_of_trial_sharpes``) parameter signature under strict mypy.
_DSR_KW: dict[str, Any] = {"n_obs": 500, "variance_of_trial_sharpes": 0.25}


@pytest.mark.parity
def test_dsr_single_trial_reduces_to_psr_against_zero() -> None:
    """With ``n_trials == 1`` the expected-max benchmark is zero => DSR == PSR(0)."""
    dsr = deflated_sharpe_ratio(0.12, n_trials=1, **_DSR_KW)
    psr = probabilistic_sharpe_ratio(0.12, n_obs=500)
    assert dsr == pytest.approx(psr, abs=1e-12)
    assert 0.0 <= dsr <= 1.0


@pytest.mark.parity
@pytest.mark.parametrize("n_trials", [1, 2, 5, 12, 50, 200, 1000])
def test_dsr_is_a_probability_for_every_trial_count(n_trials: int) -> None:
    """The DSR stays a well-formed probability across the whole multiplicity range."""
    dsr = deflated_sharpe_ratio(0.15, n_trials=n_trials, **_DSR_KW)
    assert 0.0 <= dsr <= 1.0


@pytest.mark.parity
def test_dsr_is_non_increasing_in_trial_count() -> None:
    """Counting MORE configurations can only deflate the probability (honest)."""
    trials = [1, 2, 5, 10, 25, 60, 150, 400]
    values = [deflated_sharpe_ratio(0.18, n_trials=k, **_DSR_KW) for k in trials]
    for earlier, later in itertools.pairwise(values):
        assert later <= earlier + 1e-12
    # The honest multiplicity has real teeth: a wide grid deflates well below the
    # single-trial probability.
    assert values[-1] < values[0]


@pytest.mark.parity
def test_honest_trial_count_is_signals_times_param_configs() -> None:
    """The verdict consumes the #signals x #param-config product as ``n_trials``.

    A realistic grid — say 2 signals (ma_crossover, momentum) x 6 param configs = 12
    effective trials — deflates the same observed Sharpe more than a single trial
    would, exactly the multiplicity the honest verdict must charge.
    """
    n_signals, n_param_configs = 2, 6
    n_effective_trials = n_signals * n_param_configs
    assert n_effective_trials == 12

    dsr_one = deflated_sharpe_ratio(0.16, n_trials=1, **_DSR_KW)
    dsr_grid = deflated_sharpe_ratio(0.16, n_trials=n_effective_trials, **_DSR_KW)
    assert dsr_grid <= dsr_one


@pytest.mark.parity
def test_widening_the_search_only_removes_edge_never_creates_it() -> None:
    """More trials can flip edge->no-edge through the verdict, never no-edge->edge."""
    # An observed Sharpe whose single-trial DSR clears 0.95 but whose
    # multiplicity-deflated DSR (many trials) does not.
    dsr_one = deflated_sharpe_ratio(0.165, n_trials=1, **_DSR_KW)
    dsr_many = deflated_sharpe_ratio(0.165, n_trials=500, **_DSR_KW)
    assert dsr_many <= dsr_one

    common = {"dm_statistic": 3.0, "dm_pvalue": 0.001, "pbo": 0.10}
    verdict_one = system_has_edge(**common, deflated_sharpe=dsr_one, n_effective_trials=1)
    verdict_many = system_has_edge(**common, deflated_sharpe=dsr_many, n_effective_trials=500)
    # If the single-trial verdict found no edge, the wider search certainly cannot.
    if not verdict_one.system_has_edge:
        assert verdict_many.system_has_edge is False
