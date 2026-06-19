"""Lock the PURE honesty kernels (DSR / DM / verdict) so they cannot regress.

These are the FULLY-IMPLEMENTED kernels the rest of the (stubbed) pipeline will
feed; pinning their behaviour here guarantees the load-bearing honesty gate stays
correct while the sequential authors fill the surrounding stubs:

- the Deflated Sharpe is a probability in ``[0, 1]`` and is NON-INCREASING in
  ``n_trials`` (multiplicity deflation has teeth);
- the Diebold-Mariano test is signed (POSITIVE favours the system) and returns the
  honest null ``(0.0, 1.0)`` on identical series;
- the PURE ``system_has_edge`` verdict is FALSE unless the DM test is significant
  AND the DSR clears the ``1 - alpha`` CONFIDENCE level AND the PBO is below 0.5 —
  in particular a positive-but-sub-confidence DSR can NEVER flip the verdict to True.
"""

from __future__ import annotations

import numpy as np
import pytest

from algosystem.evaluation.diebold_mariano import diebold_mariano, dm_favours_system
from algosystem.evaluation.dsr import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from algosystem.evaluation.verdict import Verdict, system_has_edge


@pytest.mark.parity
def test_dsr_is_probability_and_monotone_in_trials() -> None:
    """The DSR lies in [0, 1] and is non-increasing as the multiplicity grows."""
    common = {"n_obs": 500, "variance_of_trial_sharpes": 0.25}
    dsr_few = deflated_sharpe_ratio(0.12, n_trials=1, **common)
    dsr_many = deflated_sharpe_ratio(0.12, n_trials=200, **common)
    assert 0.0 <= dsr_many <= dsr_few <= 1.0
    # PSR against zero is the single-trial DSR (no multiplicity deflation).
    psr = probabilistic_sharpe_ratio(0.12, n_obs=500)
    assert abs(psr - dsr_few) < 1e-12


@pytest.mark.parity
def test_dm_sign_convention_and_identity_null() -> None:
    """DM is positive when the system out-returns the baseline; identical => null."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(400) * 0.01
    # A higher mean net return with a NON-degenerate differential (the per-bar edge
    # itself fluctuates, so the DM differential has dispersion — a constant offset
    # is correctly rejected as a zero-variance degenerate series).
    system = base + 0.002 + rng.standard_normal(400) * 0.0005
    stat, pvalue = diebold_mariano(system, base)
    assert stat > 0.0
    assert dm_favours_system(stat, pvalue, alpha=0.05) is True
    # Pointwise-identical series: the honest null (no difference).
    stat0, pvalue0 = diebold_mariano(base, base)
    assert (stat0, pvalue0) == (0.0, 1.0)
    assert dm_favours_system(stat0, pvalue0) is False


@pytest.mark.parity
def test_verdict_requires_all_three_gates() -> None:
    """``system_has_edge`` is True iff DM-significant AND DSR>1-alpha AND PBO<0.5."""
    # All three gates pass -> edge.
    win = system_has_edge(
        dm_statistic=3.0,
        dm_pvalue=0.001,
        deflated_sharpe=0.99,
        pbo=0.10,
        n_effective_trials=12,
    )
    assert win.verdict is Verdict.SYSTEM_HAS_EDGE
    assert win.system_has_edge is True

    # A positive-but-sub-confidence DSR (0.80 < 0.95) can NEVER win, even with a
    # strongly-significant DM and a low PBO — the DSR is a CONFIDENCE level, not >0.
    sub_dsr = system_has_edge(
        dm_statistic=3.0,
        dm_pvalue=0.001,
        deflated_sharpe=0.80,
        pbo=0.10,
        n_effective_trials=12,
    )
    assert sub_dsr.verdict is Verdict.NO_ROBUST_EDGE
    assert sub_dsr.system_has_edge is False

    # PBO >= 0.5 also blocks the edge claim regardless of DM / DSR.
    high_pbo = system_has_edge(
        dm_statistic=3.0,
        dm_pvalue=0.001,
        deflated_sharpe=0.99,
        pbo=0.60,
        n_effective_trials=12,
    )
    assert high_pbo.system_has_edge is False

    # An insignificant DM also blocks it.
    weak_dm = system_has_edge(
        dm_statistic=0.3,
        dm_pvalue=0.40,
        deflated_sharpe=0.99,
        pbo=0.10,
        n_effective_trials=12,
    )
    assert weak_dm.system_has_edge is False
