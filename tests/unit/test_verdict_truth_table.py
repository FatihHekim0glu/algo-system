"""The PURE ``system_has_edge`` verdict truth table (all three gates).

``system_has_edge`` returns :attr:`Verdict.SYSTEM_HAS_EDGE` IFF ALL THREE lines of
evidence agree:

1. the Diebold-Mariano test is significant AND positively signed
   (``dm_pvalue < alpha`` AND ``dm_statistic > 0``);
2. the Deflated Sharpe clears the ``1 - alpha`` CONFIDENCE level
   (``deflated_sharpe > 1 - alpha``) — the DSR is a PROBABILITY in ``[0, 1]``, so a
   positive-but-sub-confidence DSR (e.g. ``0.80`` with ``alpha = 0.05``) must FAIL,
   never flipping the verdict to True (the load-bearing confidence-gate binding);
3. the Probability of Backtest Overfitting is below one half (``pbo < 0.5``).

If ANY gate fails the verdict is :attr:`Verdict.NO_ROBUST_EDGE` — the documented
honest-NULL. These tests enumerate the truth table, the confidence-gate binding,
the honest-null rows, the boundary (strict ``>`` / ``<``) behaviour, the
``alpha``-dependence of the DSR gate, and the input validation.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from algosystem._exceptions import ValidationError
from algosystem.evaluation.verdict import Verdict, VerdictResult, system_has_edge

# A configuration where every gate passes; individual tests knock out one gate at a
# time to confirm each is necessary. Typed ``Any`` so the ``**`` splat accepts the
# mixed float / int parameter signature under strict mypy.
_ALL_PASS: dict[str, Any] = {
    "dm_statistic": 3.0,
    "dm_pvalue": 0.001,
    "deflated_sharpe": 0.99,
    "pbo": 0.10,
    "n_effective_trials": 12,
}


@pytest.mark.unit
def test_all_three_gates_pass_yields_edge() -> None:
    """All gates clear => SYSTEM_HAS_EDGE, with the evidence echoed back."""
    res = system_has_edge(**_ALL_PASS)
    assert isinstance(res, VerdictResult)
    assert res.verdict is Verdict.SYSTEM_HAS_EDGE
    assert res.system_has_edge is True
    assert res.dm_pvalue == 0.001
    assert res.deflated_sharpe == 0.99
    assert res.pbo == 0.10
    assert res.n_effective_trials == 12


@pytest.mark.unit
def test_positive_but_subconfidence_dsr_can_never_win() -> None:
    """THE binding gate: a positive DSR below ``1 - alpha`` fails despite DM + PBO."""
    # 0.80 > 0 (a naive '> 0' test would pass) but 0.80 < 0.95 => NO edge.
    res = system_has_edge(**{**_ALL_PASS, "deflated_sharpe": 0.80})
    assert res.verdict is Verdict.NO_ROBUST_EDGE
    assert res.system_has_edge is False
    # Even a DSR exactly AT the confidence level fails (the gate is strict ``>``).
    at_boundary = system_has_edge(**{**_ALL_PASS, "deflated_sharpe": 0.95})
    assert at_boundary.system_has_edge is False
    # A hair above the level passes.
    above = system_has_edge(**{**_ALL_PASS, "deflated_sharpe": 0.95 + 1e-9})
    assert above.system_has_edge is True


@pytest.mark.unit
def test_insignificant_dm_blocks_edge() -> None:
    """An insignificant DM (p >= alpha) fails regardless of DSR / PBO (honest null)."""
    res = system_has_edge(**{**_ALL_PASS, "dm_statistic": 0.3, "dm_pvalue": 0.40})
    assert res.verdict is Verdict.NO_ROBUST_EDGE
    assert res.system_has_edge is False


@pytest.mark.unit
def test_negative_dm_sign_blocks_edge() -> None:
    """A significant but NEGATIVE DM (system underperforms) fails — sign matters."""
    res = system_has_edge(**{**_ALL_PASS, "dm_statistic": -3.0, "dm_pvalue": 0.001})
    assert res.system_has_edge is False


@pytest.mark.unit
def test_high_pbo_blocks_edge() -> None:
    """A PBO at or above 0.5 (likely overfit) fails regardless of DM / DSR."""
    assert system_has_edge(**{**_ALL_PASS, "pbo": 0.60}).system_has_edge is False
    # Exactly 0.5 also fails (the gate is strict ``< 0.5``).
    assert system_has_edge(**{**_ALL_PASS, "pbo": 0.50}).system_has_edge is False
    # Just under 0.5 still passes the PBO gate.
    assert system_has_edge(**{**_ALL_PASS, "pbo": 0.50 - 1e-9}).system_has_edge is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dm_ok", "dsr_ok", "pbo_ok", "expected"),
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
        (False, False, True, False),
        (False, True, False, False),
        (True, False, False, False),
        (False, False, False, False),
    ],
)
def test_full_truth_table_is_and_of_three_gates(
    dm_ok: bool, dsr_ok: bool, pbo_ok: bool, expected: bool
) -> None:
    """Across all 8 gate combinations the verdict is the strict AND of the three."""
    res = system_has_edge(
        dm_statistic=3.0 if dm_ok else 0.2,
        dm_pvalue=0.001 if dm_ok else 0.50,
        deflated_sharpe=0.99 if dsr_ok else 0.80,
        pbo=0.10 if pbo_ok else 0.70,
        n_effective_trials=12,
    )
    assert res.system_has_edge is expected
    assert (res.verdict is Verdict.SYSTEM_HAS_EDGE) is expected


@pytest.mark.unit
def test_dsr_gate_tracks_alpha() -> None:
    """The DSR confidence gate is ``1 - alpha``, so a stricter alpha raises the bar."""
    # DSR 0.97: clears the 0.95 gate (alpha=0.05) but NOT the 0.99 gate (alpha=0.01).
    common: dict[str, Any] = {
        "dm_statistic": 3.0,
        "dm_pvalue": 0.001,
        "pbo": 0.10,
        "n_effective_trials": 12,
    }
    assert system_has_edge(**common, deflated_sharpe=0.97, alpha=0.05).system_has_edge is True
    assert system_has_edge(**common, deflated_sharpe=0.97, alpha=0.01).system_has_edge is False


@pytest.mark.unit
def test_honest_null_default_configuration() -> None:
    """The documented honest-null shape (weak DM, sub-confidence DSR) => NO edge."""
    res = system_has_edge(
        dm_statistic=0.4,
        dm_pvalue=0.55,
        deflated_sharpe=0.42,
        pbo=0.55,
        n_effective_trials=8,
    )
    assert res.verdict is Verdict.NO_ROBUST_EDGE
    assert res.system_has_edge is False


@pytest.mark.unit
def test_verdict_to_dict_serializes_enum_value() -> None:
    """The JSON view renders the verdict enum as its stable string identifier."""
    d = system_has_edge(**_ALL_PASS).to_dict()
    assert d["verdict"] == "system_has_edge"
    assert d["system_has_edge"] is True
    assert d["n_effective_trials"] == 12
    null = system_has_edge(**{**_ALL_PASS, "pbo": 0.9}).to_dict()
    assert null["verdict"] == "no_robust_edge"


@pytest.mark.unit
@pytest.mark.parametrize("bad_pvalue", [-0.01, 1.01, math.nan])
def test_verdict_rejects_out_of_range_pvalue(bad_pvalue: float) -> None:
    """The DM p-value must be a finite probability in ``[0, 1]``."""
    with pytest.raises(ValidationError, match="dm_pvalue"):
        system_has_edge(**{**_ALL_PASS, "dm_pvalue": bad_pvalue})


@pytest.mark.unit
@pytest.mark.parametrize("bad_pbo", [-0.01, 1.5, math.inf])
def test_verdict_rejects_out_of_range_pbo(bad_pbo: float) -> None:
    """The PBO must be a finite probability in ``[0, 1]``."""
    with pytest.raises(ValidationError, match="pbo"):
        system_has_edge(**{**_ALL_PASS, "pbo": bad_pbo})


@pytest.mark.unit
def test_verdict_rejects_nonfinite_statistic_and_dsr() -> None:
    """Non-finite DM statistic / DSR are rejected before any gate is evaluated."""
    with pytest.raises(ValidationError, match="dm_statistic"):
        system_has_edge(**{**_ALL_PASS, "dm_statistic": math.nan})
    with pytest.raises(ValidationError, match="deflated_sharpe"):
        system_has_edge(**{**_ALL_PASS, "deflated_sharpe": math.inf})


@pytest.mark.unit
def test_verdict_rejects_sub_one_trial_count() -> None:
    """The honest multiplicity count must be at least one trial."""
    with pytest.raises(ValidationError, match="n_effective_trials"):
        system_has_edge(**{**_ALL_PASS, "n_effective_trials": 0})
