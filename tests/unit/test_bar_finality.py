"""Unit + property tests for the bar-finality guard.

THE ONLY-ACT-ON-CLOSED-BARS RULE: a partial / forming bar can NEVER trigger an
order. These tests pin the three guard seams the live replay and the property
suite rely on:

- :func:`is_actionable` — exactly the CLOSED bars are actionable;
- :func:`guard_order` — raises :class:`BarFinalityError` on a FORMING bar and is a
  no-op on a CLOSED bar;
- :func:`check_finality` — tallies a status sequence and reports ``ok`` (no order
  is ever attributed to a forming bar).
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from algosystem._exceptions import BarFinalityError
from algosystem.backtest.bar_finality import (
    BarFinalityReport,
    BarStatus,
    check_finality,
    guard_order,
    is_actionable,
)


# --------------------------------------------------------------------------- #
# is_actionable                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_only_closed_bars_are_actionable() -> None:
    """A CLOSED bar is actionable; a FORMING bar is never actionable."""
    assert is_actionable(BarStatus.CLOSED) is True
    assert is_actionable(BarStatus.FORMING) is False


# --------------------------------------------------------------------------- #
# guard_order                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_guard_order_allows_closed_bar() -> None:
    """Guarding an order against a CLOSED bar is a silent no-op (no raise)."""
    assert guard_order(BarStatus.CLOSED, bar_index=3) is None


@pytest.mark.unit
def test_guard_order_blocks_forming_bar() -> None:
    """A FORMING bar can NEVER trigger an order — the guard raises."""
    with pytest.raises(BarFinalityError) as excinfo:
        guard_order(BarStatus.FORMING, bar_index=7)
    # The error names the offending bar index so the failure is diagnosable.
    assert "7" in str(excinfo.value)
    assert "forming" in str(excinfo.value).lower()


@pytest.mark.property
@given(
    bar_index=st.integers(min_value=0, max_value=10_000),
    status=st.sampled_from(list(BarStatus)),
)
@settings(max_examples=100, deadline=None)
def test_guard_order_raises_iff_forming(bar_index: int, status: BarStatus) -> None:
    """For ANY bar index the guard raises exactly when the bar is FORMING."""
    if status is BarStatus.FORMING:
        with pytest.raises(BarFinalityError):
            guard_order(status, bar_index=bar_index)
    else:
        assert guard_order(status, bar_index=bar_index) is None


# --------------------------------------------------------------------------- #
# check_finality                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_check_finality_tallies_counts_and_reports_ok() -> None:
    """The report counts closed/forming bars and flags ``ok`` (guard holds)."""
    statuses = [
        BarStatus.CLOSED,
        BarStatus.CLOSED,
        BarStatus.FORMING,
        BarStatus.CLOSED,
    ]
    report = check_finality(statuses)
    assert isinstance(report, BarFinalityReport)
    assert report.n_bars == 4
    assert report.n_closed == 3
    assert report.n_forming == 1
    assert report.ok is True
    # The counts partition the bars exactly.
    assert report.n_closed + report.n_forming == report.n_bars


@pytest.mark.unit
def test_check_finality_empty_sequence() -> None:
    """An empty status sequence reports zero counts and a vacuously-true ``ok``."""
    report = check_finality([])
    assert report.n_bars == 0
    assert report.n_closed == 0
    assert report.n_forming == 0
    assert report.ok is True


@pytest.mark.unit
def test_check_finality_all_forming() -> None:
    """A sequence of only forming bars yields zero actionable bars (still ``ok``)."""
    report = check_finality([BarStatus.FORMING, BarStatus.FORMING])
    assert report.n_closed == 0
    assert report.n_forming == 2
    assert report.ok is True


@pytest.mark.property
@given(
    statuses=st.lists(st.sampled_from(list(BarStatus)), min_size=0, max_size=200),
)
@settings(max_examples=120, deadline=None)
def test_check_finality_counts_partition_the_sequence(
    statuses: list[BarStatus],
) -> None:
    """For any status sequence the closed/forming tallies partition it and ok holds."""
    report = check_finality(statuses)
    expected_closed = sum(1 for s in statuses if s is BarStatus.CLOSED)
    assert report.n_bars == len(statuses)
    assert report.n_closed == expected_closed
    assert report.n_forming == len(statuses) - expected_closed
    # The actionable set is exactly the CLOSED bars, so the guard always holds.
    assert report.ok is True


@pytest.mark.unit
def test_report_to_dict_is_json_plain() -> None:
    """``BarFinalityReport.to_dict`` yields plain int/bool fields for the API."""
    report = check_finality([BarStatus.CLOSED, BarStatus.FORMING])
    d = report.to_dict()
    assert d == {"n_bars": 2, "n_closed": 1, "n_forming": 1, "ok": True}


@pytest.mark.unit
def test_bar_status_values_are_stable_strings() -> None:
    """The enum's serialized values are the stable API-boundary identifiers."""
    assert BarStatus.CLOSED.value == "closed"
    assert BarStatus.FORMING.value == "forming"
