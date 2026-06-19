"""Bar-finality guard: an order may only be triggered by a CLOSED bar.

[TYPED STUB — signatures, docstrings, and the frozen ``BarStatus`` enum are final;
the guard bodies raise :class:`NotImplementedError` for a sequential author to
fill.]

THE ONLY-ACT-ON-CLOSED-BARS RULE. The signal at bar ``t`` may read and act on a
bar ONLY once that bar is FINAL (closed); a partial / forming bar can NEVER trigger
an order, because its close is not yet known and acting on it would be a causality
violation (look-ahead into the bar that is still forming). This module provides the
single seam the live paper-broker replay (and the property suite) call to assert
finality before emitting any order.

The mirror invariant: in the backtester the position decided at the close of bar
``t`` is applied to the ``t -> t+1`` return and the order fills at bar ``t+1``'s
OPEN — both paths must agree that a forming bar yields no order (tested).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class BarStatus(StrEnum):
    """Whether a bar is final (closed) or still forming (partial).

    The values are stable string identifiers safe to serialize across the API
    boundary.
    """

    #: The bar is CLOSED — its open/high/low/close are final and it may be acted on.
    CLOSED = "closed"

    #: The bar is still FORMING (partial) — it may NEVER trigger an order.
    FORMING = "forming"


@dataclass(frozen=True, slots=True)
class BarFinalityReport:
    """Immutable report of a bar-finality check over a bar sequence.

    Attributes
    ----------
    n_bars:
        The number of bars examined.
    n_closed:
        The number of bars marked CLOSED (eligible to trigger an order).
    n_forming:
        The number of bars marked FORMING (which may never trigger an order).
    ok:
        ``True`` iff no order was attributed to a forming bar (the guard holds).
    """

    n_bars: int
    n_closed: int
    n_forming: int
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this report."""
        return asdict(self)


def is_actionable(status: BarStatus) -> bool:
    """Return ``True`` iff a bar with ``status`` may trigger an order.

    Only :attr:`BarStatus.CLOSED` bars are actionable; a
    :attr:`BarStatus.FORMING` bar is never actionable (the bar-finality rule).

    Parameters
    ----------
    status:
        The bar's finality status.

    Returns
    -------
    bool
        ``True`` iff ``status is BarStatus.CLOSED``.

    Raises
    ------
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    raise NotImplementedError("is_actionable: typed stub — body to be authored.")


def guard_order(status: BarStatus, *, bar_index: int) -> None:
    """Assert that an order may be emitted against the bar at ``bar_index``.

    RAISES :class:`algosystem._exceptions.BarFinalityError` when ``status`` is
    :attr:`BarStatus.FORMING` — a partial bar can NEVER trigger an order. The live
    paper-broker replay calls this before emitting every order so a forming bar is
    structurally incapable of producing a fill.

    Parameters
    ----------
    status:
        The finality status of the bar the order would act on.
    bar_index:
        The bar's index (for the error message).

    Raises
    ------
    BarFinalityError
        If ``status is BarStatus.FORMING``.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    raise NotImplementedError("guard_order: typed stub — body to be authored.")


def check_finality(statuses: list[BarStatus]) -> BarFinalityReport:
    """Tally a sequence of bar statuses into a :class:`BarFinalityReport`.

    Confirms that the actionable set is exactly the CLOSED bars and reports the
    counts — the summary the API surfaces as the ``bar_finality_ok`` flag.

    Parameters
    ----------
    statuses:
        The per-bar finality statuses in time order.

    Returns
    -------
    BarFinalityReport
        The closed/forming tallies and the ``ok`` flag.

    Raises
    ------
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    raise NotImplementedError("check_finality: typed stub — body to be authored.")
