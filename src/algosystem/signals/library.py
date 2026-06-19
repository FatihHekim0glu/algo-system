"""Pure, STRICTLY CAUSAL signal library — target positions from closed bars only.

[TYPED STUB — signatures, docstrings, the frozen ``SignalSpec`` config, and the
parameter-grid contract are final; the signal bodies raise
:class:`NotImplementedError` for a sequential author to fill.]

Each signal is a PURE function mapping the bar history up to and INCLUDING the
CLOSED bar ``t`` to a target position for bar ``t+1``. THE CAUSALITY CONTRACT:

- the signal at bar ``t`` reads ONLY closed bars ``<= t`` (typically the ``close``
  series); it NEVER reads the forming/partial bar ``t+1`` or any future bar;
- the emitted position vector is for the ``t -> t+1`` holding period, so the
  backtester applies ``position.shift(1)`` and the order fills at bar ``t+1``'s
  OPEN (the no-look-ahead next-bar fill);
- perturbing the forming/future bar MUST NOT change the position emitted at ``t``
  (a property test asserts this invariance).

Signals:

- :func:`ma_crossover` — long when the fast SMA is above the slow SMA, else flat /
  short (``fast < slow``);
- :func:`momentum` — long when the trailing ``lookback``-bar return is positive,
  else flat / short;
- :func:`flat` — the always-flat baseline (zero position; the zero-edge floor).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from algosystem._exceptions import ValidationError
from algosystem._typing import FloatArray


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """Immutable specification of a single signal + its parameters.

    Used to enumerate the honest multiplicity grid (#signals x #param configs) that
    feeds the Deflated-Sharpe ``n_trials`` and the PBO/CSCV configuration matrix.

    Attributes
    ----------
    name:
        The signal name (``"ma_crossover"`` / ``"momentum"`` / ``"flat"``).
    params:
        The signal's keyword parameters (e.g. ``{"fast": 10, "slow": 50}``).
    """

    name: str
    params: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the signal name (the ``params`` mapping is taken as given)."""
        if self.name not in {"ma_crossover", "momentum", "flat"}:
            raise ValidationError(
                f"SignalSpec: unknown signal {self.name!r}; "
                "expected one of {'ma_crossover', 'momentum', 'flat'}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this spec."""
        return asdict(self)


def ma_crossover(close: pd.Series, *, fast: int = 10, slow: int = 50) -> FloatArray:
    r"""Moving-average crossover target positions (strictly causal).

    For each bar ``t``, computes the trailing simple moving averages
    ``SMA_fast(t)`` and ``SMA_slow(t)`` over the CLOSED ``close`` history ``<= t``,
    and emits ``+1`` (long) when ``SMA_fast(t) > SMA_slow(t)`` else ``-1`` (short)
    or ``0`` (flat, per the author's convention). The position at ``t`` is for the
    ``t -> t+1`` holding period; it reads NO future or forming bar. Bars before the
    slow window is full emit ``0`` (no position).

    Parameters
    ----------
    close:
        The single-asset CLOSE-price series (closed bars only).
    fast:
        The fast SMA window (``>= 1``).
    slow:
        The slow SMA window (``> fast``).

    Returns
    -------
    FloatArray
        The per-bar target-position sequence (same length as ``close``), for the
        ``t -> t+1`` holding period.

    Raises
    ------
    ValidationError
        If ``fast < 1``, ``slow <= fast``, or ``close`` is malformed.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    if fast < 1:
        raise ValidationError(f"ma_crossover: fast must be >= 1, got {fast}.")
    if slow <= fast:
        raise ValidationError(f"ma_crossover: slow ({slow}) must be > fast ({fast}).")
    raise NotImplementedError("ma_crossover: typed stub — body to be authored.")


def momentum(close: pd.Series, *, lookback: int = 20) -> FloatArray:
    r"""Time-series momentum target positions (strictly causal).

    For each bar ``t``, computes the trailing ``lookback``-bar simple return
    ``close_t / close_{t - lookback} - 1`` over the CLOSED history ``<= t`` and
    emits ``+1`` (long) when it is positive else ``-1`` (short) or ``0`` (flat, per
    the author's convention). The position at ``t`` is for the ``t -> t+1``
    holding period; it reads NO future or forming bar. Bars before the lookback is
    full emit ``0``.

    Parameters
    ----------
    close:
        The single-asset CLOSE-price series (closed bars only).
    lookback:
        The momentum lookback window (``>= 1``).

    Returns
    -------
    FloatArray
        The per-bar target-position sequence (same length as ``close``), for the
        ``t -> t+1`` holding period.

    Raises
    ------
    ValidationError
        If ``lookback < 1`` or ``close`` is malformed.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    if lookback < 1:
        raise ValidationError(f"momentum: lookback must be >= 1, got {lookback}.")
    raise NotImplementedError("momentum: typed stub — body to be authored.")


def flat(close: pd.Series) -> FloatArray:
    """Always-flat baseline target positions (the zero-edge floor).

    Emits a zero position at every bar — the trivial no-trade baseline that earns
    no return and pays no cost, the floor against which any edge claim is judged.

    Parameters
    ----------
    close:
        The single-asset CLOSE-price series (used only for its length).

    Returns
    -------
    FloatArray
        A zero vector the same length as ``close``.

    Raises
    ------
    ValidationError
        If ``close`` is malformed.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    raise NotImplementedError("flat: typed stub — body to be authored.")


def build_signal(spec: SignalSpec, close: pd.Series) -> FloatArray:
    """Dispatch a :class:`SignalSpec` to its signal function and return the positions.

    The single entry point the backtester / PBO grid uses to evaluate any
    enumerated configuration uniformly.

    Parameters
    ----------
    spec:
        The signal specification (name + params).
    close:
        The single-asset CLOSE-price series.

    Returns
    -------
    FloatArray
        The per-bar target-position sequence.

    Raises
    ------
    ValidationError
        If ``spec.name`` is unknown.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    raise NotImplementedError("build_signal: typed stub — body to be authored.")
