"""Synthetic single-asset OHLC bar processes — the honest-null testbed + sanity fixtures.

[TYPED STUB — signatures, docstrings, and the frozen result dataclass are final;
the bar-generation bodies raise :class:`NotImplementedError` for a sequential
author to fill.]

Generates a single-asset daily OHLC BAR panel (open/high/low/close with realistic
INTRABAR structure) under three regimes, all seeded through
:func:`algosystem._rng.make_rng` so a given ``(seed, n_obs, ...)`` reproduces the
bars byte-for-byte:

- :func:`gbm_regime_bars` — a regime-switching GBM close process with a realistic
  intrabar high/low envelope and bid-ask-bounce microstructure where, BY
  CONSTRUCTION, the simple MA-crossover / momentum signals have NO exploitable
  edge net of costs. This is the deployed DEFAULT: the honest NULL holds.
- :func:`learnable_trend_bars` — a single persistent positive drift (a LEARNABLE,
  tradeable trend). The SANITY fixture: an MA-crossover that works SHOULD capture
  this trend and beat buy-and-hold net of costs, proving the machinery detects a
  real edge (so the null is honest, not vacuous).
- :func:`pure_noise_bars` — a driftless random walk (white-noise returns). The
  strict null: next-bar returns are provably unforecastable.

INTRABAR INVARIANTS every generated bar MUST satisfy (asserted by the author and
the property suite): ``low <= open <= high``, ``low <= close <= high``, and
``low <= high``, with all prices strictly positive. The signal reads ONLY the
closed ``close`` series; the ``open`` is the NEXT-bar fill price for the execution
engine.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from algosystem._exceptions import ValidationError

#: Default number of bars for the shipped synthetic panel (mirrors the API default).
DEFAULT_N_OBS: int = 2000

#: Default number of latent regimes for the regime-switching GBM.
DEFAULT_N_REGIMES: int = 2

#: Anchor close level for every synthetic path (strictly positive by construction).
START_PRICE: float = 100.0

#: Probability of staying in the current latent regime each bar (sticky chain).
REGIME_STICKINESS: float = 0.98


@dataclass(frozen=True, slots=True)
class BarPath:
    """Immutable synthetic single-asset OHLC bar panel + its known regime labels.

    Attributes
    ----------
    bars:
        The ``(n_obs, 4)`` OHLC DataFrame with columns
        ``["open", "high", "low", "close"]`` (strictly positive, intrabar
        invariants hold), indexed by business day.
    regime_labels:
        The ``(n_obs,)`` integer latent-regime label per bar (in time order); a
        single nominal regime for the non-regime processes.
    kind:
        The process kind (``"gbm_regime"`` / ``"learnable_trend"`` /
        ``"pure_noise"``).
    """

    bars: pd.DataFrame
    regime_labels: tuple[int, ...]
    kind: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this path's metadata.

        The full bar panel is omitted (it is large and not part of the API
        contract); only the shape metadata and regime labels are emitted.
        """
        return {
            "n_obs": int(self.bars.shape[0]),
            "kind": str(self.kind),
            "columns": [str(c) for c in self.bars.columns],
            "regime_labels": [int(x) for x in self.regime_labels],
        }


def gbm_regime_bars(
    *,
    n_obs: int = DEFAULT_N_OBS,
    n_regimes: int = DEFAULT_N_REGIMES,
    seed: int = 7,
    base_drift: float = 0.0,
    base_vol: float = 0.01,
    intrabar_range_bps: float = 30.0,
    microstructure_bps: float = 2.0,
    start: str = "2010-01-01",
) -> BarPath:
    r"""Generate a regime-switching GBM OHLC bar panel (the honest-null DGP).

    The close log-return at bar ``t`` is :math:`\mu_{s_t} + \sigma_{s_t}\,
    \varepsilon_t` where the latent regime ``s_t`` follows a sticky Markov chain
    over ``n_regimes`` states (each with its own mild drift/vol), plus an additive
    mean-reverting bid-ask-bounce microstructure term of magnitude
    ``microstructure_bps``. Around each close, a seeded intrabar high/low envelope
    of magnitude ``intrabar_range_bps`` is drawn so the OHLC invariants
    (``low <= {open, close} <= high``) hold. BY CONSTRUCTION the per-bar drift is
    near-zero and the regime is unforecastable from the observable closed bars, so
    the simple signals have NO timing edge net of costs and the honest NULL holds.

    Parameters
    ----------
    n_obs:
        Number of daily bars (rows).
    n_regimes:
        Number of latent regimes in the sticky Markov chain (``>= 1``).
    seed:
        Master RNG seed (feeds :func:`algosystem._rng.make_rng`).
    base_drift:
        Centre of the per-regime daily close drift (kept near zero for the null).
    base_vol:
        Centre of the per-regime daily close volatility.
    intrabar_range_bps:
        Magnitude (bps) of the seeded intrabar high/low envelope around the close.
    microstructure_bps:
        Magnitude (bps) of the additive bid-ask-bounce microstructure noise.
    start:
        First business-day date for the index.

    Returns
    -------
    BarPath
        The OHLC bar panel + its known per-bar regime labels.

    Raises
    ------
    ValidationError
        If ``n_obs < 2``, ``n_regimes < 1``, or any volatility/range is negative.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    raise NotImplementedError("gbm_regime_bars: typed stub — body to be authored.")


def learnable_trend_bars(
    *,
    n_obs: int = DEFAULT_N_OBS,
    seed: int = 7,
    drift: float = 0.0008,
    vol: float = 0.01,
    intrabar_range_bps: float = 30.0,
    start: str = "2010-01-01",
) -> BarPath:
    r"""Generate an OHLC bar panel with a persistent positive drift (the LEARNABLE SANITY fixture).

    The close log-return at bar ``t`` is :math:`\mu + \sigma\,\varepsilon_t` with a
    CONSTANT positive drift ``mu = drift`` that dominates the noise on average, so
    the optimal position is persistently long and a fast/slow MA-crossover SHOULD
    capture the trend and beat buy-and-hold net of costs — the machinery-works
    sanity check (NOT the honest-null DGP). The intrabar high/low envelope is drawn
    exactly as in :func:`gbm_regime_bars` so the OHLC invariants hold.

    Parameters
    ----------
    n_obs:
        Number of daily bars.
    seed:
        Master RNG seed.
    drift:
        The constant per-bar log-drift (positive; learnable).
    vol:
        The per-bar close volatility.
    intrabar_range_bps:
        Magnitude (bps) of the seeded intrabar high/low envelope.
    start:
        First business-day date.

    Returns
    -------
    BarPath
        The trending OHLC bar panel with a single nominal regime label.

    Raises
    ------
    ValidationError
        If ``n_obs < 2`` or ``vol < 0``.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    raise NotImplementedError("learnable_trend_bars: typed stub — body to be authored.")


def pure_noise_bars(
    *,
    n_obs: int = DEFAULT_N_OBS,
    seed: int = 7,
    vol: float = 0.01,
    intrabar_range_bps: float = 30.0,
    start: str = "2010-01-01",
) -> BarPath:
    r"""Generate a driftless random-walk OHLC bar panel (the strict null).

    The close log-return at bar ``t`` is :math:`\sigma\,\varepsilon_t` with ZERO
    drift, so closes are a driftless geometric random walk and next-bar returns are
    provably unforecastable — the strictest honest-null testbed. The intrabar
    envelope is drawn as in :func:`gbm_regime_bars` so the OHLC invariants hold.

    Parameters
    ----------
    n_obs:
        Number of daily bars.
    seed:
        Master RNG seed.
    vol:
        The per-bar close volatility.
    intrabar_range_bps:
        Magnitude (bps) of the seeded intrabar high/low envelope.
    start:
        First business-day date.

    Returns
    -------
    BarPath
        The driftless OHLC bar panel with a single nominal regime label.

    Raises
    ------
    ValidationError
        If ``n_obs < 2`` or ``vol < 0``.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    raise NotImplementedError("pure_noise_bars: typed stub — body to be authored.")


def assert_ohlc_invariants(bars: pd.DataFrame) -> None:
    """Assert the per-bar OHLC invariants on a bar panel.

    Every bar MUST satisfy ``low <= open <= high``, ``low <= close <= high``, and
    all prices strictly positive — the intrabar consistency the synthetic
    generators guarantee and the property suite enforces.

    Parameters
    ----------
    bars:
        An OHLC DataFrame with columns ``["open", "high", "low", "close"]``.

    Raises
    ------
    ValidationError
        If a required column is missing or any intrabar invariant is violated.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    if not {"open", "high", "low", "close"}.issubset(set(bars.columns)):
        raise ValidationError(
            "assert_ohlc_invariants: bars must have open/high/low/close columns."
        )
    raise NotImplementedError("assert_ohlc_invariants: typed stub — body to be authored.")
