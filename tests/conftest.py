"""Shared, seeded test fixtures.

Every fixture is deterministic (driven by :func:`algosystem._rng.make_rng`) and
returns pandas objects, so tests across the suite share identical synthetic OHLC
bars with known structure:

- ``synthetic_bars`` — the deployed-default GBM-regime OHLC bar panel (the honest
  NULL: the simple signals have no exploitable edge net of costs);
- ``learnable_trend`` — a single persistent positive drift (the SANITY fixture: an
  MA-crossover that works SHOULD capture this trend and beat buy-and-hold);
- ``pure_noise`` — a driftless random walk (the strict null: nothing forecastable).

Each bar panel satisfies the intrabar invariants ``low <= {open, close} <= high``
with strictly-positive prices, mirroring the contract the synthetic generators in
:mod:`algosystem.data.synthetic` will guarantee. Importing this module has no side
effects beyond fixture registration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algosystem._rng import make_rng

_SEED = 20260619
_N_OBS = 1024
_START_PRICE = 100.0
_INTRABAR_RANGE_BPS = 30.0


def _business_index(n_obs: int, start: str = "2010-01-01") -> pd.DatetimeIndex:
    """Return an ``n_obs``-length business-day (Mon-Fri) ``DatetimeIndex``."""
    return pd.bdate_range(start=start, periods=n_obs)


def _bars_from_log_returns(
    log_returns: np.ndarray,
    gen: np.random.Generator,
    *,
    start_price: float = _START_PRICE,
) -> pd.DataFrame:
    """Build a strictly-positive OHLC bar panel from a close log-return vector.

    The close path is anchored at ``start_price``; the open of each bar is the
    prior bar's close (a gapless open, the next-bar fill price), and a seeded
    intrabar half-range envelope wraps the bar so the OHLC invariants
    ``low <= {open, close} <= high`` hold by construction.
    """
    log_returns = np.asarray(log_returns, dtype="float64").copy()
    log_returns[0] = 0.0  # anchor the first observation at start_price.
    close = start_price * np.exp(np.cumsum(log_returns))

    # Open = prior close (gapless); the first open equals the first close.
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]

    # Seeded intrabar half-range envelope around the bar's open/close extremes.
    half_range = (_INTRABAR_RANGE_BPS / 10_000.0) * close
    span_hi = np.abs(gen.standard_normal(close.size)) * half_range
    span_lo = np.abs(gen.standard_normal(close.size)) * half_range
    bar_max = np.maximum(open_, close)
    bar_min = np.minimum(open_, close)
    high = bar_max + span_hi
    low = np.maximum(bar_min - span_lo, 1e-8)  # strictly positive.

    index = _business_index(close.size)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=index,
    ).astype("float64")


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded PCG64 generator shared by tests that need raw randomness."""
    return make_rng(_SEED)


@pytest.fixture
def synthetic_bars() -> pd.DataFrame:
    """Deployed-default GBM-regime OHLC bars (the honest NULL).

    A driftless-on-average GBM close with mild regime-switching volatility plus
    microstructure (bid-ask bounce) noise, wrapped in a seeded intrabar high/low
    envelope. BY CONSTRUCTION the simple MA-crossover / momentum signals have no
    exploitable timing edge net of costs, so the honest NULL holds. Shape
    ``(1024, 4)`` OHLC, strictly positive, intrabar invariants hold, seeded for
    byte-identical reproduction.
    """
    gen = make_rng(_SEED)
    # Two-regime sticky volatility, near-zero drift (the honest-null DGP shape).
    regime = (gen.standard_normal(_N_OBS).cumsum() % 2 > 0).astype("float64")
    vol = np.where(regime > 0, 0.013, 0.008)
    drift = 0.0
    diffusion = vol * gen.standard_normal(_N_OBS)
    micro = 2e-4 * gen.standard_normal(_N_OBS)  # bid-ask-bounce microstructure noise.
    log_returns = drift + diffusion + micro
    return _bars_from_log_returns(log_returns, gen)


@pytest.fixture
def learnable_trend() -> pd.DataFrame:
    """A persistent-positive-drift OHLC bar panel (the LEARNABLE SANITY fixture).

    A constant positive log-drift dominates the noise on average, so the optimal
    position is persistently long. An MA-crossover whose machinery works SHOULD
    capture this trend and beat buy-and-hold net of costs — the machinery-works
    sanity check. Shape ``(1024, 4)`` OHLC, strictly positive, seeded.
    """
    gen = make_rng(_SEED + 1)
    drift = 0.0008  # persistent, learnable positive drift.
    vol = 0.01
    log_returns = drift + vol * gen.standard_normal(_N_OBS)
    return _bars_from_log_returns(log_returns, gen)


@pytest.fixture
def pure_noise() -> pd.DataFrame:
    """A driftless random-walk OHLC bar panel (the strict null).

    Zero drift, i.i.d. Gaussian log-returns, so next-bar returns are provably
    unforecastable — the strictest honest-null testbed, driving the anti-overfit
    regression (the signal must NOT beat buy-and-hold). Shape ``(1024, 4)`` OHLC,
    strictly positive, seeded.
    """
    gen = make_rng(_SEED + 2)
    vol = 0.01
    log_returns = vol * gen.standard_normal(_N_OBS)
    return _bars_from_log_returns(log_returns, gen)
