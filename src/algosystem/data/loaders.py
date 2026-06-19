"""Real-data loaders: Polygon PIT single-asset OHLC bars (CLI path) + synthetic default.

[TYPED STUB — signatures, docstrings, and the routing contract are final; the
loader bodies raise :class:`NotImplementedError` for a sequential author to fill.]

The default, deployed data path is the seeded synthetic OHLC bar process
(:mod:`algosystem.data.synthetic`) — no API keys, no survivorship questions, and
the honest NULL holds by construction. This module is the OFFLINE CLI path for
real data:

- :func:`load_single_asset_bars` fetches a single ticker's point-in-time daily
  bars via the vendored Polygon provider over a date span, computes per-bar close
  returns with ``pct_change(fill_method=None)`` (no forward-fill across gaps), and
  tags the provenance (``"polygon"`` / ``"synthetic"``). On any failure (no key,
  no network, ``data`` extra absent) it falls back to the deterministic synthetic
  path so the loader is usable offline and in CI.

Heavy data dependencies (httpx via the vendored Polygon provider, pyarrow,
diskcache) live behind the ``data`` extra and are imported LAZILY inside these
functions, so importing this module pulls in nothing heavy and has no side effects.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from algosystem._exceptions import ValidationError
from algosystem.data import DataSource

#: The synthetic process kinds routed by :func:`synthetic_default_bars`.
SYNTHETIC_KINDS: frozenset[str] = frozenset({"gbm_regime", "learnable_trend", "pure_noise"})


def load_single_asset_bars(
    ticker: str,
    *,
    start: date,
    end: date,
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.Series, DataSource]:
    """Load a single asset's PIT OHLC bars + per-bar close returns and tag the provenance.

    With ``data_source_pref="polygon"`` the vendored Polygon provider is tried
    first; ``"synthetic"`` (default) and any provider failure fall straight through
    to the deterministic synthetic GBM-regime bars so the loader is usable offline
    and in CI. Returns are computed with ``pct_change(fill_method=None)`` (no
    forward-fill across gaps).

    LAZY IMPORTS: the vendored Polygon provider (and ``httpx`` inside it — the
    ``data`` extra) are imported inside this function, so importing this module is
    cheap and side-effect-free.

    Parameters
    ----------
    ticker:
        The single asset symbol to fetch (e.g. ``"SPY"``).
    start, end:
        Inclusive date span.
    data_source_pref:
        ``"polygon"`` tries the real PIT provider first; ``"synthetic"`` (default)
        and ``"auto"`` resolve to the synthetic path.
    seed:
        Master RNG seed for the synthetic fallback path.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series, DataSource]
        The OHLC bar panel, the per-bar close-return series, and the resolved
        source label.

    Raises
    ------
    ValidationError
        If ``ticker`` is empty or ``end <= start``.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    if not ticker or not ticker.strip():
        raise ValidationError("load_single_asset_bars: ticker must be a non-empty symbol.")
    if end <= start:
        raise ValidationError(
            f"load_single_asset_bars: end ({end}) must be after start ({start})."
        )
    raise NotImplementedError("load_single_asset_bars: typed stub — body to be authored.")


def synthetic_default_bars(
    *,
    n_obs: int = 2000,
    seed: int = 7,
    kind: str = "gbm_regime",
) -> tuple[pd.DataFrame, pd.Series, DataSource]:
    """Build the deployed-default synthetic OHLC bars + close returns (torch-free, no network).

    Routes to the requested synthetic process (``"gbm_regime"`` = the honest-null
    default, ``"learnable_trend"`` = the sanity fixture, ``"pure_noise"`` = the
    strict null) and returns the OHLC bars, per-bar close returns, and the
    ``"synthetic"`` provenance label. The deployed request path uses this — it
    never needs a key or network.

    Parameters
    ----------
    n_obs:
        Number of bars to generate.
    seed:
        Master RNG seed.
    kind:
        One of ``{"gbm_regime", "learnable_trend", "pure_noise"}``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series, DataSource]
        The OHLC bars, per-bar close returns, and ``"synthetic"``.

    Raises
    ------
    ValidationError
        If ``kind`` is unknown or ``n_obs < 2``.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    if kind not in SYNTHETIC_KINDS:
        raise ValidationError(
            f"synthetic_default_bars: unknown kind {kind!r}; "
            f"expected one of {sorted(SYNTHETIC_KINDS)}."
        )
    if n_obs < 2:
        raise ValidationError(f"synthetic_default_bars: n_obs must be >= 2, got {n_obs}.")
    raise NotImplementedError("synthetic_default_bars: typed stub — body to be authored.")
