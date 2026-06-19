"""Unit tests for the PIT loaders (:mod:`algosystem.data.loaders`).

Covers the synthetic-default routing (the deployed path: no key, no network), the
``data_source_pref`` contract (``"synthetic"`` / ``"auto"`` resolve to synthetic;
``"polygon"`` falls back to synthetic on any provider failure), the
``pct_change(fill_method=None)`` no-lookahead return computation, the OHLC
invariants of the loaded panel, and the input-validation guards. A monkeypatched
provider exercises the genuine Polygon success branch WITHOUT touching the network.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

from algosystem._exceptions import ValidationError
from algosystem.data.loaders import (
    SYNTHETIC_KINDS,
    load_single_asset_bars,
    synthetic_default_bars,
)
from algosystem.data.synthetic import assert_ohlc_invariants


# --------------------------------------------------------------------------- #
# Import purity                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_data_modules_import_pulls_in_no_heavy_deps() -> None:
    """Importing the data modules loads no httpx / heavy / torch-family module.

    The Polygon provider's ``httpx`` (the ``data`` extra) and any torch-family
    dependency must stay LAZY — pulled in only inside the loader bodies on the
    real-data path, never at import. A fresh interpreter asserts none leaked.
    """
    forbidden = ("httpx", "torch", "onnxruntime", "sklearn", "statsmodels", "plotly")
    code = (
        "import sys\n"
        "import algosystem.data\n"
        "import algosystem.data.synthetic\n"
        "import algosystem.data.loaders\n"
        f"forbidden = {forbidden!r}\n"
        "leaked = sorted(m for m in forbidden if m in sys.modules)\n"
        "assert not leaked, f'forbidden modules imported at load: {leaked}'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, (
        f"data import-purity subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout


# --------------------------------------------------------------------------- #
# synthetic_default_bars                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("kind", sorted(SYNTHETIC_KINDS))
def test_synthetic_default_routes_each_kind(kind: str) -> None:
    """Each synthetic kind routes, returns OHLC + returns + the 'synthetic' tag."""
    bars, returns, source = synthetic_default_bars(n_obs=300, seed=7, kind=kind)
    assert source == "synthetic"
    assert list(bars.columns) == ["open", "high", "low", "close"]
    assert bars.shape == (300, 4)
    assert_ohlc_invariants(bars)
    # Returns drop the leading NaN observation (no-lookahead differencing).
    assert len(returns) == len(bars) - 1
    assert not bool(returns.isna().any())


@pytest.mark.unit
def test_synthetic_default_returns_match_pct_change_no_fill() -> None:
    """Returns equal pct_change(fill_method=None) of the close (the no-lookahead rule)."""
    bars, returns, _ = synthetic_default_bars(n_obs=200, seed=7, kind="gbm_regime")
    expected = bars["close"].pct_change(fill_method=None).iloc[1:].astype("float64")
    pd.testing.assert_series_equal(returns, expected, check_names=False)


@pytest.mark.unit
def test_synthetic_default_is_deterministic() -> None:
    """Same (seed, n_obs, kind) reproduces the bars byte-for-byte."""
    a, ra, _ = synthetic_default_bars(n_obs=150, seed=9, kind="learnable_trend")
    b, rb, _ = synthetic_default_bars(n_obs=150, seed=9, kind="learnable_trend")
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_series_equal(ra, rb)


@pytest.mark.unit
def test_synthetic_default_rejects_unknown_kind_and_short() -> None:
    """An unknown kind or n_obs < 2 is rejected."""
    with pytest.raises(ValidationError, match="unknown kind"):
        synthetic_default_bars(kind="not_a_kind")
    with pytest.raises(ValidationError, match="n_obs"):
        synthetic_default_bars(n_obs=1, kind="gbm_regime")


# --------------------------------------------------------------------------- #
# load_single_asset_bars — routing + fallback                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("pref", ["synthetic", "auto"])
def test_load_synthetic_and_auto_resolve_to_synthetic(pref: str) -> None:
    """The default + 'auto' prefs resolve to the deterministic synthetic path."""
    bars, returns, source = load_single_asset_bars(
        "SPY",
        start=date(2015, 1, 1),
        end=date(2016, 1, 1),
        data_source_pref=pref,
        seed=7,
    )
    assert source == "synthetic"
    assert_ohlc_invariants(bars)
    assert len(returns) == len(bars) - 1
    assert bars.shape[0] >= 2


@pytest.mark.unit
def test_load_polygon_falls_back_to_synthetic_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no Polygon key / no network the 'polygon' pref falls back to synthetic."""
    # Ensure the key cannot be resolved (no env var, no .env discovered).
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setattr(
        "algosystem.data_providers.polygon._load_api_key_from_dotenv",
        lambda: None,
    )
    bars, _returns, source = load_single_asset_bars(
        "SPY",
        start=date(2015, 1, 1),
        end=date(2015, 6, 1),
        data_source_pref="polygon",
        seed=7,
    )
    assert source == "synthetic"  # graceful, offline-safe fallback.
    assert_ohlc_invariants(bars)


@pytest.mark.unit
def test_load_polygon_success_path_with_stubbed_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stubbed provider exercises the real Polygon branch WITHOUT the network."""
    idx = pd.bdate_range("2020-01-01", periods=120)
    closes = pd.Series(100.0 * np.exp(np.cumsum(np.full(120, 0.001))), index=idx, name="SPY")

    class _StubProvider:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
            return pd.DataFrame({tickers[0]: closes})

    monkeypatch.setattr(
        "algosystem.data_providers.polygon.PolygonProvider",
        _StubProvider,
    )
    bars, returns, source = load_single_asset_bars(
        "SPY",
        start=date(2020, 1, 1),
        end=date(2020, 6, 1),
        data_source_pref="polygon",
    )
    assert source == "polygon"
    assert_ohlc_invariants(bars)
    # Open is the prior close (gapless); returns are pct_change(fill_method=None).
    assert bars["open"].iloc[0] == pytest.approx(bars["close"].iloc[0])
    assert bars["open"].iloc[1] == pytest.approx(bars["close"].iloc[0])
    expected = bars["close"].pct_change(fill_method=None).iloc[1:].astype("float64")
    pd.testing.assert_series_equal(returns, expected, check_names=False)


# --------------------------------------------------------------------------- #
# load_single_asset_bars — validation                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_load_rejects_empty_ticker() -> None:
    """An empty / whitespace ticker is rejected."""
    with pytest.raises(ValidationError, match="non-empty symbol"):
        load_single_asset_bars("  ", start=date(2015, 1, 1), end=date(2016, 1, 1))


@pytest.mark.unit
def test_load_rejects_non_increasing_dates() -> None:
    """end <= start is rejected."""
    with pytest.raises(ValidationError, match="must be after start"):
        load_single_asset_bars("SPY", start=date(2016, 1, 1), end=date(2015, 1, 1))
