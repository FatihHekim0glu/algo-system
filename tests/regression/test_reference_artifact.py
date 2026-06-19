"""Regression: the committed reference artifact matches a fresh pipeline run.

``src/algosystem/artifacts/reference.json`` is the precomputed, committed
deployed-default summary (plus the learnable_trend / regime_trend sanity numbers and
the pure_noise honest-null numbers) the backend can serve without recomputation. It
MUST stay in lock-step with the live pipeline: a fresh :func:`run_system` on the
same pinned config reproduces the committed ``deployed_default`` summary exactly, and
the committed sanity / null numbers carry the documented signs.

If this test fails after an intentional pipeline change, regenerate the artifact with
``python scripts/build_reference.py`` and commit the new JSON.
"""

from __future__ import annotations

import json
from importlib.resources import files

import pytest

from algosystem.serve import run_system

_TOL = 1e-9


def _load_reference() -> dict[str, object]:
    """Load the committed reference artifact from the installed package data."""
    text = files("algosystem.artifacts").joinpath("reference.json").read_text(encoding="utf-8")
    payload: dict[str, object] = json.loads(text)
    return payload


@pytest.mark.regression
def test_reference_artifact_is_committed_and_well_formed() -> None:
    """The artifact exists, is valid JSON, and carries every documented section."""
    ref = _load_reference()
    assert ref["schema_version"] == 1
    assert set(ref) >= {
        "schema_version",
        "config",
        "deployed_default",
        "learnable_trend",
        "regime_trend",
        "pure_noise",
    }


@pytest.mark.regression
def test_reference_deployed_default_matches_fresh_run() -> None:
    """A fresh ``run_system`` on the pinned config reproduces the committed summary."""
    ref = _load_reference()
    cfg = ref["config"]
    assert isinstance(cfg, dict)
    fresh = run_system(
        signal=str(cfg["signal"]),
        fast=int(cfg["fast"]),
        slow=int(cfg["slow"]),
        cost_bps=float(cfg["cost_bps"]),
        slippage_bps=float(cfg["slippage_bps"]),
        data_source_pref="synthetic",
        seed=int(cfg["seed"]),
    ).summary.to_dict()

    committed = ref["deployed_default"]
    assert isinstance(committed, dict)
    for key, expected in committed.items():
        actual = fresh[key]
        if isinstance(expected, bool) or isinstance(actual, bool):
            assert actual == expected, f"{key}: {actual!r} != {expected!r}"
        elif isinstance(expected, int | float):
            assert abs(float(actual) - float(expected)) <= _TOL, f"{key}: {actual} != {expected}"
        else:
            assert actual == expected, f"{key}: {actual!r} != {expected!r}"


@pytest.mark.regression
def test_reference_deployed_default_is_the_honest_null() -> None:
    """The committed deployed default is the documented honest-NULL (no edge, parity ~0)."""
    ref = _load_reference()
    default = ref["deployed_default"]
    assert isinstance(default, dict)
    assert default["system_has_edge"] is False
    assert default["backtest_live_parity_max_diff"] <= 1e-10
    assert default["bar_finality_ok"] is True
    assert default["data_source"] == "synthetic"


@pytest.mark.regression
def test_reference_regime_trend_is_the_tradeable_sanity() -> None:
    """The committed regime-trend numbers prove the machinery captures a real edge."""
    ref = _load_reference()
    sanity = ref["regime_trend"]
    assert isinstance(sanity, dict)
    # The long/short pipeline beats buy-and-hold, DM-significant net of costs.
    assert sanity["beats_buyhold"] is True
    assert float(sanity["dm_statistic_vs_buyhold"]) > 0.0
    assert float(sanity["dm_pvalue_vs_buyhold"]) < 0.05
    assert float(sanity["oos_sharpe"]) > float(sanity["buyhold_sharpe"])


@pytest.mark.regression
def test_reference_pure_noise_has_no_significant_edge() -> None:
    """The committed pure-noise numbers carry no DM-significant edge (the strict null)."""
    ref = _load_reference()
    noise = ref["pure_noise"]
    assert isinstance(noise, dict)
    # No DM-significant edge over buy-and-hold on driftless noise.
    assert float(noise["dm_pvalue_vs_buyhold"]) >= 0.05
