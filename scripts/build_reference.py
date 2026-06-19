"""Precompute the committed synthetic reference summary (``artifacts/reference.json``).

Run this offline to (re)generate the committed reference artifact the backend can
serve without recomputation and the regression suite pins against:

    python scripts/build_reference.py

The artifact holds, all on the seeded synthetic processes (no key, no network):

- ``deployed_default`` — the full :func:`algosystem.serve.run_system` summary on the
  deployed default (the ``gbm_regime`` honest-null DGP, ``ma_crossover`` 10/50): the
  documented honest-NULL outcome ``system_has_edge=False``;
- ``learnable_trend`` — the monotonic-uptrend long/flat sanity numbers (the
  machinery captures a clearly-positive net-of-cost Sharpe);
- ``regime_trend`` — the directional regime-trend sanity numbers where the FULL
  long/short pipeline DOES beat buy-and-hold DM-significant (the machinery detects a
  real, tradeable edge, so the null is honest not vacuous);
- ``pure_noise`` — the strict-null honest-NULL numbers (no DM-significant edge).

The figures are stripped from the artifact (they are large and rebuilt live); only
the JSON-safe scalar summary is committed. Everything is deterministic for the
pinned ``seed`` / ``n_obs``, so re-running reproduces the file byte-for-byte.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from algosystem.backtest.engine import vectorized_backtest
from algosystem.data.loaders import synthetic_default_bars
from algosystem.evaluation.diebold_mariano import diebold_mariano
from algosystem.evaluation.metrics import strategy_metrics
from algosystem.serve import _align_positions, run_system
from algosystem.signals.library import SignalSpec, build_signal

#: The pinned reference configuration (mirrors the API + CLI defaults).
_SEED = 7
_N_OBS = 2000
_COST_BPS = 5.0
_SLIPPAGE_BPS = 2.0
_FAST = 10
_SLOW = 50

_ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "src" / "algosystem" / "artifacts" / "reference.json"


def _signal_vs_buyhold(kind: str) -> dict[str, Any]:
    """Run the ma_crossover 10/50 signal vs. buy-and-hold on a synthetic DGP.

    Uses the SAME leakage-free primitives the serve path wires together (data ->
    causal signal -> vectorized backtest -> metrics -> Diebold-Mariano), so the
    reference numbers are consistent with the live pipeline.
    """
    bars, returns, _source = synthetic_default_bars(n_obs=_N_OBS, seed=_SEED, kind=kind)
    close = bars["close"]
    ret = np.asarray(returns.to_numpy(dtype="float64"), dtype="float64")
    n_ret = ret.size

    positions = _align_positions(
        build_signal(SignalSpec("ma_crossover", {"fast": _FAST, "slow": _SLOW}), close), n_ret
    )
    strat = vectorized_backtest(ret, positions, cost_bps=_COST_BPS, slippage_bps=_SLIPPAGE_BPS)
    buyhold = vectorized_backtest(
        ret, np.ones(n_ret, dtype="float64"), cost_bps=_COST_BPS, slippage_bps=_SLIPPAGE_BPS
    )
    strat_m = strategy_metrics(strat.net_returns, strat.positions)
    buyhold_m = strategy_metrics(buyhold.net_returns, buyhold.positions)
    dm_stat, dm_p = diebold_mariano(strat.net_returns, buyhold.net_returns)
    return {
        "kind": kind,
        "oos_sharpe": float(strat_m.oos_sharpe),
        "buyhold_sharpe": float(buyhold_m.oos_sharpe),
        "dm_statistic_vs_buyhold": float(dm_stat),
        "dm_pvalue_vs_buyhold": float(dm_p),
        "beats_buyhold": bool(strat_m.oos_sharpe > buyhold_m.oos_sharpe),
        "max_drawdown": float(strat_m.max_drawdown),
        "turnover": float(strat_m.turnover),
    }


def build_reference() -> dict[str, Any]:
    """Assemble the full reference payload (the deployed default + the DGP variants)."""
    deployed = run_system(
        signal="ma_crossover",
        fast=_FAST,
        slow=_SLOW,
        cost_bps=_COST_BPS,
        slippage_bps=_SLIPPAGE_BPS,
        data_source_pref="synthetic",
        seed=_SEED,
    )
    return {
        "schema_version": 1,
        "config": {
            "signal": "ma_crossover",
            "fast": _FAST,
            "slow": _SLOW,
            "cost_bps": _COST_BPS,
            "slippage_bps": _SLIPPAGE_BPS,
            "seed": _SEED,
            "n_obs": _N_OBS,
        },
        "deployed_default": deployed.summary.to_dict(),
        "learnable_trend": _signal_vs_buyhold("learnable_trend"),
        "regime_trend": _signal_vs_buyhold("regime_trend"),
        "pure_noise": _signal_vs_buyhold("pure_noise"),
    }


def main() -> None:
    """Write the reference artifact to ``src/algosystem/artifacts/reference.json``."""
    payload = build_reference()
    _ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ARTIFACT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {_ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
