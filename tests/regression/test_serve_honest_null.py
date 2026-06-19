"""Regression: the serve-time honest-NULL holds deterministically across hash seeds.

The deployed default (``ma_crossover`` 10/50 on the seeded ``gbm_regime`` honest-null
DGP) MUST report ``system_has_edge=False`` net of costs, the Deflated-Sharpe
correction, and the PBO/CSCV check — and it must do so DETERMINISTICALLY: two runs
with the same seed reproduce identical metrics, and the verdict is invariant to
``PYTHONHASHSEED`` (the pipeline never depends on dict / set iteration order).

These pin the load-bearing claim of the capstone: the integration is honest, the
null is real, and the verdict is a pure, reproducible function of the evidence.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from algosystem.serve import run_system


@pytest.mark.regression
def test_deployed_default_has_no_robust_edge() -> None:
    """The deployed default reports the honest-NULL verdict (no edge after costs)."""
    run = run_system(signal="ma_crossover", fast=10, slow=50, seed=7)
    summary = run.summary
    assert summary.system_has_edge is False
    # The verdict is False because at least one honest gate fails on the null. On
    # this DGP the DM test is insignificant AND the DSR is far below the confidence
    # level AND the PBO is high — any one suffices, but document all three.
    assert summary.dm_pvalue_vs_buyhold >= 0.05  # DM insignificant.
    assert summary.deflated_sharpe <= 0.95  # DSR below the 1 - alpha confidence gate.
    assert summary.pbo >= 0.5  # PBO at / above one half (overfit-leaning).


@pytest.mark.regression
def test_pure_noise_strict_null_has_no_edge() -> None:
    """On the strict pure-noise null (via the loader kind) the verdict stays False.

    The serve default DGP is ``gbm_regime``; this confirms the SAME pipeline on the
    strictest driftless-random-walk null also yields the honest-NULL verdict, using
    the serve primitives directly so the path matches the live run.
    """
    import numpy as np

    from algosystem.backtest.engine import vectorized_backtest
    from algosystem.data.loaders import synthetic_default_bars
    from algosystem.evaluation.diebold_mariano import diebold_mariano, dm_favours_system
    from algosystem.serve import _align_positions
    from algosystem.signals.library import SignalSpec, build_signal

    bars, returns, _src = synthetic_default_bars(n_obs=2000, seed=7, kind="pure_noise")
    close = bars["close"]
    ret = np.asarray(returns.to_numpy(dtype="float64"), dtype="float64")
    pos = _align_positions(
        build_signal(SignalSpec("ma_crossover", {"fast": 10, "slow": 50}), close), ret.size
    )
    strat = vectorized_backtest(ret, pos, cost_bps=5.0, slippage_bps=2.0)
    buyhold = vectorized_backtest(
        ret, np.ones(ret.size, dtype="float64"), cost_bps=5.0, slippage_bps=2.0
    )
    dm_stat, dm_p = diebold_mariano(strat.net_returns, buyhold.net_returns)
    # No DM-significant edge over buy-and-hold on driftless noise.
    assert dm_favours_system(dm_stat, dm_p) is False


@pytest.mark.regression
def test_run_system_is_deterministic_for_a_fixed_seed() -> None:
    """Two runs with the same seed reproduce identical metrics + verdict."""
    a = run_system(signal="ma_crossover", fast=10, slow=50, seed=7).summary
    b = run_system(signal="ma_crossover", fast=10, slow=50, seed=7).summary
    assert a.oos_sharpe == b.oos_sharpe
    assert a.buyhold_sharpe == b.buyhold_sharpe
    assert a.dm_pvalue_vs_buyhold == b.dm_pvalue_vs_buyhold
    assert a.deflated_sharpe == b.deflated_sharpe
    assert a.pbo == b.pbo
    assert a.backtest_live_parity_max_diff == b.backtest_live_parity_max_diff
    assert a.system_has_edge == b.system_has_edge


@pytest.mark.regression
@pytest.mark.parametrize("hashseed", ["0", "1", "12345"])
def test_verdict_is_invariant_to_pythonhashseed(hashseed: str) -> None:
    """The honest-NULL verdict + key metrics are invariant to ``PYTHONHASHSEED``.

    Runs ``run_system`` in a FRESH subprocess under a chosen ``PYTHONHASHSEED`` and
    confirms the verdict and the headline scalars are byte-identical — the pipeline
    never leaks dict / set iteration order into the result.
    """
    code = (
        "from algosystem.serve import run_system\n"
        "s = run_system(signal='ma_crossover', fast=10, slow=50, seed=7).summary\n"
        "print(repr((s.system_has_edge, s.oos_sharpe, s.deflated_sharpe, s.pbo, "
        "s.n_effective_trials, s.backtest_live_parity_max_diff)))\n"
    )
    outputs = []
    for hseed in (hashseed, "999"):
        env = {"PYTHONHASHSEED": hseed, "PATH": _path_env()}
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.strip())
    # Different PYTHONHASHSEED values must produce identical results.
    assert outputs[0] == outputs[1]
    assert "False" in outputs[0]  # the honest-NULL verdict.


def _path_env() -> str:
    """Return the current ``PATH`` so the subprocess can find the interpreter venv."""
    import os

    return os.environ.get("PATH", "")
