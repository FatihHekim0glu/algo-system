"""Tests for the Typer CLI (backtest / paper / compare) + its offline pipeline.

These pin the OFFLINE entry point: three sub-commands that run the leakage-free
signal -> backtest + paper-broker replay -> parity oracle -> metrics -> DM / DSR /
PBO -> PURE verdict pipeline on the seeded synthetic null. The load-bearing
assertions:

- ``compare`` runs the FULL pipeline and prints the PURE ``system_has_edge`` verdict
  (``NO`` on the synthetic null) AND the backtest<->live parity max-diff (``~0``,
  the oracle passing);
- the pipeline never depends on the still-stubbed ``serve.run_system``;
- importing :mod:`algosystem.cli` pulls in NO Typer (it is imported lazily);
- a bad request (``fast >= slow``, unknown signal) exits non-zero, not a traceback;
- the run is deterministic across seeds and reproduces byte-for-byte.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import algosystem.cli as cli_mod
from algosystem._exceptions import ValidationError
from algosystem.cli import (
    PipelineResult,
    _build_app,
    _per_obs_sharpe,
    _sample_moments,
    main,
    run_pipeline,
)
from algosystem.execution.parity import PARITY_TOL


@pytest.fixture
def runner():  # type: ignore[no-untyped-def]
    """A Typer ``CliRunner`` (imported lazily so the module stays import-pure)."""
    from typer.testing import CliRunner

    return CliRunner()


# --------------------------------------------------------------------------- #
# run_pipeline: the self-contained offline pipeline (no serve.run_system dep)    #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_run_pipeline_returns_no_edge_on_the_synthetic_null() -> None:
    """On the seeded synthetic null the PURE verdict is False (the honest NULL)."""
    result = run_pipeline(signal="ma_crossover", fast=10, slow=50, seed=7)
    assert isinstance(result, PipelineResult)
    assert result.verdict.system_has_edge is False
    assert result.data_source == "synthetic"


@pytest.mark.unit
def test_run_pipeline_parity_oracle_passes_to_tolerance() -> None:
    """The backtest<->live parity oracle passes: max-diff is at / below ``1e-10``."""
    result = run_pipeline(signal="ma_crossover", fast=10, slow=50, seed=7)
    assert result.parity_ok is True
    assert result.parity_max_diff <= PARITY_TOL


@pytest.mark.unit
def test_run_pipeline_is_deterministic_for_a_fixed_seed() -> None:
    """Two runs with the same seed reproduce identical metrics + verdict."""
    a = run_pipeline(signal="ma_crossover", fast=10, slow=50, seed=7)
    b = run_pipeline(signal="ma_crossover", fast=10, slow=50, seed=7)
    assert a.oos_sharpe == b.oos_sharpe
    assert a.dm_pvalue == b.dm_pvalue
    assert a.deflated_sharpe == b.deflated_sharpe
    assert a.pbo == b.pbo
    assert a.verdict.system_has_edge == b.verdict.system_has_edge


@pytest.mark.unit
def test_run_pipeline_dsr_is_a_probability_and_pbo_in_unit_interval() -> None:
    """The DSR is a probability in [0, 1] and the PBO is a fraction in [0, 1]."""
    result = run_pipeline(signal="ma_crossover", fast=10, slow=50, seed=7)
    assert 0.0 <= result.deflated_sharpe <= 1.0
    assert 0.0 <= result.pbo <= 1.0
    assert result.n_trials >= 1


@pytest.mark.unit
def test_run_pipeline_n_trials_counts_the_full_grid() -> None:
    """The honest multiplicity counts the FULL #signals x #param grid, not one config."""
    result = run_pipeline(signal="ma_crossover", fast=10, slow=50, seed=7)
    # The shipped grid has >= 7 configs; a config already in the grid is not double
    # counted, so the requested default stays at the grid size.
    assert result.n_trials == 7


@pytest.mark.unit
def test_run_pipeline_custom_config_extends_the_grid() -> None:
    """A selected config NOT in the shipped grid is appended (honest multiplicity)."""
    result = run_pipeline(signal="ma_crossover", fast=7, slow=33, seed=7)
    assert result.n_trials == 8  # the 7-config grid + the novel selected config.


@pytest.mark.unit
def test_run_pipeline_supports_the_momentum_signal() -> None:
    """The momentum signal path runs and still yields the honest-null verdict."""
    result = run_pipeline(signal="momentum", lookback=20, seed=7)
    assert result.signal == "momentum"
    assert result.verdict.system_has_edge is False


@pytest.mark.unit
def test_run_pipeline_rejects_fast_ge_slow() -> None:
    """``fast >= slow`` is an invalid MA-crossover config and is rejected."""
    with pytest.raises(ValidationError):
        run_pipeline(signal="ma_crossover", fast=50, slow=10, seed=7)


@pytest.mark.unit
def test_run_pipeline_rejects_unknown_signal() -> None:
    """An unknown signal name is rejected up front."""
    with pytest.raises(ValidationError, match="ma_crossover"):
        run_pipeline(signal="bogus", seed=7)


@pytest.mark.unit
@pytest.mark.parametrize("field", ["cost_bps", "slippage_bps"])
def test_run_pipeline_rejects_negative_friction(field: str) -> None:
    """Negative transaction cost / slippage is rejected."""
    with pytest.raises(ValidationError):
        run_pipeline(signal="ma_crossover", **{field: -1.0})  # type: ignore[arg-type]


@pytest.mark.unit
def test_run_pipeline_buyhold_baseline_is_computed() -> None:
    """A finite buy-and-hold Sharpe baseline is reported alongside the strategy."""
    result = run_pipeline(signal="ma_crossover", fast=10, slow=50, seed=7)
    assert np.isfinite(result.buyhold_sharpe)
    assert result.max_drawdown <= 0.0
    assert result.turnover >= 0.0


# --------------------------------------------------------------------------- #
# The Typer commands: backtest / paper / compare                                #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("command", ["backtest", "paper", "compare"])
def test_each_command_exits_zero_and_prints_the_verdict(runner, command: str) -> None:  # type: ignore[no-untyped-def]
    """All three commands run cleanly and print the PURE ``system has edge`` verdict."""
    app = _build_app()
    result = runner.invoke(app, [command, "--seed", "7"])
    assert result.exit_code == 0, result.output
    assert "system has edge     : NO" in result.output


@pytest.mark.unit
def test_compare_prints_the_parity_max_diff(runner) -> None:  # type: ignore[no-untyped-def]
    """``compare`` prints the backtest<->live parity max-diff (the oracle passing)."""
    app = _build_app()
    result = runner.invoke(app, ["compare", "--seed", "7"])
    assert result.exit_code == 0, result.output
    assert "backtest=live parity: PASS" in result.output
    assert "max-diff" in result.output


@pytest.mark.unit
def test_compare_reports_dm_dsr_pbo(runner) -> None:  # type: ignore[no-untyped-def]
    """``compare`` surfaces the DM / DSR / PBO evidence behind the verdict."""
    app = _build_app()
    result = runner.invoke(app, ["compare", "--seed", "7"])
    out = result.output
    assert "DM p-value" in out
    assert "Deflated Sharpe" in out
    assert "PBO" in out


@pytest.mark.unit
def test_compare_momentum_signal(runner) -> None:  # type: ignore[no-untyped-def]
    """``compare`` accepts the momentum signal and still reports NO edge."""
    app = _build_app()
    result = runner.invoke(app, ["compare", "--signal", "momentum", "--lookback", "20"])
    assert result.exit_code == 0, result.output
    assert "signal              : momentum" in result.output
    assert "system has edge     : NO" in result.output


@pytest.mark.unit
def test_bad_config_exits_nonzero_not_a_traceback(runner) -> None:  # type: ignore[no-untyped-def]
    """``fast >= slow`` exits non-zero (a clean CLI error), not an unhandled traceback."""
    app = _build_app()
    result = runner.invoke(app, ["backtest", "--fast", "50", "--slow", "10"])
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)


@pytest.mark.unit
def test_unknown_signal_exits_nonzero(runner) -> None:  # type: ignore[no-untyped-def]
    """An unknown signal exits non-zero as a clean CLI error."""
    app = _build_app()
    result = runner.invoke(app, ["backtest", "--signal", "bogus"])
    assert result.exit_code != 0


@pytest.mark.unit
def test_no_args_shows_help(runner) -> None:  # type: ignore[no-untyped-def]
    """Invoking with no sub-command shows the help (lists the three commands)."""
    app = _build_app()
    result = runner.invoke(app, [])
    assert "backtest" in result.output
    assert "paper" in result.output
    assert "compare" in result.output


# --------------------------------------------------------------------------- #
# Import purity: importing algosystem.cli pulls in NO typer (lazy import)        #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_importing_cli_does_not_import_typer() -> None:
    """A fresh interpreter importing ``algosystem.cli`` does not load Typer."""
    code = (
        "import sys\n"
        "import algosystem.cli\n"
        "assert 'typer' not in sys.modules, 'typer imported at module load'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout


# --------------------------------------------------------------------------- #
# Private helpers: degenerate-series fall-backs (DSR / PBO ranking safety)       #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize(
    "series",
    [
        np.array([0.01]),  # < 2 observations: no dispersion estimate.
        np.zeros(5),  # numerically flat: zero dispersion.
    ],
)
def test_per_obs_sharpe_degenerate_returns_zero(series: np.ndarray) -> None:
    """A single-observation / flat series has an undefined Sharpe -> 0.0 (conservative)."""
    assert _per_obs_sharpe(series) == 0.0


@pytest.mark.unit
def test_per_obs_sharpe_matches_mean_over_std() -> None:
    """For a non-degenerate series the per-obs Sharpe is mean / sample-std."""
    rng = np.random.default_rng(1)
    net = rng.standard_normal(100) * 0.01
    expected = float(np.mean(net) / np.std(net, ddof=1))
    assert _per_obs_sharpe(net) == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.parametrize("series", [np.zeros(5), np.array([0.01, 0.01])])
def test_sample_moments_fall_back_to_gaussian(series: np.ndarray) -> None:
    """A flat / too-short series falls back to Gaussian moments ``(0.0, 3.0)``."""
    skew, kurtosis = _sample_moments(series)
    assert skew == 0.0
    assert kurtosis == 3.0


@pytest.mark.unit
def test_sample_moments_for_a_real_series() -> None:
    """A non-degenerate series yields finite skew + FULL (non-excess) kurtosis."""
    rng = np.random.default_rng(2)
    net = rng.standard_normal(500) * 0.01
    skew, kurtosis = _sample_moments(net)
    assert np.isfinite(skew)
    assert np.isfinite(kurtosis)
    assert kurtosis > 0.0  # FULL kurtosis (Gaussian ~ 3), never the Fisher excess.


# --------------------------------------------------------------------------- #
# main(): the console-script entry point builds + dispatches the Typer app      #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_main_builds_and_dispatches_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` builds the Typer app and invokes it (the console-script seam)."""
    called: dict[str, bool] = {"built": False, "invoked": False}

    class _FakeApp:
        def __call__(self) -> None:
            called["invoked"] = True

    def _fake_build() -> _FakeApp:
        called["built"] = True
        return _FakeApp()

    monkeypatch.setattr(cli_mod, "_build_app", _fake_build)
    main()
    assert called["built"] is True
    assert called["invoked"] is True


@pytest.mark.unit
def test_cli_does_not_depend_on_serve_run_system() -> None:
    """Importing + running the CLI pipeline does not touch the stubbed ``run_system``.

    The ``compare`` pipeline is assembled in ``algosystem.cli`` from the leakage-free
    primitives; it must run to completion even though ``serve.run_system`` is still a
    ``NotImplementedError`` stub. We assert the pipeline returns a verdict without
    raising the stub's error.
    """
    result = run_pipeline(signal="ma_crossover", fast=10, slow=50, seed=7)
    assert result.verdict.verdict.value in {"system_has_edge", "no_robust_edge"}
