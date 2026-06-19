"""Import-purity smoke test: ``import algosystem`` pulls in nothing heavy / networked.

The package contract is ZERO import-time side effects: importing ``algosystem`` (and
its public submodules) must NOT import any network / heavy / serving dependency at
module load — ``httpx``, ``statsmodels``, ``plotly``, ``typer``, and crucially any
torch / onnx / onnxruntime / sklearn / sb3 / gymnasium (this is a TORCH-FREE
capstone). Those are imported LAZILY inside the functions that need them. This test
imports the package in a FRESH subprocess interpreter and asserts none of the
forbidden modules ended up in ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: Modules that MUST NOT be imported as a side effect of ``import algosystem``.
#: The torch/onnx/sklearn/sb3/gymnasium group is the TORCH-FREE-capstone guard; the
#: httpx/statsmodels/plotly/typer group is the lazy-heavy-dependency guard.
_FORBIDDEN_AT_IMPORT: tuple[str, ...] = (
    "torch",
    "onnx",
    "onnxruntime",
    "sklearn",
    "stable_baselines3",
    "gymnasium",
    "httpx",
    "statsmodels",
    "plotly",
    "typer",
)


@pytest.mark.unit
def test_import_algosystem_is_side_effect_free() -> None:
    """A fresh interpreter importing ``algosystem`` loads no forbidden module."""
    forbidden = ", ".join(repr(m) for m in _FORBIDDEN_AT_IMPORT)
    code = (
        "import sys\n"
        "import algosystem\n"
        f"forbidden = ({forbidden},)\n"
        "leaked = sorted(m for m in forbidden if m in sys.modules)\n"
        "assert not leaked, f'forbidden modules imported at load: {leaked}'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"import-purity subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout


@pytest.mark.unit
def test_public_api_is_importable() -> None:
    """The curated ``__all__`` names are all importable from the top-level package."""
    import algosystem

    missing = [name for name in algosystem.__all__ if not hasattr(algosystem, name)]
    assert not missing, f"names in __all__ missing from the package: {missing}"


@pytest.mark.unit
def test_no_torch_in_installed_distribution() -> None:
    """Defensive: the import graph of the public submodules stays torch-free."""
    code = (
        "import importlib, sys\n"
        "for mod in (\n"
        "    'algosystem.signals.library',\n"
        "    'algosystem.backtest.engine',\n"
        "    'algosystem.execution.paper_broker',\n"
        "    'algosystem.execution.parity',\n"
        "    'algosystem.evaluation.metrics',\n"
        "    'algosystem.evaluation.pbo',\n"
        "    'algosystem.evaluation.verdict',\n"
        "    'algosystem.serve',\n"
        "    'algosystem.plots',\n"
        "    'algosystem.cli',\n"
        "):\n"
        "    importlib.import_module(mod)\n"
        "banned = ('torch', 'onnxruntime', 'onnx', 'sklearn')\n"
        "leaked = sorted(m for m in banned if m in sys.modules)\n"
        "assert not leaked, f'torch-family modules imported: {leaked}'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"torch-free subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout
