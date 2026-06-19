"""Command-line interface (Typer): backtest / paper / compare on the algo-system pipeline.

[TYPED STUB — the Typer app, command signatures, and docstrings are final; the
command bodies raise :class:`NotImplementedError` for a sequential author to fill.]

The CLI is the OFFLINE entry point (the deployed default uses the FastAPI
``run_system`` path). Three commands:

- ``backtest`` — run the vectorized purged walk-forward backtest of a signal on the
  synthetic default (or real PIT bars via ``--data-source polygon``) and print the
  OOS metrics + the PURE verdict;
- ``paper`` — replay the same signal bar-by-bar through the simulated paper broker
  and print the "live" equity summary;
- ``compare`` — run BOTH paths and print the backtest<->live parity max-diff (the
  oracle), the metrics, the DM / DSR / PBO, and the verdict.

Typer (the ``dev`` extra) is imported LAZILY inside :func:`main` so importing this
module has no side effects and pulls in nothing heavy. Demos are behind the
``__main__`` guard. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer


def _build_app() -> typer.Typer:
    """Construct the Typer application with the backtest / paper / compare commands.

    LAZY IMPORT: ``typer`` (the ``dev`` extra) is imported inside this builder so
    importing :mod:`algosystem.cli` is cheap and side-effect-free.

    Returns
    -------
    typer.Typer
        The configured Typer app.

    Raises
    ------
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    raise NotImplementedError("_build_app: typed stub — body to be authored.")


def main() -> None:
    """Console-script entry point (``algo-system`` -> ``algosystem.cli:main``).

    Builds the Typer app (lazy ``typer`` import) and dispatches to the requested
    sub-command. Wired to the ``[project.scripts]`` entry in ``pyproject.toml``.

    Raises
    ------
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    raise NotImplementedError("main: typed stub — body to be authored.")


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
