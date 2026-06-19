"""Probability of Backtest Overfitting via CSCV (Bailey et al., 2017).

[TYPED STUB — signatures, docstrings, and the frozen ``PBOResult`` are final; the
CSCV bodies raise :class:`NotImplementedError` for a sequential author to fill.]

The Combinatorially-Symmetric Cross-Validation (CSCV) estimate of the Probability
of Backtest Overfitting answers: of all the configurations tried, how often does
the IN-SAMPLE best configuration under-perform the MEDIAN out-of-sample? A high PBO
(``>= 0.5``) means the selection procedure is more likely than not picking an
in-sample-overfit artifact rather than a genuinely-best out-of-sample
configuration.

THE PROCEDURE (Bailey et al. 2017):

1. take an ``(T, N)`` matrix of per-bar performance (e.g. per-bar net returns) for
   ``N`` configurations over ``T`` bars;
2. split the ``T`` bars into ``S`` contiguous, equal blocks and form all
   :math:`\binom{S}{S/2}` symmetric partitions into an in-sample (IS) half and a
   complementary out-of-sample (OOS) half;
3. for each partition, find the IS-best configuration ``n*`` (highest IS Sharpe),
   compute its OOS rank, map the rank to a relative rank ``omega in (0, 1)``, and
   the logit ``lambda = ln(omega / (1 - omega))``;
4. the PBO is the fraction of partitions with ``lambda <= 0`` (the IS-best config
   landed in the bottom OOS half) — i.e. ``P(lambda <= 0)``.

The PBO feeds the PURE ``system_has_edge`` verdict: an edge claim requires
``pbo < 0.5`` (alongside DM-significance and a DSR clearing the ``1 - alpha``
confidence level). Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from algosystem._exceptions import ValidationError
from algosystem._typing import FloatArray


@dataclass(frozen=True, slots=True)
class PBOResult:
    """Immutable result of a CSCV Probability-of-Backtest-Overfitting estimate.

    Attributes
    ----------
    pbo:
        The Probability of Backtest Overfitting in ``[0, 1]`` — the fraction of
        symmetric partitions whose IS-best configuration landed in the bottom OOS
        half (``lambda <= 0``).
    logits:
        The per-partition logit ``lambda = ln(omega / (1 - omega))`` of the OOS
        relative rank of each partition's IS-best configuration.
    n_partitions:
        The number of symmetric IS/OOS partitions evaluated
        (:math:`\\binom{S}{S/2}`).
    n_configs:
        The number ``N`` of configurations compared.
    n_splits:
        The number ``S`` of contiguous blocks the bars were split into.
    """

    pbo: float
    logits: FloatArray
    n_partitions: int
    n_configs: int
    n_splits: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        out = asdict(self)
        out["logits"] = [float(x) for x in np.asarray(self.logits).ravel()]
        return out


def probability_of_backtest_overfitting(
    performance: FloatArray,
    *,
    n_splits: int = 16,
) -> PBOResult:
    r"""Estimate the Probability of Backtest Overfitting via CSCV.

    Takes a ``(T, N)`` matrix of per-bar performance for ``N`` configurations over
    ``T`` bars, splits the bars into ``n_splits`` contiguous equal blocks, forms all
    :math:`\binom{S}{S/2}` symmetric in-sample/out-of-sample partitions, and for
    each partition finds the IS-best configuration (highest in-sample Sharpe), maps
    its out-of-sample rank to a relative rank ``omega`` and a logit
    ``lambda = ln(omega / (1 - omega))``. The PBO is the fraction of partitions with
    ``lambda <= 0`` — the IS-best config landed in the bottom OOS half.

    Parameters
    ----------
    performance:
        A ``(T, N)`` matrix of per-bar performance (e.g. per-bar net returns), one
        column per configuration. ``N >= 2`` and ``T`` large enough to split into
        ``n_splits`` non-empty blocks.
    n_splits:
        The number ``S`` of contiguous blocks (must be even and ``>= 2``).

    Returns
    -------
    PBOResult
        The PBO, the per-partition logits, and the partition / config / split
        counts.

    Raises
    ------
    ValidationError
        If ``performance`` is not 2-D with ``N >= 2``, ``n_splits`` is odd / ``< 2``,
        or ``T`` is too short to form ``n_splits`` non-empty blocks.
    NotImplementedError
        Always (this is a typed stub for a sequential author).
    """
    matrix = np.asarray(performance, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValidationError(
            f"probability_of_backtest_overfitting: performance must be 2-D (T, N), "
            f"got ndim={matrix.ndim}."
        )
    n_obs, n_configs = matrix.shape
    if n_configs < 2:
        raise ValidationError(
            f"probability_of_backtest_overfitting: need >= 2 configurations, got {n_configs}."
        )
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValidationError(
            f"probability_of_backtest_overfitting: n_splits must be even and >= 2, "
            f"got {n_splits}."
        )
    if n_obs < n_splits:
        raise ValidationError(
            f"probability_of_backtest_overfitting: T ({n_obs}) must be >= n_splits "
            f"({n_splits}) to form non-empty blocks."
        )
    raise NotImplementedError(
        "probability_of_backtest_overfitting: typed stub — body to be authored."
    )
