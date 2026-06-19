"""Unit tests for the SIMULATED bar-by-bar paper broker (the "live" path).

These lock the paper-broker contract independently of the parity oracle:

- config validation (finite, non-negative friction; positive initial equity);
- the next-bar-open fill record (a decision at bar ``t`` fills at bar ``t+1``);
- bar-finality (a forming bar can never trigger a fill — structurally guarded);
- the result accounting against a transparent bar-by-bar reference replay;
- the JSON-serializable ``to_dict`` shapes for the API boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from algosystem._exceptions import BarFinalityError, InsufficientDataError, ValidationError
from algosystem.backtest.bar_finality import BarStatus, guard_order
from algosystem.execution.paper_broker import (
    Fill,
    PaperBrokerConfig,
    PaperBrokerResult,
    replay,
)


def _reference_curve(
    returns: np.ndarray,
    positions: np.ndarray,
    *,
    cost_bps: float,
    slippage_bps: float,
    initial_position: float = 0.0,
    initial_equity: float = 1.0,
) -> np.ndarray:
    """A transparent bar-by-bar reference replay of the fill-timing contract."""
    friction = (cost_bps + slippage_bps) / 1e4
    prev = initial_position
    wealth = initial_equity
    curve = []
    for t in range(len(positions) - 1):
        target = float(positions[t])
        cost = friction * abs(target - prev)
        net = target * float(returns[t + 1]) - cost
        wealth *= 1.0 + net
        curve.append(wealth)
        prev = target
    return np.asarray(curve, dtype="float64")


# --------------------------------------------------------------------------- #
# Config validation                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_config_defaults_are_the_documented_friction() -> None:
    """The default config is 5 bps cost + 2 bps slippage, flat, wealth index 1.0."""
    cfg = PaperBrokerConfig()
    assert cfg.cost_bps == 5.0
    assert cfg.slippage_bps == 2.0
    assert cfg.initial_position == 0.0
    assert cfg.initial_equity == 1.0
    assert cfg.to_dict() == {
        "cost_bps": 5.0,
        "slippage_bps": 2.0,
        "initial_position": 0.0,
        "initial_equity": 1.0,
    }


@pytest.mark.unit
@pytest.mark.parametrize("field", ["cost_bps", "slippage_bps"])
def test_config_rejects_negative_friction(field: str) -> None:
    """Negative cost / slippage is rejected at construction."""
    with pytest.raises(ValidationError):
        PaperBrokerConfig(**{field: -1.0})


@pytest.mark.unit
@pytest.mark.parametrize("field", ["cost_bps", "slippage_bps", "initial_position", "initial_equity"])
def test_config_rejects_non_finite(field: str) -> None:
    """Non-finite friction / position / equity is rejected at construction."""
    with pytest.raises(ValidationError):
        PaperBrokerConfig(**{field: float("nan")})


@pytest.mark.unit
def test_config_rejects_non_positive_initial_equity() -> None:
    """The starting wealth must be strictly positive (a wealth index)."""
    with pytest.raises(ValidationError):
        PaperBrokerConfig(initial_equity=0.0)
    with pytest.raises(ValidationError):
        PaperBrokerConfig(initial_equity=-5.0)


# --------------------------------------------------------------------------- #
# Replay: shapes, fills, and the next-bar-open timing                           #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_replay_returns_n_minus_one_scored_bars() -> None:
    """With ``N`` input bars the replay scores exactly ``N - 1`` bars / fills."""
    returns = np.array([0.0, 0.01, -0.02, 0.03, 0.0])
    positions = np.array([0.0, 1.0, 1.0, -1.0, 0.0])
    res = replay(returns, positions, PaperBrokerConfig(cost_bps=0.0, slippage_bps=0.0))
    assert isinstance(res, PaperBrokerResult)
    assert res.n_bars == 4
    assert res.net_returns.shape == (4,)
    assert res.equity_curve.shape == (4,)
    assert res.positions.shape == (4,)
    assert len(res.fills) == 4


@pytest.mark.unit
def test_fill_bar_index_is_the_next_bar() -> None:
    """A decision read at the close of bar ``t`` fills at the OPEN of bar ``t+1``."""
    returns = np.array([0.0, 0.01, 0.02])
    positions = np.array([1.0, -1.0, 0.0])
    res = replay(returns, positions, PaperBrokerConfig(cost_bps=0.0, slippage_bps=0.0))
    assert [f.bar_index for f in res.fills] == [1, 2]
    assert [f.target_position for f in res.fills] == [1.0, -1.0]


@pytest.mark.unit
def test_initial_position_drives_the_first_turnover_charge() -> None:
    """Opening from a non-flat book charges the first trade against ``initial_position``."""
    returns = np.array([0.0, 0.0, 0.0])
    positions = np.array([1.0, 1.0, 1.0])
    # Already long at the start => the first fill trades 0 (no turnover, no cost).
    res = replay(
        returns,
        positions,
        PaperBrokerConfig(cost_bps=10.0, slippage_bps=5.0, initial_position=1.0),
    )
    assert res.fills[0].traded == pytest.approx(0.0)
    assert res.fills[0].cost == pytest.approx(0.0)
    assert res.turnover == pytest.approx(0.0)


@pytest.mark.unit
def test_initial_equity_scales_the_curve() -> None:
    """A different starting wealth index scales the whole curve proportionally."""
    returns = np.array([0.0, 0.05, 0.05])
    positions = np.array([1.0, 1.0, 1.0])
    base = replay(returns, positions, PaperBrokerConfig(cost_bps=0.0, slippage_bps=0.0))
    scaled = replay(
        returns,
        positions,
        PaperBrokerConfig(cost_bps=0.0, slippage_bps=0.0, initial_equity=1000.0),
    )
    np.testing.assert_allclose(scaled.equity_curve, base.equity_curve * 1000.0)


# --------------------------------------------------------------------------- #
# Accounting vs a transparent reference                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_replay_matches_a_handwritten_reference_replay() -> None:
    """The broker's equity curve equals an explicit bar-by-bar reference."""
    rng = np.random.default_rng(7)
    returns = rng.standard_normal(150) * 0.01
    positions = rng.integers(-1, 2, size=150).astype("float64")
    res = replay(returns, positions, PaperBrokerConfig(cost_bps=5.0, slippage_bps=2.0))
    expected = _reference_curve(returns, positions, cost_bps=5.0, slippage_bps=2.0)
    np.testing.assert_allclose(res.equity_curve, expected, rtol=0.0, atol=1e-12)


@pytest.mark.unit
def test_turnover_is_total_one_way_traded() -> None:
    """Turnover sums ``|pi_t - pi_{t-1}|`` over the path (pi_{-1} = initial_position)."""
    returns = np.zeros(5)
    positions = np.array([1.0, -1.0, -1.0, 0.0, 0.0])
    res = replay(returns, positions, PaperBrokerConfig(cost_bps=0.0, slippage_bps=0.0))
    # 0->1 (1) + 1->-1 (2) + -1->-1 (0) + -1->0 (1) = 4 over the 4 scored bars.
    assert res.turnover == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# Bar-finality: a forming bar can never produce a fill                          #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_guard_order_blocks_a_forming_bar() -> None:
    """The bar-finality guard the broker calls rejects a forming/partial bar."""
    guard_order(BarStatus.CLOSED, bar_index=3)  # closed -> fine.
    with pytest.raises(BarFinalityError, match="forming"):
        guard_order(BarStatus.FORMING, bar_index=3)


@pytest.mark.unit
def test_every_fill_is_a_closed_bar_decision() -> None:
    """Each emitted fill corresponds to a CLOSED decision bar (none from a forming bar)."""
    returns = np.array([0.0, 0.01, 0.02, 0.03])
    positions = np.array([1.0, 1.0, 0.0, -1.0])
    res = replay(returns, positions, PaperBrokerConfig(cost_bps=0.0, slippage_bps=0.0))
    # The decision indices (t) are 0..n_bars-1; all are closed bars (the forming
    # final bar pi_{N-1} has no next bar and yields no fill).
    assert all(0 <= f.bar_index <= len(positions) - 1 for f in res.fills)
    assert len(res.fills) == positions.size - 1


# --------------------------------------------------------------------------- #
# Input validation                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_replay_rejects_length_mismatch() -> None:
    """Misaligned returns / positions are rejected."""
    with pytest.raises(ValidationError, match="same length"):
        replay(np.array([0.0, 0.01, 0.02]), np.array([1.0, 0.0]))


@pytest.mark.unit
def test_replay_rejects_too_few_bars() -> None:
    """Fewer than two bars cannot fill a single next-bar-open order."""
    with pytest.raises(InsufficientDataError):
        replay(np.array([0.0]), np.array([1.0]))


@pytest.mark.unit
def test_replay_rejects_non_finite_inputs() -> None:
    """NaN/Inf in returns or positions would silently desync the curves => rejected."""
    with pytest.raises(ValidationError):
        replay(np.array([0.0, np.nan, 0.01]), np.array([1.0, 1.0, 0.0]))
    with pytest.raises(ValidationError):
        replay(np.array([0.0, 0.01, 0.02]), np.array([1.0, np.inf, 0.0]))


@pytest.mark.unit
def test_replay_rejects_a_non_config_object() -> None:
    """A wrong ``config`` type is rejected (defensive guard)."""
    with pytest.raises(ValidationError):
        replay(np.array([0.0, 0.01]), np.array([1.0, 0.0]), config="nope")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Serialization for the API boundary                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_result_to_dict_is_json_shaped() -> None:
    """``to_dict`` yields plain Python lists/floats (JSON-serializable)."""
    returns = np.array([0.0, 0.01, 0.02])
    positions = np.array([1.0, 0.0, -1.0])
    res = replay(returns, positions, PaperBrokerConfig(cost_bps=5.0, slippage_bps=2.0))
    d = res.to_dict()
    assert isinstance(d["net_returns"], list)
    assert all(isinstance(x, float) for x in d["equity_curve"])
    assert isinstance(d["fills"], list)
    assert d["fills"][0]["bar_index"] == 1
    assert d["n_bars"] == 2
    assert d["meta"]["cost_bps"] == 5.0


@pytest.mark.unit
def test_fill_to_dict_round_trips_fields() -> None:
    """A ``Fill`` serializes its four fields verbatim."""
    fill = Fill(bar_index=4, target_position=-1.0, traded=-2.0, cost=0.0003)
    assert fill.to_dict() == {
        "bar_index": 4,
        "target_position": -1.0,
        "traded": -2.0,
        "cost": 0.0003,
    }
