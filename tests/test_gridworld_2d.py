import pytest
import jax.numpy as jnp
from rl_david_silver_2015.mdp.tabular_mdp import TabularMDP
from rl_david_silver_2015.mdp.gridworld_2d import create_gridworld_2d_tabular  # adjust import path as needed


def test_gridworld_basic_2x2():
    # 2×2 grid, terminals at (0,0) and (1,1)
    gw = create_gridworld_2d_tabular(
        shape=(2, 2),
        terminal_states=[(0, 0), (1, 1)],
        step_cost=-1.0,
        gamma=0.8,
    )
    assert isinstance(gw, TabularMDP)
    assert gw.n_states == 4
    assert gw.n_actions == 4
    assert pytest.approx(float(gw.gamma)) == 0.8

    T = gw.transition
    R = gw.reward

    # State 0 (idx=0) is terminal: absorb, zero reward
    assert jnp.allclose(T[0, :, 0], jnp.ones(4))
    assert jnp.allclose(R[0, :], jnp.zeros(4))

    # State 3 (idx=3) is terminal: absorb, zero reward
    assert jnp.allclose(T[3, :, 3], jnp.ones(4))
    assert jnp.allclose(R[3, :], jnp.zeros(4))

    # State 1 (idx=1) at (0,1):
    #   up (0)    → stays in 1
    #   down (1)  → 3
    #   left (2)  → 0
    #   right(3)  → stays in 1
    expected_T1 = jnp.array([
        [0.0, 1.0, 0.0, 0.0],  # up
        [0.0, 0.0, 0.0, 1.0],  # down
        [1.0, 0.0, 0.0, 0.0],  # left
        [0.0, 1.0, 0.0, 0.0],  # right
    ], dtype=jnp.float32)
    assert jnp.array_equal(T[1], expected_T1)

    # Rewards for state 1: landing in terminal gives 0, otherwise step_cost
    expected_R1 = jnp.array([-1.0, 0.0, 0.0, -1.0], dtype=jnp.float32)
    assert jnp.allclose(R[1], expected_R1)

def test_invalid_gamma():
    # Negative gamma -> ValueError
    with pytest.raises(ValueError):
        create_gridworld_2d_tabular(
            shape=(2, 2),
            terminal_states=[(0, 0)],
            step_cost=0.0,
            gamma=-0.1,
        )
    # Gamma > 1 -> ValueError
    with pytest.raises(ValueError):
        create_gridworld_2d_tabular(
            shape=(2, 2),
            terminal_states=[(0, 0)],
            step_cost=0.0,
            gamma=1.1,
        )