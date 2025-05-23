import jax.numpy as jnp
from typing import Sequence, Tuple, List
from rl_david_silver_2015.mdp.tabular_mdp import TabularMDP

def create_gridworld_2d_tabular(
    shape: Tuple[int, int] = (4, 4),
    terminal_states: Sequence[Tuple[int, int]] = ((0, 0), (3, 3)),
    step_cost: float = -1.0,
    gamma: float = 0.9,
) -> TabularMDP:
    """
    Build a deterministic 2D gridworld as a TabularMDP.
    - shape: (H, W) grid
    - terminal_states: list of (row, col) that are absorbing with zero reward
    - step_cost: reward for any non‐terminal transition
    - gamma: discount factor
    """
    H, W = shape
    n_states = H * W
    n_actions = 4  # 0=Up,1=Down,2=Left,3=Right

    # Map (i,j) ↔ state index
    idx = jnp.arange(n_states).reshape(H, W)

    # Initialize transition tensor and reward matrix
    T = jnp.zeros((n_states, n_actions, n_states), dtype=jnp.float32)
    R = jnp.full((n_states, n_actions), step_cost, dtype=jnp.float32)

    for i in range(H):
        for j in range(W):
            s = int(idx[i, j])

            # If this is terminal, absorb with zero reward
            if (i, j) in terminal_states:
                T = T.at[s, :, s].set(1.0)
                R = R.at[s, :].set(0.0)
                continue

            # helper to set a deterministic transition
            def _set(a, ni, nj):
                ns = int(idx[ni, nj])
                return T.at[s, a, ns].set(1.0)

            # Up
            if i > 0:
                T = _set(0, i - 1, j)
            else:
                T = T.at[s, 0, s].set(1.0)
            # Down
            if i < H - 1:
                T = _set(1, i + 1, j)
            else:
                T = T.at[s, 1, s].set(1.0)
            # Left
            if j > 0:
                T = _set(2, i, j - 1)
            else:
                T = T.at[s, 2, s].set(1.0)
            # Right
            if j < W - 1:
                T = _set(3, i, j + 1)
            else:
                T = T.at[s, 3, s].set(1.0)

            # Zero reward for moves that land in a terminal state
            for a in range(n_actions):
                nxt = int(jnp.argmax(T[s, a]))  # deterministic
                ni, nj = divmod(nxt, W)
                if (ni, nj) in terminal_states:
                    R = R.at[s, a].set(0.0)

    return TabularMDP(
        n_states=n_states,
        n_actions=n_actions,
        transition=T,
        reward=R,
        gamma=gamma,
    )
