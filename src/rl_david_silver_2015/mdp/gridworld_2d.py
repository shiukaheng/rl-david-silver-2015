import jax.numpy as jnp
from typing import Callable, Sequence, Tuple, List
from rl_david_silver_2015.mdp.tabular_mdp import TabularMDP

def create_gridworld_2d_tabular(
    shape: Tuple[int, int] = (4, 4),
    terminal_states: Sequence[Tuple[int, int]] = ((0, 0), (3, 3)),
    step_cost: float = -1.0,
    gamma: float = 0.9,
) -> Tuple[TabularMDP, Callable[[int], Tuple[int, int]], Callable[[Tuple[int, int]], int]]:
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

    def idx_to_state(s: int) -> Tuple[int, int]:
        """Convert state index to (row, col) tuple."""
        return divmod(s, W)

    def state_to_idx(state: Tuple[int, int]) -> int:
        """Convert (row, col) tuple to state index."""
        return state[0] * W + state[1]    

    return TabularMDP(
        n_states=n_states,
        n_actions=n_actions,
        transition=T,
        reward=R,
        gamma=gamma,
    ), idx_to_state, state_to_idx

ACTION_TO_INDEX = {
    'up': 0,
    'down': 1,
    'left': 2,
    'right': 3,
}