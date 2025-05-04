from typing import List, Tuple
import jax.numpy as jnp
from mdp import MDP


def create_grid_world_2d(
    shape: Tuple[int, int] = (4, 4),
    terminal_states: List[Tuple[int, int]] = [(0, 0), (3, 3)],
    step_cost: float = -1.0,
    gamma: float = 0.9,
) -> MDP:
    H, W = shape
    idx  = jnp.arange(H * W).reshape(H, W)           # state‑id lookup
    actions = jnp.array([0, 1, 2, 3])                # U D L R
    T = jnp.zeros((H * W, 4, H * W))
    R = jnp.full((H * W, 4), step_cost)

    for i in range(H):
        for j in range(W):
            s = idx[i, j]

            if (i, j) in terminal_states:
                T = T.at[s, :, s].set(1.0)           # absorb
                R = R.at[s].set(0.0)
                continue

            # helper to register deterministic successor
            def set_sa(a, ni, nj):
                ns = idx[ni, nj]
                return T.at[s, a, ns].set(1.0)

            # Up
            if i > 0:  T = set_sa(0, i - 1, j)
            else:      T = T.at[s, 0, s].set(1.0)
            # Down
            if i < H - 1: T = set_sa(1, i + 1, j)
            else:         T = T.at[s, 1, s].set(1.0)
            # Left
            if j > 0:  T = set_sa(2, i, j - 1)
            else:      T = T.at[s, 2, s].set(1.0)
            # Right
            if j < W - 1: T = set_sa(3, i, j + 1)
            else:         T = T.at[s, 3, s].set(1.0)

            # zero reward when **entering** a terminal cell
            for a in range(4):
                ns = jnp.argmax(T[s, a])             # deterministic so argmax ok
                ni, nj = divmod(ns, W)
                if (ni, nj) in terminal_states:
                    R = R.at[s, a].set(0.0)

    mdp = MDP(
        states          = idx,
        actions         = actions,
        transition      = T,
        reward          = R,
        gamma           = jnp.array(gamma),
    )
    return mdp
