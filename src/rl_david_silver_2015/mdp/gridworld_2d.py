"""
Factory for a deterministic 4-direction grid world.
Walls (obstacles) block movement; terminals are absorbing.
"""

from typing import Tuple
import jax
import jax.numpy as jnp
import jax.nn as nn
from jaxtyping import Float, Array, Int

from rl_david_silver_2015.mdp.tabular_2d_mdpf import Tabular2DMDP


# ---------------------------------------------------------------------------
#  core builder – pure JAX, no Python ints inside
# ---------------------------------------------------------------------------
def _grid_core(
    height: int, width: int,
    terminal_cells: Int[Array, "n 2"],
    obstacle_cells: Int[Array, "m 2"],
    movement_cost:  Float[Array, ""],
    gamma:          Float[Array, ""],
) -> Tabular2DMDP:
    """
    Returns a Tabular2DMDP with:
      • 4 deterministic actions 0=up,1=right,2=down,3=left
      • Obstacles behave as walls (blocked; agent stays where it is)
      • Terminal states are absorbing with 0 reward
    """
    n_actions = 4
    n_states  = height * width

    # ---- geometry ---------------------------------------------------------
    deltas = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)   # (A,2)

    ys, xs = jnp.meshgrid(jnp.arange(height, dtype=jnp.int32),
                          jnp.arange(width, dtype=jnp.int32),
                          indexing='ij')
    coords = jnp.stack([ys, xs], axis=-1)                                     # (H,W,2)

    next_xy = coords[:, :, None, :] + deltas[None, None, :, :]                # (H,W,A,2)
    next_xy = jnp.clip(next_xy,
                       jnp.array([0, 0], dtype=jnp.int32),
                       jnp.array([height - 1, width - 1], dtype=jnp.int32))

    # ---- flat indices -----------------------------------------------------
    flat_states = ys * width + xs                                                # (H,W)
    flat_next   = next_xy[..., 0] * width + next_xy[..., 1]                      # (H,W,A)

    # Obstacles as walls: if target ∈ obstacles → stay put
    obs_flat  = obstacle_cells[:, 0] * width + obstacle_cells[:, 1]             # (m,)
    hit_wall  = jnp.isin(flat_next, obs_flat).astype(bool)                  # (H,W,A)
    flat_next = jnp.where(hit_wall, flat_states[:, :, None], flat_next)     # redirect

    # ---- transition tensor -----------------------------------------------
    P1d = nn.one_hot(flat_next.reshape(-1), n_states, dtype=jnp.float32)     # (HWA,N)
    P1d = P1d.reshape(height, width, n_actions, n_states)
    P   = P1d.reshape(height, width, n_actions, height, width)

    # ---- masks for terminals ---------------------------------------------
    term_flat  = terminal_cells[:, 0] * width + terminal_cells[:, 1]
    term_mask  = jnp.isin(flat_states, term_flat).astype(bool)               # (H,W)

    eye = jnp.eye(n_states, dtype=jnp.float32).reshape(height, width, 1, height, width)
    eye = jnp.broadcast_to(eye, P.shape)

    P = jnp.where(term_mask[:, :, None, None, None], eye, P)  # terminals absorb

    # ---- rewards ----------------------------------------------------------
    R = -movement_cost * jnp.ones((height, width, n_actions), dtype=jnp.float32)
    R = jnp.where(term_mask[:, :, None], 0.0, R)

    return Tabular2DMDP(P, R, gamma, terminal_flat=term_flat)


# ---------------------------------------------------------------------------
#  thin JIT wrapper – grid size is STATIC for JIT
# ---------------------------------------------------------------------------
create_2d_gridworld = jax.jit(
    _grid_core,
    static_argnums=(0, 1)   # height, width
)
