from jaxtyping import Int, Array
import jax.numpy as jnp

from rl_david_silver_2015.mdp.gridworld_2d import create_2d_gridworld
from rl_david_silver_2015.mdp.tabular_2d_mdpf import Tabular2DMDPFramework
from rl_david_silver_2015.mdp.sampler_generator import (
    sample_mdp_batched_generator,
    BatchedAgentUpdate,
    BatchedEnvironmentUpdate,
)

mdp = create_2d_gridworld(
    height          = 5,
    width           = 5,
    terminal_cells  = jnp.array([[0, 4], [4, 0]]),
    obstacle_cells  = jnp.array([[2, 2]]),
    movement_cost   = jnp.array(0.1),
    gamma           = jnp.array(0.9),
)

# uniform random policy over the 25 states
policy = jnp.ones((5 * 5, 4)) / 4

for u in sample_mdp_batched_generator(
    mdp           = mdp,
    policy        = policy,
    mdp_framework = Tabular2DMDPFramework,
    s_1s          = jnp.array([[0, 0],[1, 0],[0, 1],[1, 1]], dtype=jnp.int32),   # start at (0,0)
    max_n         = 500,
):
    if isinstance(u, BatchedAgentUpdate):
        print(f"Agent  : t={u.seq_idx}, a={u.a_t}")
    else:
        print(f"Env    : t={u.seq_idx}, r={u.r_tp1}, s'={u.s_tp1}, done={u.terminal}")
