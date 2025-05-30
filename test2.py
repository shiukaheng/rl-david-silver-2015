import jax
from jaxtyping import Int, Array
import jax.numpy as jnp

from rl_david_silver_2015.mdp.gridworld_2d import create_2d_gridworld
from rl_david_silver_2015.mdp.tabular_2d_mdpf import Tabular2DMDPFramework
from rl_david_silver_2015.mdp.sampler import jit_sample_mdp_n_steps

mdp = create_2d_gridworld(
    height          = 100,
    width           = 100,
    terminal_cells  = jnp.array([[0, 0], [99, 99]]),  # two terminals
    obstacle_cells  = jnp.array([[50, 50]]),  # one obstacle
    movement_cost   = jnp.array(0.1),
    gamma           = jnp.array(0.9),
)

policy = jnp.ones((100 * 100, 4)) / 4  # uniform random policy over the 10,000 states

def generate_random_starting_states(n: int, height: int, width: int) -> Int[Array, "n 2"]:
    """Generate n random starting states within the grid."""
    return jax.random.randint(
        jax.random.PRNGKey(0), (n, 2), minval=0, maxval=jnp.array([height, width])
    )

f, a, r, s, t =jit_sample_mdp_n_steps(
    mdp = mdp,
    policy = policy,
    mdp_framework = Tabular2DMDPFramework,
    initial_state = generate_random_starting_states(10, 100, 100),  # 10 random starting states
    n_steps= 1000,
)

print(f"Final states: {f}")
print(f"Actions: {a}")
print(f"Rewards: {r}")
print(f"States: {s}")
print(f"Terminals: {t}")