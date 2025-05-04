import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Tuple
import jax

def value_iteration(
    mdp,
    theta: float = 1e-10,
    max_iters: int = 1000
) -> Tuple[Float[Array, "n_states"], Float[Array, "n_states n_actions"]]:
    """
    Performs Value Iteration to compute optimal value function V* and optimal policy π*.
    Returns:
        V_star: Optimal value function
        pi_star: Deterministic optimal policy (one-hot)
    """
    V = jnp.zeros(mdp.n_states)

    for i in range(max_iters):
        Q = jnp.zeros((mdp.n_states, mdp.n_actions))
        for a in range(mdp.n_actions):
            r_sa = mdp.reward[:, a]  # shape: (n_states,)
            p_sas = mdp.transition[:, a, :]  # shape: (n_states, n_states)
            Q = Q.at[:, a].set(r_sa + mdp.gamma * (p_sas @ V))

        V_new = jnp.max(Q, axis=1)
        delta = jnp.max(jnp.abs(V_new - V))
        V = V_new

        if delta < theta:
            break

    # Extract greedy policy π*(s) = argmax_a Q(s, a)
    best_actions = jnp.argmax(Q, axis=1)
    pi_star = jax.nn.one_hot(best_actions, mdp.n_actions)

    return V, pi_star


def value_iteration_vectorized(
    mdp,
    theta: float = 1e-10,
    max_iters: int = 1000
) -> Tuple[Float[Array, "n_states"], Float[Array, "n_states n_actions"]]:
    """
    Fully vectorized Value Iteration using the Bellman optimality equation.
    Returns:
        V*: Optimal state value function
        π*: Deterministic optimal policy as a one-hot matrix
    """
    V = jnp.zeros(mdp.n_states)

    for _ in range(max_iters):
        # Compute Q(s, a) = R(s, a) + γ ∑_s' P(s'|s,a) * V(s')
        Q = mdp.reward + mdp.gamma * jnp.einsum("san,n->sa", mdp.transition, V)

        # Bellman optimality update: V(s) = max_a Q(s,a)
        V_new = jnp.max(Q, axis=1)

        # Check convergence
        delta = jnp.max(jnp.abs(V_new - V))
        V = V_new
        if delta < theta:
            break

    # Derive optimal policy π*(s) = argmax_a Q(s,a)
    best_actions = jnp.argmax(Q, axis=1)
    pi_star = jax.nn.one_hot(best_actions, mdp.n_actions)

    return V, pi_star
