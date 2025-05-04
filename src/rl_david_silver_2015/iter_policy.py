from rl_david_silver_2015.eval_policy import eval_policy_vectorized
from rl_david_silver_2015.mdp import MDP, Policy, create_random_policy


import jax
import jax.numpy as jnp
from jaxtyping import Array


from typing import Tuple


def iter_policy(mdp: MDP, theta=1e-10, max_iters=1000) -> Tuple[Policy, Array]:
    """
    Full Policy Iteration: iterates evaluation and greedy improvement until convergence.
    Returns:
    - Optimal policy π*
    - Corresponding value function V*
    """
    # Start with a random policy
    policy = create_random_policy(mdp)
    policy_stable = False

    for iteration in range(max_iters):
        # Policy Evaluation
        V = eval_policy_vectorized(mdp, policy, theta)

        # Policy Improvement
        new_policy = jnp.zeros_like(policy)

        for s in range(mdp.n_states):
            q_sa = jnp.zeros(mdp.n_actions)
            for a in range(mdp.n_actions):
                p_sas = mdp.transition[s, a]  # shape: (n_states,)
                r_sa = mdp.reward[s, a]
                q_sa = q_sa.at[a].set(jnp.sum(p_sas * (r_sa + mdp.gamma * V)))
            # Greedy improvement: set policy[s] to one-hot of best action
            best_a = jnp.argmax(q_sa)
            new_policy = new_policy.at[s].set(jax.nn.one_hot(best_a, mdp.n_actions))

        # Check if policy has changed
        if jnp.allclose(new_policy, policy):
            policy_stable = True
            break

        policy = new_policy

    if not policy_stable:
        print("⚠️ Policy did not converge within iteration limit.")

    return policy, V