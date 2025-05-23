from rl_david_silver_2015.mdp import MDP, Policy


import jax.numpy as jnp
from jaxtyping import Array


def eval_policy(mdp: MDP, policy: Policy, theta=1e-10, max_iters=1000) -> Array:
    """
    Iterative Policy Evaluation: Computes V^π using the Bellman expectation equation.
    """
    V = jnp.zeros(mdp.n_states)

    def bellman_update(V):
        """
        One sweep over all states updating V(s) under current policy.
        """
        V_new = jnp.zeros_like(V)
        for s in range(mdp.n_states):
            v_s = 0.0
            for a in range(mdp.n_actions):
                pi_a = policy[s, a]
                r_sa = mdp.reward[s, a]
                p_sas = mdp.transition[s, a]  # shape: (n_states,)
                v_s += pi_a * jnp.sum(p_sas * (r_sa + mdp.gamma * V))
            V_new = V_new.at[s].set(v_s)
        return V_new

    for i in range(max_iters):
        V_new = bellman_update(V)
        delta = jnp.max(jnp.abs(V_new - V))
        V = V_new
        if delta < theta:
            break

    return V


def eval_policy_vectorized(mdp: MDP, policy: Policy, theta=1e-10, max_iters=1000) -> Array:
    """
    Iterative Policy Evaluation: Fully vectorized version without loops.
    """
    V = jnp.zeros(mdp.n_states)

    for i in range(max_iters):
        # Compute the expected immediate reward: Rπ[s] = ∑_a π(a|s) * R[s,a]
        R_pi = jnp.sum(policy * mdp.reward, axis=1)  # shape: (n_states,)

        # Compute Pπ[s, s'] = ∑_a π(a|s) * P[s, a, s']
        P_pi = jnp.einsum("ia,ias->is", policy, mdp.transition)  # shape: (n_states, n_states)

        # Bellman update: V_new = R_pi + γ * P_pi @ V
        V_new = R_pi + mdp.gamma * P_pi @ V

        delta = jnp.max(jnp.abs(V_new - V))
        V = V_new
        if delta < theta:
            break

    return V


def eval_policy_closed_form(mdp: MDP, policy: Policy) -> Array:
    """
    Evaluate policy analytically using matrix algebra:
    V = (I - γ * Pπ)^(-1) * Rπ
    """
    n_states = mdp.n_states

    # Compute expected rewards Rπ: shape (n_states,)
    R_pi = jnp.sum(policy * mdp.reward, axis=1)

    # Compute expected transitions Pπ: shape (n_states, n_states)
    P_pi = jnp.einsum("ia,ias->is", policy, mdp.transition)

    # Solve linear system: V = (I - γ * Pπ)^-1 * Rπ
    I = jnp.eye(n_states)
    A = I - mdp.gamma * P_pi
    V = jnp.linalg.solve(A, R_pi)

    return V