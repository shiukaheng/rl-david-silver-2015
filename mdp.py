from typing import NamedTuple
from jaxtyping import Array, Float, Int
import jax.numpy as jnp


class MDP(NamedTuple):
    """
    A Markov Decision Process (MDP) is defined by a tuple (S, A, P, R, gamma).
    S: set of states
    A: set of actions
    P: transition function
    R: reward function
    gamma: discount factor
    """
    states: Int[Array, "n_states"]
    actions: Int[Array, "n_actions"]
    transition: Float[Array, "n_states n_actions n_states"]
    reward: Float[Array, "n_states n_actions"]
    gamma: Float[Array, ""]

    def validate(self):
        # Check shape
        assert self.transition.ndim == 3, f"Transition matrix must be 3D (s, a, s'), got {T.ndim}D"
        n_states, n_actions, n_states_2 = self.transition.shape
        assert n_states == n_states_2, "Transition matrix last dim must equal number of states"

        # Check transition probabilities
        probs_sum = jnp.sum(self.transition, axis=2)
        if not jnp.allclose(probs_sum, 1.0, atol=1e-5):
            bad_sa = jnp.where(jnp.abs(probs_sum - 1.0) > 1e-5)
            raise ValueError(f"Transition probabilities must sum to 1 for each (s,a). "
                            f"Failed at indices: {bad_sa}")

        # Check gamma
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"Discount factor gamma={self.gamma} must be in [0, 1]")

        print("✅ MDP validation passed.")