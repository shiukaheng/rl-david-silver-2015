from rl_david_silver_2015.mdp.constants import DEFAULT_RANDOM_KEY


import jax.numpy as jnp
from jaxtyping import Array, Float
import jax
from jax.experimental.checkify import checkify
import flax.struct as struct



from dataclasses import dataclass

from rl_david_silver_2015.mdp.policy import Policy
from rl_david_silver_2015.mdp.tabular_types import TabularState, TabularAction, BatchTabularState, BatchTabularAction


@dataclass(frozen=True)
class TabularPolicy(Policy[TabularState, TabularAction, BatchTabularState, BatchTabularAction]):

    action_probabilities: Float[Array, "n_states n_actions"]

    def __post_init__(self):
        self.assert_policy_valid()

    def assert_policy_valid(self, threshold: float = 1e-5):
        """
        Checks whether policy probabilities are valid.
        Raises ValueError if any policy probabilities do not sum to 1.
        """
        # Check shape
        assert self.action_probabilities.ndim == 2, f"Policy matrix must be 2D (s, a), got {self.action_probabilities.ndim}D"
        # Check policy probabilities
        probs_sum = jnp.sum(self.action_probabilities, axis=1)
        ok = jnp.allclose(probs_sum, 1.0, atol=threshold)
        checkify(ok, f"Policy probabilities do not sum to 1.0: {probs_sum} (threshold: {threshold})")

    @jax.jit
    def sample(self, s_t: TabularState, random_key: Array = DEFAULT_RANDOM_KEY) -> TabularAction:
        """ Sample an action from the policy given a state. """
        s_batched = jnp.expand_dims(s_t, axis=0) 
        return_batched = self.sample_batched(s_batched, random_key)

        # Unwrap
        return return_batched[0]
    
    @jax.jit
    def sample_batched(self, s_t: BatchTabularState, random_key: Array = DEFAULT_RANDOM_KEY) -> BatchTabularAction:
        """ Sample actions for a batch of states with vectorized keys. """
        batch_size = s_t.shape[0]
        keys = jax.random.split(random_key, batch_size)

        # Extract the action probabilities for the given states and convert to logits
        action_probs = self.action_probabilities[s_t]
        transition_logits = jnp.where(
            action_probs > 0,
            jnp.log(action_probs),
            -jnp.inf
        ) # shape (b, n_actions)

        # Sample actions from the categorical distribution
        a_tp1 = jax.vmap(jax.random.categorical)(keys, transition_logits)
        return a_tp1
    
    @staticmethod
    def from_uniform(n_states: int, n_actions: int) -> "TabularPolicy":
        """ Create a uniform policy. """
        action_probabilities = jnp.full((n_states, n_actions), 1.0 / n_actions)
        return TabularPolicy(action_probabilities=action_probabilities)