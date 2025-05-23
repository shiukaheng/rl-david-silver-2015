from rl_david_silver_2015.mdp.constants import DEFAULT_RANDOM_KEY
from rl_david_silver_2015.mdp.generic_types import Gamma
from rl_david_silver_2015.mdp.mdp import MDP, BatchedMDPReturn, MDPReturn


import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


from dataclasses import dataclass
from functools import cached_property

from rl_david_silver_2015.mdp.tabular_policy import TabularPolicy
from rl_david_silver_2015.mdp.tabular_types import TabularAction, TabularState, BatchTabularState, BatchTabularAction



@dataclass(frozen=True)
class TabularMDP(MDP[TabularState, TabularAction, BatchTabularState, BatchTabularAction]):
    """
    Markov Decision Process (MDP) defined by a tuple (S, A, P, R, gamma).
    S: set of states
    A: set of actions
    P: transition function
    R: reward function
    gamma: discount factor

    States an here are only represented by indices. This can be extended to include more information.
    """

    n_states: int
    n_actions: int
    transition: Float[Array, "n_states n_actions n_states"]
    reward: Float[Array, "n_states n_actions"]
    gamma: Gamma

    def __post_init__(self):
        self.assert_mdp_valid()

    def assert_mdp_valid(self):
        """
        Checks whether MDP transition probabilities are valid.
        Raises ValueError if any transition probabilities do not sum to 1.
        """

        # Check shape
        assert self.transition.ndim == 3, f"Transition matrix must be 3D (s, a, s'), got {self.transition.ndim}D"
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

    @cached_property
    def terminal_state_indices(self) -> jnp.ndarray:
        """
        Returns the indices of terminal states.

        A state is terminal if for all actions,
        the transition vector is a one-hot vector pointing to itself.
        """
        eye = jnp.eye(self.n_states)  # shape: (n_states, n_states)
        # Compare each (s,a) transition to identity row of s
        # Result: (n_states, n_actions, n_states) == (n_states, 1, n_states)
        # → broadcast identity: (n_states, n_actions, n_states)
        match_self = jnp.all(self.transition == eye[:, None, :], axis=-1)
        # Now match_self[s, a] is True if action a at state s is self-transition
        # Check if all actions at state s are self-transition
        terminal_mask = jnp.all(match_self, axis=-1)  # shape: (n_states,)
        return jnp.nonzero(terminal_mask, size=self.n_states)[0]

    def sample(self, s_t: TabularState, a_t: TabularAction, random_key: Array = DEFAULT_RANDOM_KEY) -> MDPReturn[TabularState]:
        """
        Scalar version of sampling that delegates to the batched version.
        """
        # Wrap into batch of size 1
        s_batched = jnp.expand_dims(s_t, axis=0)  # shape (1,)
        a_batched = jnp.expand_dims(a_t, axis=0)  # shape (1,)
        return_batched = self.sample_batched(s_batched, a_batched, random_key)

        # Unwrap the single-element batch
        return MDPReturn(
            r_tp1=return_batched.r_tp1[0],
            s_tp1=return_batched.s_tp1[0],
            terminal=return_batched.terminal[0],
        )

    def sample_batched(
        self,
        s_t: BatchTabularState,
        a_t: BatchTabularAction,
        random_key: Array = DEFAULT_RANDOM_KEY
    ) -> BatchedMDPReturn[TabularState]:
        """
        Vectorized transition sampling for batches of states and actions, using logits,
        while ensuring transitions with zero probability are never sampled.
        """
        batch_size = s_t.shape[0]
        keys = jax.random.split(random_key, batch_size)

        # Extract transition probabilities
        transition_probs = self.transition[s_t, a_t]  # shape: (b, n_states)
        transition_logits = jnp.where(
            transition_probs > 0,
            jnp.log(transition_probs),
            -jnp.inf
        )  # shape: (b, n_states)

        # Sample next states using categorical distribution over logits
        s_tp1 = jax.vmap(jax.random.categorical)(keys, transition_logits)

        # Get rewards and terminal status
        r_tp1 = self.reward[s_t, a_t]  # shape: (b,)
        terminal = jnp.isin(s_tp1, self.terminal_state_indices)  # shape: (b,)

        return BatchedMDPReturn(r_tp1=r_tp1, s_tp1=s_tp1, terminal=terminal)
    
    def create_uniform_policy(self) -> TabularPolicy:
        """
        Create a uniform policy for this MDP.
        """
        return TabularPolicy.from_uniform(self.n_states, self.n_actions)