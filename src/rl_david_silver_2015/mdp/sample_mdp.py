from typing import Any, Generic, Iterator, NamedTuple, Sequence
from jaxtyping import Array, Int
import jax.numpy as jnp
import jax

from rl_david_silver_2015.mdp.constants import DEFAULT_RANDOM_KEY
from rl_david_silver_2015.mdp.generic_types import ActionType, BatchedReward, BatchedTerminal, BatchedActionType, BatchedStateType, StateType
from rl_david_silver_2015.mdp.mdp import MDP
from rl_david_silver_2015.mdp.policy import Policy

class AgentUpdate(NamedTuple, Generic[ActionType]):
    idx: Int[Array, "..."] # m is number of current episodes, M is total number of episodes
    a_t: BatchedActionType

class EnvironmentUpdate(NamedTuple, Generic[StateType]):
    idx: Int[Array, "..."]
    r_tp1: BatchedReward
    s_tp1: BatchedStateType
    terminal: BatchedTerminal

def default_batcher(x: Sequence[Any]) -> Any:
    """
    Takes a sequence of JAX arrays and stacks them into a single batched array.
    Assumes all arrays are the same shape.
    """
    assert len(x) > 0, "Input sequence must not be empty."
    first_shape = x[0].shape
    for i, arr in enumerate(x):
        assert isinstance(arr, jnp.ndarray), f"Element {i} is not a JAX array."
        assert arr.shape == first_shape, f"Shape mismatch at index {i}: expected {first_shape}, got {arr.shape}"
    return jnp.stack(x, axis=0)  # shape: (batch, ...)

def default_unbatcher(x: Any) -> Sequence[Any]:
    """
    Takes a batched JAX array and splits it back into a sequence of arrays.
    Assumes the batch dimension is the first.
    """
    assert isinstance(x, jnp.ndarray), "Input must be a JAX array."
    return [x[i] for i in range(x.shape[0])]

def sample_mdp_batched_generator(
        mdp: MDP, pi: Policy, s_1s: BatchedStateType, 
        max_n: int = 1000, random_key: Array = DEFAULT_RANDOM_KEY,
    ) -> Iterator[AgentUpdate[ActionType] | EnvironmentUpdate[StateType]]:
    """
    Allows sampling of a batch of episodes from the MDP.
    Also allows for early termination of subset episodes if they reach terminal states.
    """
    # Future work: Turn this into a class that allows dynamically adding new episodes.
    # Future work: Turn this into a JIT-able function? - Or not. This function is meant to be for fast BUT INTERACTIVE use cases where we visualize the sampling process.
    assert isinstance(s_1s, jnp.ndarray), "This sampler assumes that actions and samples are represented as JAX arrays."
    s_t = s_1s # current state
    seq_idx = jnp.arange(s_1s.shape[0]) # sequence for each episode being run in batch
    seq_idx_mask = jnp.ones_like(seq_idx, dtype=bool) # mask for the sequence index
    t = 1
    
    while bool(jnp.any(seq_idx_mask)) and t < max_n:
        random_key, action_key, state_key = jax.random.split(random_key, 3)
        seq_subset = seq_idx[seq_idx_mask]
        s_t_masked = s_t[seq_idx_mask]
        # Sample actions for the current states
        a_t = pi.sample_batched(s_t_masked, action_key)
        yield AgentUpdate(seq_subset, a_t)
        # Sample next states and rewards
        mdp_return = mdp.sample_batched(s_t_masked, a_t, state_key)
        yield EnvironmentUpdate(seq_subset, mdp_return.r_tp1, mdp_return.s_tp1, mdp_return.terminal)
        s_t = s_t.at[seq_subset].set(mdp_return.s_tp1)
        new_terminal_idx = seq_subset[mdp_return.terminal]
        # Update the sequence index mask by masking new terminal states
        seq_idx_mask = seq_idx_mask.at[new_terminal_idx].set(False)
        t += 1