from typing import Generic, Iterator, NamedTuple, TypeVar, cast
from jaxtyping import Array, Int
import jax
import jax.numpy as jnp
from rl_david_silver_2015.mdp.common import BatchedTerminal, RandomKey
from rl_david_silver_2015.mdp.abstract_mdpf import (
    AbstractMDPFramework,
    BatchedStateType,
    MDPType,
    PolicyType,
)
from rl_david_silver_2015.mdp.common import (
    DEFAULT_RANDOM_KEY,
    BatchedActionType,
    BatchedReward,
    MDPSample,
)

MDPFrameworkType = TypeVar("MDPFrameworkType", bound="AbstractMDPFramework")


class BatchedAgentUpdate(NamedTuple, Generic[BatchedActionType]):
    seq_idx: Int[Array, "..."]
    a_t: BatchedActionType


class BatchedEnvironmentUpdate(NamedTuple, Generic[BatchedStateType]):
    seq_idx: Int[Array, "..."]
    r_tp1: BatchedReward
    s_tp1: BatchedStateType
    terminal: BatchedTerminal


def sample_mdp_batched_generator(
    mdp: MDPType,
    policy: PolicyType,
    mdp_framework: type[
        AbstractMDPFramework[
            MDPType,
            PolicyType,
            BatchedStateType,
            BatchedActionType
        ]
    ],
    s_1s: BatchedStateType,
    max_n: int = 1000,
    random_key: Array = DEFAULT_RANDOM_KEY,
) -> Iterator[
    BatchedAgentUpdate[BatchedActionType] | BatchedEnvironmentUpdate[BatchedStateType]
]:
    """
    Allows sampling of a batch of episodes from the MDP.
    Also allows for early termination of subset episodes if they reach terminal states.
    """
    terminal_predicate = mdp_framework.get_terminal_state_predicate(mdp)

    # Future work: Turn this into a class that allows dynamically adding new episodes.
    # Future work: Turn this into a JIT-able function? - Or not. This function is meant to be for fast BUT INTERACTIVE use cases where we visualize the sampling process.
    assert isinstance(
        s_1s, jnp.ndarray
    ), "This sampler assumes that actions and samples are represented as JAX arrays."
    s_t = s_1s  # current state
    seq_idx = jnp.arange(s_1s.shape[0])  # sequence for each episode being run in batch
    seq_idx_mask = jnp.ones_like(seq_idx, dtype=bool)  # mask for the sequence index
    t = 1

    while bool(jnp.any(seq_idx_mask)) and t < max_n:

        random_key, action_key, state_key = jax.random.split(random_key, 3)

        assert isinstance(random_key, RandomKey)
        assert isinstance(action_key, RandomKey)
        assert isinstance(state_key, RandomKey)

        seq_subset = seq_idx[seq_idx_mask]
        s_t_masked: BatchedStateType = cast(BatchedStateType, s_t[seq_idx_mask])

        # Sample actions for the current states
        a_t = mdp_framework.sample_policy(policy, s_t_masked, action_key)
        yield BatchedAgentUpdate(seq_subset, a_t)

        # Sample next states and rewards
        mdp_return = mdp_framework.sample_mdp(mdp, s_t_masked, a_t, state_key)
        is_terminal = terminal_predicate(mdp_return.s_tp1)
        yield BatchedEnvironmentUpdate(
            seq_subset, mdp_return.r_tp1, mdp_return.s_tp1, is_terminal
        )

        s_t = s_t.at[seq_subset].set(mdp_return.s_tp1)
        new_terminal_idx = seq_subset[is_terminal]

        # Update the sequence index mask by masking new terminal states
        seq_idx_mask = seq_idx_mask.at[new_terminal_idx].set(False)
        t += 1


if __name__ == "__main__":
    # Example usage
    from rl_david_silver_2015.mdp.sampler_generator import sample_mdp_batched_generator
    from rl_david_silver_2015.mdp.tabular_mdpf import (
        SAMPLE_TABULAR_STARTING_STATE,
        SAMPLE_TABULAR_MDP,
        SAMPLE_TABULAR_POLICY,
        TabularMDPFramework,
    )

    for x in sample_mdp_batched_generator(
        SAMPLE_TABULAR_MDP,
        SAMPLE_TABULAR_POLICY,
        TabularMDPFramework,
        SAMPLE_TABULAR_STARTING_STATE,
    ):
        print(x)
