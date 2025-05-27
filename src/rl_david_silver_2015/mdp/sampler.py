from typing import Tuple
from rl_david_silver_2015.mdp.abstract_mdpf import (
    AbstractMDPFramework,
    MDPType,
    PolicyType,
)
from rl_david_silver_2015.mdp.common import (
    DEFAULT_RANDOM_KEY,
    BatchedActionType,
    BatchedReward,
    BatchedStateType,
    BatchedTerminal,
    MDPSample,
    RandomKey,
)
import jax
from jax import lax


def jit_sample_mdp_n_steps(
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
    initial_state: BatchedStateType,
    n_steps: int = 1000,
    random_key: RandomKey = DEFAULT_RANDOM_KEY,
) -> Tuple[
    BatchedStateType,  # final state
    BatchedActionType,  # [n_steps, batch_size]
    BatchedReward,  # [n_steps, batch_size]
    BatchedStateType,  # [n_steps, batch_size]
    BatchedTerminal,  # [n_steps, batch_size]
]:
    """
    Fully JIT-compatible MDP sampler that runs for a fixed number of steps.
    Tracks actions, rewards, states, terminals at each step.
    """

    batch_size = initial_state.shape[0]
    terminal_predicate = mdp_framework.get_terminal_state_predicate(mdp)

    def scan_step(
        carry: Tuple[BatchedStateType, RandomKey],
        _,
    ) -> Tuple[
        Tuple[BatchedStateType, RandomKey],
        Tuple[BatchedActionType, BatchedReward, BatchedStateType, BatchedTerminal],
    ]:
        s_t, key = carry
        key, action_key, state_key = jax.random.split(key, 3)

        a_t = mdp_framework.sample_policy(policy, s_t, action_key)
        sample: MDPSample[BatchedStateType] = mdp_framework.sample_mdp(
            mdp, s_t, a_t, state_key
        )
        terminal = terminal_predicate(sample.s_tp1)

        return (sample.s_tp1, key), (a_t, sample.r_tp1, sample.s_tp1, terminal)

    (final_state, _), (actions, rewards, states, terminals) = lax.scan(
        scan_step,
        (initial_state, random_key),
        xs=None,
        length=n_steps,
    )

    return final_state, actions, rewards, states, terminals


if __name__ == "__main__":
    # Example usage
    import jax
    from rl_david_silver_2015.mdp.sampler import jit_sample_mdp_n_steps
    from rl_david_silver_2015.mdp.tabular_mdpf import (
        SAMPLE_TABULAR_STARTING_STATE,
        SAMPLE_TABULAR_MDP,
        SAMPLE_TABULAR_POLICY,
        TabularMDPFramework,
    )

    r = jax.jit(jit_sample_mdp_n_steps, static_argnames=["mdp_framework"])(
        SAMPLE_TABULAR_MDP,
        SAMPLE_TABULAR_POLICY,
        TabularMDPFramework,
        SAMPLE_TABULAR_STARTING_STATE,
    )

    print(r)
