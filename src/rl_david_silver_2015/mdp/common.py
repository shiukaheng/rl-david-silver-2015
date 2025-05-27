from typing import Callable, Generic, NamedTuple, TypeVar

from jaxtyping import Array, Float, Bool
import jax

MDPType = TypeVar("MDPType")
PolicyType = TypeVar("PolicyType")
BatchedStateType = TypeVar("BatchedStateType", bound=Array)  # Batched states
BatchedActionType = TypeVar("BatchedActionType", bound=Array)  # Batched actions
BatchedReward = Float[Array, "b"]  # Batched rewards
BatchedTerminal = Bool[Array, "b"]  # Batched terminal status
RandomKey = Array  # Typically a JAX PRNG key


class MDPSample(NamedTuple, Generic[BatchedStateType]):
    r_tp1: BatchedReward  # rewards for next states
    s_tp1: BatchedStateType  # next states


MDPSamplingFunction = Callable[
    [MDPType, BatchedStateType, BatchedActionType, RandomKey],
    MDPSample[BatchedStateType],
]
PolicySamplingFunction = Callable[
    [PolicyType, BatchedStateType, RandomKey], BatchedActionType
]


class BatchedAgentUpdate(NamedTuple, Generic[BatchedActionType]):
    a_t: BatchedActionType


class BatchedEnvironmentUpdate(NamedTuple, Generic[BatchedStateType]):
    r_tp1: BatchedReward
    s_tp1: BatchedStateType
    terminal: BatchedTerminal


BatchTerminalStateEvaluator = Callable[[BatchedStateType], BatchedTerminal]

DEFAULT_RANDOM_KEY = jax.random.PRNGKey(0)
