from typing import Generic, NamedTuple, Protocol
from jaxtyping import Array

from rl_david_silver_2015.mdp.constants import DEFAULT_RANDOM_KEY
from rl_david_silver_2015.mdp.generic_types import Gamma, StateType, ActionType, Reward, Terminal, BatchedStateType, BatchedActionType, BatchedReward, BatchedTerminal

# Protocol for further extension later on (e.g., for continuous MDPs)

class MDPReturn(NamedTuple, Generic[StateType]):
    r_tp1: Reward
    s_tp1: StateType
    terminal: Terminal

class BatchedMDPReturn(NamedTuple, Generic[BatchedStateType]):
    r_tp1: BatchedReward
    s_tp1: BatchedStateType
    terminal: BatchedTerminal

class MDP(Protocol, Generic[StateType, ActionType, BatchedStateType, BatchedActionType]):
    """
    A protocol for any MDP-like object.
    """
    gamma: Gamma
    def sample(self, s_t: StateType, a_t: ActionType, random_key: Array = DEFAULT_RANDOM_KEY) -> MDPReturn[StateType]:
        pass
    def sample_batched(self, s_t: BatchedStateType, a_t: BatchedActionType, random_key: Array = DEFAULT_RANDOM_KEY) -> BatchedMDPReturn[BatchedStateType]:
        pass