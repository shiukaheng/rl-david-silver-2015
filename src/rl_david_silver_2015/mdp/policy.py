from typing import Generic, Protocol
from jaxtyping import Array
from rl_david_silver_2015.mdp.generic_types import StateType, ActionType, BatchedStateType, BatchedActionType
from rl_david_silver_2015.mdp.constants import DEFAULT_RANDOM_KEY

class Policy(Protocol, Generic[StateType, ActionType, BatchedStateType, BatchedActionType]):
    """ A protocol for any policy-like object. """
    def sample(self, state: StateType, random_key: Array = DEFAULT_RANDOM_KEY) -> ActionType:
        """ Sample an action from the policy given a state. """
        pass
    def sample_batched(self, states: BatchedStateType, random_key: Array = DEFAULT_RANDOM_KEY) -> BatchedActionType:
        """ Sample actions for a batch of states with vectorized keys. """
        pass