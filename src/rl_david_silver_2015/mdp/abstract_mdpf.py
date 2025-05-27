from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

from rl_david_silver_2015.mdp.common import (
    BatchedActionType,
    BatchedStateType,
    BatchedTerminal,
    MDPSample,
    RandomKey,
)

MDPType = TypeVar("MDPType")
PolicyType = TypeVar("PolicyType")
TerminalPredicate = Callable[[BatchedStateType], BatchedTerminal]


class AbstractMDPFramework(
    ABC,
    Generic[
        MDPType, PolicyType, BatchedStateType, BatchedActionType
    ],
):
    @staticmethod
    @abstractmethod
    def get_terminal_state_predicate(mdp: MDPType) -> TerminalPredicate:
        """
        Get the terminal states of the MDP.
        """
        raise NotImplementedError(
            "This method should be implemented to return terminal states."
        )

    @staticmethod
    @abstractmethod
    def sample_mdp(
        mdp: MDPType,
        states: BatchedStateType,
        actions: BatchedActionType,
        random_key: RandomKey,
    ) -> MDPSample[BatchedStateType]:
        """
        Sample the next state and reward from the MDP given the current state and action.
        """
        raise NotImplementedError(
            "This method should be implemented to sample from the MDP."
        )

    @staticmethod
    @abstractmethod
    def sample_policy(
        policy: PolicyType, states: BatchedStateType, random_key: RandomKey
    ) -> BatchedActionType:
        """
        Sample an action from the policy given the current state.
        """
        raise NotImplementedError(
            "This method should be implemented to sample from the policy."
        )
