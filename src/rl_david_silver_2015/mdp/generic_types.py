from typing import TypeVar
from jaxtyping import Array, Float, Bool

StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")
# RewardType = TypeVar("RewardType", default=Float[Array, ""])
# TerminalType = TypeVar("TerminalType", default=Bool[Array, ""])
Reward = Float[Array, ""]
Terminal = Bool[Array, ""]

BatchedStateType = TypeVar("BatchedStateType")
BatchedActionType = TypeVar("BatchedActionType")
# BatchedRewardType = TypeVar("BatchedRewardType", default=Float[Array, "b"])
# BatchedTerminalType = TypeVar("BatchedTerminalType", default=Bool[Array, "b"])
BatchedReward = Float[Array, "b"]
BatchedTerminal = Bool[Array, "b"]

Gamma = Float[Array, ""]