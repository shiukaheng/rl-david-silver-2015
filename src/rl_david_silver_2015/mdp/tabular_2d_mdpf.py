from dataclasses import dataclass
from typing import Tuple, Callable
import jax.numpy as jnp
from jaxtyping import Array, Int, Float, Bool
from jax.experimental.checkify import checkify

from rl_david_silver_2015.mdp.abstract_mdpf import AbstractMDPFramework
from rl_david_silver_2015.mdp.common import MDPSample, RandomKey
from rl_david_silver_2015.mdp.tabular_mdpf import (
    TabularMDP,
    TabularPolicy,
    TabularBatchedState,
    TabularBatchedAction,
    TabularMDPFramework,
)

# Type aliases for 2D wrapper
Tabular2DPolicy = Float[Array, "state_x state_y actions"]
Tabular2DBatchedState = Int[Array, "b 2"]
Tabular2DBatchedAction = TabularBatchedAction


def coordinates_to_indices(
    coords: Tabular2DBatchedState,
    shape: Tuple[int, int]
) -> TabularBatchedState:
    """
    Convert 2D coordinates (row, col) to flat indices.
    """
    return coords[:, 0] * shape[1] + coords[:, 1]


def indices_to_coordinates(
    indices: TabularBatchedState,
    shape: Tuple[int, int]
) -> Tabular2DBatchedState:
    """
    Convert flat indices back to 2D coordinates (row, col).
    """
    return jnp.column_stack((indices // shape[1], indices % shape[1]))


def tabular_2d_policy_to_1d(policy: Tabular2DPolicy) -> TabularPolicy:
    """
    Flatten a (state_x, state_y, actions) policy to (n_states, actions).
    """
    # Use -1 to let JAX infer the first dimension
    return jnp.reshape(policy, (-1, policy.shape[-1]))


@dataclass(frozen=True)
class Tabular2DMDP:
    mdp1d: TabularMDP
    # Grid shape as Python ints for JIT safety
    shape: Tuple[int, int]

    @staticmethod
    def create(
        prob2d: Float[Array, "state_x state_y actions state_x state_y"],
        reward2d: Float[Array, "state_x state_y actions"],
        gamma: Float[Array, ""]
    ) -> "Tabular2DMDP":
        """
        Build a 2D MDP from raw 2D probabilities and rewards.

        Args:
            prob2d: transition tensor of shape (sx, sy, a, sx, sy)
            reward2d: reward tensor of shape (sx, sy, a)
            gamma: discount factor scalar
        Returns:
            A Tabular2DMDP wrapping a flat TabularMDP
        """
        sx, sy, a = prob2d.shape[0], prob2d.shape[1], prob2d.shape[2]
        # Validate reward shape
        if reward2d.shape != (sx, sy, a):
            raise ValueError(f"reward2d shape {reward2d.shape} must be ({{sx}}, {{sy}}, {{a}})")
        # Validate prob2d target grid dims
        if prob2d.shape[3:] != (sx, sy):
            raise ValueError(f"prob2d target dims {prob2d.shape[3:]} must be ({{sx}}, {{sy}})")
        # Number of states
        n = sx * sy
        # Rearrange and flatten transitions to (n, a, n)
        #  prob2d: (sx, sy, a, sx, sy) -> (sx, sy, sx, sy, a)
        reordered = jnp.transpose(prob2d, (0, 1, 3, 4, 2))
        flat = jnp.reshape(reordered, (n, n, a))
        P1d = jnp.transpose(flat, (0, 2, 1))  # (n_states, n_actions, n_states)
        # Flatten rewards to (n, a)
        R1d = jnp.reshape(reward2d, (n, a))
        # Build 1D MDP
        mdp1d = TabularMDP(P=P1d, R=R1d, gamma=gamma)
        return Tabular2DMDP(mdp1d=mdp1d, shape=(sx, sy))


class Tabular2DMDPFramework(
    AbstractMDPFramework[
        Tabular2DMDP,
        Tabular2DPolicy,
        Tabular2DBatchedState,
        Tabular2DBatchedAction,
    ]
):

    @staticmethod
    def get_terminal_state_predicate(
        mdp2d: Tabular2DMDP
    ) -> Callable[[Tabular2DBatchedState], Bool[Array, "b"]]:
        """
        Return a function that tests if 2D states are terminal.
        """
        # Precompute the 1D terminal predicate once
        pred1d = TabularMDPFramework.get_terminal_state_predicate(mdp2d.mdp1d)

        def is_terminal(states2d: Tabular2DBatchedState) -> Bool[Array, "b"]:
            flat_idx = coordinates_to_indices(states2d, mdp2d.shape)
            return pred1d(flat_idx)

        return is_terminal

    @staticmethod
    def sample_mdp(
        mdp2d: Tabular2DMDP,
        states2d: Tabular2DBatchedState,
        actions2d: Tabular2DBatchedAction,
        key: RandomKey,
    ) -> MDPSample[Tabular2DBatchedState]:
        """
        Sample next states and rewards for a batch of 2D states and actions.
        """
        flat_idx = coordinates_to_indices(states2d, mdp2d.shape)
        sample1d = TabularMDPFramework.sample_mdp(
            mdp2d.mdp1d, flat_idx, actions2d, key
        )
        next_states2d = indices_to_coordinates(sample1d.s_tp1, mdp2d.shape)
        return MDPSample(s_tp1=next_states2d, r_tp1=sample1d.r_tp1)

    @staticmethod
    def sample_policy(
        policy2d: Tabular2DPolicy,
        states2d: Tabular2DBatchedState,
        key: RandomKey,
    ) -> Tabular2DBatchedAction:
        """
        Sample actions from a 2D-stacked policy for a batch of 2D states.
        """
        # Flatten policy
        policy1d = tabular_2d_policy_to_1d(policy2d)
        # Extract first two dims as Python ints
        dim0, dim1 = policy2d.shape[0], policy2d.shape[1]
        shape2d: Tuple[int, int] = (int(dim0), int(dim1))
        # Convert 2D states to flat indices
        flat_idx = coordinates_to_indices(states2d, shape2d)
        # Delegate to 1D framework
        return TabularMDPFramework.sample_policy(policy1d, flat_idx, key)

SAMPLE_2D_TABULAR_MDP = jnp.array()