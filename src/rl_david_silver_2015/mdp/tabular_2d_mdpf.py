from dataclasses import dataclass
from typing import Tuple, Callable
import jax
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


@jax.tree_util.register_pytree_node_class
class Tabular2DMDP:
    """
    2D-grid wrapper for a flat TabularMDP.  Initialize with 2D transition and reward arrays.
    This class is a JAX PyTree: P, R, and gamma are leaves; shape is auxiliary.
    """
    def __init__(
        self,
        prob2d: Float[Array, "state_x state_y actions state_x state_y"],
        reward2d: Float[Array, "state_x state_y actions"],
        gamma: Float[Array, ""]
    ):
        # shape extraction
        sx, sy, a = int(prob2d.shape[0]), int(prob2d.shape[1]), int(prob2d.shape[2])
        if reward2d.shape != (sx, sy, a):
            raise ValueError(f"reward2d shape {reward2d.shape} must be ({sx}, {sy}, {a})")
        if prob2d.shape[3:] != (sx, sy):
            raise ValueError(f"prob2d target dims {prob2d.shape[3:]} must be ({sx}, {sy})")
        # flatten dimensions
        n = sx * sy
        reordered = jnp.transpose(prob2d, (0, 1, 3, 4, 2))  # (sx, sy, sx, sy, a)
        flat = jnp.reshape(reordered, (n, n, a))             # (sx*sy, sx*sy, a)
        P1d = jnp.transpose(flat, (0, 2, 1))                  # (n, a, n)
        R1d = jnp.reshape(reward2d, (n, a))                   # (n, a)
        self.mdp1d = TabularMDP(P=P1d, R=R1d, gamma=gamma)
        self.shape = (sx, sy)

    def tree_flatten(self):
        # P, R, gamma are leaves; shape is auxiliary static data
        P, R, gamma = self.mdp1d.P, self.mdp1d.R, self.mdp1d.gamma
        return (P, R, gamma), self.shape

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Reconstruct from leaves and aux_data
        sx_sy = aux_data
        P, R, gamma = children
        mdp1d = TabularMDP(P=P, R=R, gamma=gamma)
        return cls(mdp1d=mdp1d, shape=sx_sy)


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