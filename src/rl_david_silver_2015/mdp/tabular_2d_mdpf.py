from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Bool

from rl_david_silver_2015.mdp.abstract_mdpf import (
    AbstractMDPFramework,
    MDPSample,
    RandomKey,
)
from rl_david_silver_2015.mdp.tabular_mdpf import (
    TabularMDP,
    TabularPolicy,
    TabularBatchedState,
    TabularBatchedAction,
    TabularMDPFramework,
)

# ---------------------------------------------------------------------------
#  Utility helpers – unchanged
# ---------------------------------------------------------------------------
Tabular2DPolicy        = Float[Array, "h w a"]
Tabular2DBatchedState  = Int[Array, "b 2"]
Tabular2DBatchedAction = TabularBatchedAction


def coordinates_to_indices(coords: Tabular2DBatchedState,
                           shape: Tuple[int, int]) -> TabularBatchedState:
    """Map (row, col) → flat index."""
    return coords[:, 0] * shape[1] + coords[:, 1]


def indices_to_coordinates(indices: TabularBatchedState,
                           shape: Tuple[int, int]) -> Tabular2DBatchedState:
    """Map flat index → (row, col)."""
    return jnp.column_stack((indices // shape[1], indices % shape[1]))


def tabular_2d_policy_to_1d(policy: Tabular2DPolicy) -> TabularPolicy:
    """(H, W, A) → (H·W, A)."""
    return policy.reshape(-1, policy.shape[-1])


# ---------------------------------------------------------------------------
#  Tabular2DMDP  (now stores explicit terminal set)
# ---------------------------------------------------------------------------
@jax.tree_util.register_pytree_node_class
class Tabular2DMDP:
    """
    2-D grid wrapper around a 1-D TabularMDP.

    Leaves:   mdp1d.P, mdp1d.R, mdp1d.gamma, terminal_flat
    Aux data: grid shape  (so shape is static for JIT)
    """

    # regular constructor for users -----------------------------------------
    def __init__(self,
                 P_full: Float[Array, "h w a h w"],
                 R_full: Float[Array, "h w a"],
                 gamma:  Float[Array, ""],
                 terminal_flat: Int[Array, "n_terminals"]):
        h, w, a = P_full.shape[:3]          # dims as Python ints
        self.shape          = (h, w)
        self.terminal_flat  = terminal_flat
        self.mdp1d          = TabularMDP(
            P = P_full.reshape(h * w, a, h * w),
            R = R_full.reshape(h * w, a),
            gamma = gamma,
        )

    # PyTree plumbing --------------------------------------------------------
    def tree_flatten(self):
        P, R, gamma = self.mdp1d.P, self.mdp1d.R, self.mdp1d.gamma
        return (P, R, gamma, self.terminal_flat), self.shape

    @classmethod
    def tree_unflatten(cls, aux, children):
        shape = aux
        P, R, gamma, terminal_flat = children
        h, w = shape
        return cls(
            P_full = P.reshape(h, w, P.shape[1], h, w),
            R_full = R.reshape(h, w, R.shape[1]),
            gamma  = gamma,
            terminal_flat = terminal_flat,
        )


# ---------------------------------------------------------------------------
#  Framework specialised to the 2-D grid
# ---------------------------------------------------------------------------
class Tabular2DMDPFramework(
    AbstractMDPFramework[
        Tabular2DMDP,
        Tabular2DPolicy,
        Tabular2DBatchedState,
        Tabular2DBatchedAction,
    ]
):
    # --- terminal predicate comes from mdp.terminal_flat -------------------
    @staticmethod
    def get_terminal_state_predicate(
        mdp2d: Tabular2DMDP
    ) -> Callable[[Tabular2DBatchedState], Bool[Array, "b"]]:
        terminal_set = mdp2d.terminal_flat
        def is_terminal(states2d: Tabular2DBatchedState) -> Bool[Array, "b"]:
            flat = coordinates_to_indices(states2d, mdp2d.shape)
            return jnp.isin(flat, terminal_set)
        return is_terminal

    # --- sample_mdp --------------------------------------------------------
    @staticmethod
    def sample_mdp(
        mdp2d: Tabular2DMDP,
        states2d: Tabular2DBatchedState,
        actions:  Tabular2DBatchedAction,
        key:      RandomKey,
    ) -> MDPSample[Tabular2DBatchedState]:
        flat   = coordinates_to_indices(states2d, mdp2d.shape)
        sample = TabularMDPFramework.sample_mdp(mdp2d.mdp1d, flat, actions, key)
        next2d = indices_to_coordinates(sample.s_tp1, mdp2d.shape)
        return MDPSample(r_tp1=sample.r_tp1, s_tp1=next2d)

    # --- sample_policy -----------------------------------------------------
    @staticmethod
    def sample_policy(
        policy2d: Tabular2DPolicy,
        states2d: Tabular2DBatchedState,
        key:      RandomKey,
    ) -> Tabular2DBatchedAction:
        policy1d = tabular_2d_policy_to_1d(policy2d)
        flat     = coordinates_to_indices(states2d, policy2d.shape[:2])
        return TabularMDPFramework.sample_policy(policy1d, flat, key)
