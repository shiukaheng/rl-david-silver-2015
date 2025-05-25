from dataclasses import dataclass
from typing import NamedTuple
from rl_david_silver_2015.mdp3.abstract_mdpf import AbstractMDPFramework
from jaxtyping import Array, Float, Int, Bool
import jax.numpy as jnp
import jax

from rl_david_silver_2015.mdp3.common import MDPSample, RandomKey

class TabularMDP(NamedTuple):
    """
    Information to represent a Markov Decision Process (MDP).
    S, A are implicitly defined by the indices for states and actions.
    """
    P: Float[Array, "n_states n_actions n_states"]  # transition function
    R: Float[Array, "n_states n_actions"]  # reward function
    gamma: Float[Array, ""]  # discount factor

    @property
    def n_states(self) -> int:
        return self.P.shape[0]
    
    @property
    def n_actions(self) -> int:
        return self.P.shape[1]
    
TabularPolicy = Float[Array, "n_states n_actions"]  # Policy represented as a matrix of action probabilities

TabularBatchedState = Int[Array, "b"]  # Batched states
TabularBatchedAction = Int[Array, "b"]  # Batched actions

@dataclass(frozen=True)
class TabularTerminalPredicate:
    terminal_states: Int[Array, "n_terminal_states"]
    def __call__(self, states: TabularBatchedState) -> Bool[Array, "b"]:
        """
        Evaluate if the given states are terminal.
        """
        return jnp.isin(states, self.terminal_states)

class TabularMDPFramework(AbstractMDPFramework[TabularMDP, TabularPolicy, TabularBatchedState, TabularBatchedAction, TabularTerminalPredicate]):
    
    @staticmethod
    def get_terminal_state_predicate(
        mdp: TabularMDP
    ) -> TabularBatchedState:
        """
        Get the terminal states of the MDP.
        """
        eye = jnp.eye(mdp.P.shape[0])  # Identity matrix of shape (n_states, n_states)
        match_self = jnp.all(mdp.P == eye[:, None, :], axis=-1)
        terminal_mask = jnp.all(match_self, axis=-1)  # shape: (n_states,)
        return TabularTerminalPredicate(jnp.nonzero(terminal_mask, size=mdp.n_states)[0])

    @staticmethod
    def sample_mdp(
        mdp: TabularMDP, 
        states: TabularBatchedState, 
        actions: TabularBatchedAction,
        random_key: RandomKey
    ) -> MDPSample[TabularBatchedState]:
        """
        Vectorized transition sampling for batches of states and actions, using logits,
        while ensuring transitions with zero probability are never sampled.
        """

        batch_size = states.shape[0]
        keys = jax.random.split(random_key, batch_size)

        # Extract transition probabilities
        transition_probs = mdp.P[states, actions]  # shape: (b, n_states)
        transition_logits = jnp.where(
            transition_probs > 0,
            jnp.log(transition_probs),
            -jnp.inf
        )  # shape: (b, n_states)

        # Sample next states using categorical distribution over logits
        next_states = jax.vmap(jax.random.categorical)(keys, transition_logits)

        # Get rewards and terminal status
        rewards = mdp.R[states, actions]  # shape: (b,)

        return MDPSample(r_tp1=rewards, s_tp1=next_states)

    @staticmethod
    def sample_policy(
        policy: TabularPolicy, 
        states: TabularBatchedState, 
        random_key: RandomKey
    ) -> TabularBatchedAction:
        """
        Sample an action from the policy given the current state.
        """
        batch_size = states.shape[0]
        keys = jax.random.split(random_key, batch_size)

        # Extract action probabilities for the given states
        action_probs = policy[states]  # shape: (b, n_actions)

        # Turn action probabilities into logits
        action_logits = jnp.where(
            action_probs > 0,
            jnp.log(action_probs),
            -jnp.inf
        )

        # Sample actions using categorical distribution over logits
        actions = jax.vmap(jax.random.categorical)(keys, action_logits)
        return actions
    
"""
Sample Tabular MDP with 3 states and 2 actions (starting state, left or right, left state, right state)

State 0: Starting state
State 1: Left state
State 2: Right state

Action 1: Move left
Action 2: Move right

Terminal states: State 1 and State 2
"""
SAMPLE_TABULAR_MDP = TabularMDP(
    P=jnp.array([
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # State 0 transitions
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],  # State 1 transitions
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]   # State 2 transitions
    ]),
    R=jnp.array([
        [0.0, 0.0],  # Rewards for actions in state 0
        [-1.0, -1.0],  # Rewards for actions in state 1
        [-1.0, -1.0]   # Rewards for actions in state 2
    ]),
    gamma=jnp.array(0.9)  # Discount factor
)

SAMPLE_TABULAR_STARTING_STATE = jnp.array([0, 0])  # Starting state for the sample

SAMPLE_TABULAR_POLICY = jnp.array([
    [0.5, 0.5],  # Policy for state 0: equal probability for left and right
    [0.5, 0.5],  # Policy for state 1: equal probability for left and right
    [0.5, 0.5]   # Policy for state 2: equal probability for left and right
])