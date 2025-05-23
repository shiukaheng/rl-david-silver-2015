from typing import Generator, NamedTuple, Tuple
from jaxtyping import Array, Float, Int
import jax.numpy as jnp
import jax

class MDP(NamedTuple):
    """
    A Markov Decision Process (MDP) is defined by a tuple (S, A, P, R, gamma).
    S: set of states
    A: set of actions
    P: transition function
    R: reward function
    gamma: discount factor
    """
    states: Int[Array, "n_states"]
    actions: Int[Array, "n_actions"]
    transition: Float[Array, "n_states n_actions n_states"]
    reward: Float[Array, "n_states n_actions"]
    gamma: Float[Array, ""]

    def validate(self):
        # Check shape
        assert self.transition.ndim == 3, f"Transition matrix must be 3D (s, a, s'), got {self.transition.ndim}D"
        n_states, n_actions, n_states_2 = self.transition.shape
        assert n_states == n_states_2, "Transition matrix last dim must equal number of states"

        # Check transition probabilities
        probs_sum = jnp.sum(self.transition, axis=2)
        if not jnp.allclose(probs_sum, 1.0, atol=1e-5):
            bad_sa = jnp.where(jnp.abs(probs_sum - 1.0) > 1e-5)
            raise ValueError(f"Transition probabilities must sum to 1 for each (s,a). "
                            f"Failed at indices: {bad_sa}")

        # Check gamma
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"Discount factor gamma={self.gamma} must be in [0, 1]")

        print("✅ MDP validation passed.")

    def idx_to_xy(self, idx: int) -> Tuple[int, int]:
        """
        Convert a state index to (x, y) coordinates.
        """
        H, W = self.states.shape
        return divmod(idx, W)
    
    def xy_to_idx(self, xy: Tuple[int, int]) -> int:
        """
        Convert (x, y) coordinates to a state index.
        """
        x, y = xy
        H, W = self.states.shape
        if 0 <= x < H and 0 <= y < W:
            return x * W + y
        else:
            raise ValueError(f"Coordinates {xy} are out of bounds for grid of shape {self.states.shape}.")
    
    def iterate_states(self) -> Generator[Tuple[int, Tuple[int, int]], None, None]:
        """
        Iterate over all states in the MDP.
        """
        for s in range(self.states.shape[0] * self.states.shape[1]):
            yield s, self.idx_to_xy(s)

    def iterate_actions(self) -> Generator[int, None, None]:
        """
        Iterate over all actions in the MDP.
        """
        for a in range(self.actions.shape[0]):
            yield a

    @property
    def n_states(self) -> int:
        """
        Number of states in the MDP.
        """
        return self.states.shape[0] * self.states.shape[1]
    
    @property
    def n_actions(self) -> int:
        """
        Number of actions in the MDP.
        """
        return self.actions.shape[0]

Policy = Float[Array, "n_states n_actions"]

def create_random_policy(mdp: MDP) -> Policy:
    """
    Create a random policy for the given MDP.
    The policy is a matrix of shape (n_states, n_actions) where each row sums to 1.
    """
    n_states, n_actions = mdp.transition.shape[0], mdp.transition.shape[1]
    policy = jax.random.uniform(jax.random.PRNGKey(0), (n_states, n_actions))
    policy /= jnp.sum(policy, axis=1, keepdims=True)
    return policy

def validate_policy(mdp: MDP, policy: Policy):
    """
    Validate the given policy against the MDP.
    The policy must have the same number of states and actions as the MDP.
    """
    n_states, n_actions = mdp.transition.shape[0], mdp.transition.shape[1]
    assert policy.shape == (n_states, n_actions), f"Policy shape {policy.shape} does not match MDP shape {(n_states, n_actions)}"
    assert jnp.all(jnp.isclose(jnp.sum(policy, axis=1), 1.0)), "Policy rows must sum to 1"
    print("✅ Policy validation passed.")