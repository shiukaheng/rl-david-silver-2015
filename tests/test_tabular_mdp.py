import pytest
import jax
import jax.numpy as jnp
from rl_david_silver_2015.mdp.tabular_mdp import TabularMDP
from rl_david_silver_2015.mdp.mdp import MDPReturn, BatchedMDPReturn


@pytest.fixture
def simple_mdp():
    # Deterministic 2-state, 2-action MDP
    # state 0: a=0->0, a=1->1
    # state 1: a=0->1, a=1->0
    n_states, n_actions = 2, 2
    T = jnp.zeros((2, 2, 2))
    T = T.at[0, 0, 0].set(1.0)
    T = T.at[0, 1, 1].set(1.0)
    T = T.at[1, 0, 1].set(1.0)
    T = T.at[1, 1, 0].set(1.0)
    R = jnp.array([[0.0, 10.0], [-1.0, 1.0]])
    gamma = 0.9
    return TabularMDP(n_states, n_actions, T, R, gamma)


def test_terminal_state_indices_correct():
    # Build a 2-state MDP where only state 0 is terminal
    T = jnp.zeros((2, 2, 2))
    # state 0 self-loops for both actions
    T = T.at[0, 0, 0].set(1.0)
    T = T.at[0, 1, 0].set(1.0)
    # state 1 always transitions to 0
    T = T.at[1, 0, 0].set(1.0)
    T = T.at[1, 1, 0].set(1.0)
    R = jnp.zeros((2, 2))
    mdp = TabularMDP(2, 2, T, R, gamma=0.5)

    terminals = mdp._calculate_terminal_state_indices
    # Unique terminal states should be {0}
    unique = jnp.unique(terminals)
    assert jnp.array_equal(unique, jnp.array([0]))


def test_sample_batched_deterministic(simple_mdp):
    key = jax.random.PRNGKey(0)
    s_t = jnp.array([0, 1])
    a_t = jnp.array([1, 0])
    # Call the un-jitted Python implementation to avoid jit hashing issues
    raw_fn = TabularMDP.sample_batched.__wrapped__
    out: BatchedMDPReturn = raw_fn(simple_mdp, s_t, a_t, key)

    # Next states should be [1, 1]
    assert jnp.array_equal(out.s_tp1, jnp.array([1, 1]))
    # Rewards should match R[s_t, a_t]
    assert jnp.array_equal(out.r_tp1, jnp.array([10.0, -1.0]))
    # Neither should be terminal
    assert not out.terminal.any()


def test_sample_scalar_via_unwrapped(simple_mdp):
    key = jax.random.PRNGKey(1)
    # Use unwrapped batched sampler and unwrap manually
    raw_batched = TabularMDP.sample_batched.__wrapped__
    batched_ret: BatchedMDPReturn = raw_batched(
        simple_mdp,
        jnp.array([1]),
        jnp.array([1]),
        key
    )
    # Manually unwrap the batch of size 1
    scalar_ret = MDPReturn(
        r_tp1=batched_ret.r_tp1[0],
        s_tp1=batched_ret.s_tp1[0],
        terminal=batched_ret.terminal[0],
    )

    # For (s=1, a=1) next state is 0 with reward 1.0
    assert scalar_ret.s_tp1 == 0
    assert pytest.approx(scalar_ret.r_tp1) == 1.0
    # Since next state 0 is terminal, expect True
    assert bool(scalar_ret.terminal) is True


def test_invalid_transition_sum():
    # Transition probabilities don't sum to 1
    T_bad = jnp.ones((2, 2, 2))
    R = jnp.zeros((2, 2))
    with pytest.raises(ValueError):
        TabularMDP(2, 2, T_bad, R, gamma=0.5)


def test_invalid_gamma():
    # Valid single-state self-loop
    T = jnp.zeros((1, 1, 1)).at[0, 0, 0].set(1.0)
    R = jnp.zeros((1, 1))
    # gamma out of bounds
    with pytest.raises(ValueError):
        TabularMDP(1, 1, T, R, gamma=-0.1)
    with pytest.raises(ValueError):
        TabularMDP(1, 1, T, R, gamma=1.1)
