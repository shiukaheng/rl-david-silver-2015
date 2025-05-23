import pytest
import jax
import jax.numpy as jnp
from rl_david_silver_2015.mdp.tabular_policy import TabularPolicy


def test_from_uniform():
    n_states, n_actions = 3, 4
    policy = TabularPolicy.from_uniform(n_states, n_actions)
    expected = jnp.full((n_states, n_actions), 1.0 / n_actions)
    assert jnp.allclose(policy.action_probabilities, expected)


def test_assert_policy_valid_pass():
    probs = jnp.array([
        [0.2, 0.8],
        [0.5, 0.5],
    ])
    policy = TabularPolicy(action_probabilities=probs)
    assert policy.action_probabilities.shape == (2, 2)


def test_assert_policy_valid_fail():
    bad_probs = jnp.array([
        [0.5, 0.6],  # sums to 1.1
        [0.3, 0.3],  # sums to 0.6
    ])
    with pytest.raises(ValueError):
        TabularPolicy(action_probabilities=bad_probs)


@pytest.fixture
def uniform_policy():
    return TabularPolicy.from_uniform(n_states=5, n_actions=3)


def test_sample_batched_unwrapped(uniform_policy):
    key = jax.random.PRNGKey(0)
    states = jnp.array([0, 1, 2])
    raw_fn = TabularPolicy.sample_batched.__wrapped__
    actions = raw_fn(uniform_policy, states, key)
    assert actions.shape == (3,)
    assert jnp.all((actions >= 0) & (actions < 3))


def test_sample_scalar_via_batched(uniform_policy):
    key = jax.random.PRNGKey(1)
    raw_batched = TabularPolicy.sample_batched.__wrapped__
    # Batch of size 1 for state=2
    batched = raw_batched(uniform_policy, jnp.array([2]), key)
    # unwrap
    action = batched[0]
    assert isinstance(action, jnp.ndarray)
    assert 0 <= int(action) < 3