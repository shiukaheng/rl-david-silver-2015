"""
Practical for lecture 3 of David Silver's Reinforcement Learning course.
We demonstrate what we can do given a MDP (Markov Decision Process) and a policy.
"""

from rl_david_silver_2015.gridworld2d import create_gridworld2d
from rl_david_silver_2015.gridworld2d_tui import gridworld2d_policy_rollout_tui
from rl_david_silver_2015.policy_iteration import policy_iteration_vectorized
from rl_david_silver_2015.value_iteration import value_iteration_vectorized

if __name__ == "__main__":

    # Create a grid world MDP
    mdp = create_gridworld2d(shape=(4, 4), terminal_states=[(0, 0), (3, 3)], step_cost=-1.0, gamma=0.9)
    mdp.validate()

    # Perform policy iteration
    optimal_policy, optimal_value = policy_iteration_vectorized(mdp)
    gridworld2d_policy_rollout_tui(mdp, optimal_policy, V=optimal_value, terminal_states=[(0, 0), (3, 3)])

    optimal_value2, optimal_policy2 = value_iteration_vectorized(mdp)
    gridworld2d_policy_rollout_tui(mdp, optimal_policy2, V=optimal_value2, terminal_states=[(0, 0), (3, 3)])