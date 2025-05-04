"""
Practical for lecture 3 of David Silver's Reinforcement Learning course.
We demonstrate what we can do given a MDP (Markov Decision Process) and a policy.
"""

from rl_david_silver_2015.gridworld2d import create_gridworld2d
from rl_david_silver_2015.gridworld2d_tui import gridworld2d_policy_rollout_tui
from rl_david_silver_2015.iter_policy import iter_policy

if __name__ == "__main__":

    # Create a grid world MDP
    mdp = create_gridworld2d(shape=(4, 4), terminal_states=[(0, 0), (3, 3)], step_cost=-1.0, gamma=0.9)
    mdp.validate()

    # Perform policy iteration
    optimal_policy, optimal_value = iter_policy(mdp)

    gridworld2d_policy_rollout_tui(mdp, optimal_policy, V=optimal_value, terminal_states=[(0, 0), (3, 3)])