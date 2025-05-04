"""
Practical for lecture 3 of David Silver's Reinforcement Learning course.
We demonstrate what we can do given a MDP (Markov Decision Process) and a policy.
"""

from typing import List, NamedTuple, Tuple
from jaxtyping import Array, Float, Int
import jax.numpy as jnp

from grid_grid_world_2d import create_grid_world_2d
from interactive_gridworld import interactive_gridworld

if __name__ == "__main__":
    # Create a grid world MDP
    mdp = create_grid_world_2d(shape=(4, 4), terminal_states=[(0, 0), (3, 3)], step_cost=-1.0, gamma=0.9)
    mdp.validate()

    # Print the MDP
    print("States:\n", mdp.states)
    print("Actions:\n", mdp.actions)
    print("Transition matrix:\n", mdp.transition)
    print("Reward tensor:\n", mdp.reward)
    print("Discount factor:\n", mdp.gamma)

    # Run the grid world movement simulation
    interactive_gridworld(mdp, start_pos=(0, 1), terminal_states=[(0, 0), (3, 3)])