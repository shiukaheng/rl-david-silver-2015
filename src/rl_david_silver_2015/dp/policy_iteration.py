"""
Policy iteration:
- Input: Tabular MDP, discount factor, and initial policy.
- Output: An optimal policy and its value function.

Algorithm outline:
1. Initialize the policy arbitrarily.
2. Repeat until convergence:
   a. Policy evaluation: Compute the value function for the current policy.
   b. Policy improvement: Update the policy based on the value function using a greedy approach.
"""