import curses
from curses import wrapper
from typing import Tuple, List
import jax.numpy as jnp

from rl_david_silver_2015.mdp.tabular_mdp import TabularMDP
from rl_david_silver_2015.mdp.gridworld_2d import create_gridworld_2d_tabular  # updated gridworld creator


def gridworld2d_tui_tabular(
    shape: Tuple[int, int] = (4, 4),
    terminal_states: List[Tuple[int, int]] = [(0, 0), (3, 3)],
    start_pos: Tuple[int, int] = (0, 1),
    step_cost: float = -1.0,
    gamma: float = 0.9,
):
    """
    Launch a curses-based TUI for a deterministic gridworld implemented via TabularMDP.

    - Uses create_gridworld_2d_tabular(...) to build the MDP.
    - WASD keys to move, Q to quit.
    - Displays steps and total reward in real-time.
    """
    # Build the TabularMDP
    mdp: TabularMDP = create_gridworld_2d_tabular(
        shape=shape,
        terminal_states=terminal_states,
        step_cost=step_cost,
        gamma=gamma,
    )

    H, W = shape
    # flatten index grid for lookup
    idx_grid = jnp.arange(H * W).reshape(H, W)

    # movement mapping: key -> action index, dy/dx
    key2action = {
        ord('w'): (0, (-1, 0)),  # Up
        ord('s'): (1, (1, 0)),   # Down
        ord('a'): (2, (0, -1)),  # Left
        ord('d'): (3, (0, 1)),   # Right
    }

    agent_pos = list(start_pos)
    total_reward = 0.0
    steps = 0
    
    def draw_grid(stdscr):
        stdscr.clear()
        for i in range(H):
            for j in range(W):
                ch = '.'
                if (i, j) in terminal_states:
                    ch = 'T'
                if (i, j) == tuple(agent_pos):
                    ch = '@'
                stdscr.addstr(i, j * 2, ch)
        stdscr.addstr(H + 1, 0, 'Use WASD to move, Q to quit.')
        stdscr.addstr(H + 2, 0, f'Steps: {steps}   Total Reward: {total_reward}')
        stdscr.refresh()

    def main(stdscr):
        nonlocal agent_pos, total_reward, steps
        curses.curs_set(0)
        draw_grid(stdscr)

        while True:
            key = stdscr.getch()
            if key in (ord('q'), ord('Q')):
                break
            if key not in key2action:
                continue

            action_idx, (dy, dx) = key2action[key]
            ny, nx = agent_pos[0] + dy, agent_pos[1] + dx
            # check bounds
            if not (0 <= ny < H and 0 <= nx < W):
                continue

            s = int(idx_grid[agent_pos[0], agent_pos[1]])
            # determine next state via MDP transition (deterministic)
            probs = mdp.transition[s, action_idx]
            ns = int(jnp.argmax(probs))
            # compute coords
            ni, nj = divmod(ns, W)

            steps += 1
            # reward lookup
            r = float(mdp.reward[s, action_idx])
            total_reward += r

            agent_pos = [ni, nj]
            draw_grid(stdscr)

            # if moved into terminal
            if (ni, nj) in terminal_states:
                stdscr.addstr(H + 3, 0, f'Reached terminal at {ni,nj}. Press any key to exit.')
                stdscr.getch()
                break

    wrapper(main)

if __name__ == "__main__":
    gridworld2d_tui_tabular()