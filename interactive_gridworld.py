import curses
from curses import wrapper
from typing import Tuple, List

def interactive_gridworld(mdp, start_pos: Tuple[int, int]=(0, 1), terminal_states: List[Tuple[int, int]] = [(0, 0), (3, 3)]):
    grid = mdp.states
    H, W = grid.shape
    actions = {'w': (-1, 0), 's': (1, 0), 'a': (0, -1), 'd': (0, 1)}
    agent_pos = list(start_pos)
    step_cost = -1  # default step cost shown to user
    total_reward = 0
    steps = 0

    def draw_grid(stdscr):
        stdscr.clear()
        for i in range(H):
            for j in range(W):
                ch = "."
                if (i, j) in terminal_states:
                    ch = "T"
                if (i, j) == tuple(agent_pos):
                    ch = "@"
                stdscr.addstr(i, j * 2, ch)
        stdscr.addstr(H + 1, 0, f"Use WASD to move. Q to quit.")
        stdscr.addstr(H + 2, 0, f"Steps: {steps}   Total Reward: {total_reward}")
        stdscr.refresh()

    def main(stdscr):
        nonlocal agent_pos, total_reward, steps
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        draw_grid(stdscr)

        while True:
            key = stdscr.getch()
            if key == ord('q'):
                break

            if chr(key) in actions:
                dy, dx = actions[chr(key)]
                ny, nx = agent_pos[0] + dy, agent_pos[1] + dx

                if 0 <= ny < H and 0 <= nx < W:
                    agent_pos = [ny, nx]
                    steps += 1
                    if (ny, nx) in terminal_states:
                        total_reward += 0  # no reward after terminal
                        draw_grid(stdscr)
                        stdscr.addstr(H + 3, 0, f"Reached terminal state at {ny,nx}. Press any key to exit.")
                        stdscr.getch()
                        break
                    else:
                        total_reward += step_cost

                draw_grid(stdscr)

    wrapper(main)
