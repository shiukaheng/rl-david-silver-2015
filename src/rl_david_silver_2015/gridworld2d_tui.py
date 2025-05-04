import curses
from curses import wrapper
from typing import Tuple, List
from jaxtyping import Array, Float
import time
import jax.numpy as jnp

def gridworld2d_tui(mdp, start_pos: Tuple[int, int]=(0, 1), terminal_states: List[Tuple[int, int]] = [(0, 0), (3, 3)]):
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

def gridworld2d_policy_rollout_tui(
    mdp,
    policy: Float[Array, "n_states n_actions"],
    V: Float[Array, "n_states"],
    terminal_states: List[Tuple[int, int]] = [(0, 0), (3, 3)],
    max_steps: int = 100
):
    grid = mdp.states
    H, W = grid.shape
    action_to_delta = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # U D L R
    action_names = ['↑', '↓', '←', '→']
    steps = 0
    total_reward = 0

    def draw_grid(stdscr, pos, highlight=False):
        stdscr.clear()
        for i in range(H):
            for j in range(W):
                ch = "."
                if (i, j) in terminal_states:
                    ch = "T"
                if (i, j) == tuple(pos):
                    ch = "@" if not highlight else "X"
                stdscr.addstr(i, j * 2, ch)

    def format_vector(vec):
        return " ".join(f"{v:.2f}" for v in vec)

    def main(stdscr):
        nonlocal steps, total_reward
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        stdscr.nodelay(False)

        # Interactive start state selection
        pos = [0, 0]
        while True:
            draw_grid(stdscr, pos, highlight=True)
            stdscr.addstr(H + 1, 0, "Use arrow keys to select start. Enter to begin. Q to quit.")
            stdscr.refresh()
            key = stdscr.getch()
            if key == ord('q'):
                return
            elif key == curses.KEY_UP and pos[0] > 0:
                pos[0] -= 1
            elif key == curses.KEY_DOWN and pos[0] < H - 1:
                pos[0] += 1
            elif key == curses.KEY_LEFT and pos[1] > 0:
                pos[1] -= 1
            elif key == curses.KEY_RIGHT and pos[1] < W - 1:
                pos[1] += 1
            elif key in [10, 13]:
                break

        agent_pos = pos
        stdscr.nodelay(False)

        # Step-by-step debug rollout
        while steps < max_steps:
            draw_grid(stdscr, agent_pos)
            s_idx = mdp.states[agent_pos[0], agent_pos[1]]
            pi_s = policy[s_idx]
            V_s = V[s_idx]

            # Compute Q-values
            Q = jnp.zeros(mdp.n_actions)
            for a in range(mdp.n_actions):
                r = mdp.reward[s_idx, a]
                p_next = mdp.transition[s_idx, a]
                Q = Q.at[a].set(jnp.sum(p_next * (r + mdp.gamma * V)))

            # Greedy action
            a_star = int(jnp.argmax(pi_s))
            action = action_names[a_star]
            dy, dx = action_to_delta[a_star]

            stdscr.addstr(H + 1, 0, f"State index: {s_idx}")
            stdscr.addstr(H + 2, 0, f"Value V(s): {V_s:.4f}")
            stdscr.addstr(H + 3, 0, f"Policy π(s): {format_vector(pi_s)}")
            stdscr.addstr(H + 4, 0, f"Q-values:     {format_vector(Q)}")
            stdscr.addstr(H + 5, 0, f"Chosen action: {action}")
            stdscr.addstr(H + 6, 0, f"Steps: {steps}   Total Reward: {total_reward:.2f}")
            stdscr.addstr(H + 8, 0, "Press SPACE to step. Q to quit.")
            stdscr.refresh()

            key = stdscr.getch()
            if key == ord('q'):
                break
            if key != ord(' '):
                continue

            # Perform action
            ny, nx = agent_pos[0] + dy, agent_pos[1] + dx
            if 0 <= ny < H and 0 <= nx < W:
                agent_pos = [ny, nx]
                steps += 1
                total_reward += float(mdp.reward[s_idx, a_star])
            else:
                stdscr.addstr(H + 9, 0, f"Invalid move from {tuple(agent_pos)}. Ending.")
                stdscr.getch()
                break

            if tuple(agent_pos) in terminal_states:
                draw_grid(stdscr, agent_pos)
                stdscr.addstr(H + 10, 0, f"Reached terminal state at {tuple(agent_pos)}. Press any key.")
                stdscr.getch()
                break

    wrapper(main)