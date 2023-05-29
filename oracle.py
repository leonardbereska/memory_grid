import numpy as np
from collections import deque
import pygame
import time

from maze import GridMaze, grid_mazes
from env import GridMazeEnv
# from memory_grid.maze import GridMaze, grid_mazes
# from memory_grid.env import GridMazeEnv


def find_shortest_path(start, finish, maze):
    """
    Find the shortest path from start to finish in a maze.
    :param start: tuple of (x, y) coordinates of the start cell
    :param finish: tuple of (x, y) coordinates of the finish cell
    :param maze: numpy array of the maze
    :return: list of (x, y) coordinates of the shortest path from start to finish
    """
    # from https://github.com/jurgisp/memory-maze/blob/main/memory_maze/oracle.py
    height, width = maze.shape
    queue = deque()
    visited = np.zeros_like(maze, dtype=bool)
    backtrace = np.zeros(maze.shape + (2,), dtype=int)
    
    # convert to booleans (True means the cell is free)
    # transpose the maze because numpy arrays are indexed by y, x
    maze = maze.T
    maze = maze != '#'
    # print('maze: ', maze)

    queue.append(start)
    visited[start] = True

    # Explore the maze using breadth-first search
    while queue:
        x, y = queue.popleft()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < width and 0 <= new_y < height and maze[new_y, new_x] and not visited[new_y, new_x]:
                queue.append((new_x, new_y))
                visited[new_y, new_x] = True
                backtrace[new_y, new_x, :] = np.array([x, y])
                if (new_x, new_y) == finish:
                    break

    # Check if the destination cell is reachable
    final_x, final_y = finish
    if not visited[final_y, final_x]:
        raise ValueError('Destination seems not reachable')
    
    # Reconstruct the path from the finish to the start
    path = []
    path.append((final_x, final_y))
    while (final_x, final_y) != start:
        final_x, final_y = backtrace[final_y, final_x]
        path.append((final_x, final_y))
        # time out
        if len(path) > 100:
            raise ValueError('Backtrace is too long')
    path.reverse()
    return path


class GridMazeVisualizer:
    def __init__(self, env, n_episodes, seed=None, full_view=True):
        self.env = env
        self.seed = seed
        self.full_view = full_view
        self.action_keys = {(1, 0): 2, (0, -1): 3, (-1, 0): 0, (0, 1): 1}
        self.action_plan = []
        self.episode_rewards = []
        self.n_episodes = n_episodes
    
    def get_oracle_action(self):
        if not self.action_plan:
            start = tuple(self.env.agent_position)
            current_target = self.env.target_positions[self.env.current_target_id]
            finish = tuple(current_target)
            try:
                path = find_shortest_path(start, finish, self.env.maze.copy())
            except ValueError:
                self.reset()
                continue

            actions = [np.array(next_position) - np.array(current_position) for current_position, next_position in zip(path[:-1], path[1:])]
            self.action_plan = actions

        action = self.action_plan.pop(0)
        action = self.action_keys[tuple(action)]
        return action

    def run(self):
        self.reset()
        while len(self.episode_rewards) < self.n_episodes: 
            action = self.get_oracle_action()
            self.step(action)

        return self.episode_rewards

    def step(self, action):
        _, _, terminated, truncated, _ = self.env.step(action)
        if terminated:
            print('terminated')
            self.reset()
        elif truncated:
            print('total reward', self.env.total_reward)
            print('truncated')
            self.episode_rewards.append(self.env.total_reward)
            self.reset()
        else:
            self.env.render(full_view=self.full_view)

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render(full_view=self.full_view)


def evaluate_oracle(run_oracle=False):
    # save the episode rewards
    n_episodes = 100
    env_names = ['GridMaze7x7', 'GridMaze9x9', 'GridMaze11x11', 'GridMaze13x13', 'GridMaze15x15']
    if run_oracle:
        for env_name in env_names:
            env = GridMazeEnv(env_name, render_mode='rgb_array')
            gui = GridMazeVisualizer(env, n_episodes=n_episodes, seed=0, full_view=True)
            episode_rewards = gui.run()
            # save
            np.save('{}_oracle_rewards.npy'.format(env_name), episode_rewards)

    # load the episode rewards
    for env_name in env_names:
        episode_rewards = np.load('{}_oracle_rewards.npy'.format(env_name))

        sem = np.std(episode_rewards)/np.sqrt(n_episodes)
        mean = np.mean(episode_rewards)

        # mean ± sem
        print('mean ± sem for {}: {} ± {}'.format(env_name, np.round(mean, -int(np.floor(np.log10(sem)))), np.round(sem, -int(np.floor(np.log10(sem))))))


if __name__ == '__main__':
    evaluate_oracle()

