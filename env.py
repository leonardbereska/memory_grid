import gym
import time
import pygame
import numpy as np
from collections import namedtuple

from memory_grid.maze import GridMaze, grid_mazes
# from maze import GridMaze, grid_mazes

rand_specs = namedtuple('rand_specs', 'seed random_maze random_targets random_agent_position')

class GridMazeEnv():
    """
    Grid-based environment for memory tasks.
    This environment is designed for memory tasks and features a 2D grid layout, as opposed to the 3D layout of the standard memory environment. The agent's observation is partial and egocentric, but from an overhead perspective. Movement is restricted to discrete grid positions, rather than continuous motion.
    This environment is simpler and more compact than the standard memory environment, making it a good choice for quick prototyping or testing memory-related algorithms.
    """

    def __init__(self, env_name, random_specs=None, view_distance=1, render_mode='human'):
        self.view_distance = view_distance
        self.observation_image_size = 2 * self.view_distance + 1
        
        if random_specs is None:
            random_specs = rand_specs(None, True, True, True) 
        self.random_seed = random_specs.seed 
        self.random_maze = random_specs.random_maze
        self.random_targets = random_specs.random_targets
        self.random_agent_position = random_specs.random_agent_position 
       
        self.maze_seed = None if self.random_maze else self.random_seed 
        self.agent_seed = None if self.random_agent_position else self.random_seed
        self.target_seed = None if self.random_targets else self.random_seed 

        self.maze_specs = grid_mazes[env_name]
        self.n_targets = self.maze_specs.n_targets
        self.max_episode_steps = self.maze_specs.max_episode_steps

        self.current_target_id = None
        
        self.grid_maze = GridMaze(self.maze_specs, seed=self.random_seed)
        self.initialize_maze_env()

        self.action_space = gym.spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.action_space.seed(self.random_seed)  # seed for action_space sampler

        self.observable_entities = [' ', '#'] + [str(t) for t in range(self.n_targets)]  # ' ' empty space, '#' wall
        image_size = 2 * self.view_distance + 1 + 2  # size of the image, e.g. 3x3, (2 for padding with target id)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, image_size, image_size), dtype=np.uint8)

        self.render_mode = render_mode
        self.window = None

    def seed(self, seed=None):
        self.random_seed = seed
        self.grid_maze.seed(seed)
        self.action_space.seed(seed)
        return seed

    def initialize_maze_env(self):
        self.maze = self.grid_maze.entity_layer
        self.agent_position = self.grid_maze.agent_position
        self.target_positions = self.grid_maze.target_positions
        self.current_target_id = self.sample_new_target_id()
        self.total_reward = 0
        self.steps = 0

    def reset(self, seed=None):
        if seed is not None:
            self.grid_maze = GridMaze(self.maze_specs, seed=seed)
        elif self.random_maze: 
            self.grid_maze = GridMaze(self.maze_specs, seed=self.maze_seed)
        self.initialize_maze_env()

        if self.random_agent_position: 
            rng = np.random.default_rng(seed=self.agent_seed)
            self.agent_position = self.grid_maze.sample_agent_position(rng=rng)
        if self.random_targets: 
            rng = np.random.default_rng(seed=self.target_seed)
            self.target_positions = self.grid_maze.sample_target_positions(rng=rng)

        return self.get_observation(), {}

    def is_wall(self, position):
        return self.maze[position[0]][position[1]] == '#'

    def is_target(self, position):
        return any([np.array_equal(position, target_position) for target_position in self.target_positions])

    def sample_new_target_id(self):
        available_target_ids = [t for t in range(self.n_targets) if t != self.current_target_id]
        return np.random.choice(available_target_ids)

    def step(self, action):
        self.update_agent_position(action)
        reward = self.get_reward()
        observation = self.get_observation()
        terminated = False
        truncated = False
        if self.steps >= self.max_episode_steps:
            truncated = True
        info = {}
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def update_agent_position(self, action):
        new_position = self.agent_position.copy()
        if action == 0:
            new_position[0] -= 1  # up
        elif action == 1:
            new_position[1] += 1  # right
        elif action == 2:
            new_position[0] += 1  # down
        elif action == 3:
            new_position[1] -= 1  # left
        else:
            raise ValueError('Invalid action: {}'.format(action))

        if not self.is_wall(new_position):  # only walk if new position is not a wall
            self.agent_position = new_position

    def get_reward(self):
        reward = 0
        current_target_position = self.target_positions[self.current_target_id]
        if np.array_equal(self.agent_position, current_target_position):
            reward = self.reward_and_give_new_target()
        return reward

    def reward_and_give_new_target(self):
        self.current_target_id = self.sample_new_target_id()

        reward = 1
        self.total_reward += reward
        return reward

    def observe_field_of_view(self):
        """
        Get observation of the agent's surroundings and the current target id.
        return: 1D array of entity chars (e.g. [' ', '#', ' ', ' ', '1', ' ', ' ', ' ', '2'])
        """
        a = self.agent_position
        v = self.view_distance  # for example, v=1: 3x3, v=2: 5x5 space around agent

        maze = self.maze.copy()
        
        # mark target positions in maze
        for i, target_position in enumerate(self.target_positions):
            maze[target_position[0], target_position[1]] = str(i)

        # add a border of walls around the maze to avoid index out of bounds errors
        maze = np.pad(maze, v - 1, 'constant', constant_values='#')

        # shift all coordinates by v-1 to account for the added border
        observation = maze[a[0] - 1:a[0] + 2 * v, a[1] - 1:a[1] + 2 * v]
        return observation

    def entity_to_color(self, entity):
        """
        Convert entity char to RGB color. 
        return: 1D array of RGB values, e.g. np.array([255, 0, 0]).
        """
        entity_to_color = {' ': np.array([255, 255, 255]),  # empty spaces are white
                           '#': np.array([0, 0, 0]),        # walls are black
                           '0': np.array([255, 0, 0]),      # target 0 is red
                           '1': np.array([0, 255, 0]),      # target 1 is green
                           '2': np.array([0, 0, 255]),      # target 2 is blue
                           '3': np.array([255, 255, 0]),    # target 3 is yellow
                           '4': np.array([0, 255, 255]),    # target 4 is cyan
                           '5': np.array([255, 0, 255])}    # target 5 is magenta
        return entity_to_color[entity]


    def get_observation(self):
        visible_entities = self.observe_field_of_view()

        # surround the visible entities with the current target id as prompt
        visible_entities = np.pad(visible_entities, 1, 'constant', constant_values=str(self.current_target_id))

        image = np.zeros((visible_entities.shape[0], visible_entities.shape[1], 3), dtype=np.uint8)
        for i in range(visible_entities.shape[0]):
            for j in range(visible_entities.shape[1]):
                image[i, j] = self.entity_to_color(visible_entities[i, j])

        image = np.swapaxes(image, 2, 0)  # swap axes to match PyTorch format (C, H, W)
        return image

    def print_maze(self):
        print_maze(self.maze.copy(), self.agent_position, self.target_positions)

    def render(self, full_view=True):

        if self.render_mode == 'rgb_array':
            img = self.get_observation()
            return img

        elif self.render_mode == 'human':
            maze = self.maze.copy()

            def get_maze_image(maze):
                a = self.agent_position
                v = self.view_distance
                image = np.zeros((maze.shape[0], maze.shape[1], 3))
                # mark target positions in maze
                for i, target_position in enumerate(self.target_positions):
                    maze[target_position[0], target_position[1]] = str(i)

                for i in range(maze.shape[0]):
                    for j in range(maze.shape[1]):
                        image[i, j] = self.entity_to_color(maze[i, j])

                        if abs(i - a[0]) > v or abs(j - a[1]) > v:  # color non-observed area 
                            image[i, j] = image[i, j] * 0.5  # zero for black, 0.5 for gray 

                return image

            if full_view:
                image = get_maze_image(maze)
            else:
                image = self.get_observation()
                image = np.transpose(image, (2, 1, 0))  # convert to (width, height, channels) for pygame

            time.sleep(0.03)

            target_id = self.current_target_id
            target_color = self.entity_to_color(str(target_id))

            image = np.transpose(image, (1, 0, 2))  # convert to (width, height, channels) for pygame
            screen_size = 640  

            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode((screen_size, screen_size))
                pygame.display.set_caption('Maze')

            image_surface = pygame.surfarray.make_surface(image)
            image_surface = pygame.transform.scale(image_surface, (screen_size, screen_size))
            self.window.blit(image_surface, (0, 0))

            def create_marker_surface(marker_size, target_color):
                marker_surface = pygame.Surface((marker_size, marker_size), pygame.SRCALPHA)
                pygame.draw.circle(marker_surface, target_color, (marker_size // 2, marker_size // 2), marker_size // 2)
                return marker_surface

            def calculate_agent_position(agent_position, screen_size, maze_shape):
                factor = screen_size / maze_shape[0]
                agent_position = (factor * agent_position[1], factor * agent_position[0])
                return agent_position

            def center_marker(agent_position, marker_size, factor):
                offset = factor / 2 - marker_size / 2 + 1  # shift marker to center of grid cell
                agent_position = (agent_position[0] + offset, agent_position[1] + offset)
                return agent_position

            if full_view:
                marker_size = 50 / maze.shape[0] * 11  # scale marker size with maze size
                factor = screen_size / maze.shape[0]

                # Create surfaces for image and marker
                marker_surface = create_marker_surface(marker_size, target_color)
                agent_position = calculate_agent_position(self.agent_position, screen_size, maze.shape)
                agent_position = center_marker(agent_position, marker_size, factor)

                # Blit image and marker surfaces to window
            else:
                marker_size = 400 / (2 * self.view_distance + 1)  # scale marker size with maze size
                marker_surface = create_marker_surface(marker_size, target_color)
                # place agent in the center of the screen
                agent_position = (screen_size / 2 - marker_size / 2, screen_size / 2 - marker_size / 2)

            if full_view:
                self.window.blit(marker_surface, agent_position)
                font = pygame.font.Font('freesansbold.ttf', 32)
                text = font.render('Score: ' + str(self.total_reward), True, (128, 128, 128))
                self.window.blit(text, (10, 10))

            pygame.display.update()

    def close(self):
        if self.window:
            pygame.display.quit()
            pygame.quit()


def same_maze_layout(env1, env2):
    return (env1.maze == env2.maze).all()

def same_agent_position(env1, env2):
    return (env1.agent_position == env2.agent_position).all()

def same_target_positions(env1, env2):
    for t1, t2 in zip(env1.target_positions, env2.target_positions):
        if not (t1 == t2).all():
            return False
    return True 

def test_all_random_no_seed():
    all_random = rand_specs(None, True, True, True)    
    for choice in choices:
        env1 = GridMazeEnv(choice, all_random)
        env2 = GridMazeEnv(choice, all_random)
        assert not same_maze_layout(env1, env2)
        assert not same_agent_position(env1, env2)
        assert not same_target_positions(env1, env2)

def test_all_random_with_seed():
    all_random_with_seed = rand_specs(42, True, True, True)    
    for choice in choices:
        env1 = GridMazeEnv(choice, all_random_with_seed)
        env2 = GridMazeEnv(choice, all_random_with_seed)
        assert same_maze_layout(env1, env2)
        assert same_agent_position(env1, env2)
        assert same_target_positions(env1, env2)
        env3 = GridMazeEnv(choice, all_random_with_seed)
        env3.reset()
        assert not same_maze_layout(env1, env3)
        assert not same_agent_position(env1, env3)
        assert not same_target_positions(env1, env3)

def test_all_fixed():
    all_fixed = rand_specs(42, False, False, False)
    for choice in choices:
        env1 = GridMazeEnv(choice, all_fixed)
        env2 = GridMazeEnv(choice, all_fixed)
        assert same_maze_layout(env1, env2)
        assert same_agent_position(env1, env2)
        assert same_target_positions(env1, env2)
        env3 = GridMazeEnv(choice, all_fixed)
        env3.reset()
        assert same_maze_layout(env1, env3)
        assert same_agent_position(env1, env3)
        assert same_target_positions(env1, env3)

def test_maze_fixed():
    for agent_random in [True, False]:
        for target_random in [True, False]:
            agent_target = rand_specs(42, False, target_random, agent_random) 
            for choice in choices:
                env1 = GridMazeEnv(choice, agent_target)
                env2 = GridMazeEnv(choice, agent_target)
                assert same_maze_layout(env1, env2)
                assert same_agent_position(env1, env2)
                assert same_target_positions(env1, env2)
                env3 = GridMazeEnv(choice, agent_target)
                env3.reset()
                assert same_maze_layout(env1, env3)
                if not agent_random:
                    assert same_agent_position(env1, env3) 
                if not target_random:
                    assert same_target_positions(env1, env3) 


if __name__ == '__main__':
    test_all_random_no_seed()
    test_all_random_with_seed()
    test_all_fixed()
    test_maze_fixed()


