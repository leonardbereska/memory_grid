import labmaze
import numpy as np
from collections import namedtuple

# The mazes' specifications are the same as in the 3D Memory Maze (Pasukonis et al. 2022).
maze_specs = namedtuple('maze_specs', 'maze_size n_targets max_rooms room_min_size room_max_size max_episode_steps')
grid_mazes = {'GridMaze7x7': maze_specs(7, 2, 6, 3, 5, 100),
              'GridMaze9x9': maze_specs(9, 3, 6, 3, 5, 200),
              'GridMaze11x11': maze_specs(11, 4, 6, 3, 5, 300),
              'GridMaze13x13': maze_specs(13, 5, 6, 3, 5, 400),
              'GridMaze15x15': maze_specs(15, 6, 9, 3, 3, 500)}


class GridMaze():
    """
    2D discrete grid maze generated with labmaze. The specification of the maze is given by maze_specs, taken from the Memory Maze (Pasukonis et al. 2022) environment.
    seed: seed for the random maze generation
    returns maze layout as entity_layer, target_positions, agent_position as class attributes
    """

    def __init__(self, maze_specs, seed=None):
        self.seed(seed)
        self.n_targets = maze_specs.n_targets
        self.maze_size = maze_specs.maze_size
        self.max_rooms = maze_specs.max_rooms
        self.room_min_size = maze_specs.room_min_size
        self.room_max_size = maze_specs.room_max_size

        self.maze = self.generate_maze()
        self.set_entity_layer()

    def seed(self, seed):
        self.random_seed = seed
        np.random.seed(seed)

    def generate_maze(self):
        maze = labmaze.RandomMaze(
            height=self.maze_size + 2,  # add outer walls (1 on each side)
            width=self.maze_size + 2,
            max_rooms=self.max_rooms,
            room_min_size=self.room_min_size,
            room_max_size=self.room_max_size,
            spawns_per_room=3,
            objects_per_room=1,
            max_variations=26,
            simplify=True,
            random_seed=self.random_seed)
        return maze

    def set_entity_layer(self):
        self.entity_layer = self.maze.entity_layer
        self.target_positions = self.place_targets()
        self.agent_position = self.place_agent()
        self.update_entity_layer()

    def regenerate(self):
        self.maze.regenerate()
        self.set_entity_layer()

    def place_targets(self):
        possible_target_positions = np.argwhere(self.entity_layer == 'G')
        counter = 0
        while self.n_targets > len(possible_target_positions):
            self.maze.regenerate()
            self.entity_layer = self.maze.entity_layer
            possible_target_positions = np.argwhere(self.entity_layer == 'G')
            counter += 1
            if counter > 100:
                raise ValueError('Could not place targets, maze too small.')

        idx = np.random.choice(len(possible_target_positions), size=self.n_targets, replace=False)
        target_positions = list(possible_target_positions[idx])
        return target_positions

    def place_agent(self):
        possible_spawn_positions = np.argwhere(self.entity_layer == 'P')
        idx = np.random.choice(len(possible_spawn_positions), size=1, replace=False)[0]
        agent_position = possible_spawn_positions[idx]
        return agent_position

    def update_entity_layer(self):
        # remove possible target positions
        self.entity_layer[self.entity_layer == 'G'] = ' '
        # remove possible agent spawn positions
        self.entity_layer[self.entity_layer == 'P'] = ' '
        # mark walls with '#'
        self.entity_layer[self.entity_layer == '*'] = '#'

    def print_maze(self):
        maze_layout = self.entity_layer
        maze_layout[self.agent_position[0], self.agent_position[1]] = 'A'
        for target_id, target_position in enumerate(self.target_positions):
            maze_layout[target_position[0], target_position[1]] = str(target_id)
        print(maze_layout)


def test_maze_sizes():
    print('GridMaze 7x7')
    maze = GridMaze(grid_mazes['GridMaze7x7'])
    maze.print_maze()
    print('GridMaze 9x9')
    maze = GridMaze(grid_mazes['GridMaze9x9'])
    maze.print_maze()
    print('GridMaze 11x11')
    maze = GridMaze(grid_mazes['GridMaze11x11'])
    maze.print_maze()
    print('GridMaze 13x13')
    maze = GridMaze(grid_mazes['GridMaze13x13'])
    maze.print_maze()
    print('GridMaze 15x15')
    maze = GridMaze(grid_mazes['GridMaze15x15'])
    maze.print_maze()


def test_regenerate():
    print('Regenerate')
    maze = GridMaze(grid_mazes['GridMaze7x7'], seed=0)
    maze.print_maze()
    maze.regenerate()
    maze.print_maze()


def test_fixing_seed():
    print('Fix seed 42')
    maze = GridMaze(grid_mazes['GridMaze7x7'], seed=42)
    maze.print_maze()
    maze = GridMaze(grid_mazes['GridMaze7x7'], seed=42)
    maze.print_maze()


if __name__ == '__main__':
    test_maze_sizes()
    test_regenerate()
    test_fixing_seed()
