# Memory Grid 
The Memory Grid is an environment specifically designed to evaluate the spatial long-range memory capabilities of reinforcement learning (RL) agents. It is based on the [Memory Maze](https://github.com/jurgisp/memory-maze) by [Pasukonis et al. 2022](https://arxiv.org/abs/2210.13383) with the same goal of isolating the long-range memory capacity of RL agents from confounding factors such as exploration. 
Compared to the Memory Maze, the Memory Grid is a simplified, discrete gridworld environment, representing a 2D top-view instead of a continuous 3D environment. This simplification reduces computational cost while still capturing the key feature of long-range spatial memory.

### Maze Types

| Grid Maze 7x7 | Grid Maze 9x9 | Grid Maze 11x11 | Grid Maze 13x13 | Grid Maze 15x15 |
|------------|--------------|--------------|--------------|--------------|
| 2 targets | 3 targets | 4 targets | 5 targets | 6 targets |
| ![7x7](https://github.com/leonardbereska/memory_grid/assets/34320299/018acfa0-2117-4a45-bd10-835f67300cb8) | ![9x9](https://github.com/leonardbereska/memory_grid/assets/34320299/6d007ebb-3e73-4365-b300-07f1e8c08d3f) | ![11x11](https://github.com/leonardbereska/memory_grid/assets/34320299/cec7b9a0-d30a-4315-99ac-a3bdfd588c71) | ![13x13](https://github.com/leonardbereska/memory_grid/assets/34320299/446abddb-14ad-42d3-aad6-a89270c24840) | ![15x15](https://github.com/leonardbereska/memory_grid/assets/34320299/f70c2324-b68c-4be5-9acb-09f450dc9529) |


In addition to the standard maze types (9x9, 11x11, 13x13, 15x15), we also include a 7x7 maze with 2 goals. The number of goals and maze generation procedure and parameters are the same as in the Memory Maze, also generated with [labmaze](https://github.com/deepmind/labmaze), the same algorithm as used by [DmLab-30](https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/dmlab30). 

This environment is designed for memory tasks and features a 2D grid layout, as opposed to the 3D layout of the standard memory environment. The agent's observation is partial and egocentric, but from a topview perspective. Movement is restricted to discrete grid positions, rather than continuous motion.



- walls are black, empty cells white, the targets are square and have different colors
- the agent is roun
d and has the color of the current target



### Task: Treasure Hunt
In the Memory Grid environment, the agent is placed in a maze with multiple targets of different colors. The agent's goal is to navigate to the corresponding target based on a given color prompt.

Here are the details of the treasure hunt task in the Memory Grid:
- The agent receives a reward of +1 for reaching a target, after which a new target is chosen randomly.
- Touching wrong targets and walls has no effect.
- The agent's observation is partial and egocentric, from a top-view perspective, and movement is restricted to discrete grid positions (up, down, left, right).
- The episode ends after a fixed number of steps, and the total reward earned is equal to the number of targets reached.

Agents with long-range spatial memory should be able to remember the location of targets that are out of sight and navigate directly to them. Agents without long-range spatial memory have randomly traverse the maze to find the target.


## Installation

```
conda create -n grid python=3.11.3
conda activate grid
pip install -r requirements.txt
```	

## Usage 

### Play the Game
The Memory Grid provides an interactive GUI for human players. You can play the game yourself by running the following command:

```
python gui.py --size 9 --full_view --view_distance 2  --rand_name mxtoaos42
```

Here are the command line arguments you can use to customize the game experience:
- `--size`: Specifies the size of the grid.
- `--full_view`: Specifies whether the agent can see the entire grid or only the cells in its field of view.
- `--view_distance`: Specifies the distance of the agent's field of view.
- `--rand_name`: Specifies a randomization for the maze. Each letter in the argument stands for a different element: 'm' for maze, 't' for target, 'a' for agent, 's' for seed. The number '42' represents the seed value. The 'o' and 'x' behind each letter (m/t/a) indicate if the entity is randomized ('o') or fixed ('x').
You can choose to play from the perspective of an agent with a limited field of view or from the perspective of an omniscient agent with a full view of the entire grid.

Agent's perspective (partially observable)

https://github.com/leonardbereska/memory_grid/assets/34320299/b887f0ba-ec50-4c13-be28-dbb506271c70

Fullview

https://github.com/leonardbereska/memory_grid/assets/34320299/bf6783b4-a828-4a04-8073-2a4fc02d9f89
