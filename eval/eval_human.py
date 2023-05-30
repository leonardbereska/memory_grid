# Set up experiment to collect human results
# The subject plays the GUI of the Memory Grid game for 10 rounds for each of the 5 environments
# The results are saved in a csv file, the mean and sem are calculated and displayed

import os
import sys
import time
import csv
import numpy as np
import pandas as pd
import pygame
import argparse
from collections import namedtuple



# Import the game
from gui import GridMazeGUI
from env import GridMazeEnv


# Set up the experiment
# Set up the game
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_view', action='store_true', default=False)  
    parser.add_argument('--view_distance', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=10)
    args = parser.parse_args()

    rand_specs = namedtuple('rand_specs', 'seed random_maze random_targets random_agent_position')
    random_specs = rand_specs(seed=args.seed, random_maze=True, random_targets=True, random_agent_position=True)
    np.random.seed(random_specs.seed)  # set seed for reproducibility
    # define dict to store results, keys are the environment size and values are the rewards
    env_rewards = {}

    for size in [7, 9, 11, 13, 15]:
        env_name = 'GridMaze{}x{}'.format(size, size)
        env = GridMazeEnv(env_name, random_specs=random_specs, view_distance=args.view_distance, render_mode='human')
        gui = GridMazeGUI(env, seed=args.seed, full_view=args.full_view, n_episodes=args.n_episodes)
        rewards = gui.run()
        env_rewards[env_name] = rewards

    # Save the results
    # Create a dataframe from the results
    df = pd.DataFrame.from_dict(env_rewards)
    # Save the dataframe as a csv file
    df.to_csv('human_results.csv')
    # Calculate the means and sems for each environment
    means = df.mean(axis=0)
    sems = df.sem(axis=0)
    # Print the means and sems
    print('means', means)
    print('sems', sems)



