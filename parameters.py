"""
This file stores all relevant test parameters in a singular location for easy adjustment
"""

# General test parameters
s_runs = 1
vis_running = 0  # 1 keeps visualizer from closing (use 0 for multiple runs)
train_new_policies = 1  # 1 = train new set, 0 = re-use trained set

# Rover Domain
n_poi = 3
n_obstacles = 3
n_configs = 10  # The number of configurations used to train policies
n_steps = 40  # Number of time steps rover can move
min_dist = 1.0  # Minimum cut off for distance based value metrics
obs_rad = 4.0  # Minimum radius at which POI may be observed
x_dim = 30.0
y_dim = 30.0
angle_resolution = 90
sensor_type = 'summed'  # summed, density, closest

# Neural Network parameters for rover navigation control
n_inputs = 8
n_hnodes = 10
n_outputs = 2

# Neural Network parameters from autonomous brain
brain_inputs = 8
brain_hnodes = 10
brain_outputs = 1
brain_gen = 1000
n_policies = (2 * n_poi) + n_obstacles

# CCEA
population_size = 60
mutation_rate = 0.1
mutation_prob = 0.3
epsilon = 0.1
generations = 600
num_elites = 5  # Number of elites carried over during selection

