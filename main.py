from ccea import Ccea
from reward_functions import calc_global_reward
from multi_reward import go_away_poi_reward, go_towards_poi_reward, avoid_obstacle_reward
from rover_domain import RoverDomain
from brain import Brain
from Visualizer.visualizer import run_visualizer
from agent import Rover
import csv
import os
import sys
import numpy as np
import warnings
import pickle
from parameters import s_runs, generations, brain_gen, n_poi, n_obstacles, n_configs, n_policies, train_new_policies, \
    n_steps


def save_reward_history(reward_history, file_name):
    """
    Saves the reward history of the rover teams to create plots for learning performance
    :param reward_history:
    :param file_name:
    :return:
    """
    dir_name = 'Output_Data/'  # Intended directory for output files
    save_file_name = os.path.join(dir_name, file_name)

    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(['Performance'] + reward_history)


def save_rover_path(rover_path):  # Save path rovers take using best policy found
    """
    Records the path each rover takes using best policy from CCEA (used by visualizer)
    :param rover_path:  trajectory tracker
    :return:
    """
    dir_name = 'Output_Data/'  # Intended directory for output files

    rpath_name = os.path.join(dir_name, 'Rover_Paths')
    rover_file = open(rpath_name, 'wb')
    pickle.dump(rover_path, rover_file)
    rover_file.close()


def save_trained_policy(filename, policy):
    """
    Save a trained NN policy to a pickle file
    :param filename:
    :param policy:
    :return:
    """
    dir_name = 'Trained_Policies/'
    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)
    path_name = os.path.join(dir_name, filename)

    outfile = open(path_name, 'wb')
    pickle.dump(policy, outfile)
    outfile.close()


def use_saved_policy(filename):
    """
    Re-use a pre-trained policy stored within a pickle file
    :param filename:
    :return:
    """
    dir_name = 'Trained_Policies/'
    path_name = os.path.join(dir_name, filename)
    infile = open(path_name, 'rb')
    policy = pickle.load(infile)
    infile.close()

    return policy


def train_policies():
    """
    Train set of Rover Control policies for the brain to choose
    :return: Dictionary of trained policies
    """
    print("Training Rover Control Policies")

    policies = {}
    pol_id = 0
    # Train POI policies
    for poi_id in range(n_poi):
        weights1 = train_towards_poi_policy(poi_id)
        policies["Policy{0}".format(pol_id)] = weights1
        save_trained_policy('TowardsPOI{0}'.format(poi_id), weights1)
        print("Policy: ", pol_id, " Trained.")
        pol_id += 1

        weights2 = train_away_from_poi_policy(poi_id)
        policies["Policy{0}".format(pol_id)] = weights2
        save_trained_policy('AwayFromPOI{0}'.format(poi_id), weights2)
        print("Policy: ", pol_id, " Trained.")
        pol_id += 1

    # Train Obstacle Avoidance Policy
    for obs_id in range(n_obstacles):
        weights3 = train_obstacle_avoid_policy(obs_id)
        policies["Policy{0}".format(pol_id)] = weights3
        save_trained_policy('AvoidObstacle{0}'.format(obs_id), weights3)
        print("Policy: ", pol_id, " Trained.")
        pol_id += 1

    return policies


def test_policy(policy_id, policy_type):
    """
    Test a selected, trained policy in the rover domain
    :param policy_id:
    :param policy_type:
    :return:
    """
    rd = RoverDomain()
    rd.use_saved_poi_configuration()
    rd.use_saved_obstacle_config()
    rd.use_saved_rover_test_config()
    rov = Rover(0)

    if policy_type == "Towards":
        chosen_pol = use_saved_policy('TowardsPOI{0}'.format(policy_id))
    elif policy_type == "Away":
        chosen_pol = use_saved_policy('AwayFromPOI{0}'.format(policy_id))
    else:
        chosen_pol = use_saved_policy('AvoidObstacle{0}'.format(policy_id))

    rov.reset_rover(rd.rover_test_config)  # Reset rover to initial conditions
    rov.get_network_weights(chosen_pol)  # Apply best set of weights to network
    rd.update_rover_path(rov, -1)

    visualizer_rover_path = np.zeros((s_runs, (n_steps + 1), 3))
    for step_id in range(n_steps):
        rov.rover_sensor_scan(rd.pois, rd.obstacles, n_poi, rd.n_obstacles)
        rov.step(rd.world_x, rd.world_y)
        rd.update_rover_path(rov, step_id)

    visualizer_rover_path[0] = rd.rover_path.copy()
    save_rover_path(visualizer_rover_path)
    run_visualizer()


def train_away_from_poi_policy(poi_id):
    """
    Train NN control policy to travel away from targeted POI
    :param poi_id:
    :return:
    """
    # Create dictionary for each instance of rover and corresponding NN and EA population
    rd = RoverDomain()
    rd.use_saved_poi_configuration()
    rd.use_saved_obstacle_config()
    rd.use_saved_rover_training_configs()
    rov = Rover(0)
    ea = Ccea(rov.n_inputs, rov.n_hnodes, rov.n_outputs)
    ea.create_new_population()

    for gen in range(generations):
        ea.reset_fitness()
        if gen == 0:
            policy_id = 0
        else:
            policy_id = ea.n_elites
        while policy_id < ea.pop_size:
            for config_id in range(n_configs):
                rov.reset_rover(rd.rover_configs[config_id])  # Reset rover to initial conditions
                rov.get_network_weights(ea.population["pop{0}".format(policy_id)])  # Apply network weights from CCEA
                rd.update_rover_path(rov, -1)  # Record starting position of each rover

                for step_id in range(n_steps):
                    rov.rover_sensor_scan(rd.pois, rd.obstacles, n_poi, n_obstacles)
                    rov.step(rd.world_x, rd.world_y)
                    rd.update_rover_path(rov, step_id)

                    # Update fitness of policies using reward information
                    ea.fitness[policy_id] += go_away_poi_reward(rd.pois[poi_id], rov, rd.world_x, rd.world_y)
            ea.fitness[policy_id] /= n_configs
            policy_id += 1

        # Choose new parents and create new offspring population
        ea.down_select()

    best_pol_id = np.argmax(ea.fitness)
    best_policy = ea.population["pop{0}".format(best_pol_id)]

    return best_policy


def train_towards_poi_policy(poi_id):
    """
    Train a NN control policy to go towards targeted POI
    :param poi_id:
    :return:
    """
    # Create dictionary for each instance of rover and corresponding NN and EA population
    rd = RoverDomain()
    rd.use_saved_poi_configuration()
    rd.use_saved_obstacle_config()
    rd.use_saved_rover_training_configs()
    rov = Rover(0)
    ea = Ccea(rov.n_inputs, rov.n_hnodes, rov.n_outputs)
    ea.create_new_population()

    for gen in range(generations):
        ea.reset_fitness()
        if gen == 0:
            policy_id = 0
        else:
            policy_id = ea.n_elites
        while policy_id < ea.pop_size:
            for config_id in range(n_configs):
                rov.reset_rover(rd.rover_configs[config_id])  # Reset rover to initial conditions
                rov.get_network_weights(ea.population["pop{0}".format(policy_id)])  # Apply network weights from CCEA
                rd.update_rover_path(rov, -1)  # Record starting position of each rover

                for step_id in range(n_steps):
                    rov.rover_sensor_scan(rd.pois, rd.obstacles, n_poi, n_obstacles)
                    rov.step(rd.world_x, rd.world_y)
                    rd.update_rover_path(rov, step_id)

                    # Update fitness of policies using reward information
                    ea.fitness[policy_id] += go_towards_poi_reward(rd.pois[poi_id], rov)
            ea.fitness[policy_id] /= n_configs
            policy_id += 1

        # Choose new parents and create new offspring population
        ea.down_select()

    best_pol_id = np.argmax(ea.fitness)
    best_policy = ea.population["pop{0}".format(best_pol_id)]

    return best_policy


def train_obstacle_avoid_policy(obs_id):
    """
    Train a NN control policy that avoids obstacles
    :return:
    """

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rd = RoverDomain()
    rd.use_saved_poi_configuration()
    rd.use_saved_obstacle_config()
    rd.use_saved_rover_training_configs()
    rov = Rover(0)
    ea = Ccea(rov.n_inputs, rov.n_hnodes, rov.n_outputs)
    ea.create_new_population()

    for gen in range(generations):
        ea.reset_fitness()
        if gen == 0:
            policy_id = 0
        else:
            policy_id = ea.n_elites
        while policy_id < ea.pop_size:
            for config_id in range(n_configs):
                rov.reset_rover(rd.rover_configs[config_id])  # Reset rover to initial conditions
                rov.get_network_weights(ea.population["pop{0}".format(policy_id)])  # Apply network weights from CCEA
                rd.update_rover_path(rov, -1)  # Record starting position of each rover

                for step_id in range(n_steps):
                    rov.rover_sensor_scan(rd.pois, rd.obstacles, n_poi, n_obstacles)
                    rov.step(rd.world_x, rd.world_y)
                    rd.update_rover_path(rov, step_id)

                    # Update fitness of policies using reward information
                    ea.fitness[policy_id] += avoid_obstacle_reward(obs_id, rd.obstacles, rov, rd.world_x, rd.world_y)
            ea.fitness[policy_id] /= n_configs
            policy_id += 1

        # Choose new parents and create new offspring population
        ea.down_select()

    best_pol_id = np.argmax(ea.fitness)
    best_policy = ea.population["pop{0}".format(best_pol_id)]

    return best_policy


def pick_policy(brain_output, policies):
    """
    Choose a policy from the policy bank based on the brain's output
    :param brain_output: Output from brain decision making NN
    :param policies: Bank of pre-trained policies to choose from
    :return:
    """
    pol_choice = brain_output * n_policies
    for pol_id in range(n_policies):
        if pol_choice >= pol_id and pol_choice < pol_id+1:
            rover_policy = policies["Policy{0}".format(pol_id)]

            return rover_policy


def multi_reward_learning_single():
    """
    Trains the brain to choose between trained rover control policies
    :return:
    """

    policies = {}
    if train_new_policies == 1:
        policies = train_policies()
    else:
        pol_id = 0
        for poi_id in range(n_poi):
            policies['Policy{0}'.format(pol_id)] = use_saved_policy('TowardsPOI{0}'.format(poi_id))
            pol_id += 1
            policies['Policy{0}'.format(pol_id)] = use_saved_policy('AwayFromPOI{0}'.format(poi_id))
            pol_id += 1
        for obs_id in range(n_obstacles):
            policies['Policy{0}'.format(pol_id)] = use_saved_policy("AvoidObstacle{0}".format(obs_id))
            pol_id += 1

    print("Training The Brain")
    visualizer_rover_path = np.zeros((s_runs, (n_steps + 1), 3))
    for srun in range(s_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Create dictionary for each instance of rover and corresponding NN and EA population
        rd = RoverDomain()
        rd.use_saved_poi_configuration()
        rd.use_saved_obstacle_config()
        rd.use_saved_rover_training_configs()
        rd.use_saved_rover_test_config()
        br = Brain()
        rov = Rover(0)
        ea = Ccea(br.n_inputs, br.n_hnodes, br.n_outputs)
        ea.create_new_population()

        # Train the Brain
        reward_history = []
        for gen in range(brain_gen):
            ea.reset_fitness()
            if gen == 0:
                policy_id = 0
            else:
                policy_id = ea.n_elites
            while policy_id < ea.pop_size:  # Each policy in CCEA is tested in teams
                for config_id in range(n_configs):
                    rov.reset_rover(rd.rover_configs[config_id])  # Reset rover to initial conditions
                    br.get_weights(ea.population["pop{0}".format(policy_id)])  # Apply network weights from CCEA
                    rd.update_rover_path(rov, -1)  # Record starting position of each rover

                    for step_id in range(n_steps):
                        rov.rover_sensor_scan(rd.pois, rd.obstacles, n_poi, n_obstacles)
                        state_vector = []
                        for bracket in range(8):
                            state_vector.append(rov.sensor_readings[bracket])

                        # Pick policy using brain's decision
                        br.get_inputs(state_vector)
                        br.get_outputs()
                        rov_policy = pick_policy(br.output_layer[0, 0], policies)
                        rov.get_network_weights(rov_policy)
                        rov.step(rd.world_x, rd.world_y)
                        rd.update_rover_path(rov, step_id)

                        # Update fitness of policies using reward information
                        global_reward = rd.step_based_global_reward(rov)
                        ea.fitness[policy_id] += global_reward

                ea.fitness[policy_id] /= n_configs
                policy_id += 1

            # Testing Phase (test best policies found so far) ---------------------------------------------------------
            rov.reset_rover(rd.rover_test_config)  # Reset rover to initial conditions
            policy_id = np.argmax(ea.fitness)
            br.get_weights(ea.population["pop{0}".format(policy_id)])  # Apply best set of weights to network
            rd.update_rover_path(rov, -1)

            for step_id in range(n_steps):
                rov.rover_sensor_scan(rd.pois, rd.obstacles, n_poi, n_obstacles)
                state_vector = []
                for bracket in range(8):
                    state_vector.append(rov.sensor_readings[bracket])

                # Pick policy using brain's decision
                br.get_inputs(state_vector)
                br.get_outputs()
                rov_policy = pick_policy(br.output_layer[0, 0], policies)
                rov.get_network_weights(rov_policy)
                rov.step(rd.world_x, rd.world_y)
                rd.update_rover_path(rov, step_id)

            global_reward = calc_global_reward(rd.rover_path, rd.pois, rd.obstacles)
            reward_history.append(global_reward)

            if gen == (brain_gen - 1):  # Save path at end of final generation
                visualizer_rover_path[srun] = rd.rover_path.copy()
                save_rover_path(visualizer_rover_path)

            # Choose new parents and create new offspring population
            ea.down_select()

        save_reward_history(reward_history, "Multi_Reward.csv")
    run_visualizer()


def main(reward_type="Visual"):
    """
    reward_type:
    :return:
    """

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if reward_type == "Multi":
        multi_reward_learning_single()
    elif reward_type == "Train":
        train_policies()
    elif reward_type == "Test":
        test_policy(0, "Avoid")  # [Policy ID, Policy Type]
    elif reward_type == "Visual":
        run_visualizer()
    elif reward_type == "Create":
        rd = RoverDomain()
        rd.create_new_poi_config()
        # rd.use_saved_poi_configuration()
        rd.create_obstacle_configs()
        # rd.use_saved_obstacle_config()
        rd.create_rover_training_configs()
        rd.create_rover_test_config()
    else:
        sys.exit('Incorrect Reward Type')


main(reward_type="Visual")  # Run the program
