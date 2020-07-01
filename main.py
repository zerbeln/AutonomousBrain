from ccea import Ccea
from reward_functions import calc_global_reward
from multi_reward import go_away_poi_reward, go_towards_poi_reward
from rover_domain import RoverDomain
from brain import Brain
from Visualizer.visualizer import run_visualizer
from agent import Rover
import csv; import os; import sys
import numpy as np
import warnings
import pickle


def get_parameters(brain_training=0):
    """
    Create dictionary of parameters needed for simulation
    :return:
    """
    parameters = {}

    # Test Parameters
    parameters["s_runs"] = 1
    parameters["new_world_config"] = 0  # 1 = Create new environment, 0 = Use existing environment
    parameters["running"] = 0  # 1 keeps visualizer from closing (use 0 for multiple runs)

    # Rover Domain Parameters
    parameters["n_rovers"] = 1
    parameters["n_poi"] = 2
    if brain_training == 0:  # Training individual policies
        parameters["n_steps"] = 30
        parameters["n_configs"] = 6  # The number of configurations used to train policies
    else:
        parameters["n_steps"] = 60  # Training the brain
        parameters["n_configs"] = 1  # The number of configurations used to train policies
    parameters["min_dist"] = 1.0
    parameters["obs_rad"] = 4.0
    parameters["c_req"] = 1  # Number of rovers required to observe a single POI
    parameters["x_dim"] = 30.0
    parameters["y_dim"] = 30.0
    parameters["angle_resolution"] = 90
    parameters["sensor_type"] = 'summed'

    # Neural Network Parameters
    parameters["n_inputs"] = 4
    parameters["n_hnodes"] = 6
    parameters["n_outputs"] = 2

    # Brain Parameters
    parameters["brain_inputs"] = 8
    parameters["brain_hnodes"] = 10
    parameters["brain_outputs"] = 1
    parameters["brain_generations"] = 1000
    parameters["n_policies"] = 2 * parameters["n_poi"]

    # CCEA Parameters
    parameters["pop_size"] = 100
    parameters["m_rate"] = 0.1
    parameters["m_prob"] = 0.3
    parameters["epsilon"] = 0.1
    parameters["generations"] = 500
    parameters["n_elites"] = 5

    return parameters


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


def save_rover_path(p, rover_path):  # Save path rovers take using best policy found
    """
    Records the path each rover takes using best policy from CCEA (used by visualizer)
    :param p:  parameter dict
    :param rover_path:  trajectory tracker
    :return:
    """
    dir_name = 'Output_Data/'  # Intended directory for output files

    rpath_name = os.path.join(dir_name, 'Rover_Paths.txt')

    rpath = open(rpath_name, 'a')
    for rov_id in range(p["n_rovers"]):
        for t in range(p["n_steps"]+1):
            rpath.write('%f' % rover_path[t, rov_id, 0])
            rpath.write('\t')
            rpath.write('%f' % rover_path[t, rov_id, 1])
            rpath.write('\t')
        rpath.write('\n')
    rpath.write('\n')
    rpath.close()


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


def test_policies(reward_type):

    print("Reward Type: ", reward_type)
    p = get_parameters(brain_training=1)

    poi_id = 1
    weights = train_towards_poi_policy(poi_id)
    # weights = train_away_from_poi_policy(poi_id)
    # save_trained_policy('NNPolicy', weights)
    # weights = use_saved_policy('NNPolicy')

    rd = RoverDomain(p)
    rd.use_saved_rover_config('Rover_Config.csv')
    rd.use_saved_poi_configuration()
    rov = Rover(p, 0)

    rov.reset_rover(rd.rover_configs[0, 0])  # Reset rover to initial conditions
    rov.get_network_weights(weights)  # Apply best set of weights to network
    rd.update_rover_path(rov, 0, -1)

    for step_id in range(p["n_steps"]):
        rov.rover_sensor_scan(rov, rd.pois, p["n_rovers"], p["n_poi"])
        rov.step(p["x_dim"], p["y_dim"])
        rd.update_rover_path(rov, 0, step_id)

    save_rover_path(p, rd.rover_path)
    run_visualizer(p)


def train_away_from_poi_policy(poi_id):
    """
    Train NN control policy to travel away from targeted POI
    :param poi_id:
    :return:
    """
    p = get_parameters()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rd = RoverDomain(p)
    if p["new_world_config"] == 1:
        rd.create_rover_training_configs()
    else:
        rd.use_saved_rover_config('rover_policy_training.csv')
    rd.use_saved_poi_configuration()
    rov = Rover(p, 0)
    ea = Ccea(p, p["n_inputs"], p["n_hnodes"], p["n_outputs"])
    ea.create_new_population()

    for gen in range(p["generations"]):
        ea.reset_fitness()
        for config_id in range(p["n_configs"]):
            for policy_id in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                rov.reset_rover(rd.rover_configs[config_id, 0])  # Reset rover to initial conditions
                rov.get_network_weights(ea.population["pop{0}".format(policy_id)])  # Apply network weights from CCEA
                rd.update_rover_path(rov, 0, -1)  # Record starting position of each rover

                for step_id in range(p["n_steps"]):
                    rov.rover_sensor_scan(rov, rd.pois, p["n_rovers"], p["n_poi"])
                    rov.step(p["x_dim"], p["y_dim"])
                    rd.update_rover_path(rov, 0, step_id)

                # Update fitness of policies using reward information
                ea.fitness[policy_id] += go_away_poi_reward(poi_id, rd.pois, rov)

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
    p = get_parameters()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rd = RoverDomain(p)
    if p["new_world_config"] == 1:
        rd.create_rover_training_configs()
    else:
        rd.use_saved_rover_config('rover_policy_training.csv')
    #rd.create_new_poi_config()
    rd.use_saved_poi_configuration()
    rov = Rover(p, 0)
    ea = Ccea(p, p["n_inputs"], p["n_hnodes"], p["n_outputs"])
    ea.create_new_population()

    for gen in range(p["generations"]):
        ea.reset_fitness()
        for config_id in range(p["n_configs"]):
            for policy_id in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                rov.reset_rover(rd.rover_configs[config_id, 0])  # Reset rover to initial conditions
                rov.get_network_weights(ea.population["pop{0}".format(policy_id)])  # Apply network weights from CCEA
                rd.update_rover_path(rov, 0, -1)  # Record starting position of each rover

                for step_id in range(p["n_steps"]):
                    rov.rover_sensor_scan(rov, rd.pois, p["n_rovers"], p["n_poi"])
                    rov.step(p["x_dim"], p["y_dim"])
                    rd.update_rover_path(rov, 0, step_id)

                # Update fitness of policies using reward information
                ea.fitness[policy_id] += go_towards_poi_reward(poi_id, rd.pois, rov)

        # Choose new parents and create new offspring population
        ea.down_select()

    best_pol_id = np.argmax(ea.fitness)
    best_policy = ea.population["pop{0}".format(best_pol_id)]

    return best_policy


def pick_policy(brain_output, n_policies, policies):
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


def multi_reward_learning_single(reward_type):
    print("Reward Type: ", reward_type)

    p = get_parameters(brain_training=1)
    assert (p["new_world_config"] == 0)

    for srun in range(p["s_runs"]):  # Perform statistical runs
        print("Run: %i" % srun)
        # Train bank of agent policies
        policies = {}
        for poi_id in range(p["n_poi"]):
            policies["Policy{0}".format(2*poi_id)] = train_towards_poi_policy(poi_id)
            policies["Policy{0}".format(2*poi_id+1)] = train_away_from_poi_policy(poi_id)

        # Create dictionary for each instance of rover and corresponding NN and EA population
        rd = RoverDomain(p)
        rd.use_saved_rover_config('Rover_Config.csv')
        rd.use_saved_poi_configuration()
        br = Brain(p)
        rov = Rover(p, 0)
        ea = Ccea(p, p["brain_inputs"], p["brain_hnodes"], p["brain_outputs"])
        ea.create_new_population()

        # Train the Brain
        reward_history = []
        for gen in range(p["brain_generations"]):
            for policy_id in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                rov.reset_rover(rd.rover_configs[0, 0])  # Reset rover to initial conditions
                br.get_weights(ea.population["pop{0}".format(policy_id)])  # Apply network weights from CCEA
                rd.update_rover_path(rov, 0, -1)  # Record starting position of each rover

                for step_id in range(p["n_steps"]):
                    rov.rover_sensor_scan(rov, rd.pois, p["n_rovers"], p["n_poi"])
                    state_vector = []
                    for bracket in range(4):
                        state_vector.append(rov.sensor_readings[bracket])
                    for poi_id in range(p["n_poi"]):
                        state_vector.append(go_towards_poi_reward(poi_id, rd.pois, rov))
                        state_vector.append(go_away_poi_reward(poi_id, rd.pois, rov))

                    # Pick policy using brain's decision
                    br.get_inputs(state_vector)
                    br.get_outputs()
                    rov_policy = pick_policy(br.output_layer[0, 0], p["n_policies"], policies)
                    rov.get_network_weights(rov_policy)
                    rov.step(p["x_dim"], p["y_dim"])
                    rd.update_rover_path(rov, 0, step_id)

                # Update fitness of policies using reward information
                global_reward = calc_global_reward(p, rd.rover_path, rd.pois)
                ea.fitness[policy_id] = global_reward

            # Testing Phase (test best policies found so far) ---------------------------------------------------------
            rov.reset_rover(rd.rover_configs[0, 0])  # Reset rover to initial conditions
            policy_id = np.argmax(ea.fitness)
            br.get_weights(ea.population["pop{0}".format(policy_id)])  # Apply best set of weights to network
            rd.update_rover_path(rov, 0, -1)

            for step_id in range(p["n_steps"]):
                rov.rover_sensor_scan(rov, rd.pois, p["n_rovers"], p["n_poi"])
                state_vector = []
                for bracket in range(4):
                    state_vector.append(rov.sensor_readings[bracket])
                for poi_id in range(p["n_poi"]):
                    state_vector.append(go_towards_poi_reward(poi_id, rd.pois, rov))
                    state_vector.append(go_away_poi_reward(poi_id, rd.pois, rov))

                # Pick policy using brain's decision
                br.get_inputs(state_vector)
                br.get_outputs()
                rov_policy = pick_policy(br.output_layer[0, 0], p["n_policies"], policies)
                rov.get_network_weights(rov_policy)
                rov.step(p["x_dim"], p["y_dim"])
                rd.update_rover_path(rov, 0, step_id)

            global_reward = calc_global_reward(p, rd.rover_path, rd.pois)
            reward_history.append(global_reward)

            if gen == (p["brain_generations"] - 1):  # Save path at end of final generation
                save_rover_path(p, rd.rover_path)

            # Choose new parents and create new offspring population
            ea.down_select()

        save_reward_history(reward_history, "Multi_Reward.csv")
    run_visualizer(p)


def visualizer_only():
    """
    Runs the PyGame visualizer to observe rover behaviors
    :return:
    """
    p = get_parameters()
    run_visualizer(p)


def main(reward_type="Test"):
    """
    reward_type:
    :return:
    """

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if reward_type == "Multi":
        multi_reward_learning_single(reward_type)
    elif reward_type == "Test":
        test_policies(reward_type)
    else:
        sys.exit('Incorrect Reward Type')


main(reward_type="Test")  # Run the program
