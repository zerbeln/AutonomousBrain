from ccea import Ccea
from reward_functions import calc_global_reward, calc_difference_reward, calc_dpp_reward
from multi_reward import go_away_poi_reward, go_towards_poi_reward
from rover_domain import RoverDomain
from brain import Brain
from Visualizer.visualizer import run_visualizer
from agent import Rover
import csv; import os; import sys
import numpy as np
import warnings


def get_parameters():
    """
    Create dictionary of parameters needed for simulation
    :return:
    """
    parameters = {}

    # Test Parameters
    parameters["s_runs"] = 1
    parameters["new_world_config"] = 0  # 1 = Create new environment, 0 = Use existing environment
    parameters["running"] = 1  # 1 keeps visualizer from closing (use 0 for multiple runs)

    # Neural Network Parameters
    parameters["n_inputs"] = 4
    parameters["n_hnodes"] = 6
    parameters["n_outputs"] = 2

    # Brain Parameters
    parameters["brain_inputs"] = 8
    parameters["brain_hnodes"] = 10
    parameters["brain_outputs"] = 1
    parameters["brain_generations"] = 300
    parameters["brain_steps"] = 25

    # CCEA Parameters
    parameters["pop_size"] = 100
    parameters["m_rate"] = 0.1
    parameters["m_prob"] = 0.3
    parameters["epsilon"] = 0.1
    parameters["generations"] = 300
    parameters["n_elites"] = 10

    # Rover Domain Parameters
    parameters["n_rovers"] = 1
    parameters["n_poi"] = 2
    parameters["n_steps"] = 25
    parameters["min_dist"] = 1.0
    parameters["obs_rad"] = 4.0
    parameters["c_req"] = 1
    parameters["x_dim"] = 30.0
    parameters["y_dim"] = 30.0
    parameters["angle_resolution"] = 90
    parameters["sensor_type"] = 'summed'

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


def rovers_global_only(reward_type):
    p = get_parameters()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rv = {}
    rd = RoverDomain(p)
    for rv_id in range(p["n_rovers"]):
        rv["AG{0}".format(rv_id)] = Rover(p, rv_id, rd.rover_positions[rv_id])
        rv["EA{0}".format(rv_id)] = Ccea(p, p["n_inputs"], p["n_hnodes"], p["n_outputs"])

    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["c_req"])

    for srun in range(p["s_runs"]):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        for rv_id in range(p["n_rovers"]):
            rv["EA{0}".format(rv_id)].create_new_population()
        reward_history = []

        for gen in range(p["generations"]):
            # print("Gen: %i" % gen)
            for rv_id in range(p["n_rovers"]):
                rv["EA{0}".format(rv_id)].select_policy_teams()
                rv["EA{0}".format(rv_id)].reset_fitness()
            for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                rd.clear_rover_path()
                for rv_id in range(p["n_rovers"]):
                    rv["AG{0}".format(rv_id)].reset_rover()  # Reset rover to initial conditions
                    policy_id = int(rv["EA{0}".format(rv_id)].team_selection[team_number])
                    weights = rv["EA{0}".format(rv_id)].population["pop{0}".format(policy_id)]
                    rv["AG{0}".format(rv_id)].get_network_weights(weights)  # Apply network weights from CCEA
                    rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, -1)  # Record starting position of each rover

                for step_id in range(p["n_steps"]):
                    for rv_id in range(p["n_rovers"]):  # Rover scans environment
                        rv["AG{0}".format(rv_id)].rover_sensor_scan(rv, rd.pois, p["n_rovers"], p["n_poi"])
                    for rv_id in range(p["n_rovers"]):  # Rover processes scan information and acts
                        rv["AG{0}".format(rv_id)].step(p["x_dim"], p["y_dim"])
                        rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, step_id)

                # Update fitness of policies using reward information
                global_reward = calc_global_reward(p, rd.rover_path, rd.pois)
                for rv_id in range(p["n_rovers"]):
                    policy_id = int(rv["EA{0}".format(rv_id)].team_selection[team_number])
                    rv["EA{0}".format(rv_id)].fitness[policy_id] = global_reward
            # print(rv["EA0"].fitness)

            # Testing Phase (test best policies found so far)
            rd.clear_rover_path()
            for rv_id in range(p["n_rovers"]):
                rv["AG{0}".format(rv_id)].reset_rover()  # Reset rover to initial conditions
                policy_id = np.argmax(rv["EA{0}".format(rv_id)].fitness)
                weights = rv["EA{0}".format(rv_id)].population["pop{0}".format(policy_id)]
                rv["AG{0}".format(rv_id)].get_network_weights(weights)  # Apply best set of weights to network
                rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, -1)

            for step_id in range(p["n_steps"]):
                for rv_id in range(p["n_rovers"]):  # Rover scans environment
                    rv["AG{0}".format(rv_id)].rover_sensor_scan(rv, rd.pois, p["n_rovers"], p["n_poi"])
                for rv_id in range(p["n_rovers"]):  # Rover processes information froms can and acts
                    rv["AG{0}".format(rv_id)].step(p["x_dim"], p["y_dim"])
                    rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, step_id)

            global_reward = calc_global_reward(p, rd.rover_path, rd.pois)
            reward_history.append(global_reward)

            if gen == (p["generations"] - 1):  # Save path at end of final generation
                save_rover_path(p, rd.rover_path)

            # Choose new parents and create new offspring population
            for rv_id in range(p["n_rovers"]):
                rv["EA{0}".format(rv_id)].down_select()

        save_reward_history(reward_history, "Global_Reward.csv")

    run_visualizer(p)


def rovers_difference_rewards(reward_type):
    p = get_parameters()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rv = {}
    rd = RoverDomain(p)
    for rv_id in range(p["n_rovers"]):
        rv["AG{0}".format(rv_id)] = Rover(p, rv_id, rd.rover_positions[rv_id])
        rv["EA{0}".format(rv_id)] = Ccea(p, p["n_inputs"], p["n_hnodes"], p["n_outputs"])

    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["c_req"])

    for srun in range(p["s_runs"]):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        for rv_id in range(p["n_rovers"]):  # Randomly initialize ccea populations
            rv["EA{0}".format(rv_id)].create_new_population()
        reward_history = []

        for gen in range(p["generations"]):
            # print("Gen: %i" % gen)
            for rv_id in range(p["n_rovers"]):
                rv["EA{0}".format(rv_id)].select_policy_teams()
            for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                for rv_id in range(p["n_rovers"]):
                    rv["AG{0}".format(rv_id)].reset_rover()  # Reset rover to initial conditions
                    policy_id = int(rv["EA{0}".format(rv_id)].team_selection[team_number])
                    weights = rv["EA{0}".format(rv_id)].population["pop{0}".format(policy_id)]
                    rv["AG{0}".format(rv_id)].get_network_weights(weights)  # Apply network weights from CCEA
                    rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, -1)  # Record starting position of each rover

                for step_id in range(p["n_steps"]):
                    for rv_id in range(p["n_rovers"]):  # Rover scans environment
                        rv["AG{0}".format(rv_id)].rover_sensor_scan(rv, rd.pois, p["n_rovers"], p["n_poi"])
                    for rv_id in range(p["n_rovers"]):  # Rover processes scan information and acts
                        rv["AG{0}".format(rv_id)].step(p["x_dim"], p["y_dim"])
                        rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, step_id)

                # Update fitness of policies using reward information
                global_reward = calc_global_reward(p, rd.rover_path, rd.pois)
                difference_rewards = calc_difference_reward(p, rd.rover_path, rd.pois, global_reward)
                for rv_id in range(p["n_rovers"]):
                    policy_id = int(rv["EA{0}".format(rv_id)].team_selection[team_number])
                    rv["EA{0}".format(rv_id)].fitness[policy_id] = difference_rewards[rv_id]

            # Testing Phase (test best policies found so far) ---------------------------------------------------------
            for rv_id in range(p["n_rovers"]):
                rv["AG{0}".format(rv_id)].reset_rover()  # Reset rover to initial conditions
                policy_id = np.argmax(rv["EA{0}".format(rv_id)].fitness)
                weights = rv["EA{0}".format(rv_id)].population["pop{0}".format(policy_id)]
                rv["AG{0}".format(rv_id)].get_network_weights(weights)  # Apply best set of weights to network
                rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, -1)

            for step_id in range(p["n_steps"]):
                # Rover scans environment
                for rv_id in range(p["n_rovers"]):
                    rv["AG{0}".format(rv_id)].rover_sensor_scan(rv, rd.pois, p["n_rovers"], p["n_poi"])

                # Rover processes information from scan and acts
                for rv_id in range(p["n_rovers"]):
                    rv["AG{0}".format(rv_id)].step(p["x_dim"], p["y_dim"])
                    rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, step_id)

            global_reward = calc_global_reward(p, rd.rover_path, rd.pois)
            reward_history.append(global_reward)

            if gen == (p["generations"] - 1):  # Save path at end of final generation
                save_rover_path(p, rd.rover_path)

            # Choose new parents and create new offspring population
            for rv_id in range(p["n_rovers"]):
                rv["EA{0}".format(rv_id)].down_select()

        save_reward_history(reward_history, "Difference_Reward.csv")
    run_visualizer(p)


def test_policies(reward_type):

    print("Reward Type: ", reward_type)
    p = get_parameters()

    poi_id = 0
    weights = train_towards_poi(poi_id)
    # weights = train_away_from_poi(poi_id)

    rd = RoverDomain(p)
    rov = Rover(p, 0, rd.rover_positions[0])

    rov.reset_rover()  # Reset rover to initial conditions
    rov.get_network_weights(weights)  # Apply best set of weights to network
    rd.update_rover_path(rov, 0, -1)

    for step_id in range(p["n_steps"]):
        rov.rover_sensor_scan(rov, rd.pois, p["n_rovers"], p["n_poi"])
        rov.step(p["x_dim"], p["y_dim"])
        rd.update_rover_path(rov, 0, step_id)

    save_rover_path(p, rd.rover_path)
    run_visualizer(p)


def train_away_from_poi(poi_id):
    p = get_parameters()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rd = RoverDomain(p)
    rov = Rover(p, 0, rd.rover_positions[0])
    ea = Ccea(p, p["n_inputs"], p["n_hnodes"], p["n_outputs"])
    ea.create_new_population()

    for gen in range(p["generations"]):
        for policy_id in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
            rov.reset_rover()  # Reset rover to initial conditions
            rov.get_network_weights(ea.population["pop{0}".format(policy_id)])  # Apply network weights from CCEA
            rd.update_rover_path(rov, 0, -1)  # Record starting position of each rover

            for step_id in range(p["n_steps"]):
                rov.rover_sensor_scan(rov, rd.pois, p["n_rovers"], p["n_poi"])
                rov.step(p["x_dim"], p["y_dim"])
                rd.update_rover_path(rov, 0, step_id)

            # Update fitness of policies using reward information
            ea.fitness[policy_id] = go_away_poi_reward(poi_id, rd.pois, rov)

        # Choose new parents and create new offspring population
        ea.down_select()

    best_pol_id = np.argmax(ea.fitness)
    best_policy = ea.population["pop{0}".format(best_pol_id)]

    return best_policy


def train_towards_poi(poi_id):
    p = get_parameters()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rd = RoverDomain(p)
    rov = Rover(p, 0, rd.rover_positions[0])
    ea = Ccea(p, p["n_inputs"], p["n_hnodes"], p["n_outputs"])
    ea.create_new_population()

    for gen in range(p["generations"]):
        for policy_id in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
            rov.reset_rover()  # Reset rover to initial conditions
            rov.get_network_weights(ea.population["pop{0}".format(policy_id)])  # Apply network weights from CCEA
            rd.update_rover_path(rov, 0, -1)  # Record starting position of each rover

            for step_id in range(p["n_steps"]):
                rov.rover_sensor_scan(rov, rd.pois, p["n_rovers"], p["n_poi"])
                rov.step(p["x_dim"], p["y_dim"])
                rd.update_rover_path(rov, 0, step_id)

            # Update fitness of policies using reward information
            ea.fitness[policy_id] = go_towards_poi_reward(poi_id, rd.pois, rov)

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
    if brain_output < 1:
        rover_policy = policies["Towards0"]
    elif 1 <= brain_output < 2:
        rover_policy = policies["Towards1"]
    elif 2 <= brain_output < 3:
        rover_policy = policies["Away0"]
    else:
        rover_policy = policies["Away1"]

    return rover_policy


def multi_reward_learning_single(reward_type):
    p = get_parameters()
    assert(p["new_world_config"] == 0)

    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["c_req"])

    for srun in range(p["s_runs"]):  # Perform statistical runs
        print("Run: %i" % srun)

        # Train bank of agent policies
        policies = {}
        for poi_id in range(p["n_poi"]):
            policies["Towards{0}".format(poi_id)] = train_towards_poi(poi_id)
            policies["Away{0}".format(poi_id)] = train_away_from_poi(poi_id)

        # Create dictionary for each instance of rover and corresponding NN and EA population
        rd = RoverDomain(p)
        br = Brain(p)
        rov = Rover(p, 0, rd.rover_positions[0])
        ea = Ccea(p, p["brain_inputs"], p["brain_hnodes"], p["brain_outputs"])
        ea.create_new_population()

        # Train the Brain
        reward_history = []
        for gen in range(p["brain_generations"]):
            for policy_id in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                rov.reset_rover()  # Reset rover to initial conditions
                br.get_weights(ea.population["pop{0}".format(policy_id)])  # Apply network weights from CCEA
                rd.update_rover_path(rov, 0, -1)  # Record starting position of each rover

                for step_id in range(p["brain_steps"]):
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
                    chosen_pol = br.output_layer[0, 0] * 4
                    rov_policy = pick_policy(chosen_pol, policies)
                    rov.get_network_weights(rov_policy)
                    rov.step(p["x_dim"], p["y_dim"])
                    rd.update_rover_path(rov, 0, step_id)

                # Update fitness of policies using reward information
                global_reward = calc_global_reward(p, rd.rover_path, rd.pois)
                ea.fitness[policy_id] = global_reward

            # Testing Phase (test best policies found so far) ---------------------------------------------------------
            rov.reset_rover()  # Reset rover to initial conditions
            policy_id = np.argmax(ea.fitness)
            br.get_weights(ea.population["pop{0}".format(policy_id)])  # Apply best set of weights to network
            rd.update_rover_path(rov, 0, -1)

            for step_id in range(p["brain_steps"]):
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
                chosen_pol = br.output_layer[0, 0] * 4
                rov_policy = pick_policy(chosen_pol, policies)
                rov.get_network_weights(rov_policy)
                rov.step(p["x_dim"], p["y_dim"])
                rd.update_rover_path(rov, 0, step_id)

            global_reward = calc_global_reward(p, rd.rover_path, rd.pois)
            reward_history.append(global_reward)

            if gen == (p["generations"] - 1):  # Save path at end of final generation
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


def main(reward_type="Global"):
    """
    reward_type: Global, Difference, or DPP
    :return:
    """

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if reward_type == "Global":
        rovers_global_only(reward_type)
    elif reward_type == "Difference":
        rovers_difference_rewards(reward_type)
    elif reward_type == "Multi":
        multi_reward_learning_single(reward_type)
    elif reward_type == "Test":
        test_policies(reward_type)
    else:
        sys.exit('Incorrect Reward Type')


main(reward_type="Multi")  # Run the program
