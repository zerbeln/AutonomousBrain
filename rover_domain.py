import numpy as np
import math
import random
import os
import csv


class RoverDomain:

    def __init__(self, p):
        # World attributes
        self.world_x = p["x_dim"]
        self.world_y = p["y_dim"]
        self.n_poi = p["n_poi"]
        self.c_req = p["c_req"]
        self.min_dist = p["min_dist"]
        self.obs_radius = p["obs_rad"]
        self.rover_steps = p["n_steps"]
        self.n_configs = p["n_configs"]
        self.n_obstacles = p["n_obstacles"]

        # Rover path trace for trajectory-wide global reward computation and vizualization purposes
        self.rover_path = np.zeros(((self.rover_steps + 1), 3))

        # POI position and value vectors
        self.pois = np.zeros((self.n_poi, 3))  # [X, Y, Val]
        self.rover_configs = np.zeros((self.n_configs, 3))  # [X, Y, Theta]
        self.rover_test_config = np.zeros(3)
        self.obstacles = np.zeros((self.n_obstacles, 3))  # [X, Y, Radius]

        # User Defined Parameters:
        self.poi_observations = np.zeros(self.n_poi)  # Used for spatial coupling of POIs

    def step_based_global_reward(self, rov):
        """
        Calculates the global reward at each time step
        :param rover_positions:
        :return:
        """

        global_reward = -1.0  # Step cost

        for poi_id in range(self.n_poi):
            # Calculate distance between agent and POI
            x_distance = self.pois[poi_id, 0] - rov.rover_x
            y_distance = self.pois[poi_id, 1] - rov.rover_y
            distance = math.sqrt((x_distance ** 2) + (y_distance ** 2))

            if distance < self.min_dist:
                distance = self.min_dist

            # Check if agent observes poi and update observer count if true
            if distance < self.obs_radius:
                global_reward += self.pois[poi_id, 2]/distance

        for obs_id in range(self.n_obstacles):
            x_distance = self.obstacles[obs_id, 0] - rov.rover_x
            y_distance = self.obstacles[obs_id, 1] - rov.rover_y
            distance = math.sqrt((x_distance ** 2) + (y_distance ** 2))

            if distance < self.obstacles[obs_id, 2]:
                global_reward -= 1000.0

        return global_reward

    def clear_rover_path(self):
        """
        Resets the rover path tracker
        :return:
        """
        self.rover_path = np.zeros(((self.rover_steps + 1), 3))  # Tracks rover trajectories

    def update_rover_path(self, rover, step_id):
        """
        Update the array tracking the path of each rover
        :param rover: instance of a rover
        :param step_id: Current time step for the simulation
        :return:
        """
        self.rover_path[step_id+1, 0] = rover.rover_x
        self.rover_path[step_id+1, 1] = rover.rover_y
        self.rover_path[step_id+1, 2] = rover.rover_theta

    # ROVER POSITION FUNCTIONS ----------------------------------------------------------------------------------------
    def create_rover_test_config(self):
        """
        Creates the rover starting configurations for testing rovers policies
        :return:
        """
        self.rover_test_config = np.zeros(3)  # [X, Y, Theta]
        self.init_rover_pos_random(0)
        self.save_rover_test_config()

    def create_rover_training_configs(self):
        """
        Create n number of world configurations to train the bank of policies
        :return:
        """
        self.rover_configs = np.zeros((self.n_configs, 3))  # [X, Y, Theta]

        for config_id in range(self.n_configs):
            self.init_rover_pos_random(config_id)
        self.save_rover_training_configs()

    def use_saved_rover_test_config(self):
        """
        Use a stored initial configuration from a CSV file
        :return:
        """
        dir_name = 'Output_Data/'  # Intended directory for output files
        rov_file_name = os.path.join(dir_name, 'Rover_Config.csv')
        config_input = []
        with open(rov_file_name) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        # Assign values to variables that track the rover's initial conditions
        self.rover_test_config[0] = float(config_input[0][0])
        self.rover_test_config[1] = float(config_input[0][1])
        self.rover_test_config[2] = float(config_input[0][2])

    def use_saved_rover_training_configs(self):
        """
        Use stored rover training configurations (stored in a CSV file)
        :return:
        """
        dir_name = 'Output_Data/'  # Intended directory for output files
        rov_file_name = os.path.join(dir_name, 'Rover_Training_Configs.csv')
        config_input = []
        with open(rov_file_name) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        # Assign values to variables that track the rover's initial conditions
        for con_id in range(self.n_configs):
            self.rover_configs[con_id, 0] = float(config_input[con_id][0])
            self.rover_configs[con_id, 1] = float(config_input[con_id][1])
            self.rover_configs[con_id, 2] = float(config_input[con_id][2])

    def save_rover_training_configs(self):
        """
        Saves rover positions to a csv file in a folder called Output_Data
        :Output: CSV file containing rover training configurations
        """
        dir_name = 'Output_Data/'  # Intended directory for output files

        if not os.path.exists(dir_name):  # If Data directory does not exist, create it
            os.makedirs(dir_name)

        pfile_name = os.path.join(dir_name, 'Rover_Training_Configs.csv')

        row = np.zeros(3)
        with open(pfile_name, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for con_id in range(self.n_configs):
                row[0] = self.rover_configs[con_id, 0]
                row[1] = self.rover_configs[con_id, 1]
                row[2] = self.rover_configs[con_id, 2]
                writer.writerow(row[:])

    def save_rover_test_config(self):
        """
        Saves rover positions to a csv file in a folder called Output_Data
        :Output: CSV file containing rover starting positions
        """
        dir_name = 'Output_Data/'  # Intended directory for output files

        if not os.path.exists(dir_name):  # If Data directory does not exist, create it
            os.makedirs(dir_name)

        pfile_name = os.path.join(dir_name, 'Rover_Config.csv')

        row = np.zeros(3)
        with open(pfile_name, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row[0] = self.rover_configs[0, 0]
            row[1] = self.rover_configs[0, 1]
            row[2] = self.rover_configs[0, 2]
            writer.writerow(row[:])

    def init_rover_pos_random(self, con_id):  # Randomly set rovers on map
        """
        Rovers given random starting positions and orientations
        :param con_id: Configuration identifier number
        :return:
        """
        x = random.uniform(0.0, self.world_x - 1.0)
        y = random.uniform(0.0, self.world_y - 1.0)
        self.rover_configs[con_id, 2] = random.uniform(0.0, 360.0)

        # Make sure rover doesn't start off in range of POI or obstacle
        count = self.n_poi + self.n_obstacles
        while count > 0:
            for poi_id in range(self.n_poi):
                x_dist = self.pois[poi_id, 0] - x
                y_dist = self.pois[poi_id, 1] - y
                dist = math.sqrt(x_dist**2 + y_dist**2)

                if dist > self.obs_radius:
                    count -= 1

            for obs_id in range(self.n_obstacles):
                x_dist = self.obstacles[obs_id, 0] - x
                y_dist = self.obstacles[obs_id, 1] - y
                dist = math.sqrt(x_dist ** 2 + y_dist ** 2)

                if dist > self.obstacles[obs_id, 2]:
                    count -= 1

            if count > 0:
                x = random.uniform(0.0, self.world_x - 1.0)
                y = random.uniform(0.0, self.world_y - 1.0)
                count = self.n_poi + self.n_obstacles

        self.rover_configs[con_id, 0] = x
        self.rover_configs[con_id, 1] = y

    def init_rover_pos_random_concentrated(self, con_id, radius=5.0):
        """
        Rovers given random starting positions within a radius of the center. Starting orientations are random.
        :param radius: maximum radius from the center rovers are allowed to start in
        :return:
        """
        # Origin of constraining circle
        center_x = self.world_x/2.0
        center_y = self.world_y/2.0

        x = random.uniform(0.0, self.world_x-1.0)  # Rover X-Coordinate
        y = random.uniform(0.0, self.world_y-1.0)  # Rover Y-Coordinate

        # Make sure coordinates are within the bounds of the constraining circle
        while x > (center_x + radius) or x < (center_x - radius):
            x = random.uniform(0.0, self.world_x-1.0)
        while y > (center_y + radius) or y < (center_y - radius):
            y = random.uniform(0.0, self.world_y-1.0)

        self.rover_configs[con_id, 0] = x
        self.rover_configs[con_id, 1] = y
        self.rover_configs[con_id, 2] = random.uniform(0.0, 360.0)

    # POI POSITION FUNCTIONS ------------------------------------------------------------------------------------------
    def create_new_poi_config(self):
        """
        Set POI positions and POI values, clear the rover path tracker
        :return: none
        """
        self.pois = np.zeros((self.n_poi, 3))
        self.init_poi_pos_random()
        self.init_poi_vals_random()
        self.save_poi_configuration()

    def save_poi_configuration(self):
        """
        Saves POI configuration to a csv file in a folder called Output_Data
        :Output: One CSV file containing POI postions and POI values
        """
        dir_name = 'Output_Data/'  # Intended directory for output files

        if not os.path.exists(dir_name):  # If Data directory does not exist, create it
            os.makedirs(dir_name)

        pfile_name = os.path.join(dir_name, 'POI_Config.csv')

        with open(pfile_name, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for poi_id in range(self.n_poi):
                writer.writerow(self.pois[poi_id, :])

    def use_saved_poi_configuration(self):
        """
        Re-use POI configuration stored in a CSV file in folder called Output_Data
        :return:
        """
        config_input = []
        with open('Output_Data/POI_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        for poi_id in range(self.n_poi):
            self.pois[poi_id, 0] = float(config_input[poi_id][0])
            self.pois[poi_id, 1] = float(config_input[poi_id][1])
            self.pois[poi_id, 2] = float(config_input[poi_id][2])

    def init_poi_pos_random(self):  # Randomly set POI on the map
        """
        POI positions set randomly across the map
        :return:
        """
        for poi_id in range(self.n_poi):
            self.pois[poi_id, 0] = random.uniform(0, self.world_x-1.0)
            self.pois[poi_id, 1] = random.uniform(0, self.world_y-1.0)

    def init_poi_pos_circle(self):
        """
        POI positions are set in a circle around the center of the map at a specified radius.
        :return:
        """
        radius = 15.0
        interval = float(360/self.n_poi)

        x = self.world_x/2.0
        y = self.world_y/2.0
        theta = 0.0

        for poi_id in range(self.n_poi):
            self.pois[poi_id, 0] = x + radius*math.cos(theta*math.pi/180)
            self.pois[poi_id, 1] = y + radius*math.sin(theta*math.pi/180)
            theta += interval

    def init_poi_pos_two_poi(self):
        """
        Sets two POI on the map, one on each side and aligned with the center
        :return:
        """
        assert(self.n_poi == 2)

        self.pois[0, 0] = 1.0
        self.pois[0, 1] = self.world_y/2.0
        self.pois[1, 0] = (self.world_x-2.0)
        self.pois[1, 1] = self.world_y/2.0

    def init_poi_pos_four_corners(self):  # Statically set 4 POI (one in each corner)
        """
        Sets 4 POI on the map (one in each corner)
        :return:
        """
        assert(self.n_poi == 4)  # There must only be 4 POI for this initialization

        # Bottom left
        self.pois[0, 0] = 2.0
        self.pois[0, 1] = 2.0

        # Top left
        self.pois[1, 0] = 2.0
        self.pois[1, 1] = (self.world_y - 2.0)

        # Bottom right
        self.pois[2, 0] = (self.world_x - 2.0)
        self.pois[2, 1] = 2.0

        # Top right
        self.pois[3, 0] = (self.world_x - 2.0)
        self.pois[3, 1] = (self.world_y - 2.0)

    # POI VALUE FUNCTIONS --------------------------------------------------------------------------------------------
    def init_poi_vals_random(self):
        """
        POI values randomly assigned 1-10
        :return:
        """
        for poi_id in range(self.n_poi):
            self.pois[poi_id, 2] = float(random.randint(10, 100))

    def init_poi_vals_fixed(self, val=70.0):
        """
        Set all POIs to a static, fixed value
        :return:
        """
        for poi_id in range(self.n_poi):
            self.pois[poi_id, 2] = val

    # OBSTACLE FUNCTIONS ---------------------------------------------------------------------------------------------
    def create_obstacle_configs(self):
        """
        Create new obstacle configurations
        :return:
        """
        self.random_obstacles()
        self.save_obstacle_config()

    def use_saved_obstacle_config(self):
        """
        Re-use obstacle configuration stored in a CSV file in folder called Output_Data
        :return:
        """
        config_input = []
        with open('Output_Data/Obs_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        for obs_id in range(self.n_obstacles):
            self.obstacles[obs_id, 0] = float(config_input[obs_id][0])
            self.obstacles[obs_id, 1] = float(config_input[obs_id][1])
            self.obstacles[obs_id, 2] = float(config_input[obs_id][2])

    def save_obstacle_config(self):
        """
        Saves obstacle configuration to a csv file in a folder called Output_Data
        :Output: One CSV file containing Obstacle configurations
        """
        dir_name = 'Output_Data/'  # Intended directory for output files

        if not os.path.exists(dir_name):  # If Data directory does not exist, create it
            os.makedirs(dir_name)

        pfile_name = os.path.join(dir_name, 'Obs_Config.csv')

        with open(pfile_name, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for obs_id in range(self.n_obstacles):
                writer.writerow(self.obstacles[obs_id, :])

    def random_obstacles(self):
        """
        Create obstacles in random positions that the rover must avoid
        :return:
        """
        for obs_id in range(self.n_obstacles):
            x = random.uniform(0.0, self.world_x-1.0)
            y = random.uniform(0.0, self.world_y-1.0)
            self.obstacles[obs_id, 2] = random.randint(2, 5)

            # Make sure obstacle doesn't start off in range of POI
            count = self.n_poi
            while count > 0:
                for poi_id in range(self.n_poi):
                    x_dist = self.pois[poi_id, 0] - x
                    y_dist = self.pois[poi_id, 1] - y
                    dist = math.sqrt(x_dist ** 2 + y_dist ** 2)

                    if dist > self.obs_radius:
                        count -= 1

                if count > 0:
                    x = random.uniform(0.0, self.world_x - 1.0)
                    y = random.uniform(0.0, self.world_y - 1.0)
                    count = self.n_poi

            self.obstacles[obs_id, 0] = x
            self.obstacles[obs_id, 1] = y
