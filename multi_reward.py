import numpy as np
import math


def furthest_poi_reward(n_pois, pois, rover, obs_rad):

    distances = np.zeros(n_pois)
    max_dist = 0.0
    max_poi_id = -1
    reward = 0.0

    for poi_id in range(n_pois):
        x_dist_i = pois[poi_id, 0] - rover.rx_init
        y_dist_i = pois[poi_id, 1] - rover.ry_init
        x_dist = pois[poi_id, 0] - rover.rover_x
        y_dist = pois[poi_id, 1] - rover.rover_y

        initial_dist = math.sqrt(x_dist_i**2 + y_dist_i**2)
        distances[poi_id] = math.sqrt(x_dist**2 + y_dist**2)
        if initial_dist > max_dist:
            max_dist = initial_dist
            max_poi_id = poi_id

    if distances[max_poi_id] < obs_rad:
        reward = pois[max_poi_id, 2]/distances[max_poi_id]

    return reward


def closest_poi_reward(n_pois, pois, rover, obs_rad):

    distances = np.zeros(n_pois)
    min_dist = 1000.0
    min_poi_id = -1
    reward = 0.0

    for poi_id in range(n_pois):
        x_dist_i = pois[poi_id, 0] - rover.rx_init
        y_dist_i = pois[poi_id, 1] - rover.ry_init
        x_dist = pois[poi_id, 0] - rover.rover_x
        y_dist = pois[poi_id, 1] - rover.rover_y

        initial_dist = math.sqrt(x_dist_i ** 2 + y_dist_i ** 2)
        distances[poi_id] = math.sqrt(x_dist ** 2 + y_dist ** 2)
        if initial_dist < min_dist:
            min_dist = initial_dist
            min_poi_id = poi_id

    if distances[min_poi_id] < obs_rad:
        reward = pois[min_poi_id, 2] / distances[min_poi_id]

    return reward


def go_towards_poi_reward(poi_id, pois, rover):
    x_dist = pois[poi_id, 0] - rover.rover_x
    y_dist = pois[poi_id, 1] - rover.rover_y
    dist = x_dist**2 + y_dist**2

    reward = -dist

    return reward


def go_away_poi_reward(poi_id, pois, rover):
    x_dist = pois[poi_id, 0] - rover.rover_x
    y_dist = pois[poi_id, 1] - rover.rover_y
    dist = x_dist**2 + y_dist**2

    reward = dist

    return reward

