import numpy as np
import math


def go_towards_poi_reward(poi_id, pois, rover):
    """
    Rover receives a reward encouraging it to move closer to a particular POI
    :param poi_id:
    :param pois:
    :param rover:
    :return:
    """
    x_dist = pois[poi_id, 0] - rover.rover_x
    y_dist = pois[poi_id, 1] - rover.rover_y
    dist = x_dist**2 + y_dist**2

    reward = -dist

    return reward


def go_away_poi_reward(poi_id, pois, rover):
    """
    Rover receives a reward encouraging it to move further away from a particular POI
    :param poi_id:
    :param pois:
    :param rover:
    :return:
    """
    x_dist = pois[poi_id, 0] - rover.rover_x
    y_dist = pois[poi_id, 1] - rover.rover_y
    dist = x_dist**2 + y_dist**2

    reward = dist

    return reward


def avoid_obstacle_reward(obs_id, obstacles, rover):
    """
    Rover receives reward encouraging it to avoid an obstacle
    :param obs_id:
    :param obstacles:
    :param rover:
    :return:
    """
    x_dist = obstacles[obs_id, 0] - rover.rover_x
    y_dist = obstacles[obs_id, 1] - rover.rover_y
    dist = x_dist**2 + y_dist**2

    reward = dist

    return reward
