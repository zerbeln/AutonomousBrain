import math


def go_towards_poi_reward(poi_pos, rover):
    """
    Rover receives a reward encouraging it to move closer to a particular POI
    :param poi_pos:
    :param rover:
    :return:
    """
    x_dist = poi_pos[0] - rover.rover_x
    y_dist = poi_pos[1] - rover.rover_y
    dist = math.sqrt(x_dist**2 + y_dist**2)

    reward = -dist

    return reward


def go_away_poi_reward(poi_pos, rover, wx, wy):
    """
    Rover receives a reward encouraging it to move further away from a particular POI
    :param poi_pos:
    :param rover:
    :param wx: world x-dimension
    :param wy: world y-dimension
    :return:
    """
    x_dist = poi_pos[0] - rover.rover_x
    y_dist = poi_pos[1] - rover.rover_y
    dist = math.sqrt(x_dist**2 + y_dist**2)

    if 0.0 > rover.rover_x or rover.rover_x > wx:
        dist = -1000.00
    elif 0.0 > rover.rover_y or rover.rover_y > wy:
        dist = -1000.00

    reward = dist

    return reward


def avoid_obstacle_reward(obs_id, obstacles, rover, wx, wy):
    """
    Rover receives reward encouraging it to avoid an obstacle
    :param obs_id:
    :param obstacles:
    :param rover:
    :param wx: world x-dimension
    :param wy: world y-dimension
    :return:
    """
    x_dist = obstacles[obs_id, 0] - rover.rover_x
    y_dist = obstacles[obs_id, 1] - rover.rover_y
    dist = math.sqrt(x_dist**2 + y_dist**2)

    if 0.0 > rover.rover_x or rover.rover_x > wx:
        dist = -1000.00
    elif 0.0 > rover.rover_y or rover.rover_y > wy:
        dist = -1000.00

    reward = dist

    return reward
