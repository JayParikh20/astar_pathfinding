#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, Quaternion, Point
from nav_msgs.msg import Odometry
import tf

import numpy as np
import matplotlib.pyplot as plt

grid_size = (20, 18)
yaw_thresh = 1

#     -9 -8 -7 -6 -5 -4 -3 -2 -1 | 1  2  3  4  5  6  7  8  9
#      0  1  2  3  4  5  6  7  8 | 9 10 11 12 13 14 15 16 17
map = [0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 1, 0, 1, 0, 0, 0,  # 0  10
       0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 1, 0, 1, 0, 0, 0,  # 1  9
       0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  # 2  8
       1, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  # 3  7
       0, 1, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  # 4  6
       0, 0, 1, 0, 0, 0, 1, 1, 1,  1, 1, 1, 0, 0, 0, 0, 0, 0,  # 5  5
       0, 0, 1, 0, 0, 0, 1, 1, 1,  1, 1, 1, 0, 0, 0, 0, 0, 0,  # 6  4
       0, 0, 0, 1, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 1, 1, 0,  # 7  3
       0, 0, 0, 0, 1, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 1, 1, 1,  # 8  2
       0, 0, 0, 0, 1, 1, 0, 0, 0,  0, 0, 0, 0, 0, 0, 1, 1, 1,  # 9  1
       #                         0                             # -----
       0, 0, 0, 0, 0, 1, 0, 0, 0,  0, 0, 0, 0, 0, 0, 1, 1, 1,  # 10 -1
       0, 0, 0, 0, 0, 1, 1, 0, 0,  0, 0, 0, 0, 0, 0, 0, 1, 0,  # 11 -2
       0, 0, 0, 0, 0, 0, 1, 1, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0,  # 12 -3
       0, 0, 0, 0, 0, 0, 0, 1, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  # 13 -4
       0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  # 14 -5
       0, 0, 0, 0, 0, 0, 0, 0, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0,  # 15 -6
       0, 0, 0, 0, 0, 0, 0, 0, 1,  1, 0, 0, 0, 1, 1, 1, 1, 0,  # 16 -7
       0, 0, 0, 0, 0, 0, 0, 0, 1,  1, 1, 0, 0, 1, 1, 1, 1, 0,  # 17 -8
       0, 0, 0, 0, 0, 0, 0, 1, 1,  1, 0, 0, 0, 1, 1, 1, 1, 0,  # 18 -9
       0, 0, 0, 0, 0, 0, 0, 0, 1,  1, 0, 0, 0, 1, 1, 1, 1, 1]  # 19 -10

hardcoded_node_path = [(1, 11), (2, 12), (3, 13), (4, 14), (5, 14),
                       (6, 14), (7, 14), (8, 14), (9, 13), (10, 12),
                       (10, 11), (10, 10), (10, 9), (10, 8), (10, 7),
                       (11, 7), (12, 7), (12, 6), (12, 5), (12, 4), (12, 3), (13, 2), (13, 1), (13, 0)]


def node_to_world_pos(node: np.ndarray):
    center_x = grid_size[1] // 2  # 9
    center_y = grid_size[0] // 2  # 10
    if (node[0] < grid_size[1] // 2):
        x = (node[0] - center_x)
    else:
        x = (node[0] - center_x) + 1

    if (node[1] < grid_size[0] // 2):
        y = (center_y - node[1])
    else:
        y = (center_y - node[1]) - 1
    return np.array([x, y])


def center_offset(vec):
    offset_x = 0
    offset_y = 0
    return np.array([vec[0] + offset_x, vec[1] + offset_y])


def world_pos_to_node(pos: np.ndarray):
    center_x = grid_size[1] // 2  # 9
    center_y = grid_size[0] // 2  # 10
    if (pos[0] > 0):
        i = center_x + pos[0] - 1
    else:
        i = center_x + pos[0]

    if (pos[1] > 0):
        j = center_y - pos[1]
    else:
        j = center_y - pos[1] - 1
    return np.array([i, j])


class Node:
    # pos is the position of node indices
    pos: np.ndarray = np.array([])
    gcost: int = 0
    hcost: int = 0
    fcost: int = 0
    parent = None

    # To remove same node when calling list.remove function
    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.pos[0] == other.pos[0] and self.pos[1] == other.pos[1]  # refine as needed


def astar_pathfinder(bot_pos_in_node, goal_pos_in_node):
    open_nodes: list = []
    closed_nodes: list = []
    rospy.loginfo(f"bot_node: {bot_pos_in_node}, goal_node: {goal_pos_in_node}")
    bot_node = Node()
    bot_node.pos = bot_pos_in_node
    open_nodes.append(bot_node)
    while (open_nodes):
        current_node: Node = Node()
        min_fcost = np.inf
        for node in open_nodes:
            gcost = np.linalg.norm(node_to_world_pos(node.pos) - node_to_world_pos(bot_pos_in_node))
            hcost = np.linalg.norm(node_to_world_pos(goal_pos_in_node) - node_to_world_pos(node.pos))
            fcost = gcost + hcost
            if (fcost < min_fcost):
                min_fcost = fcost
                current_node = node
                current_node.gcost = gcost
                current_node.hcost = hcost
                current_node.fcost = fcost
        open_nodes.remove(current_node)
        closed_nodes.append(current_node)

        curr2goal_dist = np.linalg.norm(node_to_world_pos(goal_pos_in_node) - node_to_world_pos(current_node.pos))
        # rospy.loginfo(f"Working: {curr2goal_dist}")
        if (curr2goal_dist < 0.5):
            node_path = []
            search_node = current_node
            while True:
                if (search_node.parent is None):
                    break
                node_path.append(search_node.pos)
                search_node = search_node.parent
            return np.flip(np.array(node_path), axis=0)

        st_path_blocked = [False, False, False, False]

        def check_straight_elements(m, n):
            if (m == -1 and n == 1):
                return st_path_blocked[0] and st_path_blocked[1]
            if (m == 1 and n == 1):
                return st_path_blocked[1] and st_path_blocked[2]
            if (m == 1 and n == -1):
                return st_path_blocked[2] and st_path_blocked[3]
            if (m == -1 and n == -1):
                return st_path_blocked[3] and st_path_blocked[0]

        for index, (nx, ny) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]):
            neighbor_node = Node()
            neighbor_node.pos = np.array([current_node.pos[0] + nx, current_node.pos[1] + ny])
            if (neighbor_node.pos[0] < 0 or neighbor_node.pos[1] < 0):
                continue
            elif (neighbor_node.pos[0] >= grid_size[1] or neighbor_node.pos[1] >= grid_size[0]):
                continue
            elif (neighbor_node in closed_nodes):
                continue
            elif (map2d[neighbor_node.pos[0], (grid_size[0] - 1) - neighbor_node.pos[1]] == 1):
                if (index < 4):
                    st_path_blocked[index] = True
                continue
            # checks if straight path is blocked
            # elif (index >= 4 and check_straight_elements(nx, ny)):
            #     continue

            neighbor_node.gcost = current_node.gcost + np.linalg.norm(node_to_world_pos(neighbor_node.pos) - node_to_world_pos(current_node.pos))
            neighbor_node.hcost = np.linalg.norm(node_to_world_pos(goal_pos_in_node) - node_to_world_pos(neighbor_node.pos))
            neighbor_node.fcost = neighbor_node.gcost + neighbor_node.hcost
            neighbor_node.parent = current_node

            for node in open_nodes:
                if (neighbor_node == node):
                    if (neighbor_node.gcost > node.gcost):
                        continue

            open_nodes.append(neighbor_node)


class Turtlebot:
    def __init__(self):
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pos = np.array([])
        self.ort = Quaternion()
        self.start_pos = None

    def odom_callback(self, data):
        self.pos = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
        self.ort = data.pose.pose.orientation
        if (self.start_pos is None):
            self.start_pos = data.pose.pose.position

        pass


def init(world_path: list):
    turtle_bot = Turtlebot()
    rate = rospy.Rate(10)

    current_node = 0
    while not rospy.is_shutdown():
        if (turtle_bot.pos.shape[0] <= 0):
            continue
        local_goal = np.array(world_path[current_node])
        distance = np.linalg.norm(turtle_bot.pos - local_goal)
        goal_yaw = np.arctan2(turtle_bot.pos[1] - local_goal[1], turtle_bot.pos[0] - local_goal[0])
        bot_yaw = tf.transformations.euler_from_quaternion((turtle_bot.ort.x,
                                                            turtle_bot.ort.y,
                                                            turtle_bot.ort.z,
                                                            turtle_bot.ort.w))[2]
        rel_yaw = np.arctan2(np.sin(goal_yaw - bot_yaw), np.cos(goal_yaw - bot_yaw))
        rel_yaw = rel_yaw + np.pi
        if (rel_yaw > np.pi):
            rel_yaw = np.abs(np.pi - rel_yaw) - np.pi

        if (distance < 0.05 and current_node < len(world_path) - 1):
            current_node += 1
        twist_msg = Twist()
        yaw_kp = np.min([np.abs(rel_yaw), 2]) / 2
        if (current_node == len(world_path) - 1):
            dist_kp = np.min([distance, 0.5]) / 0.5
        else:
            dist_kp = 1
        if (rel_yaw < -np.deg2rad(yaw_thresh) and distance > 0.05):
            twist_msg.angular.z = -10 * yaw_kp

        elif (rel_yaw > np.deg2rad(yaw_thresh) and distance > 0.05):
            twist_msg.angular.z = 10 * yaw_kp

        elif (distance > 0.05):
            twist_msg.linear.x = 2.0 * dist_kp
        else:
            twist_msg.linear.x = 0.0
        turtle_bot.cmd_pub.publish(twist_msg)

        rate.sleep()


if __name__ == '__main__':
    rospy.init_node('astar', anonymous=True)
    start_pos = world_pos_to_node(np.array([-8, -2]))
    goalx = rospy.get_param('~goalx')
    goaly = rospy.get_param('~goaly')
    goal_pos = world_pos_to_node(np.array([round(goalx + 0.01), round(goaly)]))
    map2d = np.array(map).reshape((grid_size[0], grid_size[1])).T
    map2d = np.flip(map2d, axis=1)
    rospy.loginfo(f"start position i:{start_pos[0]}, j:{start_pos[1]}")
    rospy.loginfo(f"goal position i:{goal_pos[0]}, j:{goal_pos[1]}")
    astar_path = astar_pathfinder(start_pos, goal_pos)
    astar_path = np.vstack((start_pos, astar_path))
    world_astar_path = []
    for node in astar_path:
        world_astar_path.append(node_to_world_pos(node))
    world_astar_path = np.array(world_astar_path)
    rospy.loginfo(f"\nA-star Path:\n {world_astar_path}")

    hardcoded_world_path = []
    for node in hardcoded_node_path:
        hardcoded_world_path.append(center_offset(node_to_world_pos(node)))
    rospy.loginfo(f"hardcoded offset path: {hardcoded_world_path}")

    # init(hardcoded_world_path)
    plt.plot(np.where(map2d == 0)[0], np.where(map2d == 0)[1], marker='.', color='blue', linestyle="")
    plt.plot(np.where(map2d == 1)[0], np.where(map2d == 1)[1], marker='s', color='black', linestyle="")
    plt.plot(astar_path[:, 0], (grid_size[0] - 1) - astar_path[:, 1], marker='o', color='red')
    # plt.plot(start_pos[0], start_pos[1], marker='x', color='red')
    plt.show()
