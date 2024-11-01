#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion
import json
import os
from math import degrees


def readJson():
    filepath = os.path.dirname(__file__)
    with open(filepath + '/goals.json', 'r') as f:
        goals = json.load(f)
    return goals

def save_goals(data):
    filepath = os.path.dirname(__file__)
    with open(filepath + '/goals.json', 'w') as f:
        json.dump(data, f, indent=4)

def goalCallback(goal_msg):
    global num_goals

    x = goal_msg.pose.position.x
    y = goal_msg.pose.position.y
    q = goal_msg.pose.orientation

    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    num_goals += 1
    data = readJson()
    data[str(num_goals)] = [x, y, degrees(yaw)]
    save_goals(data)
    print(f"Saved goal {num_goals}")



if __name__ == "__main__":
    rospy.init_node("save_goals")
    rospy.Subscriber("/move_base_simple/goal", PoseStamped, goalCallback)

    # saved_goals = readJson()
    num_goals = 0

    rospy.spin()
