#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Pose, PoseArray
from tf.transformations import quaternion_from_euler
import json
import math
import os


def readGoals():
    filepath = os.path.dirname(__file__)
    with open(filepath + "/goals.json", 'r') as f:
        return json.load(f)

def createPoseArray(goals):
    for goal_id in goals.keys():
        goal = goals[goal_id]
        x = goal[0]
        y = goal[1]
        yaw = math.radians(goal[2])
        _, _, z, w = quaternion_from_euler(0, 0, yaw)

        goal_pose = Pose()
        goal_pose.position.x = x
        goal_pose.position.y = y
        goal_pose.orientation.z = z
        goal_pose.orientation.w = w

        goal_pose_array.poses.append(goal_pose)


if __name__ == "__main__":
    rospy.init_node("display_goals")
    goalsPub = rospy.Publisher("/goal_poses", PoseArray, queue_size=10)
    goal_pose_array = PoseArray()
    goal_pose_array.header.frame_id = "map"

    goals = readGoals()
    createPoseArray(goals)

    while not rospy.is_shutdown():
        goalsPub.publish(goal_pose_array)
        rospy.sleep(1)