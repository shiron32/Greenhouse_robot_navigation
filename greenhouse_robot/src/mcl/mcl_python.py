#!/usr/bin/env python

import cv2
import math
import copy
import rospy
import numpy as np
from tf import transformations
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, PoseStamped


#####################
# TOOL
#####################


def xyzrpy2mat(x, y, z, roll, pitch, yaw):
    rot_vec = np.zeros((3, 1), dtype=np.float32)
    rot_vec[0] = roll
    rot_vec[1] = pitch
    rot_vec[2] = yaw
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    result = np.zeros((4, 4), dtype=np.float32)
    result[0:3, 0:3] = rot_mat
    result[0, 3] = x
    result[1, 3] = y
    result[2, 3] = z
    result[3, 3] = 1
    return result


def mat2xyzrpy(mat):
    result = np.zeros(6)
    result[0] = mat[0, 3]
    result[1] = mat[1, 3]
    result[2] = mat[2, 3]
    rot_mat = mat[0:3, 0:3]
    rot_vec, _ = cv2.Rodrigues(rot_mat)
    result[3] = rot_vec[0]
    result[4] = rot_vec[1]
    result[5] = rot_vec[2]
    return result


#####################
# MCL
#####################


class Particle:
    def __init__(self):
        self.pose = np.eye(4, dtype=np.float32)
        self.score = np.float32(0.0)
        self.scan = np.zeros((4, 4), dtype=np.float32)


class mcl:
    def __init__(self):
        self.gen = np.random.default_rng(1)
        self.imageResolution = None
        self.mapCenterX = None
        self.mapCenterY = None
        self.particles = []
        self.maxProbParticle = Particle()
        self.gridMap = None
        self.gridMapCV = None
        self.poseMap = None
        self.particlesMap = None
        self.odomBefore = None
        self.numOfParticle = 2500
        self.repropagateCountNeeded = 1
        self.odomCovariance = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02]).astype(np.float32)
        self.tf_laser2robot = np.array([[1, 0, 0, 0],
                                        [0, -1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]]).astype(np.float32)
        self.isOdomInitialized = False
        self.predictionCounter = 0

        self.bridge = CvBridge()
        self.showmap_pub = rospy.Publisher("/image/showmap", Image, queue_size=10)
        self.posemap_pub = rospy.Publisher("/image/posemap", Image, queue_size=10)
        self.particle_pub = rospy.Publisher("/particles", PoseArray, queue_size=10)

        particle_pub_timer = rospy.Timer(rospy.Duration(0.5), self.publishParticles)

        self.getMap()
        self.initializeParticles()
        self.showInMap()

        self.pose_estimated = (0,0)

    def convertToPose(self):
        particles = []
        for p in self.particles:
            p_ = Pose()
            x, y, z, roll, pitch, yaw = mat2xyzrpy(p.pose)
            p_.position.x = x
            p_.position.y = y

            quat = transformations.quaternion_from_euler(roll, pitch, yaw)
            p_.orientation.z = quat[2]
            p_.orientation.w = quat[3]
            particles.append(p_)
        return particles
    
    def publishParticles(self, event):
        if len(self.particles) != 0:
            particles = PoseArray()
            particles.header.frame_id = "map"
            particles.poses = self.convertToPose()
            self.particle_pub.publish(particles)

    def getMap(self):
        map_msg = rospy.wait_for_message("/map", OccupancyGrid, timeout=10)

        i = 0
        self.gridMap = np.zeros((map_msg.info.height, map_msg.info.width), dtype=np.byte)
        self.gridMapCV = np.zeros((map_msg.info.height, map_msg.info.width), dtype=np.ubyte)
        for y in range(map_msg.info.height):
            for x in range(map_msg.info.width):            
                self.gridMap[y, x] = map_msg.data[i]
                self.gridMapCV[y, x] = 0 if map_msg.data[i] == -1 else map_msg.data[i]
                i += 1
        
        # cv2.imshow("image", self.gridMapCV)
        # cv2.waitKey(0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1, 1))
        self.gridMapCV = cv2.dilate(self.gridMapCV, kernel)
        _, self.poseMap = cv2.threshold(self.gridMapCV, 50, 255, cv2.THRESH_BINARY_INV)
        self.poseMap = cv2.cvtColor(self.poseMap, cv2.COLOR_GRAY2BGR)
        self.particlesMap = self.poseMap.copy()

        self.mapCenterX = int(map_msg.info.origin.position.x + map_msg.info.width * map_msg.info.resolution / 2.0) + 0.75
        self.mapCenterY = int(map_msg.info.origin.position.y + map_msg.info.height * map_msg.info.resolution / 2.0)
        self.imageResolution = map_msg.info.resolution

    def initializeParticles(self):
        rospy.loginfo("Starting particle initialization")
        self.particles = []
        
        i = 0
        while i != self.numOfParticle:
            particle_temp = Particle()
            randomX = self.gen.uniform(self.mapCenterX - self.gridMap.shape[1] * self.imageResolution / 2.0,
                                        self.mapCenterX + self.gridMap.shape[1] * self.imageResolution / 2.0)
            randomY = self.gen.uniform(self.mapCenterY - self.gridMap.shape[0] * self.imageResolution / 2.0,
                                        self.mapCenterY + self.gridMap.shape[0] * self.imageResolution / 2.0)
            randomTheta = self.gen.uniform(-math.pi, math.pi)
            ptX = int((randomX - self.mapCenterX + (self.gridMap.shape[1] * self.imageResolution) / 2) / self.imageResolution)
            ptY = int((randomY - self.mapCenterY + (self.gridMap.shape[0] * self.imageResolution) / 2) / self.imageResolution)
            if self.gridMap[ptY, ptX] != 0:
                continue
            particle_temp.pose = xyzrpy2mat(randomX, randomY, 0, 0, 0, randomTheta)
            particle_temp.score = 1 / self.numOfParticle
            self.particles.append(particle_temp)
            i += 1
        rospy.loginfo("Completed")

    def prediction(self, diffPose):
        diff_xyzrpy = mat2xyzrpy(diffPose)

        delta_trans = math.sqrt(pow(diff_xyzrpy[0], 2) + pow(diff_xyzrpy[1], 2))
        delta_rot1 = math.atan2(diff_xyzrpy[1], diff_xyzrpy[0])
        delta_rot2 = diff_xyzrpy[5] - delta_rot1

        if delta_rot1 > math.pi:
            delta_rot1 -= (2 * math.pi)
        if delta_rot1 < -math.pi:
            delta_rot1 += (2 * math.pi)
        if delta_rot2 > math.pi:
            delta_rot2 -= (2 * math.pi)
        if delta_rot2 < -math.pi:
            delta_rot2 += (2 * math.pi)

        # Add noises to trans/rot1/rot2
        trans_noise_coeff = self.odomCovariance[2] * abs(delta_trans) + self.odomCovariance[3] * abs(delta_rot1 + delta_rot2)
        rot1_noise_coeff = self.odomCovariance[0] * abs(delta_rot1) + self.odomCovariance[1] * abs(delta_trans)
        rot2_noise_coeff = self.odomCovariance[0] * abs(delta_rot2) + self.odomCovariance[1] * abs(delta_trans)

        for i in range(self.numOfParticle):
            gaussian_distribution = self.gen.normal(0, 1)

            delta_trans = delta_trans + gaussian_distribution * trans_noise_coeff
            delta_rot1 = delta_rot1 + gaussian_distribution * rot1_noise_coeff
            delta_rot2 = delta_rot2 + gaussian_distribution * rot2_noise_coeff

            x = delta_trans * np.cos(delta_rot1) + gaussian_distribution * self.odomCovariance[4]
            y = delta_trans * np.sin(delta_rot1) + gaussian_distribution * self.odomCovariance[5]
            theta = delta_rot1 + delta_rot2 + gaussian_distribution * self.odomCovariance[0] * (math.pi / 180.0)

            diff_odom_w_noise = xyzrpy2mat(x, y, 0, 0, 0, -theta)
            pose_t_plus_1 = self.particles[i].pose @ diff_odom_w_noise
            self.particles[i].pose = pose_t_plus_1

    def weightning(self, laser):
        maxScore = np.float32(0.0)
        scoreSum = np.float32(0.0)

        for i in range(self.numOfParticle):
            transLaser = (self.particles[i].pose @ self.tf_laser2robot) @ laser
            calcedWeight = np.float32(0.0)

            for j in range(transLaser.shape[1]):
                ptX = int((transLaser[0, j] - self.mapCenterX + (self.gridMap.shape[1] * self.imageResolution) / 2) / self.imageResolution)
                ptY = int((transLaser[1, j] - self.mapCenterY + (self.gridMap.shape[0] * self.imageResolution) / 2) / self.imageResolution)

                if ptX < 0 or ptX >= self.gridMap.shape[1] or ptY < 0 or ptY >= self.gridMap.shape[0]:
                    continue
                else:
                    calcedWeight += np.float32(self.gridMapCV[ptY, ptX] / np.float32(100.0))

            self.particles[i].score = np.float32(self.particles[i].score + (calcedWeight / np.float32(transLaser.shape[1])))
            scoreSum += self.particles[i].score

            if maxScore < self.particles[i].score:
                self.maxProbParticle = self.particles[i]
                self.maxProbParticle.scan = laser
                maxScore = self.particles[i].score

        # scoreArr = np.array([particle.score for particle in self.particles]).astype(np.float32)
        # scoreSum = np.sum(scoreArr).astype(np.float32)
        # scoreNorm = scoreArr / scoreSum
        # for i, particle in enumerate(self.particles):
        #     particle.score = scoreNorm[i]

        for i in range(self.numOfParticle):
            self.particles[i].score = self.particles[i].score / scoreSum

    def resampling(self):
        particleScores = []
        particleSampled = []
        scoreBaseline = np.float32(0.0)
    
        for particle in self.particles:
            scoreBaseline += particle.score
            particleScores.append(scoreBaseline)

        for i in range(self.numOfParticle):
            darted = self.gen.uniform(0, scoreBaseline)
            particleIndex = next(x[0] for x in enumerate(particleScores) if x[1] > darted)
            selectedParticle = copy.deepcopy(self.particles[particleIndex])
            selectedParticle.score = 1.0 / self.numOfParticle
            particleSampled.append(selectedParticle)

        self.particles = particleSampled

    def showInMap(self):
        showMap = self.particlesMap.copy()

        for i in range(self.numOfParticle):
            xPos = int((self.particles[i].pose[0, 3] - self.mapCenterX + (self.gridMap.shape[1] * self.imageResolution) / 2) / self.imageResolution)
            yPos = int((self.particles[i].pose[1, 3] - self.mapCenterY + (self.gridMap.shape[0] * self.imageResolution) / 2) / self.imageResolution)
            cv2.circle(showMap, (xPos, yPos), 1, (255, 0, 0), -1)

        if self.maxProbParticle.score > 0:
            x_all = 0
            y_all = 0
            for i in range(self.numOfParticle):
                x_all += self.particles[i].pose[0, 3] * self.particles[i].score
                y_all += self.particles[i].pose[1, 3] * self.particles[i].score
            xPos = int((x_all - self.mapCenterX + (self.gridMap.shape[1] * self.imageResolution) / 2) / self.imageResolution)
            yPos = int((y_all - self.mapCenterY + (self.gridMap.shape[0] * self.imageResolution) / 2) / self.imageResolution)
            
            self.pose_estimated = (xPos, yPos)

            cv2.circle(showMap, (xPos, yPos), 2, (0, 0, 255), -1)
            cv2.circle(self.poseMap, (xPos, yPos), 1, (0, 0, 255), -1)

            transLaser = (self.maxProbParticle.pose @ self.tf_laser2robot) @ self.maxProbParticle.scan

            for i in range(transLaser.shape[1]):
                xPos = int((transLaser[0, i] - self.mapCenterX + (self.gridMap.shape[1] * self.imageResolution) / 2) / self.imageResolution)
                yPos = int((transLaser[1, i] - self.mapCenterY + (self.gridMap.shape[0] * self.imageResolution) / 2) / self.imageResolution)
                cv2.circle(showMap, (xPos, yPos), 1, (0, 255, 255), -1)

        self.showmap_pub.publish(self.bridge.cv2_to_imgmsg(showMap, "bgr8"))
        self.posemap_pub.publish(self.bridge.cv2_to_imgmsg(self.poseMap, "bgr8"))

    def updateData(self, pose, laser):
        if not self.isOdomInitialized:
            self.odomBefore = pose
            self.isOdomInitialized = True
            return

        diffOdom = np.linalg.inv(self.odomBefore) @ pose

        self.prediction(diffOdom)
        self.weightning(laser)
        self.predictionCounter += 1
        if self.predictionCounter == self.repropagateCountNeeded:
            self.resampling()
            self.predictionCounter = 0
        self.showInMap()
        
        self.odomBefore = pose

    def getPositionError(self):
        x_all = 0
        y_all = 0
        scores = 0
        for i in range(self.numOfParticle):
            x_all += self.particles[i].pose[0, 3] * self.particles[i].score
            y_all += self.particles[i].pose[1, 3] * self.particles[i].score
            scores += self.particles[i].score
        return x_all/scores, y_all/scores


#####################
# MCL NODE
#####################


class MclNode(object):
    def __init__(self):
        self.mclocalizer = mcl()
        self.vec_poses = []
        self.vec_poses_time = []
        self.vec_lasers = []
        self.vec_lasers_time = []

        self.subscribe_laser = rospy.Subscriber('scan', LaserScan, self.callback_laser, queue_size=10)
        self.subscribe_pose = rospy.Subscriber('/odom', Odometry, self.callback_pose, queue_size=10)

        self.odom_estimate_pub = rospy.Publisher('/odom_estimate', Odometry, queue_size=10)
        self.odom_estimate = Odometry()
        self.odom_estimate.header.frame_id = "map" 

    def check_data(self):
        while (len(self.vec_poses) != 0 and len(self.vec_lasers) != 0):
            if (np.abs(self.vec_poses_time[0] - self.vec_lasers_time[0]) > 30.0):
                if (self.vec_poses_time[0] > self.vec_lasers_time[0]):
                    self.vec_lasers.pop(0)
                    self.vec_lasers_time.pop(0)
                else:
                    self.vec_poses.pop(0)
                    self.vec_poses_time.pop(0)
            else:
                mat_poses = np.asarray(self.vec_poses[0], dtype=np.float32)
                mat_lasers = np.asarray(self.vec_lasers[0], dtype=np.float32)
                self.mclocalizer.updateData(mat_poses, mat_lasers)
                self.vec_lasers.pop(0)
                self.vec_lasers_time.pop(0)
                self.vec_poses.pop(0)
                self.vec_poses_time.pop(0)

    def callback_laser(self, msg):
        scanQuantity = int((msg.angle_max - msg.angle_min) / msg.angle_increment + 1)
        eigenLaser = np.ones((4, 0), dtype=np.float32)
        scanEffective = 0
        for i in range(scanQuantity):
            dist = msg.ranges[i]
            if dist > 0.12 and dist < 10.0:
                scanEffective += 1
                eigenLaser = np.lib.pad(eigenLaser, ((0,0),(0,1)), 'constant', constant_values=(0))
                eigenLaser[0, scanEffective-1] = dist * np.cos(msg.angle_min + msg.angle_increment * i)
                eigenLaser[1, scanEffective-1] = dist * np.sin(msg.angle_min + msg.angle_increment * i)
                eigenLaser[2, scanEffective-1] = 0.0
                eigenLaser[3, scanEffective-1] = 1.0
        self.vec_lasers.append(eigenLaser)
        self.vec_lasers_time.append(msg.header.stamp.to_sec())
        self.check_data()

    def callback_pose(self, msg):
        position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=np.float32)
        orientation = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w], dtype=np.float32)
        
        # Convert quaternion to rotation matrix
        rotation_matrix = transformations.quaternion_matrix(orientation)[:3, :3]

        # Convert rotation matrix to Euler angles
        euler_angles = transformations.euler_from_matrix(rotation_matrix, 'sxyz')
        self.current_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, euler_angles[2]], dtype=np.float32)
        err_pose = self.mclocalizer.getPositionError()

        eigenPose = transformations.quaternion_matrix(orientation).astype(np.float32)
        eigenPose[0:3, 3] = position
        self.vec_poses.append(eigenPose)
        self.vec_poses_time.append(msg.header.stamp.to_sec())
        
        self.odom_estimate.pose.pose.position.x = err_pose[0] + self.current_position[0] 
        self.odom_estimate.pose.pose.position.y = err_pose[1] + self.current_position[1] 
        self.odom_estimate.pose.pose.orientation.w = 1.0

        self.odom_estimate_pub.publish(self.odom_estimate)


def main():
    rospy.init_node('mcl_python')
    MclNode()
    rospy.spin()


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
