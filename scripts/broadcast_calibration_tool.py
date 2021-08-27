#!/usr/bin/env python

import sys
import os
import rospy
import tf2_ros
import tf.transformations as tr
import moveit_commander
from moveit_commander.conversions import pose_to_list
import mmint_utils

from geometry_msgs.msg import TransformStamped, Pose
from moveit_msgs.msg import AttachedCollisionObject, PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive

package_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/bubble_control')[0], 'bubble_control')


def broadcast_calibration_tool():

    path_to_tool_configuration = os.path.join(package_path, 'config', 'calibration_tool.yaml')
    cfg = mmint_utils.load_cfg(path_to_tool_configuration)
    rospy.init_node('calibration_tool_collision_node', anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)
    current_scene = moveit_commander.PlanningSceneInterface(ns='med')

    rospy.sleep(1.0)

    co = CollisionObject()
    co.header.frame_id = '/med_base'
    co.id = 'calibration_tool'
    co.operation = co.ADD
    # Define a the table as a box:
    ct_sp = SolidPrimitive() # calibration tool solid primitive
    ct_sp.type = ct_sp.CYLINDER
    ct_sp.dimensions = [cfg['tool_size']['h'], 0.5 * cfg['tool_size']['diameter']]  # cylinder_height, cylinder_radius
    ct_pose = Pose()
    ct_pose.position.x = cfg['tool_center']['x']
    ct_pose.position.y = cfg['tool_center']['y']
    ct_pose.position.z = cfg['tool_size']['h'] * .5 #assume no offset, just lies on top of the table
    ct_pose.orientation.x = 0.0
    ct_pose.orientation.y = 0.0
    ct_pose.orientation.z = 0.0
    ct_pose.orientation.w = 1.0
    co.primitives.append(ct_sp)
    co.primitive_poses.append(ct_pose)
    current_scene.add_object(co)
    rospy.sleep(5.0)
    static_broadcaster = tf2_ros.StaticTransformBroadcaster()
    calibration_tool_tf = TransformStamped()
    calibration_tool_tf.header.stamp = rospy.Time.now()
    calibration_tool_tf.header.frame_id = 'world'
    calibration_tool_tf.child_frame_id = 'calibration_tool'
    pos = [cfg['tool_center']['x'], cfg['tool_center']['y'], cfg['tool_size']['h']*.5]
    quat = [0,0,0,1]
    calibration_tool_tf.transform.translation.x = pos[0]
    calibration_tool_tf.transform.translation.y = pos[1]
    calibration_tool_tf.transform.translation.z = pos[2]
    calibration_tool_tf.transform.rotation.x = quat[0]
    calibration_tool_tf.transform.rotation.y = quat[1]
    calibration_tool_tf.transform.rotation.z = quat[2]
    calibration_tool_tf.transform.rotation.w = quat[3]
    static_broadcaster.sendTransform(calibration_tool_tf)
    rospy.spin()


if __name__ == '__main__':
    broadcast_calibration_tool()