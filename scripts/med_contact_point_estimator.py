#! /usr/bin/env python

import rospy
import numpy as np
import threading
from tf import transformations as tr

from geometry_msgs.msg import WrenchStamped
from visualization_msgs.msg import Marker

from arm_robots.med import Med
from arc_utilities.listener import Listener
from arc_utilities.tf2wrapper import TF2Wrapper
from victor_hardware_interface_msgs.msg import ControlMode
from mmint_camera_utils.recorders.wrench_recorder import WrenchRecorder


class ContactPointMarkerPublisher(object):
    def __init__(self, rate=5.0):
        self.marker_color = np.array([1.0, 0, 0, 1.0]) # red
        self.marker_type = Marker.SPHERE
        self.marker_size = .02
        self.contact_point_marker_publisher = rospy.Publisher('contact_point_marker_publisher', Marker, queue_size=100)
        self.rate = rospy.Rate(rate)
        self.show = False
        self.alive = True
        self.lock = threading.Lock()
        self.publisher_thread = threading.Thread(target=self._publish_loop)
        self.publisher_thread.start()

    def _publish_loop(self):
        while not rospy.is_shutdown():
            marker = self._get_marker()
            if self.show:
                marker.action = Marker.ADD
            else:
                marker.action = Marker.DELETE
            self.contact_point_marker_publisher.publish(marker)
            self.rate.sleep()
            with self.lock:
                if not self.alive:
                    return

    def _get_marker(self):
        marker = Marker()
        marker.header.frame_id = 'tool_contact_point'
        marker.header.stamp = rospy.Time()
        marker.id = 0
        marker.type = self.marker_type
        marker.action = Marker.ADD
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.marker_size
        marker.scale.y = self.marker_size
        marker.scale.z = self.marker_size
        marker.color.r = self.marker_color[0]
        marker.color.g = self.marker_color[1]
        marker.color.b = self.marker_color[2]
        marker.color.a = self.marker_color[3]
        return marker

    def finish(self):
        with self.lock:
            self.alive = False
        self.publisher_thread.join()


class ToolContactPointEstimator(object):

    def __init__(self, force_threshold=4.0):
        self.grasp_pose_joints = [0.7613740469101997, 1.1146166859754167, -1.6834551714751782, -1.6882417308401203,
                             0.47044861033517205, 0.8857417788890095, 0.8497585444122142]
        self.force_threshold = force_threshold
        rospy.init_node('tool_contact_point_estimator', anonymous=True)
        self.med = self._get_med()
        self.wrench_listener = Listener('/med/wrench', WrenchStamped)
        self.wrench_recorder = WrenchRecorder('/med/wrench', wait_for_data=True)
        self.tf2_listener = TF2Wrapper()
        self.calibration_wrench = None
        self.contact_point_marker_publisher = ContactPointMarkerPublisher()

    def set_grasp_pose(self):
        self.med.plan_to_joint_config(self.med.arm_group, self.grasp_pose_joints)

    def _get_med(self):
        med = Med(display_goals=False)
        med.connect()
        return med

    def get_wrench(self):
        wrench_stamped_wrist = self.wrench_listener.get(block_until_data=True)
        wrench_stamped_world = self.tf2_listener.transform_to_frame(wrench_stamped_wrist, target_frame='world',
                                                               timeout=rospy.Duration(nsecs=int(5e8)))
        return wrench_stamped_world

    def _down_stop_signal(self, feedback):
        wrench_stamped_world = self.get_wrench()
        measured_fz = wrench_stamped_world.wrench.force.z
        calibration_fz = self.calibration_wrench.wrench.force.z
        fz =  measured_fz - calibration_fz
        flag_force = np.abs(fz) >= np.abs(self.force_threshold)
        if flag_force:
            print('force z: {} (measured: {}, calibration: {}) --- flag: {}'.format(fz, measured_fz, calibration_fz, flag_force))
            # activate contact detector pub
            self.contact_point_marker_publisher.show = True
        return flag_force

    def _up_signal(self, feedback):
        wrench_stamped_world = self.get_wrench()
        measured_fz = wrench_stamped_world.wrench.force.z
        calibration_fz = self.calibration_wrench.wrench.force.z
        fz = measured_fz - calibration_fz
        flag_no_force = np.abs(fz) < np.abs(self.force_threshold)
        if flag_no_force:
            self.contact_point_marker_publisher.show = False # deactivate the force flag
        out_flag = False
        return out_flag

    def lower_down(self):

        lowering_z = 0.065 # we could go as low as 0.06
        # Lower down
        current_grasp_pose = self.tf2_listener.get_transform('world', 'grasp_frame')
        current_grasp_pos = current_grasp_pose[:3, 3]
        target_pos = np.append(current_grasp_pos[:2], lowering_z)

        # Update the calibration
        self.calibration_wrench = self.get_wrench()

        self.med.set_execute(False)
        plan_result = self.med.plan_to_position_cartesian(self.med.arm_group, 'grasp_frame', target_position=target_pos)
        self.med.set_execute(True)
        self.med.follow_arms_joint_trajectory(plan_result.planning_result.plan.joint_trajectory, stop_condition=self._down_stop_signal)

    def raise_up(self):
        z_value = 0.35
        current_grasp_pose = self.tf2_listener.get_transform('world', 'grasp_frame')
        current_grasp_pos = current_grasp_pose[:3, 3]
        target_pos = np.append(current_grasp_pos[:2], z_value)
        self.med.set_execute(False)
        plan_result = self.med.plan_to_position_cartesian(self.med.arm_group, 'grasp_frame', target_position=target_pos)
        self.med.set_execute(True)
        self.med.follow_arms_joint_trajectory(plan_result.planning_result.plan.joint_trajectory,
                                              stop_condition=self._up_signal)

    def rotate_along_axis_angle(self, axis, angle, frame='grasp_frame'):
        # get current frame pose:
        current_frame_pose = self.tf2_listener.get_transform('world', frame)
        current_frame_pos = current_frame_pose[:3, 3].copy()
        rotation_matrix = tr.quaternion_matrix(tr.quaternion_about_axis(angle, axis))
        new_pose_matrix = rotation_matrix @ current_frame_pose
        new_position = current_frame_pos
        new_quat = tr.quaternion_from_matrix(new_pose_matrix)
        target_pose = np.concatenate([new_position, new_quat])
        self.med.plan_to_pose(self.med.arm_group, frame, target_pose=list(target_pose), frame_id='world')

    def estimate_motion(self):
        self.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.1)
        self.set_grasp_pose()
        rospy.sleep(2.0)
        self.lower_down()
        rospy.sleep(3.0)
        # move on the plane
        move_dist = 0.2
        current_grasp_pose = self.tf2_listener.get_transform('world', 'grasp_frame')
        current_grasp_pos = current_grasp_pose[:3, 3]
        target_pos = current_grasp_pos + move_dist * np.array([0, 1, 0])
        self.med.plan_to_position_cartesian(self.med.arm_group, 'grasp_frame', target_position=target_pos)
        rospy.sleep(2.0)
        self.raise_up()
        rospy.sleep(2.0)
        # rotate
        self.rotate_along_axis_angle(axis=np.array([1,0,0]), angle=np.pi/8, frame='grasp_frame')
        rospy.sleep(2.0)
        # lower down again
        self.lower_down()
        rospy.sleep(4.0)
        self.raise_up()
        rospy.sleep(2.0)
        # rotate
        self.rotate_along_axis_angle(axis=np.array([1, 0, 0]), angle=-np.pi / 8, frame='grasp_frame')
        rospy.sleep(2.0)
        # lower down again
        self.lower_down()
        rospy.sleep(4.0)
        self.raise_up()

    def close(self):
        self.contact_point_marker_publisher.finish()


def contact_point_estimation_with_actions():
    force_threshold = 4.0
    tcpe = ToolContactPointEstimator(force_threshold=force_threshold)
    tcpe.estimate_motion()
    tcpe.close()





if __name__ == '__main__':
    contact_point_estimation_with_actions()