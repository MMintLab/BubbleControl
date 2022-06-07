#! /usr/bin/env python
import rospy
import numpy as np
from bubble_utils.bubble_med.bubble_med import BubbleMed
import tf.transformations as tr
from geometry_msgs.msg import PoseStamped
from victor_hardware_interface_msgs.msg import MotionCommand, ControlMode
from victor_hardware_interface.victor_utils import get_cartesian_impedance_params, send_new_control_mode
from bubble_utils.bubble_med.aux.load_confs import load_robot_configurations
from mmint_camera_utils.ros_utils.utils import matrix_to_pose, pose_to_matrix


class CartesianMed(BubbleMed):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.robot_confs = load_robot_configurations()
        self.home_joints = self._get_home_joints()
        self.command_frame = 'med_base'
        self.cartesian_impedance_frame = 'med_kuka_link_ee'
        # self.cartesian_impedance_frame = 'grasp_frame'

    def _get_home_joints(self):
        home_joints = self.robot_confs['home_conf']# Override the method if needed
        return home_joints

    def home_robot(self):
        self.plan_to_joint_config(self.arm_group, self.home_joints)

    def zero_robot(self):
        # bring the robot up (straight) by setting all joints to zero
        self.set_robot_conf('zero_conf')

    def set_robot_conf(self, conf_name):
        joint_conf = self.robot_confs[conf_name]
        self.plan_to_joint_config(self.arm_group, joint_conf)

    def set_cartesian_impedance(self, velocity, x_stiffness=7500, y_stiffness=7500, z_stiffnes=5000):
        cartesian_impedance_params = get_cartesian_impedance_params(velocity=velocity * 40)  # we multiply by 40 because in get_control_mode they do the same...
        cartesian_impedance_params.cartesian_impedance_params.cartesian_stiffness.z = z_stiffnes  # by default is 5000
        cartesian_impedance_params.cartesian_impedance_params.cartesian_stiffness.x = x_stiffness
        cartesian_impedance_params.cartesian_impedance_params.cartesian_stiffness.y = y_stiffness
        cartesian_impedance_params.cartesian_impedance_params.cartesian_damping.a = 0.5
        cartesian_impedance_params.cartesian_impedance_params.cartesian_damping.b = 0.5
        cartesian_impedance_params.cartesian_impedance_params.cartesian_damping.c = 0.5
        # import pdb; pdb.set_trace()
        send_new_control_mode(arm='med', msg=cartesian_impedance_params)

    def set_joint_position_control(self, vel=0.1, **kwargs):
        self.set_control_mode(control_mode=ControlMode.JOINT_POSITION, vel=vel, **kwargs)

    # TESTING FUNCTIONS ------------------------------------------------------------------------------------------------

    def raw_cartesian_command(self, target_pose, ref_frame='med_base', frame_id='grasp_frame'):
        target_pose_ci = self._tr_pose_to_cartesian_impedance_frame(target_pose, frame_id)
        target_pose_stamped = PoseStamped()
        target_pose_stamped.pose.position.x = target_pose_ci[0]
        target_pose_stamped.pose.position.y = target_pose_ci[1]
        target_pose_stamped.pose.position.z = target_pose_ci[2]
        target_pose_stamped.pose.orientation.x = target_pose_ci[3]
        target_pose_stamped.pose.orientation.y = target_pose_ci[4]
        target_pose_stamped.pose.orientation.z = target_pose_ci[5]
        target_pose_stamped.pose.orientation.w = target_pose_ci[6]
        target_pose_stamped.header.frame_id = ref_frame
        self._raw_cartesian_command_from_pose_msg(target_pose_stamped)

    def _raw_cartesian_command_from_pose_msg(self, target_pose_stamped):
        motion_command = MotionCommand()
        # command_frame_original = self.sensor_frames[self.active_arm]
        motion_command.header.frame_id = self.command_frame
        motion_command.control_mode.mode = ControlMode.CARTESIAN_IMPEDANCE
        target_in_arm_frame = self.tf_wrapper.tf_buffer.transform(target_pose_stamped, self.command_frame)
        motion_command.cartesian_pose = target_in_arm_frame.pose
        # pub = self.motion_command_publisher[self.active_arm]
        pub = self.arm_command_pub
        while pub.get_num_connections() < 1:
            rospy.sleep(0.01)
        pub.publish(motion_command)

    def _tr_pose_to_cartesian_impedance_frame(self, pose, frame_id):
        # tranform the pose from world to frame_id to world to cartesian impedance frame
        # (assume fixed tf between frame_id and cartesian impedance frame)
        w_X_f = pose_to_matrix(pose)
        f_X_ci = self.tf_wrapper.get_transform(parent=frame_id, child=self.cartesian_impedance_frame)
        w_X_ci = w_X_f @ f_X_ci
        pose_ci = matrix_to_pose(w_X_ci)
        return pose_ci


def raw_cartesian_test():
    # frame_ee = 'grasp_frame'
    frame_ee = 'med_kuka_link_ee'
    rospy.init_node('bubble_med')
    med = CartesianMed(display_goals=False)
    med.connect()
    med.set_joint_position_control()
    med.home_robot()
    target_pos = np.array([0.6, 0, 0.2])
    target_quat = np.array([-np.cos(np.pi / 4), np.cos(np.pi / 4), 0, 0])
    target_pose = np.concatenate([target_pos, target_quat])
    med.plan_to_pose(med.arm_group, ee_link_name='grasp_frame', target_pose=list(target_pose), frame_id='med_base')
    med.set_cartesian_impedance(1, x_stiffness=5000, y_stiffness=5000, z_stiffnes=100)
    _ = input('press enter to continue')
    X_tr = med.tf_wrapper.get_transform(parent='med_base', child=frame_ee)
    # import pdb; pdb.set_trace()
    target_pos = X_tr[:3,3]
    target_quat = tr.quaternion_from_matrix(X_tr)
    delta_pos = np.array([0.001, 0.00, 0.])
    delta_quat = tr.quaternion_about_axis(angle=np.pi/45, axis=np.array([1,0,0]))

    for i in range(10):
        target_pos = target_pos + delta_pos
        target_quat = tr.quaternion_multiply(delta_quat, target_quat)
        # target_quat_i = tr.quaternion_multiply(target_quat, delta_quat)
        target_pose_i = np.concatenate([target_pos, target_quat])
        med.raw_cartesian_command(target_pose_i, ref_frame='med_base', frame_id=frame_ee)
        _ = input('press enter to continue')


def raw_cartesian_impedance_grasp_frame_test():
    rospy.init_node('bubble_med')
    med = CartesianMed(display_goals=False)
    med.cartesian_impedance_frame = 'grasp_frame'
    med.connect()
    med.set_joint_position_control()
    med.home_robot()
    target_pos = np.array([0.6, 0, 0.2])
    target_quat = np.array([-np.cos(np.pi / 4), np.cos(np.pi / 4), 0, 0])
    target_pose = np.concatenate([target_pos, target_quat])
    med.plan_to_pose(med.arm_group, ee_link_name='grasp_frame', target_pose=list(target_pose), frame_id='med_base')
    med.set_cartesian_impedance(1, x_stiffness=100, y_stiffness=5000, z_stiffnes=100)
    _ = input('press enter to continue')
    delta_pos = np.array([0., 0, 0])
    delta_quat = tr.quaternion_about_axis(angle=np.pi / 16, axis=np.array([1, 0, 0]))
    target_pos_i = target_pos + delta_pos
    target_quat_i = tr.quaternion_multiply(delta_quat, target_quat)
    target_pose_i = np.concatenate([target_pos_i, target_quat_i])
    med.raw_cartesian_command(target_pose_i, ref_frame='med_base', frame_id='grasp_frame')
    _ = input('press enter to continue')


if __name__ == '__main__':
    raw_cartesian_test()
    # raw_cartesian_impedance_grasp_frame_test()