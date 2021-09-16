import os
import sys
import numpy as np
import threading
import copy
import rospy
import tf
import tf.transformations as tr
import mmint_utils

from arm_robots.med import Med
from mmint_camera_utils.point_cloud_parsers import PicoFlexxPointCloudParser
from mmint_camera_utils.tf_recording import save_tfs

from victor_hardware_interface.victor_utils import Stiffness
from victor_hardware_interface_msgs.msg import ControlMode, MotionStatus

from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Vector3
from control_msgs.msg import FollowJointTrajectoryFeedback
from visualization_msgs.msg import Marker
from bubble_control.bubble_data_collection.data_collector_base import DataCollectorBase

from bubble_control.bubble_data_collection.bubble_data_collection_base import BubbleDataCollectionBase
from bubble_control.bubble_drawer.bubble_drawer import BubbleDrawer

# TODO: Fix the filecode

class BubbleDrawingDataCollection(BubbleDataCollectionBase):

    def __init__(self, *args, **kwargs):
        self.bubble_drawer = BubbleDrawer(object_topic='estimated_object', wrench_topic='/med/wrench', force_threshold=5., reactive=False) # TODO: Pass these parameters to the consturctor
        super().__init__(*args, **kwargs)
        self.last_undeformed_fc = None

    def _record_gripper_calibration(self):
        self.bubble_drawer.set_grasp_pose()
        _ = input('Press enter to open the gripper and calibrate the bubbles')
        self.med.open_gripper()
        self._record(fc=self.filecode)
        self.last_undeformed_fc = self.filecode
        self.filecode += 1
        _ = input('Press enter to close the gripper')
        self.med.set_grasping_force(5.0)
        self.med.gripper.move(25.0)
        self.med.grasp(20.0, 30.0)
        rospy.sleep(2.0)
        print('Calibration is done')
        self.bubble_drawer.home_robot()

    def collect_data(self, num_data):
        print('Calibration undeformed state, please follow the instructions')
        self._record_gripper_calibration()
        out = super().collect_data(num_data)
        self.bubble_drawer.home_robot()
        return out

    def _get_med(self):
        return self.bubble_drawer.med

    def _get_legend_column_names(self):
        column_names = ['UndeformedFC', 'InitialStateFC',  'FinalStateFC', 'GraspForce', 'GraspWidth', 'Action', 'Scene']
        return column_names

    def _get_legend_lines(self, data_params):
        legend_lines = []
        for i, init_fc_i in enumerate(data_params['initial_fc']):
            final_fc_i = data_params['final_fc'][i]
            grasp_force_i = data_params['grasp_force'][i]
            grasp_width_i = data_params['grasp_width'][i]
            action_i = data_params['action']
            scene_i = self.scene_name
            line_i = [self.last_undeformed_fc, init_fc_i, final_fc_i,  grasp_force_i, grasp_width_i, action_i, scene_i]
            legend_lines.append(line_i)
        return legend_lines

    def _collect_data_sample(self, params=None):
        """
        Adjust the robot so the object has a constant pose (target_pose) in the reference ref_frame
        returns:
            data_params: <dict> containing the parameters and information of the collected data
        """
        data_params_keys = ['initial_fc', 'final_fc', 'grasp_force', 'grasp_width', 'action']
        data_params = dict(zip(data_params_keys, [[] for x in data_params_keys]))

        # Sample drawing parameters:
        drawing_area_center_point = np.array([0.55, 0.])
        drawing_area_size = np.array([.1, .1])
        start_point_i = np.random.uniform(drawing_area_center_point - drawing_area_size,
                                          drawing_area_center_point + drawing_area_size, (2,))
        direction_i = np.random.uniform(0, 2 * np.pi)  # assume planar motion only
        length_i = np.random.uniform(0.01, 0.1)
        end_point_i = start_point_i + length_i * np.array([np.cos(direction_i), np.sin(direction_i)])
        grasp_width_i = 20
        # Sample the fcs:
        init_fc = 2*self.filecode - 1
        final_fc = 2*self.filecode

        action_i = {'start_point': start_point_i,
                    'direction': direction_i,
                    'length': length_i,
                    }

        # Set the grasp width
        self.med.gripper.move(grasp_width_i, 10.0)

        # TODO: read grasp force
        grasp_force_i = 0

        # Init the drawing
        draw_height = self.bubble_drawer._init_drawing(start_point_i)
        # record init state:
        self._record(fc=init_fc)
        # draw
        self.bubble_drawer._draw_to_point(end_point_i, draw_height)

        # record final_state
        self._record(fc=final_fc)

        # raise the arm at the end
        # Raise the arm when we reach the last point
        self.bubble_drawer._end_raise(end_point_i)
        data_params['initial_fc'].append(init_fc)
        data_params['final_fc'].append(final_fc)
        data_params['grasp_force'].append(grasp_force_i)
        data_params['grasp_width'].append(grasp_width_i)
        data_params['action'].append(action_i)

        return data_params