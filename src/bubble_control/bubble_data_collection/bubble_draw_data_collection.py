import abc

import numpy as np
import rospy
from collections import OrderedDict
import gym
import copy
import tf.transformations as tr

from arc_utilities.listener import Listener
from bubble_utils.bubble_data_collection.bubble_data_collection_base import BubbleDataCollectionBase
from bubble_control.bubble_drawer.bubble_drawer import BubbleDrawer
from bubble_control.aux.action_sapces import ConstantSpace, AxisBiasedDirectionSpace, QuaternionSpace
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Bool, Header
from mmint_camera_utils.ros_utils.publisher_wrapper import PublisherWrapper


class BubbleDrawingDataCollectionBase(BubbleDataCollectionBase):

    def __init__(self, *args, impedance_mode=False, reactive=False, force_threshold=4., max_searches=2000, **kwargs):
        self.impedance_mode = impedance_mode
        self.reactive = reactive
        self.force_threshold = force_threshold
        self.action_space = self._get_action_space()
        super().__init__(*args, **kwargs)
        self.aux_model_pc_publisher = PublisherWrapper(topic_name='aux_model_pc', msg_type=PointCloud2)
        self.model_listener = Listener('contact_model_pc', PointCloud2)
        self.tool_detected_listener = Listener('tool_detected', Bool)
        self.max_searches = max_searches
        self.last_undeformed_fc = None

    @abc.abstractmethod
    def _get_action_space(self):
        pass

    def _get_med(self):
        med = BubbleDrawer(object_topic='estimated_object',
                           wrench_topic='/med/wrench',
                           force_threshold=self.force_threshold,
                           reactive=self.reactive,
                           impedance_mode=self.impedance_mode)
        med.connect()
        return med

    def _record_gripper_calibration(self):
        self.med.set_grasp_pose()
        _ = input('Press enter to open the gripper and calibrate the bubbles')
        self.med.open_gripper()
        self.last_undeformed_fc = self.get_new_filecode(update_pickle=False)
        self._record(fc=self.last_undeformed_fc)
        _ = input('Press enter to close the gripper')
        self.med.set_grasping_force(5.0)
        self.med.gripper.move(25.0)
        self.med.grasp(20.0, 30.0)
        rospy.sleep(2.0)
        print('Calibration is done')
        self.med.home_robot()

    def _get_model_pc(self):
        model_pc = self.model_listener.get(block_until_data=True)
        model_points = np.array(list(pc2.read_points(model_pc)))
        return model_points

    def collect_data(self, num_data):
        print('Calibration undeformed state, please follow the instructions')
        self._record_gripper_calibration()
        out = super().collect_data(num_data)
        self.med.home_robot()
        return out

    def _get_legend_column_names(self):
        action_keys = self._sample_action().keys()
        column_names = ['Scene', 'UndeformedFC', 'InitialStateFC',  'FinalStateFC', 'GraspForce'] + ['ImpedanceMode', 'Reactive', 'ForceThreshold'] + list(action_keys)
        return column_names

    def _get_legend_lines(self, data_params):
        legend_lines = []
        init_fc_i = data_params['initial_fc']
        final_fc_i = data_params['final_fc']
        grasp_force_i = data_params['grasp_force']
        action_i = data_params['action']
        scene_i = self.scene_name
        action_keys = self._sample_action().keys()
        action_values = [action_i[k] for k in action_keys]
        line_i = [scene_i, self.last_undeformed_fc, init_fc_i, final_fc_i,  grasp_force_i] + [self.impedance_mode, self.reactive, self.force_threshold] + action_values
        legend_lines.append(line_i)
        return legend_lines

    def _sample_action(self):
        action_i = self.action_space.sample()
        return action_i

    def _do_pre_action(self, action):
        pass

    def _do_action(self, action):
        pass

    def _do_post_action(self, action):
        pass

    def _check_valid_action(self, action):
        return True

    def _check_marker_displaced(self):
        return False

    def _collect_data_sample(self, params=None):
        """
        Adjust the robot so the object has a constant pose (target_pose) in the reference ref_frame
        returns:
            data_params: <dict> containing the parameters and information of the collected data
        """
        data_params = {}

        # Sample drawing parameters:
        action_i = None
        
        
        failed = True
        no_verticality = False
        while failed:
            tool = self.tool_detected_listener.get(block_until_data=True).data
            displaced_marker = self._check_marker_displaced()
            regrasped = False
            if not tool or displaced_marker or no_verticality:
                print('Toold detected: ', tool)
                print('Displaced marker: ', displaced_marker)
                print('No verticality: ', no_verticality)
                print('No tool detected or tool displaced, going to grasp pose')
                self.med.set_robot_conf('grasp_conf')
                info_msg = '\n\t>>> We will open the gripper!\t'
                _ = input(info_msg)
                self.med.set_grasping_force(25.0)
                self.med.gripper.open_gripper()
                additional_msg = '\n We will close the gripper to a width {}mm'.format(20.0)
                _ = input(additional_msg)
                self.med.gripper.move(20.0, speed=50.0)
                no_verticality = False
                self.med.home_robot()
                regrasped = True
            
            is_valid_action = False
            i = 0
            while i < self.max_searches and not is_valid_action:
                action_i = self._sample_action(regrasped)
                is_valid_action = self._check_valid_action(action_i)
                i += 1
                # print('Looking for valid action: ', i)
            
            if i ==self.max_searches:
                failed = True
                no_verticality = True
                print('Max searches')
                continue
            # Sample the fcs:
            init_fc = self.get_new_filecode()
            final_fc = self.get_new_filecode()

            grasp_force_i = self.med.gripper.get_force()

            # Set the grasp width
            draw_quat_found, pre_success = self._do_pre_action(action_i)
            if not draw_quat_found:
                print('@' * 20 + '    Initial rotation to fulfill verticality not found    ' + '@' * 20)
                failed = True
                no_verticality = True
                continue 
            if not pre_success:
                print('@' * 20 + '    Pre-action failed    ' + '@' * 20)
                failed = True
                continue                 
            # record init state:
            self._record(fc=init_fc)

            success = self._do_action(action_i)
            if not success:
                print('@' * 20 + '    Action failed    ' + '@' * 20)
                failed = True
                continue    
            # record final_state
            self._record(fc=final_fc)

            self._do_post_action(action_i)

            data_params['initial_fc'] = init_fc
            data_params['final_fc'] = final_fc
            data_params['grasp_force'] = grasp_force_i
            data_params['action'] = action_i
            failed = False

        return data_params


class BubbleDrawingDataCollection(BubbleDrawingDataCollectionBase):

    def __init__(self, *args, prob_axis=0.08, drawing_area_center=(0.55, 0.), drawing_area_size=(.15, .15), max_delta_z =0.01,
                 max_width=25., min_width=15., init_dexterity=0.4, action_dexterity=0.1, drawing_length_limits=(0.01, 0.15),**kwargs):
        self.prob_axis = prob_axis
        self.previous_end_point = None
        self.previous_draw_height = None
        self.drawing_area_center_point = np.asarray(drawing_area_center)
        self.drawing_area_size = np.asarray(drawing_area_size)
        self.drawing_length_limits = drawing_length_limits
        self.init_dexterity = init_dexterity
        self.action_dexterity = action_dexterity
        self.max_delta_z = max_delta_z
        self.max_width = max_width
        self.min_width = min_width
        super().__init__(*args, **kwargs)

    def _get_action_space(self):
        drawing_area_center_point = self.drawing_area_center_point
        drawing_area_size = self.drawing_area_size

        action_space_dict = OrderedDict()
        action_space_dict['lift'] = gym.spaces.MultiBinary(1)
        action_space_dict['start_point'] = gym.spaces.Box(drawing_area_center_point - drawing_area_size,
                                          drawing_area_center_point + drawing_area_size, (2,), dtype=np.float64) # random uniform
        action_space_dict['end_point'] = gym.spaces.Box(drawing_area_center_point - drawing_area_size,
                                          drawing_area_center_point + drawing_area_size, (2,), dtype=np.float64) # placeholder for the limits
        action_space_dict['delta_z'] = gym.spaces.Box(-self.max_delta_z, self.max_delta_z, (1,), dtype=np.float64)
        action_space_dict['direction'] = AxisBiasedDirectionSpace(prob_axis=self.prob_axis)
        action_space_dict['length'] = gym.spaces.Box(low=self.drawing_length_limits[0], high=self.drawing_length_limits[1], shape=())
        action_space_dict['grasp_width'] = gym.spaces.Box(self.min_width, self.max_width, (1,), dtype=np.float64)
        action_space_dict['final_orientation'] = QuaternionSpace(self.action_dexterity)

        action_space = gym.spaces.Dict(action_space_dict)
        return action_space

    def _check_valid_action(self, action):
        end_point = action['end_point']
        final_orientation = action['final_orientation']
        delta_z = action['delta_z']
        # Checking whether it is potentially a drawing motion (it doesn't lift the marker)
        current_pose = self.med.get_current_pose()
        initial_pc = self._get_model_pc()
        # Gripper rotation in grasp frame
        movement_transformation = tr.quaternion_matrix(tr.quaternion_multiply(final_orientation, tr.quaternion_inverse(current_pose[3:])))
        # Move the point cloud to origin, apply rotation and convert back
        tool_frame_tf = self.tf2_listener.get_transform(parent='med_base', child='tool_frame')
        initial_pc_origin = initial_pc - tool_frame_tf[:3,3]
        transformed_pc_gf = initial_pc_origin @ movement_transformation[:3,:3].T
        self.transformed_model_pc = transformed_pc_gf + tool_frame_tf[:3,3]
        self.transformed_model_pc[:, 2] += delta_z
        valid_end_point = end_point in self.action_space['end_point'] # since the end point is computed by ourselves based on direction and length, verify that the end point is within its limits,
        tool_height = min(self.transformed_model_pc[:,2])
        # Publish upcoming pointcloud after rotation
        pc_header = Header()
        pc_header.frame_id = 'world'
        transformed_model_pc_i = pc2.create_cloud_xyz32(pc_header, self.transformed_model_pc)
        self.aux_model_pc_publisher.data = transformed_model_pc_i
        valid_rot = tool_height < 0.002 and tool_height > -0.002
        if valid_rot:
            print('Tool height: ', tool_height)
        return valid_end_point and valid_rot

    def _sample_action(self, regrasped=False):
        action_sampled = super()._sample_action()
        lift = action_sampled['lift']
        if self.previous_end_point is None or regrasped:
            action_sampled['lift'][0] = 1
        elif lift == 0:
            # if we do not lift, start at the end point
            action_sampled['start_point'] = copy.deepcopy(self.previous_end_point)
        start_point_i = action_sampled['start_point']
        length_i = action_sampled['length']
        direction_i = action_sampled['direction']
        action_sampled['end_point'] = start_point_i + length_i * np.array([np.cos(direction_i), np.sin(direction_i)])
        return action_sampled

    def _check_marker_displaced(self):
        tool_frame_tf = self.tf2_listener.get_transform(parent='grasp_frame', child='tool_frame')
        tool_axis = tool_frame_tf[:3,:3] @ np.array([0,0,1])
        return np.dot(tool_axis, np.array([0,0,1])) < 0.6


    def _find_draw_quat(self, dext):
        initial_orientation = None
        i = 0
        while i < self.max_searches and (initial_orientation is None or np.dot(tool_axis_wf, np.array([0,0,-1])) < 0.97):
            tool_frame_tf = self.tf2_listener.get_transform(parent='grasp_frame', child='tool_frame')
            yaw = -np.pi/2 + np.random.uniform(-dext, dext)
            roll = np.pi + np.random.uniform(-dext, dext)
            pitch = np.random.uniform(-dext, dext)
            initial_orientation = tr.euler_matrix(roll, pitch, yaw, axes='sxyz')
            tool_axis_wf = initial_orientation[:3,:3] @ tool_frame_tf[:3,:3] @ np.array([0,0,1])
            #print('Dot product: ', np.dot(tool_axis_wf, np.array([0,0,-1])))
            #print('Looking for initial orientation: ', i)
            i += 1
        found = (i != self.max_searches)
        return tr.quaternion_from_matrix(initial_orientation), found
        

    def _do_pre_action(self, action):
        start_point_i = action['start_point']
        grasp_width_i = action['grasp_width'][0]
        lift = action['lift']
        self.med.gripper.move(grasp_width_i, 10.0)
        # Init the drawing
        success = True
        if lift == 1:
            self.med._end_raise()
            draw_quat, draw_quat_found = self._find_draw_quat(self.init_dexterity)
            if not draw_quat_found:
                return False, True
            draw_height, success = self.med._init_drawing(start_point_i, draw_quat)
            if not success:
                return True, False
            self.med.gripper.move(35., 10.0)
            rospy.sleep(1.)
            self.med.gripper.move(grasp_width_i, 10.0)
            # No success if movement was stopped in an elevated position due to force guard 
            success = success and self.med.get_current_pose()[2] < 0.2
            self.previous_draw_height = copy.deepcopy(draw_height)
        else:
            if self.reactive:
                # Adjust the object pose:
                _, success = self.med._adjust_tool_position(start_point_i)
            draw_height = self.previous_draw_height
        action['draw_height'] = draw_height # Override action to add draw_height so it is available at _do_action
        return True, success

    def _do_action(self, action):
        end_point_i = action['end_point']
        draw_height = action['draw_height']
        rotation_i = action['final_orientation']
        delta_z = action['delta_z']
        position_i = np.insert(end_point_i, 2, draw_height+delta_z)
        pose = np.concatenate((position_i, rotation_i))
        success = self.med._draw_to_point_rotation(pose)
        return success


    def _do_post_action(self, action):
        end_point_i = action['end_point']
        self.previous_end_point = copy.deepcopy(action['end_point'])
        # raise the arm at the end
        # Raise the arm when we reach the last point
