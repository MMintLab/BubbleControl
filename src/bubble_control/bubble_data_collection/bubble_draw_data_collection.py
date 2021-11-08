import abc

import numpy as np
import rospy
from collections import OrderedDict
import gym
import copy

from bubble_utils.bubble_data_collection.bubble_data_collection_base import BubbleDataCollectionBase
from bubble_control.bubble_drawer.bubble_drawer import BubbleDrawer


class BubbleDrawingDataCollectionBase(BubbleDataCollectionBase):

    def __init__(self, *args, **kwargs):
        self.action_space = self._get_action_space()
        super().__init__(*args, **kwargs)
        self.last_undeformed_fc = None


    @abc.abstractmethod
    def _get_action_space(self):
        pass

    def _get_med(self):
        med = BubbleDrawer(object_topic='estimated_object', wrench_topic='/med/wrench', force_threshold=5., reactive=False) # TODO: Pass these parameters to the consturctor
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

    def collect_data(self, num_data):
        print('Calibration undeformed state, please follow the instructions')
        self._record_gripper_calibration()
        out = super().collect_data(num_data)
        self.med.home_robot()
        return out

    def _get_legend_column_names(self):
        action_keys = self._sample_action().keys()
        column_names = ['Scene', 'UndeformedFC', 'InitialStateFC',  'FinalStateFC', 'GraspForce'] + list(action_keys)
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
        line_i = [scene_i, self.last_undeformed_fc, init_fc_i, final_fc_i,  grasp_force_i] + action_values
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

    def _collect_data_sample(self, params=None):
        """
        Adjust the robot so the object has a constant pose (target_pose) in the reference ref_frame
        returns:
            data_params: <dict> containing the parameters and information of the collected data
        """
        data_params = {}

        # Sample drawing parameters:
        action_i = self._sample_action()
        start_point_i = action_i['start_point']
        end_point_i = action_i['end_point']
        grasp_width_i = action_i['grasp_width']
        # Sample the fcs:
        init_fc = self.get_new_filecode()
        final_fc = self.get_new_filecode()

        grasp_force_i = 0  # TODO: read grasp force

        # Set the grasp width
        self._do_pre_action(action_i)

        # record init state:
        self._record(fc=init_fc)

        # draw
        self._do_action(action_i)

        # record final_state
        self._record(fc=final_fc)

        self._do_post_action(action_i)

        data_params['initial_fc'] = init_fc
        data_params['final_fc'] = final_fc
        data_params['grasp_force'] = grasp_force_i
        data_params['action'] = action_i

        return data_params


class AxisBiasedDirectionSpace(gym.spaces.Space):
    """
    Saple space between [0,2pi) with bias towards the axis directions.
    On prob_axis, the sample will be along one of the cartesian axis directions, i.e. [0, pi/2, pi, 3pi/2]
    """
    def __init__(self, prob_axis, seed=None):
        self.prob_axis = prob_axis
        super().__init__((), np.float32, seed)

    def sample(self):
        p_axis_direction = self.np_random.random() # probability of getting an axis motion
        if p_axis_direction < self.prob_axis:
            direction_i = 0.5 * np.pi * np.random.randint(0, 4) # axis direction (0, pi/2, pi/ 3pi/2)
        else:
            direction_i = np.random.uniform(0, 2 * np.pi)  # direction as [0, 2pi)
        return direction_i

    def contains(self, x):
        return 0 <= x <= 2*np.pi


class ConstantSpace(gym.spaces.Space):
    """
    Constant space. Only has one possible value. For convenience.
    """
    def __init__(self, value, seed=None):
        self.value = value
        super().__init__((), np.float32, seed)

    def sample(self):
        return self.value

    def contains(self, x):
        return x == self.value

    def __eq__(self, other):
        return (
                isinstance(other, ConstantSpace)
                and self.value == other.value
        )


class BubbleDrawingDataCollection(BubbleDrawingDataCollectionBase):

    def __init__(self, *args, **kwargs):
        self.previous_end_point = None
        self.previous_draw_height = None
        super().__init__(*args, **kwargs)

    def _get_action_space(self):

        drawing_area_center_point = np.array([0.55, 0.])
        drawing_area_size = np.array([.1, .1])

        action_space_dict = OrderedDict()
        action_space_dict['lift'] = gym.spaces.MultiBinary(1)
        action_space_dict['start_point'] = gym.spaces.Box(drawing_area_center_point - drawing_area_size,
                                          drawing_area_center_point + drawing_area_size, (2,)) # random uniform
        action_space_dict['end_point'] = ConstantSpace(0)  # placeholder
        action_space_dict['direction'] = AxisBiasedDirectionSpace(prob_axis=0.08)
        action_space_dict['length'] = gym.spaces.Box(low=0.01, high=0.15, shape=())
        action_space_dict['grasp_width'] = ConstantSpace(20.)

        action_space = gym.spaces.Dict(action_space_dict)
        return action_space

    def _sample_action(self):
        action_sampled = super()._sample_action()
        lift = action_sampled['lift']
        if self.previous_end_point is None:
            action_sampled['lift'][0] = 1
        elif lift == 1:
            action_sampled['start_point'] = copy.deepcopy(self.previous_end_point)
        start_point_i = action_sampled['start_point']
        length_i = action_sampled['length']
        direction_i = action_sampled['direction']
        action_sampled['end_point'] = start_point_i + length_i * np.array([np.cos(direction_i), np.sin(direction_i)])
        self.previous_end_point = copy.deepcopy(action_sampled['end_point'])
        return action_sampled

    def _do_pre_action(self, action):
        start_point_i = action['start_point']
        grasp_width_i = action['grasp_width']
        lift = action['lift']
        self.med.gripper.move(grasp_width_i, 10.0)
        # Init the drawing
        if lift == 1:
            self.med._end_raise(start_point_i)
            draw_height = self.med._init_drawing(start_point_i)
            self.previous_draw_height = copy.deepcopy(draw_height)
        else:
            draw_height = self.previous_draw_height
        action['draw_height'] = draw_height # Override action to add draw_height so it is available at _do_action

    def _do_action(self, action):
        end_point_i = action['end_point']
        draw_height = action['draw_height']
        self.med._draw_to_point(end_point_i, draw_height)


    def _do_post_action(self, action):
        end_point_i = action['end_point']
        # raise the arm at the end
        # Raise the arm when we reach the last point
