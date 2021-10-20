#! /usr/bin/env python
import rospy

from bubble_utils.src.bubble_utils.bubble_data_collection.bubble_data_collection_base import BubbleDataCollectionBase


# TEST THE CODE: ------------------------------------------------------------------------------------------------------

class BubbleManualDataCollection(BubbleDataCollectionBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_undeformed_fc = None

    def _record_gripper_calibration(self):
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

    def collect_data(self, num_data):
        print('Calibration undeformed state, please follow the instructions')
        self._record_gripper_calibration()
        out = super().collect_data(num_data)
        return out

    def _get_legend_column_names(self):
        column_names = ['Scene', 'UndeformedFC', 'StateFC', 'Object', 'GraspForce']
        return column_names

    def _get_legend_lines(self, data_params):
        legend_lines = []
        state_fc_i = data_params['state_fc']
        object_i = data_params['object']
        grasp_force_i = data_params['grasp_force']
        scene_i = self.scene_name
        line_i = [scene_i, self.last_undeformed_fc, state_fc_i, object_i, grasp_force_i]
        legend_lines.append(line_i)
        return legend_lines

    def _collect_data_sample(self, params=None):
        """
        Adjust the robot so the object has a constant pose (target_pose) in the reference ref_frame
        returns:
            data_params: <dict> containing the parameters and information of the collected data
        """
        data_params = {}

        object_name = input("Enter object name after moving into contact with it")
        # Sample drawing parameters:
        state_fc = self.get_new_filecode()

        grasp_force_i = 0  # TODO: read grasp force

        # record init state:
        self._record(fc=state_fc)

        # raise the arm at the end
        # Raise the arm when we reach the last point
        data_params['state_fc'] = state_fc
        data_params['object'] = object_name
        data_params['grasp_force'] = grasp_force_i

        return data_params


def collect_data_drawing_test(supervision=False, reactive=False):
    save_path = '/home/mmint/Desktop/ycb_single_bubble'
    scene_name = 'first'

    dc = BubbleManualDataCollection(data_path=save_path, scene_name=scene_name, left=False)
    dc.collect_data(num_data=20)


if __name__ == '__main__':
    collect_data_drawing_test()
