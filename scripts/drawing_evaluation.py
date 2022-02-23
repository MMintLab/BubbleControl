#! /usr/bin/env python
import argparse
import rospy

from bubble_control.bubble_data_collection.drawing_evaluation_data_collection import DrawingEvaluationDataCollection

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Collect Data Drawing')
    parser.add_argument('save_path', type=str, help='path to save the data')
    parser.add_argument('num_data', type=int, help='Number of data samples to be collected')
    parser.add_argument('--scene_name', type=str, default='drawing_data', help='scene name for the data. For organization purposes')
    parser.add_argument('--random_action', action='store_true', help='impedance mode')
    parser.add_argument('--fixed_model', action='store_true', help='reactive mode -- adjust tool position to be straight when we start drawing')
    parser.add_argument('--debug', type=bool, default=False, help='Whether or not to visualize model predictions')
    parser.add_argument('--object_name', type=str, default='marker', help='name of the object')
    # TODO: Add more parameters if needed

    args = parser.parse_args()

    save_path = args.save_path
    scene_name = args.scene_name
    num_data = args.num_data
    random_action = args.random_action
    fixed_model = args.fixed_model
    debug = args.debug

    rospy.init_node('test_evaluation_drawing')

    dc = DrawingEvaluationDataCollection(data_path=save_path, scene_name=scene_name, fixed_model=fixed_model, random_action=random_action, debug=debug)
    dc.collect_data(num_data=num_data)