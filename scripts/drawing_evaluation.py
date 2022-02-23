#! /usr/bin/env python
import argparse
import rospy

from bubble_control.bubble_data_collection.drawing_evaluation_data_collection import DrawingEvaluationDataCollection

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Collect Data Drawing')
    parser.add_argument('save_path', type=str, help='path to save the data')
    parser.add_argument('num_data', type=int, help='Number of data samples to be collected')
    parser.add_argument('--scene_name', type=str, default='drawing_data', help='scene name for the data. For organization purposes')
    parser.add_argument('--model_name', type=str, default='random', help='code for the model used. can also be "fixed_model" or "random"')
    parser.add_argument('--load_version', type=int, default=0, help='version f for the model used. can also be "fixed_model" or "random"')
    parser.add_argument('--debug', type=bool, default=False, help='Whether or not to visualize model predictions')
    # TODO: Add more parameters if needed

    args = parser.parse_args()

    save_path = args.save_path
    scene_name = args.scene_name
    num_data = args.num_data
    debug = args.debug
    model_name = args.model_name
    load_version = args.load_version

    rospy.init_node('test_evaluation_drawing')

    dc = DrawingEvaluationDataCollection(data_path=save_path, scene_name=scene_name, model_name=model_name, load_version=load_version ,debug=debug)
    dc.collect_data(num_data=num_data)