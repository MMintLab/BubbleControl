#! /usr/bin/env python
from bubble_control.bubble_data_collection.bubble_data_collection import BubbleCalibrationDataCollection

def data_collection_test():
    bc = BubbleCalibrationDataCollection()
    bc.collect_data(2)

if __name__ == '__main__':
    data_collection_test()