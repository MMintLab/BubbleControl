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