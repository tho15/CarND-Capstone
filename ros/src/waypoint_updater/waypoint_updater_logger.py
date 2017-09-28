import time
import rospy
import numpy as np


class WaypointUpdaterLogger:
    def __init__(self, waypoint_updater, rate):
        self.waypoint_updater = waypoint_updater
        self.logging_interval = 1000 / rate
        self.last_time_logged = self.current_time()

    def log(self):
        if not self.should_log() or not self.waypoint_updater.track_waypoints.waypoints:
            return

        self.last_time_logged = self.current_time()

        line1 = 'stop waypoint index: {}; stop waypoint: {}'.format(
            self.waypoint_updater.stop_waypoint_index,
            self.waypoint_updater.track_waypoints.waypoints[
                self.waypoint_updater.stop_waypoint_index].pose.pose.position
        ).replace('\n', ' ')

        line0 = '--- waypoint_updater node '
        line0 = line0 + ('-' * (np.max([len(line1)]) - len(line0)))

        rospy.loginfo('')
        rospy.loginfo(line0)
        rospy.loginfo(line1)
        rospy.loginfo('')

    def should_log(self):
        return self.current_time() - self.last_time_logged > self.logging_interval

    @staticmethod
    def current_time():
        return int(round(time.time() * 1000))
