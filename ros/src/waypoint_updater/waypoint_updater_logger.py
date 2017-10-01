import time
import rospy
import numpy as np

WAYPOINT_UPDATER_LOG_ENABLED = True


class WaypointUpdaterLogger:
    def __init__(self, waypoint_updater, rate):
        self.waypoint_updater = waypoint_updater
        self.logging_interval = 1000 / rate
        self.last_time_logged = self.current_time()

    def log(self, next_waypoint_index, total_distance_func):
        if not self.should_log() or not self.waypoint_updater.track_waypoints.waypoints:
            return

        self.last_time_logged = self.current_time()

        line1 = 'Light: {}'.format('GREEN'
                                   if self.waypoint_updater.stop_waypoint_index == -1
                                   else 'RED')

        line2 = 'stop index: {}; wp: {}'.format(
            self.waypoint_updater.stop_waypoint_index,
            self.waypoint_updater.track_waypoints.waypoints[
                self.waypoint_updater.stop_waypoint_index].pose.pose.position
            if self.waypoint_updater.stop_waypoint_index != -1 else "N/A"
        ).replace('\n', ' ')

        line3 = 'next index: {}; wp: {}'.format(
            next_waypoint_index,
            self.waypoint_updater.track_waypoints.waypoints[
                next_waypoint_index].pose.pose.position
        ).replace('\n', ' ')

        line4 = '      dist: {};'.format(
            total_distance_func(self.waypoint_updater.track_waypoints.waypoints,
                                next_waypoint_index,
                                self.waypoint_updater.stop_waypoint_index)
        ).replace('\n', ' ')

        line0 = '--- waypoint_updater node '
        line0 = line0 + ('-' * (np.max([len(line1), len(line2), len(line3), len(line4)]) - len(line0)))

        if WAYPOINT_UPDATER_LOG_ENABLED:
            rospy.loginfo('')
            rospy.loginfo(line0)
            rospy.loginfo(line1)
            rospy.loginfo(line2)
            rospy.loginfo(line3)
            rospy.loginfo(line4)
            rospy.loginfo('')

    def should_log(self):
        return self.current_time() - self.last_time_logged > self.logging_interval

    @staticmethod
    def current_time():
        return int(round(time.time() * 1000))
