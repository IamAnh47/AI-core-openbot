import math
from config import WAYPOINT_INTERVAL, MAX_WAYPOINTS

class PathMemory:
    def __init__(self):
        self.waypoints = []
        self.last_pose = None

    def update_pose(self, current_pose):
        """
        current_pose: tuple (x, y, theta) (giả lập hoặc từ odometry/SLAM)
        Lưu waypoint nếu cách last_pose >= WAYPOINT_INTERVAL.
        """
        if self.last_pose is None:
            self.waypoints.append(current_pose)
            self.last_pose = current_pose
            return

        x0, y0, _ = self.last_pose
        x1, y1, _ = current_pose
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist >= WAYPOINT_INTERVAL:
            if len(self.waypoints) >= MAX_WAYPOINTS:
                self.waypoints.pop(0)
            self.waypoints.append(current_pose)
            self.last_pose = current_pose

    def pop_waypoint(self):
        if self.waypoints:
            return self.waypoints.pop()
        return None

    def clear(self):
        self.waypoints.clear()
        self.last_pose = None
