# Chưa hoàn thiện nên mn không ccần để ý
import numpy as np
from config import KP_PERSON, KP_OBS, MAX_SPEED, MIN_SPEED

class Controller:
    def __init__(self, frame_size=(640, 480)):
        self.frame_w, self.frame_h = frame_size

    def compute(self, bbox_person, obstacles=None):
        """
        bbox_person: [x, y, w, h]
        obstacles: list các [x, y, w, h]
        Trả về: (linear_vel, angular_vel)
        """
        x, y, w, h = bbox_person
        cx = x + w / 2

        # 1. Theo người: nếu người lệch trái/ phải
        error_p = cx - self.frame_w / 2
        ang_p   = KP_PERSON * error_p

        # 2. Né obstacle
        ang_o = 0
        lin_o = 0
        if obstacles:
            angles = []
            for obs in obstacles:
                ox, oy, ow, oh = obs
                ocx = ox + ow / 2
                error_o = (self.frame_w / 2) - ocx
                angles.append(KP_OBS * error_o)
                lin_o = -0.2  # lùi/giam toc khi obstacle
            if angles:
                ang_o = np.mean(angles)

        # 3. Xác định tốc độ
        linear_vel = MAX_SPEED
        if lin_o != 0:
            linear_vel = lin_o

        # Nếu người quá gần (bbox quá lớn) → giảm tốc
        if w > (self.frame_w * 0.5):
            linear_vel = MIN_SPEED

        angular_vel = ang_p + ang_o

        # Hạn chế giá trị
        angular_vel = np.clip(angular_vel, -0.5, 0.5)
        linear_vel  = np.clip(linear_vel, -MAX_SPEED, MAX_SPEED)

        return float(linear_vel), float(angular_vel)
