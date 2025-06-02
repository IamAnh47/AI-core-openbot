import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path tới model TFLite (inference)
# TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "weights", "yolov10n.tflite")
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "weights", "yolov10n.pt")
# Path tới file danh sách class COCO
COCO_NAMES_PATH   = os.path.join(BASE_DIR, "cfg", "coco.names")

# Thresholds cho detect
CONF_THRESHOLD   = 0.4
NMS_THRESHOLD    = 0.3  # (không dùng NMS nếu YOLOv10-N, nhưng giữ để nếu cần)

# Kích thước input cho YOLO TFLite
INPUT_SIZE       = 320

# Cấu hình tracker (DeepSORT)
MAX_AGE               = 15
MIN_HITS              = 3
IOU_THRESHOLD_TRACKER = 0.3

# Cấu hình Controller
KP_PERSON    = 0.005   # Hệ số điều khiển ngang (theo người)
KP_OBS       = 0.01    # Hệ số điều khiển né chướng ngại

MAX_SPEED    = 1.0     # Đơn vị m/s (tùy robot)
MIN_SPEED    = 0.2

# Cấu hình path memory
WAYPOINT_INTERVAL = 0.2   # (m) robot di chuyển bao nhiêu thì lưu 1 waypoint
MAX_WAYPOINTS     = 5000

# Khi mất target quá bao nhiêu frame thì bắt đầu xử lý tìm lại
MAX_LOST_FRAMES = 30

# Robot interface (SERIAL_PORT=None sẽ in ra console)
SERIAL_PORT = None
BAUDRATE    = 115200
