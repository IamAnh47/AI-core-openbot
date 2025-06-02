import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
import argparse
from detector import YoloTfliteDetector, YoloFallbackDetector
from tracker import PersonTracker
from controller import Controller
from path_memory import PathMemory
from robot_interface import RobotInterface
from utils.preprocessor import preprocess_frame
from utils.viz import draw_bboxes
from config import INPUT_SIZE, CONF_THRESHOLD, MAX_LOST_FRAMES

# ----------------- Hàm phụ để tìm idx bắt đầu -----------------
def get_start_index(raw_img_dir):
    """
    Tìm số thứ tự index lớn nhất trong data/raw/images - định dạng imgXXXXX.jpg,
    return index -> index + 1.
    Nếu thư mục trống -> 0.
    """
    all_files = os.listdir(raw_img_dir) if os.path.isdir(raw_img_dir) else []
    max_idx = 0
    for filename in all_files:
        name, ext = os.path.splitext(filename)
        if ext.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        if name.startswith("img"):
            num_part = name[3:]
            if num_part.isdigit():
                idx_val = int(num_part)
                if idx_val > max_idx:
                    max_idx = idx_val
    return max_idx

# ---- Hàm hỗ trợ để tạo thư mục lưu raw data ----
def ensure_raw_dirs():
    os.makedirs("data/raw/images", exist_ok=True)
    os.makedirs("data/raw/annotations", exist_ok=True)

# ---- Hàm hỗ trợ lưu ảnh + annotation ----
def save_frame_and_label(frame, detections, idx):
    """
    frame: ảnh BGR gốc
    detections: list dict {class_id, confidence, bbox=[x,y,w,h]}
    idx: số thứ tự (int) để sinh tên img{idx:05d}.jpg và .txt
    """
    img_name = f"img{idx:05d}.jpg"
    img_path = os.path.join("data/raw/images", img_name)
    cv2.imwrite(img_path, frame)

    H, W = frame.shape[:2]
    label_lines = []
    for det in detections:
        cls_id = det["class_id"]
        x, y, w_bbox, h_bbox = det["bbox"]
        x_center = (x + w_bbox / 2) / W
        y_center = (y + h_bbox / 2) / H
        w_norm = w_bbox / W
        h_norm = h_bbox / H
        label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    label_name = f"img{idx:05d}.txt"
    label_path = os.path.join("data/raw/annotations", label_name)
    with open(label_path, "w") as f:
        for line in label_lines:
            f.write(line + "\n")

# ---- Chức năng Data Collection ----
def run_data_collection(args):
    """
    args: chứa source, collect_duration, collect_freq
    Khi bắt đầu, hàm kiểm tra data/raw/images/, tìm idx max,
    rồi tiếp tục lưu từ idx+1 để không ghi đè.
    """
    src = 0 if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở nguồn {args.source}")
        return

    # Khởi detector
    try:
        detector = YoloTfliteDetector()
        print("[INFO] Dùng TFLite detector")
    except Exception as e:
        print(f"[WARN] TFLite load lỗi: {e}")
        detector = YoloFallbackDetector()
        print("[INFO] Dùng Ultralytics fallback")

    # Đảm bảo thư mục tồn tại và xác định idx bắt đầu
    ensure_raw_dirs()
    raw_img_dir = "data/raw/images"
    start_idx = get_start_index(raw_img_dir)
    if start_idx > 0:
        print(f"[INFO] Tìm thấy {start_idx} ảnh đã có sẵn → Tiếp tục từ idx = {start_idx + 1}")
    else:
        print("[INFO] Chưa có ảnh raw nào → Bắt đầu từ idx = 1")

    print(f"[INFO] Bắt đầu thu thập trong {args.collect_duration} giây …")
    start_time = time.time()
    idx = start_idx
    next_capture = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        elapsed = now - start_time
        if elapsed > args.collect_duration:
            break

        # Detect để lấy bounding box làm annotation
        detections = detector.detect(frame)

        if now >= next_capture:
            idx += 1
            save_frame_and_label(frame, detections, idx)
            print(f"[SAVED] img{idx:05d}.jpg + .txt (Tổng: {idx})")
            next_capture = now + 1.0 / args.collect_freq

        # Hiển thị realtime (tùy chọn :(( )
        for det in detections:
            x, y, w_bbox, h_bbox = det["bbox"]
            cv2.rectangle(frame, (x, y), (x + w_bbox, y + h_bbox), (0, 255, 0), 2)
        cv2.putText(frame, f"Recording: {int(elapsed)}/{args.collect_duration}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Data Collection Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Thu thập hoàn tất.")

# ---- Chức năng chính pipeline detect -> track→control ----
def run_main_loop(args):
    src = 0 if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở nguồn {args.source}")
        return

    try:
        detector = YoloTfliteDetector()
        print("[INFO] Dùng TFLite detector")
    except Exception as e:
        print(f"[WARN] TFLite load lỗi: {e}")
        detector = YoloFallbackDetector()
        print("[INFO] Dùng Ultralytics fallback")

    tracker    = PersonTracker()
    controller = Controller(frame_size=(640, 480))
    path_mem   = PathMemory()
    robot      = RobotInterface()

    target_id    = None
    lost_counter = 0

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracked_objs = tracker.update(detections, frame)

        if target_id is None or lost_counter > MAX_LOST_FRAMES:
            persons = [o for o in tracked_objs if o["class_id"] == 0]
            if persons:
                best = max(persons, key=lambda x: x["bbox"][2] * x["bbox"][3])
                target_id = best["track_id"]
                lost_counter = 0
                path_mem.clear()

        bbox_target = None
        obstacles   = []
        for o in tracked_objs:
            if o["class_id"] == 0 and o["track_id"] == target_id:
                bbox_target = o["bbox"]
            elif o["class_id"] != 0:
                obstacles.append(o["bbox"])

        if bbox_target:
            lost_counter = 0
            linear_vel, angular_vel = controller.compute(bbox_target, obstacles)
            robot.send(linear_vel, angular_vel)
        else:
            lost_counter += 1
            if lost_counter > MAX_LOST_FRAMES:
                wp = path_mem.pop_waypoint()
                if wp:
                    robot.send(0.0, 0.0)
                else:
                    robot.send(0.0, 0.0)
            else:
                robot.send(0.0, 0.0)

        draw_bboxes(frame, tracked_objs, target_id)
        fps = 1.0 / (time.time() - t0)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Robot AI Core", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    robot.close()
    cap.release()
    cv2.destroyAllWindows()

# ---- Hàm main, chọn chế độ collect hoặc chạy bình thường - trong README có hướng dẫn ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collect", action="store_true",
        help="Bật chế độ thu thập dữ liệu: lưu ảnh + annotation vào data/raw/"
    )
    parser.add_argument(
        "--collect_duration", type=int, default=30,
        help="Số giây thu thập khi bật --collect (mặc định 30s)"
    )
    parser.add_argument(
        "--collect_freq", type=float, default=1.0,
        help="Số frame/giây lưu khi thu thập (mặc định 1)"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Nguồn video/camera: '0' = webcam, hoặc đường dẫn file video"
    )
    args = parser.parse_args()

    if args.collect:
        run_data_collection(args)
    else:
        run_main_loop(args)

if __name__ == "__main__":
    main()
