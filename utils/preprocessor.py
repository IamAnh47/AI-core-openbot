import cv2

def preprocess_frame(frame, target_size=(640, 480)):
    """
    Resize frame về target_size (W, H).
    Nếu cần normalize hoặc thêm chức năng khác, bổ sung ở đây - hiện chưa xác định nên chuaw bổ sung gì cả
    """
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    if (w, h) != (target_w, target_h):
        frame = cv2.resize(frame, (target_w, target_h))
    return frame
