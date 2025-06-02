import cv2

def draw_bboxes(frame, tracked_objs, target_id=None):
    """
    Vẽ bounding boxes và ID lên frame:
      - Person với ID = target_id vẽ màu xanh lá
      - Các person khác vẽ màu xanh dương
      - Obstacles (class != 0) vẽ màu đỏ
    tracked_objs: list dict {track_id, class_id, confidence, bbox=[x,y,w,h]}
    """
    for o in tracked_objs:
        x, y, w, h = o["bbox"]
        tid = o["track_id"]
        if o["class_id"] == 0:
            color = (0, 255, 0) if tid == target_id else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"ID{tid}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

def draw_path(frame, waypoints):
    """
    Vẽ đường đi - waypoints lên frame (giả lập - do chưa đưa ra thực tế),
    waypoints: list of (x, y, theta) giả lập pixel coords
    """
    for wp in waypoints:
        x, y, _ = wp
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
