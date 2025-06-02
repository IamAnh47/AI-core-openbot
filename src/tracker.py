# src/tracker.py
from deep_sort_realtime.deepsort_tracker import DeepSort
from config import MAX_AGE, MIN_HITS, IOU_THRESHOLD_TRACKER

class PersonTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=MAX_AGE,
            n_init=MIN_HITS,
            max_iou_distance=IOU_THRESHOLD_TRACKER
        )

    def update(self, detections, frame):
        """
        detections: list dict {class_id, confidence, bbox=[x, y, w, h]}
        frame: ảnh BGR (numpy array)
        Trả về: list dict {track_id, class_id, confidence, bbox=[x, y, w, h]}
        """
        ds_input = []
        for det in detections:
            x, y, w, h = det["bbox"]
            conf       = det["confidence"]
            cls_id     = det["class_id"]
            # DeepSORT-Realtime cần bbox là list/tuple 4 phần tử, rồi confidence và class_id
            ds_input.append(([x, y, w, h], conf, cls_id))

        tracks = self.tracker.update_tracks(ds_input, frame=frame)
        tracked_objects = []
        for trk in tracks:
            if not trk.is_confirmed():
                continue

            tid = trk.track_id
            # Lấy bounding box từ track (ltrb → [l, t, r, b])
            l, t, r, b = [int(v) for v in trk.to_ltrb()]

            # Lấy confidence, bỏ qua nếu None
            conf_val = None
            if hasattr(trk, 'det_confidence'):
                conf_val = trk.det_confidence
            elif hasattr(trk, 'det_conf'):
                conf_val = trk.det_conf

            # Nếu conf_val là None, gán 0.0
            if conf_val is None:
                conf = 0.0
            else:
                conf = float(conf_val)

            cls_id = trk.get_det_class()  # class_id gốc, ở đây hiện tại luôn = 0 vì chỉ track person, tui sẽ update sau

            tracked_objects.append({
                "track_id": tid,
                "class_id": cls_id,
                "confidence": conf,
                "bbox": [l, t, r - l, b - t]
            })

        return tracked_objects
