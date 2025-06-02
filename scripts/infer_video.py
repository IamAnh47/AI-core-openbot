import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import time
import argparse
from ultralytics import YOLO
from src.config import CONF_THRESHOLD, INPUT_SIZE
# from cfg.coco import names

def main(video_path, output_path=None, model_path="weights/pretrain/yolov10n.pt"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Không mở được video {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    if output_path:
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()
        results = model(frame, imgsz=INPUT_SIZE, conf=CONF_THRESHOLD)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if cls_id != 0:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            color = (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"person {conf:.2f}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        fps = 1.0 / (time.time() - t0)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if out:
            out.write(frame)
        cv2.imshow("Infer Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, help="Path to save output video (optional)")
    parser.add_argument("--model",  type=str, default="weights/pretrain/yolov10n.pt", help="YOLO model .pt")
    args = parser.parse_args()
    main(args.video, args.output, args.model)
