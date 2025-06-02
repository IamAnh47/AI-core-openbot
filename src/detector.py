import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from config import (
    TFLITE_MODEL_PATH,
    INPUT_SIZE,
    CONF_THRESHOLD,
    COCO_NAMES_PATH
)

class YoloTfliteDetector:
    def __init__(
        self,
        model_path=TFLITE_MODEL_PATH,
        input_size=INPUT_SIZE,
        conf_thres=CONF_THRESHOLD
    ):
        self.input_size = input_size
        self.conf_thres = conf_thres

        #Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        #Lấy thông tin input/output index
        input_details  = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.input_index  = input_details[0]['index']
        self.output_index = output_details[0]['index']

        #Load class names (COCO 80 classes)
        with open(COCO_NAMES_PATH, 'r') as f:
            self.class_names = [x.strip() for x in f.readlines()]

    def preprocess(self, frame):
        """
        Resize + normalize hình (BGR -> RGB, scale 0–1),
        result: numpy array shape (1, input_size, input_size, 3), float32
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)
        return img

    def postprocess(self, preds, original_shape):
        """
        preds: numpy array shape (1, N, 6)
               Mỗi row: [x_center_norm, y_center_norm, w_norm, h_norm, conf, cls_id]
        original_shape: (H, W, 3)
        Trả về list các dict: {class_id, confidence, bbox: [x, y, w, h]}
        """
        H, W = original_shape[:2]
        detections = []

        for det in preds[0]:
            x, y, w, h, conf, cls = det
            if conf < self.conf_thres:
                continue
            cls_id = int(cls)
            # Chỉ lọc class “person”
            if cls_id != 0:
                continue

            # Convert normalized bbox -> pixel coords
            cx = x * W
            cy = y * H
            bw = w * W
            bh = h * H
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)

            detections.append({
                "class_id": cls_id,
                "confidence": float(conf),
                "bbox": [x1, y1, int(bw), int(bh)]
            })
        return detections

    def detect(self, frame):
        """
        Input: frame BGR (numpy array)
        Output: list dict {class_id, confidence, bbox: [x, y, w, h]}
        """
        # 1. Tiền xử lý
        input_tensor = self.preprocess(frame)

        # 2. Chạy inference
        self.interpreter.set_tensor(self.input_index, input_tensor)
        self.interpreter.invoke()

        # 3. Lấy output (Ultralytics export TFLite cho YOLOv10-N chỉ có 1 output)
        preds = self.interpreter.get_tensor(self.output_index)

        # 4. Postprocess
        return self.postprocess(preds, frame.shape)

class YoloFallbackDetector:
    """
    Nếu TFLite load lỗi, fallback dùng API Ultralytics Python.
    Update: Hiện đang lỗi TFLite nên phải dùng API
    """
    def __init__(self, model_path="yolov10n.pt", conf_thres=CONF_THRESHOLD, img_size=INPUT_SIZE):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.img_size = img_size

        with open(COCO_NAMES_PATH, 'r') as f:
            self.class_names = [x.strip() for x in f.readlines()]

    def detect(self, frame):
        """
        Input: frame BGR
        Output: list dict {class_id, confidence, bbox}
        """
        results = self.model(frame, imgsz=self.img_size, conf=self.conf_thres)[0]
        detections = []
        for b in results.boxes:
            cls_id = int(b.cls[0])
            conf   = float(b.conf[0])
            if cls_id != 0:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            w = int(x2 - x1)
            h = int(y2 - y1)
            detections.append({
                "class_id": cls_id,
                "confidence": conf,
                "bbox": [int(x1), int(y1), w, h]
            })
        return detections
